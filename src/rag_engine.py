from typing import List, Dict, Any, Iterator
from src.embedding.chroma_manager import ChromaManager
from src.generation.llm_client import LLMClient

class LegalRAGEngine:
    """
    Bộ não RAG: Kết hợp truy xuất Database và lập luận LLM để trả lời câu hỏi So sánh Pháp lý.
    """
    def __init__(self, db_manager: ChromaManager, llm_client: LLMClient, old_law_source: str = "", new_law_source: str = ""):
        self.db = db_manager
        self.llm = llm_client
        self.old_law_source = old_law_source
        self.new_law_source = new_law_source
        
        # System prompt định hướng vai trò chuyên gia phân tích luật
        self.system_prompt = (
            "Bạn là một chuyên gia Chuyên môn Pháp lý và Lập pháp xuất sắc tại Việt Nam.\n"
            "Nhiệm vụ của bạn là so sánh, đối chiếu và giải đáp các thắc mắc về các điều khoản luật dựa TRÊN NHỮNG TÀI LIỆU ĐƯỢC CUNG CẤP.\n"
            "BẮT BUỘC TUÂN THỦ CÁC QUY TẮC SAU:\n"
            "1. CHỈ dựa vào phần \"Các Trích Đoạn Pháp Lý\" được cung cấp bên dưới để trả lời.\n"
            "2. Nếu tài liệu cung cấp KHÔNG chứa đủ thông tin để trả lời, HÃY TRẢ LỜI: \"Dựa trên dữ liệu hiện tại, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi này.\", KHÔNG ĐƯỢC tự bịa ra kiến thức (No Hallucination).\n"
            "3. LUÔN LUÔN trích dẫn rõ nguồn (Tên File, Chương, Điều, Khoản) khi bạn đưa ra nhận định hoặc dẫn chứng.\n"
            "4. Khi được yêu cầu so sánh (ví dụ giữa tài liệu cũ và mới), hãy chỉ ra rõ RÀNG cái gì ĐƯỢC GIỮ NGUYÊN, cái gì BỊ XÓA BỎ, và cái gì ĐƯỢC THÊM MỚI.\n"
            "5. Câu trả lời phải khách quan, rõ ràng, trình bày dạng danh sách gạch đầu dòng (bullet points) dễ đọc."
        )

    def _build_context_prompt(self, query: str, search_results: Dict[str, Any]) -> str:
        """
        Lắp ráp kịch bản so sánh gộp chung kết quả từ Database.
        """
        old_law_blocks = []
        new_law_blocks = []
        other_blocks = []
        
        # ChromaDB trả về list of list cho n_results
        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            source = meta.get("source", "Không rõ Nguồn")
            chuong = meta.get("chuong", "Không rõ Chương")
            muc = meta.get("muc", "Không rõ Mục")
            dieu = meta.get("dieu", "Không rõ Điều")
            
            # Format 1 block thông tin
            block = f"Vị trí: {chuong} > {muc} > {dieu}\nNội dung văn bản:\n{doc}\n" + "-" * 30
            
            if self.old_law_source and source == self.old_law_source:
                old_law_blocks.append(block)
            elif self.new_law_source and source == self.new_law_source:
                new_law_blocks.append(block)
            else:
                other_blocks.append(f"Nguồn: {source}\n{block}")
                
        context_str = ""
        if old_law_blocks:
            context_str += "--- BỐI CẢNH LUẬT CŨ (TÀI LIỆU 1) ---\n" + "\n\n".join(old_law_blocks) + "\n\n"
        if new_law_blocks:
            context_str += "--- BỐI CẢNH LUẬT MỚI (TÀI LIỆU 2) ---\n" + "\n\n".join(new_law_blocks) + "\n\n"
        if other_blocks:
            context_str += "--- CÁC TÀI LIỆU KHÁC ---\n" + "\n\n".join(other_blocks) + "\n\n"
        
        final_prompt = (
            "Dưới đây là CÁC TRÍCH ĐOẠN PHÁP LÝ được rút trích có liên quan tới câu hỏi của người dùng:\n\n"
            f"{context_str}\n\n"
            "Câu hỏi hoặc Yêu cầu So sánh của người dùng:\n"
            f'"{query}"\n\n'
            "Dựa trên các trích đoạn trên, hãy phân tích và trả lời câu hỏi chi tiết theo đúng Quy tắc Chuyên gia."
        )
        return final_prompt

    def ask(self, query: str, top_k: int = 5) -> str:
        """Thực hiện một luồng RAG hoàn chỉnh trả về kết quả duy nhất."""
        print(f"[RAG] Đang dò tìm {top_k} phần tử tài liệu liên quan nhất trong CSDL...")
        results = self.db.query(query, n_results=top_k)
        
        print("[RAG] Đang khởi tạo bộ Prompt kết hợp ngữ cảnh...")
        prompt = self._build_context_prompt(query, results)
        
        print(f"[RAG] Đang chờ QA Model {self.llm.model_name} xử lý lập luận pháp lý...")
        answer = self.llm.generate_response(prompt=prompt, system_prompt=self.system_prompt)
        return answer
        
    def stream_ask(self, query: str, top_k: int = 6) -> Iterator[str]:
        """Thực hiện luồng RAG và yield text dưới dạng Stream."""
        results = self.db.query(query, n_results=top_k)
        
        # Log nhanh các nguồn tra cứu được
        docs = results.get("documents", [[]])[0]
        
        if not docs:
            yield "❌ Không tìm thấy văn bản pháp lý nào khớp với dữ liệu trong bộ nhớ."
            return
            
        prompt = self._build_context_prompt(query, results)
        
        # Bắt đầu stream câu trả lời từ LLM
        for chunk in self.llm.stream_response(prompt=prompt, system_prompt=self.system_prompt):
            yield chunk
