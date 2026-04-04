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
        
        # System prompt định hướng vai trò chuyên gia SO SÁNH 2 tài liệu pháp lý
        self.system_prompt = (
            "Bạn là chuyên gia đối chiếu và phân tích văn bản pháp lý tại Việt Nam.\n"
            f"Hệ thống đang làm việc với HAI tài liệu: [BẢN GỐC] = \"{self.old_law_source}\" và [BẢN MỚI] = \"{self.new_law_source}\".\n"
            "Nhiệm vụ duy nhất là SO SÁNH, PHÁT HIỆN SỰ THAY ĐỔI giữa 2 tài liệu dựa HOÀN TOÀN vào trích đoạn được cung cấp.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. CHỈ sử dụng nội dung từ phần \"Trích đoạn pháp lý\" bên dưới. TUYỆT ĐỐI KHÔNG dùng kiến thức ngoài (No Hallucination).\n"
            "2. Nếu trích đoạn KHÔNG đủ thông tin, chỉ trả lời đúng 1 câu: \"Không đủ dữ liệu để so sánh nội dung này.\" rồi DỪNG.\n"
            "3. LUÔN trích dẫn nguồn cụ thể (Tên file, Chương, Điều, Khoản) khi đưa ra nhận định.\n"
            "4. Kết quả so sánh PHẢI phân loại rõ ràng:\n"
            "   - ✅ GIỮ NGUYÊN: Nội dung giống nhau giữa 2 tài liệu\n"
            "   - ➕ THÊM MỚI: Nội dung có trong [BẢN MỚI] nhưng KHÔNG có trong [BẢN GỐC]\n"
            "   - ❌ XÓA BỎ: Nội dung có trong [BẢN GỐC] nhưng KHÔNG có trong [BẢN MỚI]\n"
            "   - 📝 SỬA ĐỔI: Nội dung có trong cả 2 nhưng khác nhau (nêu rõ khác ở điểm nào)\n"
            "5. Nếu KHÔNG CÓ SỰ THAY ĐỔI, chỉ ghi: \"Không có sự thay đổi về [chủ đề].\"\n"
            "6. Trình bày bằng bullet points, ngắn gọn, dễ đọc.\n"
            "7. TUYỆT ĐỐI KHÔNG thêm ghi chú phụ như '(không thay đổi so với BẢN MỚI)', '(không thay đổi so với BẢN GỐC)' vào cuối câu. Nếu nội dung giống nhau, xếp vào mục ✅ GIỮ NGUYÊN là đủ."
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
            source = meta.get("source", "")
            chuong = meta.get("chuong", "")
            muc    = meta.get("muc", "")
            dieu   = meta.get("dieu", "")
            
            # Lọc bỏ các giá trị không xác định trước khi ghép chuỗi vị trí
            _UNKNOWN_TOKENS = {"không rõ", "không xác định", "n/a", "none", ""}
            
            def _is_known(val: str) -> bool:
                return val.strip().lower() not in _UNKNOWN_TOKENS
            
            location_parts = [p for p in [chuong, muc, dieu] if _is_known(p)]
            location_str = " > ".join(location_parts) if location_parts else "Không rõ vị trí"
            
            # Format 1 block thông tin
            block = f"Vị trí: {location_str}\nNội dung văn bản:\n{doc}\n" + "-" * 30
            
            if self.old_law_source and source == self.old_law_source:
                old_law_blocks.append(block)
            elif self.new_law_source and source == self.new_law_source:
                new_law_blocks.append(block)
            else:
                other_blocks.append(f"Nguồn: {source}\n{block}")
                
        context_str = ""
        if old_law_blocks:
            context_str += "=== BẢN GỐC ===\n" + "\n\n".join(old_law_blocks) + "\n\n"
        if new_law_blocks:
            context_str += "=== BẢN MỚI ===\n" + "\n\n".join(new_law_blocks) + "\n\n"
        if other_blocks:
            context_str += "=== TÀI LIỆU KHÁC ===\n" + "\n\n".join(other_blocks) + "\n\n"
        
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
        
    def stream_ask(self, query: str, strategy_name: str = "Normal_v1 (Raw Query)", top_k: int = 6) -> Iterator[str]:
        """Thực hiện luồng RAG và yield text dưới dạng Stream thông qua một Strategy được chọn."""
        from src.query_strategies import STRATEGIES, NormalV1Strategy
        
        # Chọn lớp Strategy (Fallback về NormalV1 nếu string bị sai)
        strategy_class = STRATEGIES.get(strategy_name, NormalV1Strategy)
        strategy_instance = strategy_class()
        
        # Chuyển nhượng phân luồng thực thi cho class Strategy
        for chunk in strategy_instance.stream_execute(query=query, engine=self, top_k=top_k):
            yield chunk
