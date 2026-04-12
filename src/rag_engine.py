from typing import List, Dict, Any, Iterator
from src.indexing_strategies.base_indexing import BaseIndexingStrategy
from src.generation.llm_client import LLMClient

class LegalRAGEngine:
    """
    Bộ não RAG: Kết hợp truy xuất Database và lập luận LLM để trả lời câu hỏi So sánh Pháp lý.
    """
    def __init__(self, indexing_strategy: BaseIndexingStrategy, llm_client: LLMClient, old_law_source: str = "", new_law_source: str = ""):
        self.indexing_strategy = indexing_strategy
        self.llm = llm_client
        self.old_law_source = old_law_source
        self.new_law_source = new_law_source
        
        # System prompt định hướng vai trò chuyên gia SO SÁNH 2 tài liệu hợp đồng
        old_label = self.old_law_source or "Tài liệu 1"
        new_label = self.new_law_source or "Tài liệu 2"

        self.system_prompt = (
            "Bạn là chuyên gia đối chiếu và phân tích hợp đồng, văn bản pháp lý tại Việt Nam.\n"
            f"Hệ thống đang làm việc với HAI tài liệu:\n"
            f"  • Bản cũ: \"{old_label}\"\n"
            f"  • Bản mới: \"{new_label}\"\n\n"
            "CẤU TRÚC VÀ QUY TẮC TRẢ LỜI BẮT BUỘC:\n"
            "1. TÓM TẮT MỞ ĐẦU: Luôn mở đầu bằng một câu tóm tắt điểm khác biệt chính sách/nội dung quan trọng nhất giữa hai bản. "
            "(Ví dụ: \"Điểm khác biệt duy nhất tại Điều [X] giữa hai bản hợp đồng nằm ở khoản [Y], quy định về...\").\n\n"
            "2. TRÌNH BÀY CHI TIẾT SỰ THAY ĐỔI: Sử dụng câu dẫn \"Cụ thể sự thay đổi như sau:\" và liệt kê theo định dạng bullet points:\n"
            "   - **Bản hợp đồng cũ** (kèm thời gian hoặc tên ngắn gọn rút từ tên file): [Trình bày ngắn gọn quy định cũ]\n"
            "   - **Bản hợp đồng mới** (kèm thời gian hoặc tên ngắn gọn rút từ tên file): [Nêu rõ quy định thay đổi, nhận xét mức độ khắt khe/nới lỏng hơn...]\n"
            "     - [Bullet point: Chi tiết thay đổi 1]\n"
            "     - [Bullet point: Chi tiết thay đổi 2]\n\n"
            "3. KẾT LUẬN VỀ YẾU TỐ GIỮ NGUYÊN: Ở phần cuối cùng, hãy tóm tắt vắn tắt các nội dung "
            "còn lại của Điều/Khoản đó đã được giữ nguyên (nếu có). "
            "(Ví dụ: \"Toàn bộ các nội dung còn lại của Điều [X]... đều được giữ nguyên hoàn toàn ở cả hai phiên bản.\").\n\n"
            "4. BÁM SÁT DỮ LIỆU: Chỉ sử dụng nội dung từ trích đoạn. Nếu không đủ dữ liệu, hãy trả lời không đủ dữ liệu.\n"
            "5. BỎ QUA CÁC THAY ĐỔI HÌNH THỨC: Bỏ qua lỗi chính tả, thay đổi từ đồng nghĩa không làm đổi bản chất pháp lý."
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
        print(f"[RAG] Đang dò tìm {top_k} phần tử tài liệu liên quan nhất qua Indexing Strategy...")
        results = self.indexing_strategy.build_context(query, top_k=top_k)
        
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
