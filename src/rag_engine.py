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
            "Bạn là một Trợ lý Luật sư chuyên nghiệp. Nhiệm vụ của bạn là đọc và phân tích sự thay đổi giữa hai văn bản hợp đồng "
            "để giải đáp chính xác, rõ ràng và trung thực câu hỏi cụ thể của người dùng.\n\n"
            "=== NGUYÊN TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ 100%) ===\n"
            "- Trả lời trực tiếp vào trọng tâm câu hỏi của người dùng.\n"
            "- Chỉ sử dụng thông tin từ văn bản gốc (BẢN GỐC, BẢN MỚI) và KẾT QUẢ SO SÁNH TỰ ĐỘNG (DIFF) được cung cấp dưới đây.\n"
            "- Tuyệt đối không tự bịa đặt, suy diễn, hoặc vẽ thêm thông tin giả định nằm ngoài ngữ cảnh được cung cấp.\n"
            "- Nếu câu hỏi của người dùng hỏi về nội dung/thay đổi của một điều cụ thể, hãy trích dẫn trung thực sự thay đổi của điều đó từ văn bản và giải thích ngắn gọn, xúc tích nếu cần thiết."
        )

    def _build_context_prompt(self, query: str, search_results: Dict[str, Any], diff_text: str = "", intent: str = "SPECIFIC") -> str:
        """
        Lắp ráp kịch bản so sánh gộp chung kết quả từ Database.
        """
        # Nếu là câu hỏi Liệt kê toàn bộ (LIST_ALL) hoặc đã có diff_text, loại bỏ hoàn toàn Raw Text để chống loãng context gây hoang tưởng.
        if intent == "LIST_ALL" or diff_text:
            context_str = "(Hệ thống đã tự động lọc bỏ tài liệu thô để tối ưu hóa đối chiếu và tránh loãng ngữ cảnh)"
        else:
            old_law_blocks = []
            new_law_blocks = []
            other_blocks = []
            
            # ChromaDB trả về list of list cho n_results
            documents = search_results.get("documents", [[]])[0]
            metadatas = search_results.get("metadatas", [[]])[0]
            
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                if not meta:
                    continue
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
            
        diff_block = ""
        if diff_text:
            diff_block = (
                "=== KẾT QUẢ SO SÁNH TỰ ĐỘNG (DIFF) ===\n"
                "(Phần này do hệ thống tạo tự động, không phải từ LLM. "
                "Các từ bôi đậm là những từ ĐÃ THAY ĐỔI trong Bản mới so với Bản cũ.)\n\n"
                f"{diff_text}\n\n"
            )
        
        final_prompt = (
            "Dưới đây là KẾT QUẢ SO SÁNH TỰ ĐỘNG (DIFF) được hệ thống rút trích giữa 2 tài liệu:\n\n"
            f"{context_str}\n\n"
            f"{diff_block}"
            "Yêu cầu của người dùng:\n"
            f'"{query}"\n\n'
            "Hãy trả lời chính xác và trực tiếp câu hỏi của người dùng dựa trên các thông tin văn bản gốc và DIFF được cung cấp ở trên. "
            "TUYỆT ĐỐI TUÂN THỦ CÁC NGUYÊN TẮC CỐT LÕI TRONG SYSTEM PROMPT!"
        )
        return final_prompt

    def ask(self, query: str, top_k: int = 12) -> str:
        """Thực hiện một luồng RAG hoàn chỉnh trả về kết quả duy nhất."""
        print(f"[RAG] Đang dò tìm {top_k} phần tử tài liệu liên quan nhất qua Indexing Strategy...")
        results = self.indexing_strategy.build_context(query, top_k=top_k)
        
        print("[RAG] Đang khởi tạo bộ Prompt kết hợp ngữ cảnh...")
        prompt = self._build_context_prompt(query, results)
        
        print(f"[RAG] Đang chờ QA Model {self.llm.model_name} xử lý lập luận pháp lý...")
        answer = self.llm.generate_response(prompt=prompt, system_prompt=self.system_prompt)
        return answer
        
    def stream_ask(self, query: str, top_k: int = 12, strategy_name: str = None) -> Iterator[str]:
        """Thực hiện luồng RAG và yield text dưới dạng Stream."""
        from src.query_strategies import STRATEGIES, PairedRetrievalStrategy
        
        if strategy_name and strategy_name in STRATEGIES:
            strategy_class = STRATEGIES[strategy_name]
        else:
            strategy_class = PairedRetrievalStrategy
            
        strategy = strategy_class()
        for chunk in strategy.stream_execute(query=query, engine=self, top_k=top_k):
            yield chunk
