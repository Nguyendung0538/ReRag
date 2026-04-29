import re
import difflib
from typing import List, Dict, Any, Iterator
from src.indexing_strategies.base_indexing import BaseIndexingStrategy
from src.generation.llm_client import LLMClient
from src.diff.clause_differ import ClauseDiff, ClauseDiffer

class LegalRAGEngine:
    """
    Bộ não RAG: Kết hợp truy xuất Database và lập luận LLM để trả lời câu hỏi So sánh Pháp lý.
    Hỗ trợ Hybrid Diff + RAG: inject kết quả diff thuần túy vào prompt để LLM không bỏ sót thay đổi.
    """
    def __init__(self, indexing_strategy: BaseIndexingStrategy, llm_client: LLMClient, 
                 old_law_source: str = "", new_law_source: str = "",
                 clause_diffs: List[ClauseDiff] = None):
        self.indexing_strategy = indexing_strategy
        self.llm = llm_client
        self.old_law_source = old_law_source
        self.new_law_source = new_law_source
        self.clause_diffs = clause_diffs or []
        self.differ = ClauseDiffer()
        self.last_retrieved_context = {"documents": [[]], "metadatas": [[]]}
        
        # System prompt định hướng vai trò chuyên gia SO SÁNH 2 tài liệu hợp đồng
        old_label = self.old_law_source or "Tài liệu 1"
        new_label = self.new_law_source or "Tài liệu 2"

        self.system_prompt = (
            "Bạn là chuyên gia đối chiếu hợp đồng pháp lý.\n"
            f"Hệ thống đang làm việc với HAI tài liệu:\n"
            f"  - Bản cũ: \"{old_label}\"\n"
            f"  - Bản mới: \"{new_label}\"\n\n"

            "=== LỆNH CẤM TUYỆT ĐỐI ===\n"
            "- CẤM dùng emoji.\n"
            "- CẤM dùng bảng (table/markdown table).\n"
            "- CẤM viết Kết luận, Tổng kết, Nhận xét, Lời khuyên, Đánh giá.\n"
            "- CẤM tóm tắt hay diễn giải nội dung. Chỉ được SAO CHÉP NGUYÊN VĂN.\n"
            "- CẤM viết bất kỳ câu nào không phải là trích dẫn nguyên văn từ tài liệu.\n"
            "- DỪNG NGAY sau điểm khác biệt cuối cùng. Không viết thêm gì.\n\n"

            "=== CÁCH TRẢ LỜI ===\n"
            "Dòng 1: Một câu tóm tắt ngắn gọn về điểm khác biệt chính.\n\n"
            "Sau đó liệt kê từng điểm khác biệt:\n"
            "- Ghi rõ Điều bao nhiêu (VD: Điều 7, Điều 12.4).\n"
            "- Dòng 'Bản cũ:' PHẢI là văn bản SAO CHÉP NGUYÊN VĂN từ phần BẢN GỐC trong ngữ cảnh. KHÔNG được tự viết lại.\n"
            "- Dòng 'Bản mới:' PHẢI là văn bản SAO CHÉP NGUYÊN VĂN từ phần BẢN MỚI trong ngữ cảnh. KHÔNG được tự viết lại.\n"
            "- Trong dòng Bản mới, bôi đậm (dùng **...**) CHỈ những từ/cụm từ bị thay đổi so với Bản cũ.\n"
            "- Bỏ qua các điều khoản giống nhau.\n\n"

            "=== VÍ DỤ CHUẨN (HỌC THEO CHÍNH XÁC ĐỊNH DẠNG NÀY) ===\n\n"
            "Thay đổi quan trọng nhất: Điều 7 cụ thể hóa thời hạn thông báo từ 'kịp thời' thành 03 ngày làm việc.\n\n"
            "- Điều 7 - Thời hạn thông báo:\n"
            "  + Bản cũ: Nếu một Bên vi phạm quy định về bảo mật thông tin và không khắc phục trong thời hạn kịp thời thông báo của Bên bị vi phạm, hoặc vi phạm lần 2 thì Bên bị vi phạm có quyền chấm dứt Hợp đồng này.\n"
            "  + Bản mới: Nếu một Bên vi phạm quy định về bảo mật thông tin và không khắc phục trong thời hạn **thông báo trong vòng 03 ngày làm việc** của Bên bị vi phạm, hoặc vi phạm lần 2 thì Bên bị vi phạm có quyền chấm dứt Hợp đồng này.\n\n"
            "- Điều 15 - Số lượng bản hợp đồng:\n"
            "  + Bản cũ: Hợp đồng này được lập thành 04 (bốn) bản bằng tiếng Việt có giá trị pháp lý như nhau; mỗi Bên giữ 02 (hai) bản.\n"
            "  + Bản mới: Hợp đồng này được lập thành **06 (sáu)** bản bằng tiếng Việt có giá trị pháp lý như nhau; mỗi Bên giữ 02 (hai) bản.\n\n"

            "=== VỀ DIFF TỰ ĐỘNG ===\n"
            "Nếu trong ngữ cảnh có phần 'KẾT QUẢ SO SÁNH TỰ ĐỘNG (DIFF)', bạn PHẢI dùng nó làm cơ sở.\n"
            "Phần DIFF đã có sẵn dấu **in đậm** bao quanh các từ thay đổi trong Bản mới.\n"
            "Hãy SAO CHÉP NGUYÊN VĂN cả phần Bản cũ và Bản mới (giữ nguyên dấu **...**) từ DIFF vào câu trả lời.\n"
            "KHÔNG ĐƯỢC tự thêm hoặc bỏ bất kỳ dấu **in đậm** nào.\n"
            "KHÔNG ĐƯỢC bỏ qua bất kỳ mục nào trong DIFF."
        )

    def _extract_dieu_from_query(self, query: str) -> str | None:
        """Trích xuất tên Điều từ câu hỏi (VD: 'điều 6' → 'Điều 6')."""
        from src.ingestion.legal_chunker import LegalChunker
        match = re.search(r"điều\s+([\dIVXLCDM]+)", query, re.IGNORECASE)
        if match:
            chunker = LegalChunker()
            normalized = chunker._normalize_dieu_number(match.group(1))
            return f"Điều {normalized}"
        return None

    def _get_relevant_diffs(self, query: str) -> List[ClauseDiff]:
        """
        Lọc diff liên quan đến câu hỏi.
        - Nếu query nhắc Điều cụ thể → chỉ lấy diff cho Điều đó
        - Nếu query tổng quát → lấy tất cả diff
        """
        if not self.clause_diffs:
            return []
        
        dieu_name = self._extract_dieu_from_query(query)
        if dieu_name:
            filtered = self.differ.filter_by_dieu(self.clause_diffs, dieu_name)
            if filtered:
                return filtered
        
        # Query tổng quát hoặc không tìm thấy Điều cụ thể → trả hết
        return self.clause_diffs

    def _build_context_prompt(self, query: str, search_results: Dict[str, Any]) -> str:
        """
        Lắp ráp prompt so sánh. Chiến lược:
        - Query tổng quát (không nhắc Điều cụ thể) → dùng DIFF làm nguồn DUY NHẤT
        - Query cụ thể (nhắc Điều X) → dùng DIFF + retrieval context bổ trợ
        """
        # Kiểm tra diff
        relevant_diffs = self._get_relevant_diffs(query)
        is_specific_query = self._extract_dieu_from_query(query) is not None
        
        # Nếu có diff và query tổng quát → chỉ dùng diff, bỏ retrieval context
        if relevant_diffs and not is_specific_query:
            # Clear retrieval context — không dùng nên không hiển thị trong citation
            self.last_retrieved_context = {"documents": [[]], "metadatas": [[]]}
            
            diff_text = self.differ.format_diff_for_prompt(relevant_diffs)
            final_prompt = (
                "=== KẾT QUẢ SO SÁNH TỰ ĐỘNG (DIFF) ===\n"
                "Dưới đây là TOÀN BỘ các điểm khác biệt giữa hai tài liệu, được tính toán chính xác 100% bằng thuật toán.\n"
                "Phần Bản mới đã có sẵn dấu **in đậm** bao quanh các từ thay đổi.\n\n"
                f"{diff_text}\n\n"
                f'Câu hỏi: "{query}"\n\n'
                "NHIỆM VỤ: Hãy trình bày lại TỪNG mục thay đổi ở trên theo đúng định dạng chuẩn.\n"
                "Với mỗi mục, SAO CHÉP NGUYÊN VĂN phần Bản cũ và Bản mới (giữ nguyên dấu **in đậm**).\n"
                "KHÔNG được bỏ sót mục nào. KHÔNG được tự viết thêm nội dung ngoài trích dẫn."
            )
            return final_prompt
        
        # Query cụ thể hoặc không có diff → build retrieval context
        old_law_blocks = []
        new_law_blocks = []
        other_blocks = []
        
        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            source = meta.get("source", "")
            dieu   = meta.get("dieu", "")
            location_str = dieu if dieu else "Không rõ vị trí"
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
        
        # Inject diff cho query cụ thể
        diff_section = ""
        if relevant_diffs:
            diff_text = self.differ.format_diff_for_prompt(relevant_diffs)
            diff_section = (
                "=== KẾT QUẢ SO SÁNH TỰ ĐỘNG (DIFF) ===\n"
                "Phần Bản mới đã có sẵn dấu **in đậm** bao quanh các từ thay đổi.\n\n"
                f"{diff_text}\n\n"
            )
        
        final_prompt = (
            f"{diff_section}"
            f"{context_str}\n\n"
            f'Câu hỏi: "{query}"\n\n'
            "NHIỆM VỤ: SAO CHÉP NGUYÊN VĂN phần Bản cũ và Bản mới từ DIFF (giữ nguyên dấu **in đậm**) vào câu trả lời.\n"
            "Nếu không có DIFF, trích dẫn nguyên văn từ BẢN GỐC và BẢN MỚI, tự bôi đậm phần thay đổi."
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

    def compute_grounding_score(self, answer: str) -> float:
        """
        Tính % nội dung câu trả lời có thể truy nguyên về nguồn gốc.
        
        Sử dụng word-level overlap: đếm % từ trong câu trả lời xuất hiện
        trong tài liệu nguồn (chunks + diff). Tránh lỗi SequenceMatcher
        khi so sánh câu ngắn với chunk dài.
        """
        sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 15]
        if not sentences:
            return 0.0
        
        # Thu thập TẤT CẢ nguồn: retrieved chunks + diff text
        source_docs = list(self.last_retrieved_context.get("documents", [[]])[0])
        
        # Thêm diff text làm nguồn (quan trọng cho broad queries không qua retrieval)
        if self.clause_diffs:
            for d in self.clause_diffs:
                if d.old_text:
                    source_docs.append(d.old_text)
                if d.new_text:
                    source_docs.append(d.new_text)
        
        if not source_docs:
            return 0.0
        
        # Build tập từ từ tất cả nguồn
        source_words = set()
        for doc in source_docs:
            source_words.update(doc.lower().split())
        
        grounded = 0
        for sent in sentences:
            sent_words = sent.lower().split()
            if not sent_words:
                continue
            overlap = sum(1 for w in sent_words if w in source_words) / len(sent_words)
            if overlap >= 0.5:  # >=50% từ trong câu tìm thấy trong nguồn
                grounded += 1
        
        return (grounded / len(sentences)) * 100
