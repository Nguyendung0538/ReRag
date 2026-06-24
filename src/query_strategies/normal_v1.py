from typing import Iterator
from .base_strategy import QueryStrategy

class NormalV1Strategy(QueryStrategy):
    """
    Kỹ thuật RAG Baseline (Normal V1) — dùng để đo lường so sánh với Decompose_v1.

    Dùng NGUYÊN câu hỏi của người dùng tiêm thẳng vào ChromaDB, lấy top_k kết quả,
    build context và sinh câu trả lời. Không có bước xử lý query trước.

    Dùng chiến lược này khi:
    - Muốn đo baseline nhanh (ít latency nhất)
    - Câu hỏi đã đủ rõ ràng, chứa đúng từ khóa pháp lý
    """
    def stream_execute(self, query: str, engine, top_k: int = 6) -> Iterator[str]:
        where_filter = self._extract_metadata_filter(query)
        kwargs = {}
        if where_filter:
            kwargs["where"] = where_filter

        # Truy xuất cơ sở dữ liệu với nguyên câu hỏi gốc
        results = engine.indexing_strategy.build_context(query, top_k=top_k, **kwargs)
        
        docs = results.get("documents", [[]])[0]
        
        if not docs:
            yield "Khong tim thay van ban phap ly nao khop voi du lieu trong bo nho."
            return
            
        # Sử dụng hàm build context prompt từ Engine giúp định dạng Text khối thống nhất
        prompt = engine._build_context_prompt(query, results)
        
        # Bắt đầu stream câu trả lời từ LLM
        for chunk in engine.llm.stream_response(prompt=prompt, system_prompt=engine.system_prompt):
            yield chunk
