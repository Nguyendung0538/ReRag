import json
from typing import Iterator, List, Dict, Any
from .base_strategy import QueryStrategy


DECOMPOSE_PROMPT = """Bạn là chuyên gia hiểu ý định tìm kiếm (Search Intent) pháp lý.

Nhiệm vụ: Phân rã câu hỏi dưới đây thành các câu truy vấn con (sub-queries) để hệ thống RAG tìm kiếm dễ nhất.

Câu hỏi gốc: "{query}"

QUY TẮC BẮT BUỘC (ĐỌC KỸ TRƯỚC KHI LÀM):
1. Nếu câu hỏi siêu ngắn gọn chỉ định đích danh một Điều khoản cụ thể (VD: "điều 6 có thay đổi gì", "so sánh điều 5"): 
   -> TUYỆT ĐỐI KHÔNG bịa thêm các khía cạnh.
   -> TUYỆT ĐỐI KHÔNG được phép thêm các từ phụ họa dư thừa như "hiện hành", "mới", "cũ", "nội dung chính", "thay đổi".
   -> CHỈ YÊU CẦU trả về ĐÚNG 1 sub-query CỤT NGỦN mang tên điều khoản đó (VD: ["Điều 6"]).
2. NẾU VÀ CHỈ NẾU câu hỏi là về một CHỦ ĐỀ tổng quát phức tạp (VD: "so sánh về quy định nộp phạt"):
   -> Mới được chia nhỏ thành các khía cạnh (VD: ["mức tiền phạt", "thời hạn nộp phạt"]).
3. Ghi nhớ: Sub-query càng thuần túy và ngắn gọn, máy học càng dễ tìm.

Trả về mảng JSON thuần túy (KHÔNG dùng markdown, KHÔNG giải thích):
["query 1", "query 2"]"""


def _call_llm_sync(llm, prompt: str) -> str:
    """Gọi LLM một lần, trả về toàn bộ text (không stream)."""
    result = ""
    for chunk in llm.stream_response(prompt=prompt, system_prompt=""):
        result += chunk
    return result.strip()


def _decompose_query(llm, query: str) -> List[str]:
    """
    Dùng LLM phân rã câu hỏi thành danh sách aspect-based sub-queries.
    Fallback an toàn về câu hỏi gốc nếu LLM trả sai format.
    """
    prompt = DECOMPOSE_PROMPT.format(query=query)
    raw = _call_llm_sync(llm, prompt)

    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            valid = [q for q in parsed if isinstance(q, str) and q.strip()]
            if valid:
                return valid[:4]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: câu hỏi gốc
    return [query]


def _query_both_sources(
    indexing_strategy, sub_query: str, old_law: str, new_law: str, per_k: int, base_where: dict = None
) -> List[Dict[str, Any]]:
    """
    Với mỗi sub-query, truy vấn IndexingStrategy 2 lần — một lần filter theo tài liệu cũ,
    một lần filter theo tài liệu mới — kết hợp filter theo Điều khoản nếu có.
    """
    results = []
    for source in [old_law, new_law]:
        if source:
            where_clause = {"source": source}
            if base_where:
                where_clause = {"$and": [{"source": source}, base_where]}
            try:
                r = indexing_strategy.build_context(sub_query, top_k=per_k, where=where_clause)
                # Tránh các list trống
                if r.get("documents") and r["documents"][0]:
                    results.append(r)
            except Exception:
                pass
    # Nếu cả 2 filter không tìm được gì, fallback về query không filter source
    if not results:
        kwargs = {}
        if base_where:
            kwargs["where"] = base_where
        r = indexing_strategy.build_context(sub_query, top_k=per_k, **kwargs)
        if r.get("documents") and r["documents"][0]:
            results.append(r)
    return results


def _merge_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Gộp nhiều kết quả ChromaDB, loại trùng lặp theo fingerprint 80 ký tự đầu.
    Giữ nguyên cấu trúc để tương thích với engine._build_context_prompt().
    """
    seen = set()
    merged_docs, merged_metas, merged_distances = [], [], []

    for results in results_list:
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            fingerprint = doc[:80].strip()
            if fingerprint not in seen:
                seen.add(fingerprint)
                merged_docs.append(doc)
                merged_metas.append(meta)
                merged_distances.append(dist)

    return {
        "documents": [merged_docs],
        "metadatas": [merged_metas],
        "distances": [merged_distances],
    }


class DecomposeV1Strategy(QueryStrategy):
    """
    Kỹ thuật RAG Nâng cao — Decompose V1 (So sánh 2 tài liệu).

    Luồng xử lý:
    1. Aspect-based Decomposition: LLM phân rã câu hỏi thành 2-4 sub-queries
       nhắm đến từng KHÍA CẠNH nội dung (không nhắc tên tài liệu để tránh noise).
    2. Dual-source Retrieval: Mỗi sub-query được truy vấn 2 lần — filter riêng
       theo tài liệu cũ và tài liệu mới qua ChromaDB `where` clause.
    3. Merge & Deduplicate: Gộp tất cả kết quả, loại trùng lặp.
    4. Build Context & Stream: Tạo prompt phân loại Cũ/Mới và stream câu trả lời.

    Ưu điểm so với Normal_v1:
    - Không bỏ sót thông tin khi câu hỏi có nhiều khía cạnh
    - Cân bằng kết quả từ cả 2 tài liệu (không bị thiên về tài liệu nào)
    - Ít nhiễu từ điều khoản không liên quan
    """

    def stream_execute(self, query: str, engine, top_k: int = 6) -> Iterator[str]:
        old_law = engine.old_law_source or ""
        new_law = engine.new_law_source or ""

        # ── Bước 1: Phân rã câu hỏi thành aspect-based sub-queries ──────────
        yield "⚙️ *[Decompose] Đang phân rã câu hỏi thành các khía cạnh tìm kiếm...*\n\n"
        subqueries = _decompose_query(engine.llm, query)

        yield f"📋 *Sub-queries ({len(subqueries)} khía cạnh):*\n"
        for i, sq in enumerate(subqueries, 1):
            yield f"&nbsp;&nbsp;**[{i}]** {sq}\n"
        yield "\n"

        # Định vị filter dựa vào câu hỏi chính tổng thể gốc
        where_filter = self._extract_metadata_filter(query)

        # ── Bước 2: Dual-source retrieval — mỗi sub-query × 2 tài liệu ──────
        per_k = max(2, top_k // max(len(subqueries), 1))
        all_results: List[Dict[str, Any]] = []

        for sq in subqueries:
            results_for_sq = _query_both_sources(engine.indexing_strategy, sq, old_law, new_law, per_k, base_where=where_filter)
            all_results.extend(results_for_sq)

        # ── Bước 3: Gộp và loại trùng lặp ───────────────────────────────────
        merged = _merge_results(all_results)
        docs = merged.get("documents", [[]])[0]

        if not docs:
            yield "❌ Không tìm thấy văn bản pháp lý nào khớp với dữ liệu trong bộ nhớ."
            return

        # Thống kê số chunks từ mỗi tài liệu để debug
        metas = merged.get("metadatas", [[]])[0]
        old_count = sum(1 for m in metas if m.get("source") == old_law)
        new_count = sum(1 for m in metas if m.get("source") == new_law)
        yield (
            f"✅ *Tìm được {len(docs)} đoạn văn bản "
            f"(Cũ: {old_count}, Mới: {new_count}). Đang phân tích so sánh...*\n\n---\n\n"
        )

        # ── Bước 4: Build context prompt và stream câu trả lời ───────────────
        prompt = engine._build_context_prompt(query, merged)
        for chunk in engine.llm.stream_response(prompt=prompt, system_prompt=engine.system_prompt):
            yield chunk
