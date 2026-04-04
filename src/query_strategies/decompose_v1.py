import json
from typing import Iterator, List, Dict, Any
from .base_strategy import QueryStrategy


# ─── Prompt phân rã câu hỏi thành ASPECT-BASED sub-queries ────────────────────
# Không nhắc tên tài liệu trong sub-query để tránh ChromaDB kéo các điều khoản
# không liên quan (VD: Lời nói đầu, phạm vi áp dụng) chỉ vì trùng tên document.
DECOMPOSE_PROMPT = """Bạn là chuyên gia xây dựng truy vấn tìm kiếm pháp lý.

Nhiệm vụ: Phân rã câu hỏi so sánh pháp lý dưới đây thành TỐI ĐA 4 sub-queries.
Mỗi sub-query phải nhắm đến một KHÍA CẠNH CỤ THỂ của nội dung (điều kiện, thủ tục, độ tuổi, thời hạn, đối tượng...).

Câu hỏi gốc: "{query}"

YÊU CẦU QUAN TRỌNG:
- Viết sub-query theo dạng "tìm kiếm nội dung điều khoản" — KHÔNG nhắc tên tài liệu, không nhắc "Luật 2014" hay "Luật 2023"
- Mỗi sub-query ngắn gọn, chứa đúng thuật ngữ pháp lý để embedding tìm đúng điều khoản
- Phân tách theo KHÍA CẠNH khác nhau, không lặp lại
- Ví dụ tốt: "độ tuổi được cấp thẻ căn cước", "thủ tục cấp thẻ cho người dưới 14 tuổi"
- Ví dụ XẤU (tránh): "đối tượng cấp thẻ theo Luật 2014", "quy định Luật 2023 về..."

Trả về JSON array — CHỈ JSON thuần túy, không markdown, không giải thích:
["sub-query 1", "sub-query 2", "sub-query 3"]"""


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
    db, sub_query: str, old_law: str, new_law: str, per_k: int
) -> List[Dict[str, Any]]:
    """
    Với mỗi sub-query, truy vấn ChromaDB 2 lần — một lần filter theo tài liệu cũ,
    một lần filter theo tài liệu mới — để đảm bảo bao phủ đủ cả 2 nguồn.
    """
    results = []
    for source in [old_law, new_law]:
        if source:
            try:
                r = db.query(sub_query, n_results=per_k, where={"source": source})
                results.append(r)
            except Exception:
                # Fallback nếu source không tồn tại trong DB (ChromaDB raises on empty where)
                pass
    # Nếu cả 2 filter không tìm được gì, fallback về query không filter
    if not results:
        results = [db.query(sub_query, n_results=per_k)]
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

        # ── Bước 2: Dual-source retrieval — mỗi sub-query × 2 tài liệu ──────
        per_k = max(2, top_k // max(len(subqueries), 1))
        all_results: List[Dict[str, Any]] = []

        for sq in subqueries:
            results_for_sq = _query_both_sources(engine.db, sq, old_law, new_law, per_k)
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
