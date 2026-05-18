from typing import Iterator, Dict, Any, List
from .base_strategy import QueryStrategy

class PairedRetrievalStrategy(QueryStrategy):
    """
    Chiến thuật truy vấn song song (Paired Retrieval Strategy) — Hai pha (Two-Phase):
    1. Pha 1 (Exact Match): Query mỗi nguồn riêng biệt bằng ChromaDB filter 'source' + 'dieu' (nếu có).
    2. Pha 2 (Semantic Cross-Match): Nếu phát hiện lệch số Điều giữa 2 bản (renumbering),
       dùng nội dung tìm được của bên này để truy vấn ngữ nghĩa bên kia.
    """

    def _build_where(self, source: str, dieu_filter: dict | None) -> dict:
        """
        Xây dựng filter where cho ChromaDB.
        Hỗ trợ filter kép: source + dieu sử dụng toán tử $and của ChromaDB.
        """
        if dieu_filter and "dieu" in dieu_filter:
            dieu_val = dieu_filter["dieu"]
            return {
                "$and": [
                    {"source": {"$eq": source}},
                    {"dieu": {"$eq": dieu_val}}
                ]
            }
        return {"source": {"$eq": source}}

    def _merge_results(self, *result_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gộp nhiều kết quả query của ChromaDB và loại bỏ các chunk trùng lặp dựa trên ID.
        """
        merged_ids = []
        merged_documents = []
        merged_metadatas = []
        merged_distances = []
        
        seen_ids = set()
        
        for r in result_sets:
            if not r:
                continue
            
            ids = r.get("ids", [[]])[0]
            docs = r.get("documents", [[]])[0]
            metas = r.get("metadatas", [[]])[0]
            dists = r.get("distances", [[]])[0] if "distances" in r else [0.0] * len(ids)
            
            for i, d, m, dist in zip(ids, docs, metas, dists):
                if i not in seen_ids:
                    seen_ids.add(i)
                    merged_ids.append(i)
                    merged_documents.append(d)
                    merged_metadatas.append(m)
                    merged_distances.append(dist)
                    
        return {
            "ids": [merged_ids],
            "documents": [merged_documents],
            "metadatas": [merged_metadatas],
            "distances": [merged_distances]
        }

    def _semantic_cross_query(self, content_text: str, target_source: str, engine: Any, top_k: int) -> Dict[str, Any]:
        """
        Pha 2: Thực hiện truy vấn semantic chéo sang tài liệu đích (target_source)
        sử dụng chính nội dung text của tài liệu nguồn làm query vector, chỉ filter theo source.
        """
        where = {"source": {"$eq": target_source}}
        print(f"[Paired RAG] Chạy Semantic Cross-Query sang '{target_source}' với nội dung từ bản đối chiếu...")
        return engine.indexing_strategy.build_context(content_text, top_k=top_k, where=where)

    def _classify_intent(self, query: str, llm_client: Any) -> str:
        """
        Sử dụng LLM với prompt cực ngắn để phân loại ý định của người dùng:
        - LIST_ALL: Nếu người dùng muốn liệt kê toàn bộ thay đổi của hợp đồng.
        - SPECIFIC: Nếu câu hỏi tập trung vào một chủ đề cụ thể (như trách nhiệm, giá cả, đặt cọc, thời hạn thuê, ...).
        """
        system_prompt = (
            "Bạn là một bộ phân loại ý định câu hỏi cực kỳ chính xác. Nhiệm vụ của bạn là đọc câu hỏi của người dùng "
            "về việc so sánh 2 bản hợp đồng thuê nhà và chỉ được trả về một trong hai nhãn sau:\n"
            "- 'LIST_ALL': Nếu câu hỏi yêu cầu liệt kê toàn bộ, tổng hợp toàn bộ các thay đổi hoặc so sánh tất cả các điều khoản.\n"
            "- 'SPECIFIC': Nếu câu hỏi hỏi về một nội dung cụ thể, một chủ đề hoặc khía cạnh cụ thể (Ví dụ: giá cả, tiền thuê, đặt cọc, trách nhiệm, chấm dứt hợp đồng, phạt vi phạm, bàn giao trang thiết bị...).\n\n"
            "Chỉ trả ra đúng nhãn 'LIST_ALL' hoặc 'SPECIFIC', tuyệt đối không giải thích gì thêm."
        )
        try:
            print("[Paired RAG] Đang phân loại ý định câu hỏi qua LLM Classifier...")
            response = llm_client.generate_response(prompt=f'Câu hỏi: "{query}"', system_prompt=system_prompt)
            intent = response.strip().upper()
            if "LIST_ALL" in intent:
                return "LIST_ALL"
            return "SPECIFIC"
        except Exception as e:
            print(f"[Paired RAG] Lỗi phân loại ý định: {e}. Fallback về LIST_ALL.")
            return "LIST_ALL"

    def _format_diff_for_user(self, diff_text: str) -> str:
        """
        Định dạng lại kết quả DIFF toán học thành định dạng Markdown Bullet Points tuyệt đẹp,
        đảm bảo 100% không ảo tưởng và chính xác tuyệt đối.
        Chỉ liệt kê các điều khoản, loại bỏ phần mở đầu, căn cứ, thông tin các bên.
        """
        if not diff_text.strip():
            return "Không phát hiện bất kỳ sự thay đổi nào giữa hai tài liệu hợp đồng."
            
        blocks = diff_text.split('\n\n')
        formatted_blocks = []
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = block.split('\n')
            first_line = lines[0].strip()
            
            # Kiểm tra xem có phải là tiêu đề chính không
            if first_line.startswith("- ") and first_line.endswith(":"):
                title = first_line[2:].rstrip(":")
                title_lower = title.lower()
                
                # Chỉ giữ lại các block có tiêu đề chứa "điều" hoặc "khoản"
                if not ("điều" in title_lower or "khoản" in title_lower):
                    continue
            
            # Định dạng block này
            formatted_lines = []
            for line in lines:
                stripped = line.strip()
                if line.startswith("- ") and line.endswith(":"):
                    title = line[2:].rstrip(":")
                    formatted_lines.append(f"- **{title}**:")
                elif stripped.startswith("+ Bản cũ:") or stripped.startswith("Bản cũ:") or stripped.startswith("- Bản cũ:"):
                    content = stripped.replace("+ Bản cũ:", "").replace("Bản cũ:", "").replace("- Bản cũ:", "").strip()
                    formatted_lines.append(f"  - **Bản cũ:** {content}")
                elif stripped.startswith("+ Bản mới:") or stripped.startswith("Bản mới:") or stripped.startswith("- Bản mới:"):
                    content = stripped.replace("+ Bản mới:", "").replace("Bản mới:", "").replace("- Bản mới:", "").strip()
                    formatted_lines.append(f"  - **Bản mới:** {content}")
                else:
                    formatted_lines.append(line)
                    
            formatted_blocks.append("\n".join(formatted_lines))
            
        if not formatted_blocks:
            return "Không phát hiện bất kỳ sự thay đổi nào trong các điều khoản hợp đồng."
            
        return "\n\n".join(formatted_blocks)

    def stream_execute(self, query: str, engine: Any, top_k: int = 6) -> Iterator[str]:
        # Tránh lỗi chia nhỏ
        half_k = max(top_k // 2, 2)
        
        # 1. Trích xuất Điều nếu có từ câu hỏi
        dieu_filter = self._extract_metadata_filter(query)
        
        old_src = engine.old_law_source
        new_src = engine.new_law_source
        
        print(f"[Paired RAG] Bản cũ: '{old_src}', Bản mới: '{new_src}'")
        
        # Xác định ý định
        intent = "SPECIFIC"
        if dieu_filter:
            print(f"[Paired RAG] Phát hiện yêu cầu cụ thể theo Điều với: {dieu_filter}")
        else:
            intent = self._classify_intent(query, engine.llm)
            print(f"[Paired RAG] Phân tích ý định câu hỏi: {intent}")
            
        # === PHA 1: LẤY DỮ LIỆU TỪ TỪNG NGUỒN ===
        if dieu_filter:
            # Có filter Điều cụ thể: Dùng semantic search kết hợp metadata filter
            where_old = self._build_where(old_src, dieu_filter)
            where_new = self._build_where(new_src, dieu_filter)
            
            print(f"[Paired RAG] Pha 1: Đang tìm kiếm song song theo Điều cụ thể...")
            results_old = engine.indexing_strategy.build_context(query, top_k=half_k, where=where_old)
            results_new = engine.indexing_strategy.build_context(query, top_k=half_k, where=where_new)
        elif intent == "LIST_ALL":
            # Không có filter Điều và muốn liệt kê tất cả: Lấy TOÀN BỘ tài liệu của cả 2 nguồn
            print(f"[Paired RAG] Pha 1: Câu hỏi tổng quát dạng LIỆT KÊ — đang lấy TOÀN BỘ chunks từ cả 2 tài liệu...")
            try:
                results_old = engine.indexing_strategy.get_all_by_source(old_src)
                results_new = engine.indexing_strategy.get_all_by_source(new_src)
            except (NotImplementedError, AttributeError):
                # Fallback nếu indexing strategy không hỗ trợ get_all
                print(f"[Paired RAG] Fallback: get_all_by_source không khả dụng, dùng top_k lớn...")
                results_old = engine.indexing_strategy.build_context(query, top_k=50, where={"source": {"$eq": old_src}})
                results_new = engine.indexing_strategy.build_context(query, top_k=50, where={"source": {"$eq": new_src}})
        else:
            # Câu hỏi ngữ nghĩa cụ thể: Dùng semantic search song song để lấy context
            print(f"[Paired RAG] Pha 1: Đang tìm kiếm ngữ nghĩa song song từ 2 tài liệu...")
            results_old = engine.indexing_strategy.build_context(query, top_k=half_k, where={"source": {"$eq": old_src}})
            results_new = engine.indexing_strategy.build_context(query, top_k=half_k, where={"source": {"$eq": new_src}})
        
        docs_old = results_old.get("documents", [[]])[0] if results_old else []
        docs_new = results_new.get("documents", [[]])[0] if results_new else []
        meta_old = results_old.get("metadatas", [[]])[0] if results_old else []
        meta_new = results_new.get("metadatas", [[]])[0] if results_new else []
        
        print(f"[Paired RAG] Pha 1 hoàn tất. Bản cũ: {len(docs_old)} chunks. Bản mới: {len(docs_new)} chunks.")

        # === PHA 2: SEMANTIC CROSS-MATCH (Cho cả Điều cụ thể và Semantic Query) ===
        if dieu_filter or intent == "SPECIFIC":
            # Case 2.1: Bản cũ có nhưng bản mới không tìm thấy Điều đó (Bị đánh số lệch hoặc đổi vị trí)
            if docs_old and not docs_new:
                print(f"[Paired RAG] Pha 2: Điều mong muốn có ở Bản cũ nhưng rỗng ở Bản mới. Đang tìm chéo theo ngữ nghĩa...")
                cross_text = docs_old[0]
                results_new = self._semantic_cross_query(cross_text, new_src, engine, half_k)
                docs_new = results_new.get("documents", [[]])[0] if results_new else []
                
            # Case 2.2: Bản mới có nhưng bản cũ rỗng
            elif docs_new and not docs_old:
                print(f"[Paired RAG] Pha 2: Điều mong muốn có ở Bản mới nhưng rỗng ở Bản cũ. Đang tìm chéo theo ngữ nghĩa...")
                cross_text = docs_new[0]
                results_old = self._semantic_cross_query(cross_text, old_src, engine, half_k)
                docs_old = results_old.get("documents", [[]])[0] if results_old else []
                
            # Case 2.3: Cả hai bên đều có kết quả nhưng vẫn cross-query để bắt trường hợp renumbering/nhầm lẫn chéo
            elif docs_old and docs_new:
                print(f"[Paired RAG] Pha 2: Thực hiện Semantic Cross-Query bổ sung để tối ưu hóa đối chiếu...")
                # Lấy chunk đầu của old quét chéo sang new
                extra_new = self._semantic_cross_query(docs_old[0], new_src, engine, 1)
                # Lấy chunk đầu của new quét chéo sang old
                extra_old = self._semantic_cross_query(docs_new[0], old_src, engine, 1)
                
                results_old = self._merge_results(results_old, extra_old)
                results_new = self._merge_results(results_new, extra_new)

        # === GỘP VÀ LOẠI BỎ TRÙNG LẶP ===
        merged = self._merge_results(results_old, results_new)
        final_docs = merged.get("documents", [[]])[0]
        
        if not final_docs:
            yield "❌ Không tìm thấy văn bản pháp lý nào phù hợp để đối chiếu."
            return
            
        print(f"[Paired RAG] Tổng cộng gom được {len(final_docs)} chunks không trùng lặp cho LLM lập luận.")
        
        # 3. Tạo diff tự động mức từ/cụm từ
        from src.diff.text_diff_engine import TextDiffEngine
        diff_engine = TextDiffEngine()
        
        metas_old = results_old.get("metadatas", [[]])[0] if results_old else []
        texts_old = results_old.get("documents", [[]])[0] if results_old else []
        metas_new = results_new.get("metadatas", [[]])[0] if results_new else []
        texts_new = results_new.get("documents", [[]])[0] if results_new else []
        
        diff_text = diff_engine.diff_paired_chunks(
            chunks_old=metas_old,
            texts_old=texts_old,
            chunks_new=metas_new,
            texts_new=texts_new
        )
        
        if diff_text:
            print(f"[Paired RAG] Đã tạo thành công kết quả DIFF tự động.")

        # NẾU LÀ YÊU CẦU LIỆT KÊ TOÀN BỘ (LIST_ALL): Trả về kết quả định dạng DIFF trực tiếp
        if intent == "LIST_ALL":
            print(f"[Paired RAG] Đối với LIST_ALL: Trả về kết quả định dạng DIFF trực tiếp để tránh loãng và ảo tưởng 100%.")
            yield self._format_diff_for_user(diff_text)
            return
        
        # 4. Sử dụng helper build_context_prompt từ RAGEngine
        prompt = engine._build_context_prompt(query, merged, diff_text=diff_text, intent=intent)
        
        # 5. Stream kết quả qua LLM (với temperature 0.0)
        for chunk in engine.llm.stream_response(prompt=prompt, system_prompt=engine.system_prompt):
            yield chunk
