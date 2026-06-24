import re
import difflib
from typing import Dict, Any, List

class CitationVerifier:
    def __init__(self, old_law_source: str, new_law_source: str):
        self.old_law_source = old_law_source
        self.new_law_source = new_law_source
        
        # Regex tìm định dạng (Theo BẢN GỐC, Chương I > Điều 1) hoặc (Theo BẢN MỚI - Điều 2)
        self.citation_pattern = re.compile(r"\([Tt]heo\s+([^,:\-]+)[,:\-]\s*([^)]+)\)")

    def _get_location_str(self, meta: dict) -> str:
        chuong = meta.get("chuong", "")
        muc    = meta.get("muc", "")
        dieu   = meta.get("dieu", "")
        _UNKNOWN_TOKENS = {"không rõ", "không xác định", "n/a", "none", ""}
        
        def _is_known(val: str) -> bool:
            return val.strip().lower() not in _UNKNOWN_TOKENS
            
        location_parts = [p for p in [chuong, muc, dieu] if _is_known(p)]
        return " > ".join(location_parts) if location_parts else "Không rõ vị trí"

    def verify(self, answer: str, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Kiểm tra các trích dẫn trong answer so với search_results.
        Trả về danh sách các kết quả xác thực.
        """
        results = []
        if not search_results:
            return results
            
        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        
        # Tách câu để so sánh chéo (cross-check)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', answer) if s.strip()]
        
        citations_found = self.citation_pattern.findall(answer)
        
        for ten_ban, vi_tri in set(citations_found):
            ten_ban_clean = ten_ban.strip().upper()
            vi_tri_clean = vi_tri.strip()
            
            # Quy đổi tên bản về source
            target_source = None
            if "GỐC" in ten_ban_clean or "CŨ" in ten_ban_clean:
                target_source = self.old_law_source
            elif "MỚI" in ten_ban_clean or "SỬA ĐỔI" in ten_ban_clean:
                target_source = self.new_law_source
                
            found_doc = None
            best_doc = ""
            
            # Tìm context tương ứng
            for doc, meta in zip(documents, metadatas):
                if not meta: continue
                source = meta.get("source", "")
                
                # Bỏ qua kiểm tra source nếu target_source không rõ ràng, chỉ check vị trí
                if target_source and source != target_source:
                    continue
                    
                loc_str = self._get_location_str(meta)
                if vi_tri_clean.lower() in loc_str.lower() or loc_str.lower() in vi_tri_clean.lower():
                    found_doc = meta
                    best_doc = doc
                    break
            
            # Nếu tìm thấy nguồn, tìm câu văn chứa trích dẫn này để check similarity
            status = "unknown"
            grounding_text = ""
            ratio = 0.0
            
            if found_doc:
                # Trích xuất đoạn Grounding (tối đa 500 ký tự đầu để hiển thị)
                grounding_text = best_doc[:500] + ("..." if len(best_doc) > 500 else "")
                
                # Tìm câu có chứa trích dẫn này
                relevant_sentence = ""
                for s in sentences:
                    if vi_tri_clean in s or ten_ban in s:
                        relevant_sentence += s + " "
                        
                if relevant_sentence:
                    # Chúng ta dùng chung các từ vựng để cross-check
                    words_in_sentence = set(re.findall(r'\w+', relevant_sentence.lower()))
                    words_in_doc = set(re.findall(r'\w+', best_doc.lower()))
                    overlap = len(words_in_sentence.intersection(words_in_doc))
                    ratio = overlap / max(len(words_in_sentence), 1)
                    
                    if ratio < 0.15:
                        status = "warning" # Hallucinated content
                    else:
                        status = "verified"
                else:
                    status = "verified" # Không trích xuất được câu nhưng nguồn có tồn tại
                    ratio = 1.0
            else:
                status = "not_found"
                ratio = 0.0
                
            results.append({
                "citation": f"(Theo {ten_ban}, {vi_tri})",
                "status": status,
                "grounding": grounding_text,
                "word_ratio": ratio
            })
            
        return results
