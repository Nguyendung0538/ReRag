import difflib
from typing import List, Dict, Any, Tuple

class TextDiffEngine:
    """
    So sánh 2 đoạn văn bản ở mức word/cụm từ bằng difflib.SequenceMatcher.
    Không cần LLM, không cần internet, chạy hoàn toàn local.
    """

    # Ngưỡng tối thiểu để coi 2 đoạn văn là "có liên quan" (cùng điều)
    PAIR_MIN_RATIO = 0.4
    # Ngưỡng để bỏ qua các điều giống nhau hoàn toàn (không có gì thay đổi)
    IDENTICAL_THRESHOLD = 1.0

    def diff_texts(self, old_text: str, new_text: str) -> str:
        import re
        
        # Tách thành câu để tìm các câu giống nhau hoàn toàn
        old_sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', old_text) if s.strip()]
        new_sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', new_text) if s.strip()]
        
        matcher = difflib.SequenceMatcher(None, old_sentences, new_sentences, autojunk=False)
        
        old_out = []
        new_out = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                if not old_out or old_out[-1] != "[...]":
                    old_out.append("[...]")
                    new_out.append("[...]")
            else:
                # Nếu câu có sự thay đổi, tiến hành so sánh từng từ bên trong
                sub_old_text = " ".join(old_sentences[i1:i2])
                sub_new_text = " ".join(new_sentences[j1:j2])
                
                old_tokens = sub_old_text.split()
                new_tokens = sub_new_text.split()
                sub_matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
                
                new_rendered = self._render_new_with_bold(new_tokens, sub_matcher.get_opcodes())
                
                if sub_old_text:
                    old_out.append(sub_old_text)
                if new_rendered:
                    new_out.append(new_rendered)
                    
        final_old = " ".join(old_out).replace("[...] [...]", "[...]").replace("[...]", "").strip()
        final_new = " ".join(new_out).replace("[...] [...]", "[...]").replace("[...]", "").strip()
        
        # Clean up multiple spaces
        final_old = re.sub(r'\s+', ' ', final_old).strip()
        final_new = re.sub(r'\s+', ' ', final_new).strip()
        
        return f"  + Bản cũ: {final_old}\n  + Bản mới: {final_new}"

    def _render_new_with_bold(self, new_tokens: List[str], opcodes: List[Tuple]) -> str:
        """Ghép lại new_tokens, bọc **bold** quanh các token insert/replace."""
        parts = []
        for tag, i1, i2, j1, j2 in opcodes:
            segment = " ".join(new_tokens[j1:j2])
            if tag == "equal":
                parts.append(segment)
            elif tag in ("replace", "insert"):
                if segment:
                    parts.append(f"**{segment}**")
        return " ".join(parts)

    def diff_paired_chunks(
        self,
        chunks_old: List[Dict[str, Any]],
        texts_old: List[str],
        chunks_new: List[Dict[str, Any]],
        texts_new: List[str],
    ) -> str:
        """
        Ghép cặp chunk theo metadata `dieu`, chạy diff từng cặp.
        - Bỏ qua các điều giống nhau hoàn toàn.
        - Ghi nhận điều bị xóa và điều mới thêm vào.
        Trả về toàn bộ block diff formatted.
        """
        pairs, only_in_old, only_in_new = self._pair_chunks(chunks_old, texts_old, chunks_new, texts_new)

        if not pairs and not only_in_old and not only_in_new:
            return ""

        diff_blocks = []

        # Các điều có ở cả 2 bản — chỉ in ra nếu CÓ SỰ THAY ĐỔI
        for dieu_label, old_text, new_text in pairs:
            ratio = difflib.SequenceMatcher(None, old_text, new_text).ratio()
            if ratio >= self.IDENTICAL_THRESHOLD:
                continue  # Bỏ qua điều giống nhau hoàn toàn
            block = f"- {dieu_label}:\n{self.diff_texts(old_text, new_text)}"
            diff_blocks.append(block)

        # Các điều bị xóa ở bản mới
        for key, old_text in only_in_old:
            block = (
                f"- {key} (ĐÃ BỊ XÓA ở Bản mới):\n"
                f"  + Bản cũ: {old_text}\n"
                f"  + Bản mới: (không còn điều khoản này)"
            )
            diff_blocks.append(block)

        # Các điều mới thêm vào ở bản mới
        for key, new_text in only_in_new:
            block = (
                f"- {key} (MỚI THÊM ở Bản mới):\n"
                f"  + Bản cũ: (không có điều khoản này)\n"
                f"  + Bản mới: **{new_text}**"
            )
            diff_blocks.append(block)

        return "\n\n".join(diff_blocks)

    def _pair_chunks(
        self,
        metas_old: List[Dict[str, Any]],
        texts_old: List[str],
        metas_new: List[Dict[str, Any]],
        texts_new: List[str],
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Ghép cặp chunk theo trường 'dieu' trong metadata.
        Trả về:
          - pairs: list[(label, old_text, new_text)]
          - only_in_old: list[(key, old_text)] — bị xóa ở bản mới
          - only_in_new: list[(key, new_text)] — mới thêm ở bản mới
        """
        old_map = {}
        for meta, text in zip(metas_old, texts_old):
            if not meta:
                continue
            key = meta.get("dieu", "")
            if key and key not in old_map:
                old_map[key] = text

        new_map = {}
        for meta, text in zip(metas_new, texts_new):
            if not meta:
                continue
            key = meta.get("dieu", "")
            if key and key not in new_map:
                new_map[key] = text

        pairs = []
        matched_old_keys = set()
        matched_new_keys = set()

        # Pha 1: Exact match theo tên Điều + kiểm tra tương đồng nội dung (tránh lệch số Điều)
        for key, old_text in old_map.items():
            if key in new_map:
                ratio = difflib.SequenceMatcher(None, old_text, new_map[key]).ratio()
                if ratio >= self.PAIR_MIN_RATIO:
                    pairs.append((key, old_text, new_map[key]))
                    matched_old_keys.add(key)
                    matched_new_keys.add(key)

        # Pha 2: Semantic fallback — tìm match tốt nhất theo nội dung (chống renumbering)
        for key, old_text in old_map.items():
            if key in matched_old_keys:
                continue

            best_key = self._find_best_semantic_match(old_text, new_map, matched_new_keys)
            if best_key:
                label = f"{key} (chuyển thành {best_key} ở Bản mới)"
                pairs.append((label, old_text, new_map[best_key]))
                matched_old_keys.add(key)
                matched_new_keys.add(best_key)

        # Điều chỉ có ở bản cũ (bị xóa ở bản mới)
        only_in_old = [(k, v) for k, v in old_map.items() if k not in matched_old_keys]
        # Điều chỉ có ở bản mới (mới thêm)
        only_in_new = [(k, v) for k, v in new_map.items() if k not in matched_new_keys]

        return pairs, only_in_old, only_in_new

    def _find_best_semantic_match(self, old_text: str, new_map: Dict[str, str], already_matched: set) -> str | None:
        """Tìm chunk new có nội dung gần nhất với old_text (dùng SequenceMatcher ratio)."""
        best_ratio = 0.3
        best_key = None
        for key, new_text in new_map.items():
            if key in already_matched:
                continue
            ratio = difflib.SequenceMatcher(None, old_text, new_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
        return best_key
