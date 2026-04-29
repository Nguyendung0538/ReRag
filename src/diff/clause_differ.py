import difflib
from typing import List, Dict, Any
from collections import defaultdict
from src.ingestion.legal_chunker import DocumentChunk


class ClauseDiff:
    """
    Kết quả so sánh 1 Điều khoản giữa Bản gốc và Bản mới.
    """
    def __init__(self, dieu: str, change_type: str, old_text: str, new_text: str, diff_details: List[Dict[str, str]]):
        self.dieu = dieu
        self.change_type = change_type  # "modified", "added", "removed"
        self.old_text = old_text
        self.new_text = new_text
        self.diff_details = diff_details  # list of {type, old, new}

    def __repr__(self):
        return f"ClauseDiff(dieu='{self.dieu}', type='{self.change_type}', details={len(self.diff_details)})"


class ClauseDiffer:
    """
    So sánh 2 bộ DocumentChunks theo từng Điều khoản (metadata 'dieu').
    
    Sử dụng difflib.SequenceMatcher để phát hiện thay đổi ở cấp dòng,
    đảm bảo không bỏ sót bất kỳ thay đổi nhỏ nào (số, ngày, tên riêng).
    """

    def _group_by_dieu(self, chunks: List[DocumentChunk]) -> Dict[str, str]:
        """
        Gom text của các chunks cùng dieu lại thành 1 chuỗi.
        Trả về dict: { "Điều 1": "toàn bộ text...", ... }
        """
        groups = defaultdict(list)
        for c in chunks:
            dieu = c.metadata.get("dieu", "Không xác định")
            groups[dieu].append(c.text)
        
        return {dieu: "\n".join(texts) for dieu, texts in groups.items()}

    def _diff_text(self, old_text: str, new_text: str) -> List[Dict[str, str]]:
        """
        So sánh 2 đoạn text ở cấp dòng bằng SequenceMatcher.
        Trả về danh sách các thay đổi: [{type, old, new}, ...]
        """
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        diffs = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            diffs.append({
                "type": tag,  # 'replace', 'insert', 'delete'
                "old": "\n".join(old_lines[i1:i2]),
                "new": "\n".join(new_lines[j1:j2]),
            })
        
        return diffs

    def compare(self, old_chunks: List[DocumentChunk], new_chunks: List[DocumentChunk]) -> List[ClauseDiff]:
        """
        So sánh toàn bộ 2 bộ chunks, ghép cặp theo 'dieu', trả về danh sách ClauseDiff.
        Chỉ trả về các Điều CÓ thay đổi (bỏ qua unchanged).
        """
        old_map = self._group_by_dieu(old_chunks)
        new_map = self._group_by_dieu(new_chunks)
        
        all_dieus = sorted(set(list(old_map.keys()) + list(new_map.keys())))
        
        results = []
        for dieu in all_dieus:
            old_text = old_map.get(dieu, "")
            new_text = new_map.get(dieu, "")
            
            if not old_text and new_text:
                # Điều mới được thêm vào Bản mới
                results.append(ClauseDiff(
                    dieu=dieu, change_type="added",
                    old_text="", new_text=new_text, diff_details=[]
                ))
            elif old_text and not new_text:
                # Điều bị xóa khỏi Bản mới
                results.append(ClauseDiff(
                    dieu=dieu, change_type="removed",
                    old_text=old_text, new_text="", diff_details=[]
                ))
            elif old_text == new_text:
                # Giống hệt → bỏ qua
                continue
            else:
                # Có thay đổi → tính diff chi tiết
                diffs = self._diff_text(old_text, new_text)
                if diffs:
                    results.append(ClauseDiff(
                        dieu=dieu, change_type="modified",
                        old_text=old_text, new_text=new_text,
                        diff_details=diffs
                    ))
        
        return results

    def _highlight_changes(self, old_text: str, new_text: str) -> str:
        """
        So sánh ở cấp từ (word-level) và chèn **bold** quanh các từ thay đổi/thêm mới trong new_text.
        Trả về new_text đã có bold markers.
        """
        old_words = old_text.split()
        new_words = new_text.split()
        
        matcher = difflib.SequenceMatcher(None, old_words, new_words)
        result = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                result.extend(new_words[j1:j2])
            elif tag in ("replace", "insert"):
                result.append("**" + " ".join(new_words[j1:j2]) + "**")
            # 'delete' → từ bị xóa, không cần thêm gì vào new_text
        
        return " ".join(result)

    def format_diff_for_prompt(self, diffs: List[ClauseDiff]) -> str:
        """
        Render danh sách ClauseDiff thành text để inject vào LLM prompt.
        Chỉ output phần thay đổi (VD: 18.3) thay vì toàn bộ Điều.
        Phần Bản mới đã được pre-compute bold markers (**...**) quanh từ thay đổi.
        """
        if not diffs:
            return "Không phát hiện thay đổi nào giữa hai tài liệu."
        
        lines = []
        for d in diffs:
            if d.change_type == "added":
                lines.append(f"- [{d.dieu}] — THÊM MỚI (chỉ có trong Bản mới):")
                lines.append(f"  + Bản mới: {d.new_text[:500].strip()}")
                
            elif d.change_type == "removed":
                lines.append(f"- [{d.dieu}] — BỊ XÓA (chỉ có trong Bản gốc):")
                lines.append(f"  + Bản cũ: {d.old_text[:500].strip()}")
                
            elif d.change_type == "modified":
                lines.append(f"- [{d.dieu}] — THAY ĐỔI:")
                # Chỉ output phần thay đổi (diff_details), không dump toàn bộ Điều
                for detail in d.diff_details:
                    if detail["type"] == "replace":
                        highlighted = self._highlight_changes(detail["old"], detail["new"])
                        lines.append(f"  + Bản cũ: {detail['old'].strip()}")
                        lines.append(f"  + Bản mới: {highlighted.strip()}")
                    elif detail["type"] == "insert":
                        lines.append(f"  + Thêm mới: **{detail['new'].strip()}**")
                    elif detail["type"] == "delete":
                        lines.append(f"  + Bị xóa: {detail['old'].strip()}")
            
            lines.append("")  # Dòng trống phân cách
        
        return "\n".join(lines)

    def filter_by_dieu(self, diffs: List[ClauseDiff], dieu_name: str) -> List[ClauseDiff]:
        """Lọc diff theo tên Điều cụ thể (VD: 'Điều 6')."""
        return [d for d in diffs if d.dieu == dieu_name]
