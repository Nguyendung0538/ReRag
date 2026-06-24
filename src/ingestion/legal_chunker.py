import re
from typing import List, Dict, Any

class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        
    def __repr__(self):
        return f"DocumentChunk(dieu={self.metadata.get('dieu')}, text_len={len(self.text)})"

    def to_dict(self):
        return {
            "metadata": self.metadata,
            "content": self.text
        }

class LegalChunker:
    """
    Bộ chia text đặc thù cho văn bản hợp đồng đối chiếu.
    Sử dụng Regex để nhận diện Điều khoản, và thông tin Bên tham gia.
    """
    def __init__(self):
        # Regex hỗ trợ cả hợp đồng (số La Mã và Ả Rập)
        self.chuong_pattern = re.compile(r"^Chương\s+([IVXLCDM\d]+)(?:\s*[:\.]|\s+|$)", re.IGNORECASE)
        self.muc_pattern = re.compile(r"^Mục\s+(\d+)(?:\s*[:\.]|\s+|$)", re.IGNORECASE)
        self.dieu_pattern = re.compile(r"^Điều\s+(\d+|[IVXLCDM]+)(?:\s*[:\.]|\s+|$)", re.IGNORECASE)
        self.khoan_pattern = re.compile(r"^(\d+)\.\s+", re.IGNORECASE)
        # Regex nhận diện phần thông tin các bên (VD: I: BÊN CHO THUÊ NHÀ)
        self.party_pattern = re.compile(r"^([IVXLCDM]+)\s*[:\.]\s*(BÊN\s+.*)", re.IGNORECASE)

    def _normalize_dieu_number(self, s: str) -> str:
        s = s.upper()
        # Nếu chuỗi chỉ toàn là các kí tự số La Mã
        if all(c in 'IVXLCDM' for c in s) and not s.isdigit():
            roman_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
            int_val = 0
            for i in range(len(s)):
                if i > 0 and roman_val[s[i]] > roman_val[s[i - 1]]:
                    int_val += roman_val[s[i]] - 2 * roman_val[s[i - 1]]
                else:
                    int_val += roman_val[s[i]]
            return str(int_val)
        return s

    def chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        chunks = []
        current_chunk_lines = []
        
        current_chuong = ""
        current_muc = ""
        current_dieu = ""
        current_khoan = ""
        
        def emit_current_chunk():
            nonlocal current_chunk_lines
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines).strip()
                if chunk_text:
                    meta = base_metadata.copy()
                    # Xác định key dieu chính cho việc gom nhóm
                    group_label = "Lời nói đầu / Căn cứ"
                    if current_dieu:
                        group_label = current_dieu
                    elif current_muc:
                        group_label = current_muc
                    elif current_chuong:
                        group_label = current_chuong
                    
                    meta.update({
                        "chuong": current_chuong,
                        "muc": current_muc,
                        "dieu": group_label,
                        "khoan": current_khoan
                    })
                    chunks.append(DocumentChunk(text=chunk_text, metadata=meta))
                current_chunk_lines = []

        lines = text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Kiểm tra xem dòng hiện tại có phải thông tin các Bên không
            match_party = self.party_pattern.search(line_stripped)
            if match_party:
                emit_current_chunk()
                
                # Reset trạng thái phân cấp
                current_chuong = ""
                current_muc = ""
                current_khoan = ""
                
                party_info = match_party.group(2).strip()
                if party_info.endswith(':'):
                    party_info = party_info[:-1].strip()
                current_dieu = f"Thông tin {party_info}"
                
                current_chunk_lines.append(line)
                continue
                
            # Kiểm tra xem dòng hiện tại có chứa Chương không
            match_chuong = self.chuong_pattern.search(line_stripped)
            if match_chuong:
                emit_current_chunk()
                current_chuong = f"Chương {match_chuong.group(1)}"
                current_muc = ""
                current_dieu = ""
                current_khoan = ""
                current_chunk_lines.append(line)
                continue

            # Kiểm tra xem dòng hiện tại có chứa Mục không
            match_muc = self.muc_pattern.search(line_stripped)
            if match_muc:
                emit_current_chunk()
                current_muc = f"Mục {match_muc.group(1)}"
                current_dieu = ""
                current_khoan = ""
                current_chunk_lines.append(line)
                continue

            # Kiểm tra xem dòng hiện tại có chứa Điều không
            match_dieu = self.dieu_pattern.search(line_stripped)
            if match_dieu:
                emit_current_chunk()
                dieu_number_raw = match_dieu.group(1)
                dieu_number_normalized = self._normalize_dieu_number(dieu_number_raw)
                current_dieu = f"Điều {dieu_number_normalized}"
                current_khoan = ""
                current_chunk_lines.append(line)
                continue

            # Kiểm tra xem dòng hiện tại có chứa Khoản không
            match_khoan = self.khoan_pattern.search(line_stripped)
            if match_khoan and current_dieu:
                emit_current_chunk()
                current_khoan = f"Khoản {match_khoan.group(1)}"
                current_chunk_lines.append(line)
                continue

            # Dòng nội dung bình thường
            current_chunk_lines.append(line)
                
        # Gom phần còn lại cuối cùng
        emit_current_chunk()
            
        return chunks
