import re
from typing import List, Dict, Any

class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        
    def __repr__(self):
        return f"DocumentChunk(dieu={self.metadata.get('dieu')}, chuong={self.metadata.get('chuong')}, text_len={len(self.text)})"

    def to_dict(self):
        return {
            "metadata": self.metadata,
            "content": self.text
        }

class LegalChunker:
    """
    Bộ chia text đặc thù cho văn bản pháp lý.
    Sử dụng Regex để nhận diện Chương, Điều.
    """
    def __init__(self):
        # Regex theo implementation plan
        self.dieu_pattern = re.compile(r"^Điều\s+(\d+)\.", re.IGNORECASE)
        self.chuong_pattern = re.compile(r"^Chương\s+[IVXLCDM]+\s*[:\-\.]?", re.IGNORECASE)
        self.muc_pattern = re.compile(r"^Mục\s+\d+\s*[:\-\.]?", re.IGNORECASE)
        
    def chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        chunks = []
        current_chunk_lines = []
        pending_headers = []  # Lưu dòng chữ của Chương/Mục chờ nối vào Điều hoặc Chunk tiếp theo
        collecting_headers = False
        
        current_chuong = "Không xác định"
        chunk_chuong = "Không xác định"
        current_muc = "Không xác định"
        chunk_muc = "Không xác định"
        current_dieu = "Lời nói đầu / Không xác định"
        
        lines = text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Kiểm tra xem dòng hiện tại có chứa Chương không
            match_chuong = self.chuong_pattern.search(line_stripped)
            if match_chuong:
                current_chuong = line_stripped
                current_muc = "Không xác định"  # Reset Mục khi sang Chương mới
                collecting_headers = True
                pending_headers.append(line)
                continue
                
            # Kiểm tra xem dòng hiện tại có chứa Mục không (VD: Mục 1, Mục 2)
            match_muc = self.muc_pattern.search(line_stripped)
            if match_muc:
                current_muc = line_stripped
                collecting_headers = True
                pending_headers.append(line)
                continue
                
            # Kiểm tra xem dòng hiện tại có chứa Điều không
            match_dieu = self.dieu_pattern.search(line_stripped)
            if match_dieu:
                collecting_headers = False
                
                # Nếu đã có nội dung chữ được gom tụ, tạo thành 1 chunk kín (của Điều trước đó)
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        meta = base_metadata.copy()
                        meta.update({"chuong": chunk_chuong, "muc": chunk_muc, "dieu": current_dieu})
                        chunks.append(DocumentChunk(text=chunk_text, metadata=meta))
                    current_chunk_lines = []
                
                # Bắt đầu gom tụ đoạn mới cho Điều mới
                điều_number = match_dieu.group(1)
                current_dieu = f"Điều {điều_number}"
                chunk_chuong = current_chuong
                chunk_muc = current_muc
                
                # Nối các headers (Chương X, Mục Y) đứng liền trước nếu có
                if pending_headers:
                    current_chunk_lines.extend(pending_headers)
                    pending_headers = []
                    
                current_chunk_lines.append(line)
            else:
                # Dòng nội dung bình thường
                if collecting_headers:
                    pending_headers.append(line)
                else:
                    current_chunk_lines.append(line)
                
        # Gom phần còn lại cuối cùng (Điều cuối của văn bản)
        if current_chunk_lines or pending_headers:
            current_chunk_lines.extend(pending_headers)
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                meta = base_metadata.copy()
                meta.update({"chuong": chunk_chuong, "muc": chunk_muc, "dieu": current_dieu})
                chunks.append(DocumentChunk(text=chunk_text, metadata=meta))
            
        return chunks
