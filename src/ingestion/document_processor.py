import os
from .docx_loader import load_docx
from .pdf_loader import load_pdf
from .legal_chunker import LegalChunker, DocumentChunk
from typing import List

def process_document(file_path: str) -> List[DocumentChunk]:
    """
    Hàm tổng hợp cho pipeline xử lý tài liệu.
    Nhận tham số là file path -> Đọc text bằng Loader -> Chia theo Chunker -> Trả về danh sách DocumentChunk.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
    if ext == ".docx":
        text = load_docx(file_path)
        doc_type = "DOCX"
    elif ext == ".pdf":
        text = load_pdf(file_path)
        doc_type = "PDF"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc_type = "TXT"
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")
        
    filename = os.path.basename(file_path)
    base_metadata = {
        "source": filename,
        "type": doc_type
    }
    
    # Khởi tạo Chunker chuyên dụng cho pháp lý
    chunker = LegalChunker()
    chunks = chunker.chunk(text, base_metadata)
    
    return chunks

if __name__ == "__main__":
    # Test file mẫu nếu có, ví dụ: process_document('mau_hop_dong.docx')
    pass
