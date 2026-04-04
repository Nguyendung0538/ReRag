import os
from .docx_loader import load_docx
from .pdf_loader import load_pdf
from .legal_chunker import LegalChunker, DocumentChunk
from typing import List

def process_document(file_source, filename: str = None, ext: str = None) -> List[DocumentChunk]:
    """
    Hàm tổng hợp cho pipeline xử lý tài liệu.
    Hỗ trợ `file_source` là đường dẫn string hoặc đối tượng file-like (ví dụ UploadedFile từ Streamlit).
    """
    if isinstance(file_source, str):
        if ext is None:
            ext = os.path.splitext(file_source)[1].lower()
        if filename is None:
            filename = os.path.basename(file_source)
        if not os.path.exists(file_source):
            raise FileNotFoundError(f"Không tìm thấy file: {file_source}")
    else:
        if ext is None or filename is None:
            raise ValueError("Khi tải lên trực tiếp từ RAM, phải cung cấp `filename` và `ext`.")
        ext = ext.lower()
        
    if ext == ".docx":
        text = load_docx(file_source)
        doc_type = "DOCX"
    elif ext == ".pdf":
        text = load_pdf(file_source)
        doc_type = "PDF"
    elif ext == ".txt":
        if isinstance(file_source, str):
            with open(file_source, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = file_source.read().decode("utf-8")
        doc_type = "TXT"
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")
    base_metadata = {
        "source": getattr(file_source, 'name', filename) if filename else "unknown",
        "type": doc_type
    }
    
    # Khởi tạo Chunker chuyên dụng cho pháp lý
    chunker = LegalChunker()
    chunks = chunker.chunk(text, base_metadata)
    
    return chunks

if __name__ == "__main__":
    # Test file mẫu nếu có, ví dụ: process_document('mau_hop_dong.docx')
    pass
