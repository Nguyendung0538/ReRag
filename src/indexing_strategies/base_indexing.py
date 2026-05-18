from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.ingestion.legal_chunker import DocumentChunk

class BaseIndexingStrategy(ABC):
    """
    Khung chuẩn cho các chiến thuật nạp dữ liệu (Indexing Strategies).
    Một chiến thuật Indexing sẽ bao gồm 2 bước:
    1. index: Nạp danh sách các chunks từ tài liệu. Có thể ghi vào DB hoặc giữ ở RAM.
    2. build_context: Nhận câu hỏi, tìm và gom lại các đoạn liên quan dưới cấu trúc chuẩn.
    """
    
    @abstractmethod
    def index(self, chunks: List[DocumentChunk], **kwargs) -> bool:
        """
        Xử lý và lưu trữ dữ liệu.
        Trả về True nếu nạp thành công, False nếu thất bại (VD: Quá giới hạn token).
        """
        pass

    @abstractmethod
    def build_context(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Trả về kết quả tìm kiếm theo định dạng dict chứa documents và metadatas 
        để tương thích với RAG Engine hiện tại.
        Format chuẩn mong mỏi:
        {
            "documents": [[doc1, doc2, ...]],
            "metadatas": [[meta1, meta2, ...]]
        }
        """
        pass

    def get_all_by_source(self, source: str) -> Dict[str, Any]:
        """
        Lấy toàn bộ chunks của một nguồn tài liệu.
        Subclass có thể override nếu muốn tối ưu hoá riêng.
        Mặc định raise NotImplementedError.
        """
        raise NotImplementedError("Subclass này chưa hỗ trợ get_all_by_source.")
