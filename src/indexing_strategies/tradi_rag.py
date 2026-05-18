from typing import List, Dict, Any
from .base_indexing import BaseIndexingStrategy
from src.ingestion.legal_chunker import DocumentChunk
from src.embedding.chroma_manager import ChromaManager

class TradiRAGIndexing(BaseIndexingStrategy):
    """
    Chiến thuật nạp truyền thống: Chunk -> Vectors -> Storage.
    Giữ nguyên bộ code cũ của hệ thống.
    """
    def __init__(self, embedding_model: str = "qwen3-embedding:8b"):
        self.db_manager = ChromaManager(
            collection_name="legal_compare",
            embedding_model=embedding_model
        )
        # Khởi tạo DB luôn mỗi lần instantiate
        self.db_manager.reset_collection()

    def index(self, chunks: List[DocumentChunk], **kwargs) -> bool:
        if not chunks:
            return False
        
        self.db_manager.add_documents(chunks)
        # Giải phóng VRAM để nhường cho LLM
        if hasattr(self.db_manager, 'embedder'):
            self.db_manager.embedder.unload()
        return True

    def build_context(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Tìm kiếm `top_k` kết quả gần với vector query từ Chroma.
        Nếu muốn filter, có thể lấy parameter 'where' truyền qua kwargs.
        """
        where = kwargs.get('where', None)
        return self.db_manager.query(query, n_results=top_k, where=where)

    def get_all_by_source(self, source: str) -> Dict[str, Any]:
        """
        Lấy TOÀN BỘ chunks của một file nguồn mà không cần vector query.
        Dùng khi người dùng yêu cầu so sánh tổng thể (không hỏi về Điều cụ thể).
        """
        where = {"source": {"$eq": source}}
        return self.db_manager.get_all(where=where)
