from abc import ABC, abstractmethod
from typing import Iterator, Any

class QueryStrategy(ABC):
    """
    Interface gốc cho tất cả các Chiến thuật Truy vấn nội dung Pháp lý.
    """
    
    @abstractmethod
    def stream_execute(self, query: str, engine: Any, top_k: int = 6) -> Iterator[str]:
        """
        Thực hiện một luồng lấy context + prompt + stream text trả về UI.
        
        Args:
            query (str): Câu hỏi của người dùng.
            engine (Any): Thể hiện (instance) của LegalRAGEngine chứa LLMClient, ChromaManager, system_prompts.
            top_k (int): Số mẩu văn bản lấy từ CSDL.
            
        Returns:
            Iterator[str]: Stream các phần chữ được sinh ra từ LLM.
        """
        pass

    def _extract_metadata_filter(self, query: str) -> dict | None:
        """Phân tích câu hỏi để tự động xuất filter metadata (Ví dụ: bóc tách chính xác Điều 6)."""
        import re
        from src.ingestion.legal_chunker import LegalChunker
        match = re.search(r"điều\s+([\dIVXLCDM]+)", query, re.IGNORECASE)
        if match:
            chunker = LegalChunker()
            normalized = chunker._normalize_dieu_number(match.group(1))
            return {"dieu": f"Điều {normalized}"}
        return None
