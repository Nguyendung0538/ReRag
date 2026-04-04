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
