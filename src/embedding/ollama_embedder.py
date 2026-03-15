import ollama
from typing import List

class OllamaEmbedder:
    """
    Class giao tiếp với Ollama API cục bộ để lấy embeddings.
    Khuyến nghị dùng qwen3-embedding:8b theo setup của user.
    """
    def __init__(self, model_name: str = "qwen3-embedding:8b"):
        self.model_name = model_name
        
    def embed_text(self, text: str) -> List[float]:
        """
        Gửi đoạn text qua Ollama để sinh vector float.
        """
        response = ollama.embeddings(
            model=self.model_name,
            prompt=text
        )
        return response["embedding"]
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Nhúng hàng loạt các đoạn văn bản.
        """
        # API của python ollama hiện tại hỗ trợ embeddings cho 1 đoạn text mỗi lệnh gọi (`ollama.embeddings`).
        # Để nhúng batch, ta tạm thời dùng vòng lặp list comprehension.
        return [self.embed_text(text) for text in texts]

if __name__ == "__main__":
    # Test thử trực tiếp
    embedder = OllamaEmbedder()
    try:
        vec = embedder.embed_text("Xin chào, đây là bài test hệ thống RAG.")
        print(f"✅ Lấy vector thành công! Kích thước chiều của vector: {len(vec)}")
    except Exception as e:
        print(f"❌ Lỗi khi lấy thông tin từ Ollama: {e}\nHãy chắc chắn container/service Ollama đang chạy với model qwen3-embedding:8b")
