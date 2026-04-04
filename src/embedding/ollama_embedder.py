import ollama
import requests
from typing import List

class OllamaEmbedder:
    """
    Class giao tiếp với Ollama API cục bộ để lấy embeddings.
    Khuyến nghị dùng qwen3-embedding:8b theo setup của user.
    
    Tham số keep_alive:
    - keep_alive=0  : Ollama unload model khỏi VRAM ngay sau request → giải phóng tài nguyên cho LLM
    - keep_alive=-1 : Ollama giữ model trong VRAM mãi mãi (mặc định Ollama là 5 phút)
    - keep_alive=300: Giữ 300 giây (5 phút)
    Mặc định dùng 0 để nhường VRAM cho qwen3:8b khi không cần embed.
    """
    OLLAMA_BASE_URL = "http://localhost:11434"

    def __init__(self, model_name: str = "qwen3-embedding:8b", keep_alive: int = 0):
        self.model_name = model_name
        self.keep_alive = keep_alive

    def embed_text(self, text: str) -> List[float]:
        """Gửi đoạn text qua Ollama để sinh vector float."""
        response = ollama.embeddings(
            model=self.model_name,
            prompt=text,
            keep_alive=self.keep_alive,
        )
        return response["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Nhúng hàng loạt các đoạn văn bản.
        keep_alive=0 chỉ áp dụng sau request CUỐI CÙNG — các request giữa chừng
        vẫn giữ model trong VRAM để tránh reload liên tục.
        """
        results = []
        last_idx = len(texts) - 1
        for i, text in enumerate(texts):
            # Chỉ unload ở chunk cuối cùng
            ka = self.keep_alive if i == last_idx else -1
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text,
                keep_alive=ka,
            )
            results.append(response["embedding"])
        return results

    def unload(self):
        """
        Chủ động yêu cầu Ollama unload model khỏi VRAM ngay lập tức.
        Gọi hàm này sau khi hoàn tất ingestion để nhường VRAM cho LLM.
        """
        try:
            requests.post(
                f"{self.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": self.model_name, "prompt": "", "keep_alive": 0},
                timeout=5,
            )
            print(f"[Embedder] Model '{self.model_name}' đã được unload khỏi VRAM.")
        except Exception as e:
            print(f"[Embedder] Không thể unload model: {e}")


if __name__ == "__main__":
    embedder = OllamaEmbedder()
    try:
        vec = embedder.embed_text("Xin chào, đây là bài test hệ thống RAG.")
        print(f"✅ Lấy vector thành công! Kích thước chiều của vector: {len(vec)}")
        embedder.unload()
    except Exception as e:
        print(f"❌ Lỗi khi lấy thông tin từ Ollama: {e}\nHãy chắc chắn Ollama đang chạy với model qwen3-embedding:8b")

