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
        Nhung hang loat cac doan van ban bang batch API (ollama.embed).
        Chia nho thanh tung batch 100 de tranh timeout hoac tran VRAM.
        keep_alive=0 chi ap dung cho batch cuoi cung de giai phong bo nho.
        """
        if not texts:
            return []

        results = []
        batch_size = 100
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            sub_batch = texts[start_idx:end_idx]

            # Chi unload o batch cuoi cung
            ka = self.keep_alive if i == num_batches - 1 else -1

            response = ollama.embed(
                model=self.model_name,
                input=sub_batch,
                keep_alive=ka,
            )

            embeddings = getattr(response, "embeddings", None)
            if embeddings is None:
                embeddings = response.get("embeddings", [])

            results.extend(embeddings)

        return results

    def unload(self):
        """
        Chủ động yêu cầu Ollama unload model khỏi VRAM ngay lập tức.
        Gọi hàm này sau khi hoàn tất ingestion để nhường VRAM cho LLM.
        """
        try:
            requests.post(
                f"{self.OLLAMA_BASE_URL}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=10,
            )
            print(f"[Embedder] Model '{self.model_name}' đã được unload khỏi VRAM.")
        except Exception as e:
            print(f"[Embedder] Không thể unload model: {e}")


if __name__ == "__main__":
    embedder = OllamaEmbedder()
    try:
        vec = embedder.embed_text("Xin chào, đây là bài test hệ thống RAG.")
        print(f"Lay vector thanh cong! Kich thuoc chieu cua vector: {len(vec)}")
        embedder.unload()
    except Exception as e:
        print(f"Loi khi lay thong tin tu Ollama: {e}\nHay chac chan Ollama dang chay voi model qwen3-embedding:8b")

