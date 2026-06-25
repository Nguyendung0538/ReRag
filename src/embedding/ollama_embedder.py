from ollama import Client
import requests
from typing import List

class OllamaEmbedder:
    """
    Class giao tiep voi Ollama API cuc bo de lay embeddings.
    Khuyen nghi dung qwen3-embedding:8b theo setup cua user.
    
    Tham so keep_alive:
    - keep_alive=0  : Ollama unload model khoi VRAM ngay sau request -> giai phong tai nguyen cho LLM
    - keep_alive=-1 : Ollama giu model trong VRAM mai mai (mac dinh Ollama la 5 phut)
    - keep_alive=300: Giu 300 giay (5 phut)
    Mac dinh dung 0 de nhuong VRAM cho qwen3:8b khi khong can embed.
    """

    def __init__(self, model_name: str = "qwen3-embedding:8b", keep_alive: int = 300, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.keep_alive = keep_alive
        self.base_url = base_url
        self.client = Client(
            host=self.base_url,
            headers={"ngrok-skip-browser-warning": "true"}
        )

    def embed_text(self, text: str) -> List[float]:
        """Gui doan text qua Ollama de sinh vector float."""
        response = self.client.embeddings(
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

            response = self.client.embed(
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
        Chu dong yeu cau Ollama unload model khoi VRAM ngay lap tuc.
        Goi ham nay sau khi hoan tat ingestion de nhuong VRAM cho LLM.
        """
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=10,
            )
            print(f"[Embedder] Model '{self.model_name}' da duoc unload khoi VRAM.")
        except Exception as e:
            print(f"[Embedder] Khong the unload model: {e}")


if __name__ == "__main__":
    embedder = OllamaEmbedder()
    try:
        vec = embedder.embed_text("Xin chào, đây là bài test hệ thống RAG.")
        print(f"Lay vector thanh cong! Kich thuoc chieu cua vector: {len(vec)}")
        embedder.unload()
    except Exception as e:
        print(f"Loi khi lay thong tin tu Ollama: {e}\nHay chac chan Ollama dang chay voi model qwen3-embedding:8b")

