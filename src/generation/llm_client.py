from ollama import Client
from typing import Iterator

class LLMClient:
    """
    Module giao tiep voi nhanh sinh van ban (Text Generation) cua Ollama.
    Dung de lap luan, so sanh va tra ve cau tra loi cho nguoi dung.
    """
    def __init__(self, model_name: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = Client(
            host=self.base_url,
            headers={"ngrok-skip-browser-warning": "true"}
        )

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Gui yeu cau khoi tao cau tra loi tron ven (khong stream).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": 0.0}
        )
        return response['message']['content']

    def stream_response(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        """
        Sinh cau tra loi dang luong (stream), giup giao dien CLI hien thi chu chay lien tuc nhu ChatGPT.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        stream = self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            options={"temperature": 0.0}
        )
        
        for chunk in stream:
            yield chunk['message']['content']

if __name__ == "__main__":
    client = LLMClient()
    print("Testing LLM...")
    for text in client.stream_response("Xin chào, bạn là ai?"):
        print(text, end="", flush=True)
    print()
