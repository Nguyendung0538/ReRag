import ollama
from typing import Iterator

class LLMClient:
    """
    Module giao tiếp với nhánh sinh văn bản (Text Generation) của Ollama.
    Dùng để lập luận, so sánh và trả về câu trả lời cho người dùng.
    """
    def __init__(self, model_name: str = "qwen3:8b"):
        self.model_name = model_name

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Gửi yêu cầu khởi tạo câu trả lời trọn vẹn (không stream).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=self.model_name,
            messages=messages
        )
        return response['message']['content']

    def stream_response(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        """
        Sinh câu trả lời dạng luồng (stream), giúp giao diện CLI hiển thị chữ chạy liên tục như ChatGPT.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        stream = ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            yield chunk['message']['content']

if __name__ == "__main__":
    client = LLMClient()
    print("Testing LLM...")
    for text in client.stream_response("Xin chào, bạn là ai?"):
        print(text, end="", flush=True)
    print()
