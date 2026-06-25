# Hướng dẫn Mượn GPU miễn phí từ Google Colab cho Hệ thống RAG

Hệ thống ReRag hỗ trợ việc kết nối với máy chủ Ollama bên ngoài. Bạn có thể tận dụng GPU T4 miễn phí của Google Colab để tăng tốc độ xử lý câu hỏi và nhúng văn bản.

## Bước 1: Chuẩn bị trên Google Colab
1. Truy cập [Google Colab](https://colab.research.google.com/) và tạo một Notebook mới.
2. Đổi cấu hình phần cứng:
   - Trên thanh menu, chọn **Runtime** (Thời gian chạy) -> **Change runtime type** (Thay đổi loại thời gian chạy).
   - Ở mục Hardware accelerator (Bộ tăng tốc phần cứng), chọn **T4 GPU** và nhấn **Save** (Lưu).

## Bước 2: Cài đặt và Khởi chạy Ollama
Tạo một ô mã (Code cell) mới trên Colab, dán đoạn mã sau vào và nhấn nút Run (Chạy):

```python
# 1. Cai dat zstd (can thiet cho viec giai nen Ollama)
!sudo apt-get update && sudo apt-get install -y zstd pciutils

# 2. Cai dat thu vien Ollama cho Linux
!curl -fsSL https://ollama.com/install.sh | sh

# 3. Khoi chay Ollama Server duoi dang process ngam va ghi log ra file
!pkill ollama || true
!OLLAMA_HOST=0.0.0.0 OLLAMA_MAX_LOADED_MODELS=2 nohup ollama serve > ollama.log 2>&1 &
!sleep 5

# Kiểm tra xem Ollama đã chạy thành công chưa
!curl http://127.0.0.1:11434/

# 4. Tải các model cần thiết (Embedding và LLM)
!ollama pull qwen3-embedding:8b
!ollama pull gemma4:e4b

# 5. Nạp sẵn (Preload) các model vào VRAM GPU để không bị trễ ở câu hỏi đầu tiên
import requests

models = ["qwen3-embedding:8b", "gemma4:e4b"]
for model in models:
    print(f"Đang nạp mô hình '{model}' vào VRAM GPU...")
    try:
        if "embedding" in model:
            # Mô hình embedding cần nạp qua endpoint /api/embed
            url = "http://127.0.0.1:11434/api/embed"
            payload = {"model": model, "input": "hello", "keep_alive": -1}
        else:
            # Mô hình sinh văn bản nạp qua /api/generate
            url = "http://127.0.0.1:11434/api/generate"
            payload = {"model": model, "prompt": "", "stream": False, "keep_alive": -1}
            
        response = requests.post(url, json=payload, timeout=180)
        if response.status_code == 200:
            print(f"Đã nạp xong mô hình '{model}' thành công!")
        else:
            print(f"Lỗi khi nạp '{model}': {response.status_code}")
    except Exception as e:
        print(f"Không thể kết nối để nạp mô hình '{model}': {e}")

print("Tất cả mô hình đã sẵn sàng hoạt động trên GPU!")
```

## Bước 3: Cấu hình Ngrok để mở cổng API ra ngoài (Expose Port)
Mặc định, Colab chỉ chạy Ollama ở `localhost:11434` bên trong máy ảo. Để máy tính của bạn kết nối được tới Colab, chúng ta cần dùng **Ngrok**.

1. Đăng ký tài khoản miễn phí tại [ngrok.com](https://dashboard.ngrok.com/signup).
2. Lấy mã **Authtoken** của bạn trong phần `Your Authtoken` trên trang quản trị ngrok.
3. Tạo thêm một ô mã (Code cell) trên Colab, dán đoạn mã sau và thay thế Token của bạn vào:

```python
# 1. Cài đặt thư viện pyngrok
!pip install pyngrok

from pyngrok import ngrok

# 2. Thay thế YOUR_NGROK_AUTHTOKEN bằng mã thực tế của bạn
NGROK_TOKEN = "YOUR_NGROK_AUTHTOKEN_HERE"
ngrok.set_auth_token(NGROK_TOKEN)

# 3. Mở cổng 11434 ra Internet (sử dụng IPv4 và ghi đè Host Header)
tunnel = ngrok.connect("127.0.0.1:11434", "http", host_header="rewrite")

print("==================================================")
print("Sao chép đường link URL bên dưới và dán vào phần cài đặt của ReRag:")
print(f"Ollama API URL: {tunnel.public_url}")
print("==================================================")
```

## Bước 4: Cấu hình trên hệ thống ReRag
1. Sau khi chạy mã Ngrok ở Bước 3, bạn sẽ nhận được một đường link (Ví dụ: `https://abcd-12-34-56-78.ngrok-free.app`). Hãy sao chép (copy) đường link này.
2. Mở ứng dụng ReRag trên máy tính của bạn (`streamlit run app.py`).
3. Bấm vào nút **Thay đổi cài đặt** (Settings).
4. Dán đường link vừa copy vào ô **Ollama API URL (Dùng cho Ngrok/Colab)**.
5. Nhấn **Lưu cài đặt**.

### Lưu ý quan trọng
- Tab Colab phải được giữ mở trong suốt quá trình bạn sử dụng ứng dụng.
- **Kiem tra GPU**: Bạn có thể kiểm tra xem mô hình có thực sự chạy trên GPU hay không bằng cách tạo một ô mã mới trong Colab và chạy lệnh:
  ```bash
  !nvidia-smi
  ```
  Nếu thấy phần **GPU Memory Usage** tăng lên (ví dụ dùng khoảng 5-6GB VRAM cho mô hình qwen3-embedding) thì tức là hệ thống đang chạy hoàn toàn bằng GPU của Colab.
- Google Colab phiên bản miễn phí sẽ tự ngắt kết nối sau 12 giờ hoặc nếu bạn không tương tác với tab trình duyệt quá lâu.
- Mỗi lần khởi động lại Colab, Ngrok sẽ cấp cho bạn một **đường link mới**. Bạn cần phải cập nhật lại link đó vào giao diện cài đặt của ReRag.
