import sys
import os
from pprint import pprint

# Thêm src vào sys.path để test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from ingestion.document_processor import process_document

def main():
    if len(sys.argv) < 2:
        print("Sử dụng: python test_run.py <đường_dẫn_tới_file_docx_hoặc_pdf>")
        print("Ví dụ: python test_run.py hop_dong_mau.docx")
        sys.exit(1)

    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        sys.exit(1)

    print(f"Bắt đầu xử lý file: {file_path}...")
    try:
        chunks = process_document(file_path)
        print(f"\n✅ Đã xử lý xong! Tìm thấy tổng cộng {len(chunks)} Chunks (Điều/Khoản).")
        
        if chunks:
            result_file = "result.txt"
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(f"TỔNG SỐ CHUNKS: {len(chunks)}\n")
                f.write("="*50 + "\n\n")
                
                for i, chunk in enumerate(chunks):
                    f.write(f"[Chunk {i+1}]\n")
                    f.write(f"Metadata: {chunk.metadata}\n")
                    f.write(f"Nội dung:\n{chunk.text}\n")
                    f.write("-" * 50 + "\n\n")
                    
            print(f"\nĐã xuất toàn bộ kết quả ra file: {os.path.abspath(result_file)}")
            
    except Exception as e:
        print(f"❌ Xảy ra lỗi trong quá trình xử lý: {e}")

if __name__ == "__main__":
    main()
