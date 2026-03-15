import sys
import os
from src.ingestion.document_processor import process_document
from src.embedding.chroma_manager import ChromaManager

def main():
    if len(sys.argv) < 2:
        print("Cách sử dụng: python ingestion_pipeline.py <duong_dan_file_1> [duong_dan_file_2] ...")
        print("Ví dụ: python ingestion_pipeline.py luat_cu.docx luat_moi.pdf")
        sys.exit(1)
        
    file_paths = sys.argv[1:]
    
    # Kiểm tra file có tồn tại không
    for path in file_paths:
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy file: {path}")
            sys.exit(1)

    db_manager = ChromaManager(persist_dir="./chroma_db", collection_name="legal_compare")
    
    # Reset DB trước khi nạp tài liệu mới (theo design: up file mới -> drop DB cũ)
    print("\n🧹 Bắt đầu dọn dẹp Database pháp lý cũ...")
    db_manager.reset_collection()
    print("✅ Đã dọn dẹp xong!")
    
    all_chunks = []
    for path in file_paths:
        print(f"\n📄 Đang ráp và phân tách cấu trúc (Chunking) file: {path}...")
        try:
            chunks = process_document(path)
            print(f"   -> Đã trích xuất được {len(chunks)} chunks pháp lý (Điều/Khoản).")
            # Cần sửa đổi id, metadata, source theo chunk để đảm bảo không trùng
            for chunk in chunks:
                 chunk.metadata["source"] = os.path.basename(path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"❌ Bỏ qua file {path} do lỗi xử lý: {e}")
            
    if all_chunks:
        print(f"\n🧠 Bắt đầu Embed và nạp {len(all_chunks)} chunks vào ChromaDB qua Ollama (qwen3-embedding:8b)...")
        db_manager.add_documents(all_chunks)
        print("\n🎉 Hoàn tất quá trình nhập liệu (Ingestion). Dữ liệu đã sẵn sàng trên ổ đĩa!")
        print("Thư mục CSDL: ./chroma_db")
        
        # Test Query nhỏ
        print("\n🔍 Chạy Test Thử Nghiệm Query DB...")
        try:
            test_res = db_manager.query("thẻ căn cước công dân", n_results=1)
            print("Kết quả test:")
            print("- Câu trả lời gần nhất thuộc Điều:", test_res["metadatas"][0][0].get("dieu", "Không rõ"))
            print("- Nguồn:", test_res["metadatas"][0][0].get("source", "Không rõ"))
            print("- Nội dung trích dẫn:", test_res["documents"][0][0][:150], "...")
        except Exception as e:
            print("Lỗi khi test query rút trích:", e)
            
    else:
        print("\n⚠ Không có dữ liệu văn bản nào được nhận diện. Hủy quá trình.")

if __name__ == "__main__":
    main()
