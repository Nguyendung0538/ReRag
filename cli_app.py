import sys
import os
from src.embedding.chroma_manager import ChromaManager
from src.generation.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine
from src.ingestion.document_processor import process_document

def main():
    print("======================================================")
    print("👨‍⚖️ CHƯƠNG TRÌNH SO SÁNH VÀ HỎI ĐÁP PHÁP LÝ (LEGAL RAG) ")
    print("======================================================")
    
    print("\n📥 BƯỚC 1: TẢI LÊN VĂN BẢN ĐỂ SO SÁNH")
    old_file_path = input("Nhập đường dẫn TÀI LIỆU LUẬT CŨ (Bản gốc): ").strip().strip('"\'')
    if not os.path.exists(old_file_path):
        print(f"❌ Không tìm thấy file: {old_file_path}")
        sys.exit(1)
        
    new_file_path = input("Nhập đường dẫn TÀI LIỆU LUẬT MỚI (Bản sửa đổi/mới): ").strip().strip('"\'')
    if not os.path.exists(new_file_path):
        print(f"❌ Không tìm thấy file: {new_file_path}")
        sys.exit(1)
        
    old_law_source = os.path.basename(old_file_path)
    new_law_source = os.path.basename(new_file_path)

    print("\n🛠 BƯỚC 2: KHỞI TẠO DỮ LIỆU...")
    try:
        db_manager = ChromaManager(persist_dir="./chroma_db", collection_name="legal_compare")
        
        print("\n🧹 Đang xóa Database pháp lý cũ...")
        db_manager.reset_collection()
        
        all_chunks = []
        for path in [old_file_path, new_file_path]:
            print(f"📄 Đang xử lý file: {path}...")
            chunks = process_document(path)
            for chunk in chunks:
                 chunk.metadata["source"] = os.path.basename(path)
            all_chunks.extend(chunks)
                
        if all_chunks:
            print(f"🧠 Đang nạp {len(all_chunks)} khối văn bản vào DB (Vector Embeddings)... Vui lòng đợi trong giây lát...")
            db_manager.add_documents(all_chunks)
            print("✅ Đã nạp thành công!\n")
        else:
            print("⚠ Không trích xuất được văn bản.")
            sys.exit(1)
    except Exception as e:
         print(f"❌ Lỗi nạp dữ liệu DB: {e}")
         sys.exit(1)

    print("🤖 Khởi tạo nhánh Lập luận AI (Qwen3:8b)...")
    llm_client = LLMClient(model_name="qwen3:8b")
    
    rag_engine = LegalRAGEngine(
        db_manager=db_manager, 
        llm_client=llm_client, 
        old_law_source=old_law_source, 
        new_law_source=new_law_source
    )
    
    print("------------------------------------------------------")
    print("💡 Hướng dẫn:")
    print("- Gõ câu hỏi của bạn (Ví dụ: 'So sánh thẻ căn cước công dân giữa hai phiên bản có gì khác biệt?')")
    print("- Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print("------------------------------------------------------\n")
    
    while True:
        try:
            query = input("\n👤 Câu hỏi của bạn: ")
            query_stripped = query.strip()
            
            if not query_stripped:
                continue
            
            if query_stripped.lower() in ['exit', 'quit', 'thoát']:
                print("Tạm biệt!")
                break
                
            # Stream câu trả lời RAG
            for text_chunk in rag_engine.stream_ask(query=query_stripped, top_k=6):
                print(text_chunk, end="", flush=True)
                
        except KeyboardInterrupt:
            print("\nĐã hủy yêu cầu hiện tại.")
            continue
        except Exception as e:
            print(f"\n❌ Lỗi phát sinh trong quá trình truy vấn: {e}")

if __name__ == "__main__":
    main()
