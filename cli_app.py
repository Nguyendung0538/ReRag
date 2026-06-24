import sys
import os
from src.embedding.chroma_manager import ChromaManager
from src.generation.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine
from src.ingestion.document_processor import process_document

def main():
    print("======================================================")
    print(" CHUONG TRINH SO SANH VA HOI DAP PHAP LY (LEGAL RAG) ")
    print("======================================================")
    
    print("\n BUOC 1: TAI LEN VAN BAN DE SO SANH")
    old_file_path = input("Nhap duong dan TAI LIEU LUAT CU (Ban goc): ").strip().strip('"\'')
    if not os.path.exists(old_file_path):
        print(f" Khong tim thay file: {old_file_path}")
        sys.exit(1)
        
    new_file_path = input("Nhap duong dan TAI LIEU LUAT MOI (Ban sua doi/moi): ").strip().strip('"\'')
    if not os.path.exists(new_file_path):
        print(f" Khong tim thay file: {new_file_path}")
        sys.exit(1)
        
    old_law_source = os.path.basename(old_file_path)
    new_law_source = os.path.basename(new_file_path)

    print("\n BUOC 2: KHOI TAO DU LIEU...")
    try:
        db_manager = ChromaManager(collection_name="legal_compare")
        
        print("\n Dang xoa Database phap ly cu...")
        db_manager.reset_collection()
        
        all_chunks = []
        for path in [old_file_path, new_file_path]:
            print(f" Dang xu ly file: {path}...")
            chunks = process_document(path)
            for chunk in chunks:
                 chunk.metadata["source"] = os.path.basename(path)
            all_chunks.extend(chunks)
                
        if all_chunks:
            print(f" Dang nap {len(all_chunks)} khoi van ban vao DB (Vector Embeddings)... Vui long doi trong giay lat...")
            db_manager.add_documents(all_chunks)
            print(" Da nap thanh cong!\n")
        else:
            print(" Khong trich xuat duoc van ban.")
            sys.exit(1)
    except Exception as e:
         print(f" Loi nap du lieu DB: {e}")
         sys.exit(1)

    print(" Khoi tao nhanh Lap luan AI (Qwen3:8b)...")
    llm_client = LLMClient(model_name="qwen3:8b")
    
    rag_engine = LegalRAGEngine(
        db_manager=db_manager, 
        llm_client=llm_client, 
        old_law_source=old_law_source, 
        new_law_source=new_law_source
    )
    
    print("------------------------------------------------------")
    print(" Huong dan:")
    print("- Go cau hoi cua ban (Vi du: 'So sanh the can cuoc cong dan giua hai phien ban co gi khac biet?')")
    print("- Go 'exit' hoac 'quit' de thoat chuong trinh.")
    print("------------------------------------------------------\n")
    
    while True:
        try:
            query = input("\n Cau hoi cua ban: ")
            query_stripped = query.strip()
            
            if not query_stripped:
                continue
            
            if query_stripped.lower() in ['exit', 'quit', 'thoat']:
                print("Tam biet!")
                break
                
            # Stream cau tra loi RAG
            for text_chunk in rag_engine.stream_ask(query=query_stripped, top_k=12):
                print(text_chunk, end="", flush=True)
                
        except KeyboardInterrupt:
            print("\n Da huy yeu cau hien tai.")
            continue
        except Exception as e:
            print(f"\n Loi phat sinh trong qua trinh truy van: {e}")

if __name__ == "__main__":
    main()
