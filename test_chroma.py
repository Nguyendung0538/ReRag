import chromadb

def inspect_chromadb():
    client = chromadb.PersistentClient(path="./chroma_db")
    # Tên collection mặc định
    collections = client.list_collections()
    for col in collections:
        print(f"Collection: {col.name}")
        data = col.get()
        metadatas = data.get('metadatas', [])
        
        # Đếm số lượng chunks có dieu='Điều 6'
        dieu6_chunks = [m for m in metadatas if m and m.get('dieu') == 'Điều 6']
        print(f"Total chunks with dieu='Điều 6' in {col.name}: {len(dieu6_chunks)}")
        
        for m in dieu6_chunks:
            print(f"- Source: {m.get('source')} | Dieu: {m.get('dieu')}")
            
if __name__ == "__main__":
    inspect_chromadb()
