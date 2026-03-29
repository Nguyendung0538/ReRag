import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from .ollama_embedder import OllamaEmbedder
import os

class ChromaManager:
    """
    Quản lý bộ nhớ Vector trên RAM (In-memory).
    Vì mỗi lần chạy chúng ta đều nạp lại 2 văn bản mới nên không cần lưu ra ổ cứng,
    giúp tránh tạo ra các file rác và chạy nhanh hơn.
    """
    def __init__(self, collection_name: str = "legal_docs"):
        
        # Kết nối tới In-Memory DB (Chỉ lưu trên RAM)
        self.client = chromadb.EphemeralClient()
        self.collection_name = collection_name
        
        # Load Collection ra nếu có sẵn, dùng để Query lại sau này.
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        self.embedder = OllamaEmbedder()

    def reset_collection(self):
        """Xóa trắng Collection pháp lý cũ để nạp tài liệu mới"""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception as e:
            pass # Lỗi nếu collection không tồn tại, có thể bỏ qua
            
        self.collection = self.client.create_collection(name=self.collection_name)
        
    def add_documents(self, chunks: List[Any]):
        """
        Nhận danh sách DocumentChunk từ legal_chunker, sinh vector và nhét vào ChromaDB.
        """
        if not chunks:
            return
            
        documents = []
        metadatas = []
        ids = []
        
        # Tách thuộc tính từ Chunk
        for i, chunk in enumerate(chunks):
            documents.append(chunk.text)
            
            # Khắc phục metadata nếu bị dính object không tương thích với Chroma (chỉ cho phép kiểu str, int, float, bool)
            meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)
            
            # Tạo unique ID (ví dụ: file_name_chuong_dieu_id)
            source_name = meta.get("source", f"doc_{i}")
            chunk_dieu = meta.get("dieu", f"dieu_{i}").replace(" ", "_")
            doc_id = f"{source_name}_{chunk_dieu}_{i}"
            ids.append(doc_id)
            
        # Lấy embeddings thông qua Ollama
        print(f"Đang xử lý Embeddings cho {len(documents)} khối văn bản qua mô hình {self.embedder.model_name}...")
        embeddings = self.embedder.embed_batch(documents)
        
        # Đưa vào ChromaDB
        print("Đang lưu dữ liệu vào ChromaDB...")
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Hoàn tất nạp dữ liệu vào Database Vector cục bộ!")
        
    def query(self, text: str, n_results: int = 5) -> Dict[str, Any]:
        """Tìm kiếm các chunks gần nhất với câu hỏi chứa nội dung so sánh"""
        # Embed câu truy vấn
        query_vector = self.embedder.embed_text(text)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        return results
