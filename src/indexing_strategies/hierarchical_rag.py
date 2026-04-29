from typing import List, Dict, Any
from collections import defaultdict
from .base_indexing import BaseIndexingStrategy
from src.ingestion.legal_chunker import DocumentChunk
from src.embedding.chroma_manager import ChromaManager
from src.generation.llm_client import LLMClient
import chromadb

class HierarchicalRAGIndexing(BaseIndexingStrategy):
    """
    Chiến thuật nạp Cây Phân cấp Lai (Hybrid Hierarchical RAG).
    
    Khắc phục điểm yếu của Vectorless RAG bằng cách DÙNG LẠI VECTOR EMBEDDING, kết hợp với LLM Summary.
    1. Indexing:
       - Gom các chunk thành các cụm theo (Nguồn + Điều khoản).
       - Nhờ LLM tóm tắt từng Điều, lưu Vector Tóm Tắt vào ChromaDB `summaries_col`.
       - Nạp Vector toàn bộ văn bản chi tiết vào ChromaDB `details_col`.
    2. Retrieval:
       - Tìm câu hỏi bằng Vector trên `summaries_col` để khoanh vùng đúng 1-2 Điều tiềm năng nhất.
       - Tìm lại lần 2 bằng Vector trên `details_col` (nhưng ép vào filter của các Điều đã khoanh vùng).
    """
    def __init__(self, embedding_model: str = "qwen3-embedding:8b", llm_model: str = "qwen3:8b", db_name="hierarchical_rag"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        self.db_manager = ChromaManager(
            collection_name=f"{db_name}_details",
            embedding_model=self.embedding_model
        )
        self.db_manager.reset_collection()
        self.details_col = self.db_manager.collection
        
        # Tạo Collection riêng rẽ cho Summaries
        self.summaries_col_name = f"{db_name}_summaries"
        try:
            self.db_manager.client.delete_collection(name=self.summaries_col_name)
        except Exception:
            pass
        self.summaries_col = self.db_manager.client.create_collection(name=self.summaries_col_name)

    def index(self, chunks: List[DocumentChunk], **kwargs) -> bool:
        if not chunks:
            return False
            
        print("[Hybrid RAG] Đang phân nhóm dữ liệu theo Điều khoản...")
        # Gom text vào các nhóm (source, dieu)
        groups = defaultdict(list)
        for chunk in chunks:
            src = chunk.metadata.get("source", "Không rõ")
            dieu = chunk.metadata.get("dieu", "Không xác định")
            groups[(src, dieu)].append(chunk)

        llm = LLMClient(model_name=self.llm_model)
        summaries_docs = []
        summaries_metas = []
        summaries_ids = []
        
        idx = 0
        total_groups = len(groups)
        for (src, dieu), group_chunks in groups.items():
            idx += 1
            print(f"[Hybrid RAG] Sinh Tóm tắt nhanh cho Vector: {src} - {dieu} ({idx}/{total_groups})")
            
            # Giới hạn text đầu vào ~6000 ký tự để nạp LLM tạo tóm tắt nhanh
            full_text = "\n".join([c.text for c in group_chunks])
            truncated_text = full_text[:6000]
            
            prompt = (
                f"Hãy viết một câu tóm tắt thật ngắn gọn (tối đa 2 câu) về nội dung chính yếu của phần văn bản pháp lý sau để làm chỉ mục tìm kiếm (index): \n\n"
                f"{truncated_text}" 
            )
            
            summary_ans = llm.generate_response(prompt=prompt, system_prompt="Bạn là trợ lý pháp lý tóm tắt văn bản.")
            
            summaries_docs.append(summary_ans)
            summaries_metas.append({"source": src, "dieu": dieu})
            summaries_ids.append(f"summary_{src}_{dieu}_{idx}")

        print("[Hybrid RAG] Nạp cấu trúc Vector Tổng phân lớp Details...")
        self.db_manager.add_documents(chunks)
        
        print("[Hybrid RAG] Nhúng và nạp cấu trúc Vector Tóm tắt...")
        if summaries_docs:
            sum_embeddings = self.db_manager.embedder.embed_batch(summaries_docs)
            self.summaries_col.add(
                embeddings=sum_embeddings,
                documents=summaries_docs,
                metadatas=summaries_metas,
                ids=summaries_ids
            )
        
        if hasattr(self.db_manager, 'embedder'):
            self.db_manager.embedder.unload()
            
        return True

    def build_context(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        where = kwargs.get('where', None)
        
        # 1. Tìm summaries vector embedding để khoanh vùng Điều liên quan
        query_vector = self.db_manager.embedder.embed_text(query)
        sum_kwargs = dict(
            query_embeddings=[query_vector],
            n_results=10  # Tìm 10 Điều liên quan nhất nhờ vector tóm tắt để không bỏ sót
        )
        if where:
            sum_kwargs["where"] = where
            
        sum_results = self.summaries_col.query(**sum_kwargs)
        
        target_dieus = set()
        retrieved_metas = sum_results.get("metadatas", [[]])[0]
        for meta in retrieved_metas:
            if "dieu" in meta:
                target_dieus.add(meta["dieu"])
                
        # 2. Truy vấn Details Collection, filter theo các Điều đã khoanh vùng
        if target_dieus:
            combined_where = {}
            if where:
                combined_where.update(where)
            if len(target_dieus) == 1:
                combined_where["dieu"] = list(target_dieus)[0]
            else:
                combined_where["dieu"] = {"$in": list(target_dieus)}
                
            det_kwargs = dict(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=combined_where
            )
        else:
            # Fallback
            det_kwargs = dict(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            if where:
                det_kwargs["where"] = where

        res = self.details_col.query(**det_kwargs)
        return res
