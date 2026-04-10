from typing import List, Dict, Any
from collections import defaultdict
import json
import re

from .base_indexing import BaseIndexingStrategy
from src.ingestion.legal_chunker import DocumentChunk
from src.generation.llm_client import LLMClient

class HierarchicalRAGIndexing(BaseIndexingStrategy):
    """
    Kỹ thuật RAG Phi Vector (Vectorless RAG) mang cảm hứng từ 'PageIndex' của VectifyAI.
    
    1. Lập chỉ mục (Indexing):
       Không dùng Embedding/ChromaDB. Thay vào đó, phân vùng tài liệu thành một
       Cây Mục lục (Table of Contents / Tree) nhỏ gọn lưu trên RAM. Gán ID cho mỗi Điều.
       Mỗi node chứa: ID, Source, Chương, Mục, Điều và 120 ký tự trích dẫn nội dung (preview).
       
    2. Truy xuất ngang (Retrieval):
       Thay vì tìm từ đồng nghĩa bằng vector, chuyển cấu trúc JSON Mục lục vào trong Context của LLM.
       Hỏi LLM: "Với câu hỏi này và mục lục luật này, hãy lập luận xem nên đọc ID nào?".
       LLM trả về List [ID1, ID2]. Dùng ID bốc nguyên bản text từ RAM trả về hệ thống hiện tại.
       
    Tuyệt đối chính xác và chống mờ nhòe bối cảnh (Similarity Hallucination).
    """

    def __init__(self, embedding_model: str = "", llm_model: str = "qwen3:8b", max_chars_limit: int = 150000):
        # Mặc dù giao diện vẫn gọi embedding_model để tương thích, ta sẽ không cần dùng tới.
        self.llm_model = llm_model
        
        # Ngưỡng token an toàn để mục lục JSON không làm tràn RAM/Context của model router (150K ký tự)
        self.max_chars_limit = max_chars_limit  
        
        # In-memory storage
        self.db_nodes = {}  # dict mapping: node_id -> List[DocumentChunk]
        self.toc = defaultdict(list) # struct: { "Tên văn bản": [ {id, path, preview}, ... ] }

    def index(self, chunks: List[DocumentChunk], **kwargs) -> bool:
        if not chunks:
            return False
            
        print("[PageIndex] Khởi tạo Hệ thống Vectorless...")
        print("[PageIndex] 1. Cấu trúc hóa cây phân cấp tài liệu dựa vào metadata...")
        
        self.db_nodes = {}
        self.toc = defaultdict(list)
        
        # Gom nhóm chunks
        groups = defaultdict(list)
        for c in chunks:
            src = c.metadata.get("source", "Không phân loại")
            chuong = c.metadata.get("chuong", "")
            muc = c.metadata.get("muc", "")
            dieu = c.metadata.get("dieu", "")
            
            # Gộp chung vào cùng một node nếu cùng địa chỉ
            key = (src, chuong, muc, dieu)
            groups[key].append(c)
            
        print(f"[PageIndex] Đã gom nhóm thành {len(groups)} Nodes riêng biệt.")
        
        # Gán node_id cho từng cụm và xây table of contents
        node_id_counter = 1
        for (src, chuong, muc, dieu), group_chunks in groups.items():
            node_id = node_id_counter
            node_id_counter += 1
            
            self.db_nodes[node_id] = group_chunks
            
            # Làm sạch path
            path_parts = [p for p in [chuong, muc, dieu] if p.strip() and p.lower() not in {"không rõ", "không xác định", "n/a", "none"}]
            path = " > ".join(path_parts) if path_parts else "Nội dung chung"
            
            # Lấy teaser nội dung để model điều hướng dễ nhận diện
            combined_txt = " ".join([c.text for c in group_chunks])
            preview = combined_txt[:120].strip() + "..."
            
            self.toc[src].append({
                "id": node_id,
                "path": path,
                "preview": preview
            })

        # Đo lường dung lượng
        self.tree_index_json = json.dumps(self.toc, ensure_ascii=False)
        if len(self.tree_index_json) > self.max_chars_limit:
            print("[PageIndex ERROR] Kích thước Mục lục qua lớn so với Context Window.")
            return False
            
        print("[PageIndex] Hoàn tất nạp Cây Mục lục lên RAM (Không sử dụng Embedding DB)!")
        return True

    def build_context(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        print("[PageIndex] Nhận truy vấn. Đang yêu cầu LLM xác định vùng tài liệu...")
        llm = LLMClient(model_name=self.llm_model)
        
        where = kwargs.get('where', None)
        
        # Lọc mục lục JSON nếu đang ở chế độ Decompose query (để tối ưu lượng token chuyển cho LLM)
        filtered_toc = {}
        if where and "source" in where:
            src_filter = where["source"]
            if src_filter in self.toc:
                filtered_toc[src_filter] = self.toc[src_filter]
        else:
            filtered_toc = self.toc
            
        toc_json_str = json.dumps(filtered_toc, ensure_ascii=False, indent=2)
        
        prompt = f"""Dưới đây là BẢNG MỤC LỤC của một số tài liệu pháp lý định dạng JSON. Mỗi mục có:
- 'id': Mã số định danh.
- 'path': Nhánh cấu trúc (Chương > Mục > Điều).
- 'preview': Nội dung tóm tắt để bạn nhận diện.

MỤC LỤC TIÊU ĐIỂM:
{toc_json_str}

CÂU HỎI NGƯỜI DÙNG: 
"{query}"

NHIỆM VỤ CỦA CHUYÊN GIA ĐIỀU HƯỚNG:
Dựa trên Mục lục trên, hãy suy luận logic và đoán xem vùng nội dung nào chứa thông tin liên quan nhất đến câu hỏi.
1. Chọn ra các ID tương ứng với các điều khoản luật có khả năng chứa câu trả lời.
2. Chọn tối đa {top_k} IDs.
3. CHỈ YÊU CẦU TRẢ VỀ: Một mảng JSON chứa các số nguyên tương ứng. TUYỆT ĐỐI KHÔNG giải thích, KHÔNG viết bất kỳ chữ nào ngoài JSON Array. Đừng bọc markdown json.

VÍ DỤ OUTPUT CHUẨN:
[3, 14, 25]"""

        system_prompt = "Bạn là thuật toán tìm kiếm Vectorless xuất JSON. Chỉ trả duy nhất định dạng JSON thuần tuý."
        
        # Không dùng streaming vì ta cần trọn vẹn kết quả JSON mảng
        response = llm.generate_response(prompt=prompt, system_prompt=system_prompt)
        
        selected_ids = []
        try:
            raw = response.strip()
            # Dọn dẹp Markdown rác nếu mô hình không tuân thủ
            if "```" in raw:
                # Xóa sạch trong code block
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
                if match:
                    raw = match.group(1)
                    
            # Trích xuất đoạn giống Array nhất
            arr_match = re.search(r'\[(.*?)\]', raw, re.DOTALL)
            if arr_match:
                selected_ids = json.loads(arr_match.group(0))
                if not isinstance(selected_ids, list):
                    selected_ids = []
        except Exception as e:
            print(f"[PageIndex] Cảnh báo rủi ro bóc tách JSON (sẽ trả về mảng rỗng). Lỗi: {e}. Raw LLM: {response}")
            selected_ids = []
            
        print(f"[PageIndex] LLM Điều hướng đã quét cấu trúc thành công và chọn ra các node: {selected_ids}")
        
        # Tiến trình kéo dữ liệu từ Ram theo ID
        docs = []
        metas = []
        
        fetched_count = 0
        for sid in selected_ids:
            if isinstance(sid, int) and sid in self.db_nodes:
                group_chunks = self.db_nodes[sid]
                for c in group_chunks:
                    # Double check `where` filter lần nữa để an toàn, cho dù Mục lục đã lọc
                    meta_match = True
                    if where:
                        for k, v in where.items():
                            if c.metadata.get(k) != v:
                                meta_match = False
                                break
                    
                    if meta_match:
                        docs.append(c.text)
                        metas.append(c.metadata)
                
                fetched_count += 1
                if fetched_count >= top_k:
                    break

        return {"documents": [docs], "metadatas": [metas]}
