import streamlit as st
import os
import shutil
from src.embedding.chroma_manager import ChromaManager
from src.generation.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine
from src.ingestion.document_processor import process_document

st.set_page_config(page_title="Hệ thống Trợ lý Pháp lý RAG", page_icon="⚖️", layout="wide")

# Khởi tạo thư mục Uploads tạm
UPLOAD_DIR = "./document/temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# CSS để UI nhìn xịn hơn
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session data
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "👋 Xin chào! Vui lòng nạp 2 tài liệu pháp lý ở menu bên trái để tôi có thể hỗ trợ bạn so sánh và phân tích."}]
if "db_ready" not in st.session_state:
    st.session_state["db_ready"] = False
if "old_law_name" not in st.session_state:
    st.session_state["old_law_name"] = ""
if "new_law_name" not in st.session_state:
    st.session_state["new_law_name"] = ""

st.title("Legal comparision Local RAG")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📥 Nạp Dữ Liệu")
    st.markdown("Hệ thống chỉ lưu tập trung 2 tài liệu vào bộ nhớ để so sánh chính xác nhất.")
    
    file_old = st.file_uploader("1️⃣ Tài liệu Bản gốc", type=['docx', 'pdf', 'txt'])
    file_new = st.file_uploader("2️⃣ Tài liệu Bản sửa đổi/Bổ sung", type=['docx', 'pdf', 'txt'])
    
    if st.button("🚀 Khởi tạo Hệ thống RAG", type="primary", use_container_width=True):
        if file_old and file_new:
            # Xóa các file cũ trong temp folder
            for f in os.listdir(UPLOAD_DIR):
                os.remove(os.path.join(UPLOAD_DIR, f))
                
            path_old = os.path.join(UPLOAD_DIR, file_old.name)
            path_new = os.path.join(UPLOAD_DIR, file_new.name)
            
            with open(path_old, "wb") as f:
                f.write(file_old.getbuffer())
            with open(path_new, "wb") as f:
                f.write(file_new.getbuffer())
                
            st.session_state["old_law_name"] = file_old.name
            st.session_state["new_law_name"] = file_new.name
            
            with st.spinner("Đang cấu trúc lại văn bản và nhúng (Embed) lên ChromaDB... Xin đợi..."):
                try:
                    db_manager = ChromaManager(collection_name="legal_compare")
                    db_manager.reset_collection()
                    
                    all_chunks = []
                    for path in [path_old, path_new]:
                        chunks = process_document(path)
                        for chunk in chunks:
                             chunk.metadata["source"] = os.path.basename(path)
                        all_chunks.extend(chunks)
                    
                    if all_chunks:
                        db_manager.add_documents(all_chunks)
                        st.session_state["db_ready"] = True
                        st.success(f"✅ Đã nạp thành công văn bản!")
                    else:
                        st.error("Không thể rút trích văn bản từ 2 file này.")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            st.warning("Vui lòng tải lên ĐỦ 2 văn bản để bắt đầu!")

# ==================== MAIN CHAT ====================

# Render các tin nhắn cũ
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
# Xử lý nhập chat
if prompt := st.chat_input("Hỏi gì đó (Ví dụ: So sánh hạn sử dụng thẻ căn cước...)"):
    if not st.session_state["db_ready"]:
        st.error("⚠ Bạn cần tải lên văn bản và nhấn 'Khởi tạo Hệ thống RAG' trước khi đặt câu hỏi!")
    else:
        # Thêm câu hỏi của user vào UI
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Khởi tạo Engine nếu đã Ready
        with st.chat_message("assistant"):
            db_manager = ChromaManager(collection_name="legal_compare")
            llm_client = LLMClient(model_name="qwen3:8b")
            rag_engine = LegalRAGEngine(
                db_manager=db_manager,
                llm_client=llm_client,
                old_law_source=st.session_state["old_law_name"],
                new_law_source=st.session_state["new_law_name"]
            )
            
            # Hàm sinh generator cho st.write_stream
            def generate_reply():
                for chunk_text in rag_engine.stream_ask(query=prompt, top_k=6):
                    yield chunk_text
                    
            # Gọi stream ra UI
            full_response = st.write_stream(generate_reply)
            
        # Lưu câu trả lời của Trợ lý vào session
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
