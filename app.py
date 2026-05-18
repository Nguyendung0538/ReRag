import streamlit as st
import os
import time
from src.embedding.chroma_manager import ChromaManager
from src.generation.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine
from src.ingestion.document_processor import process_document
from src.indexing_strategies import INDEXING_STRATEGIES
from src.query_strategies import STRATEGIES

st.set_page_config(page_title="Hệ thống Trợ lý Pháp lý RAG", page_icon="⚖️", layout="wide")

# ==================== CẤU HÌNH MODEL OLLAMA ====================
LLM_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "qwen3-embedding:8b"

# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "👋 Xin chào! Vui lòng nạp 2 tài liệu pháp lý ở menu bên trái để tôi có thể hỗ trợ bạn so sánh và phân tích."}]
if "db_ready" not in st.session_state:
    st.session_state["db_ready"] = False
if "old_law_name" not in st.session_state:
    st.session_state["old_law_name"] = ""
if "new_law_name" not in st.session_state:
    st.session_state["new_law_name"] = ""

if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = LLM_MODEL
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = EMBEDDING_MODEL
if "indexing_strategy_name" not in st.session_state:
    st.session_state["indexing_strategy_name"] = list(INDEXING_STRATEGIES.keys())[0]
if "query_strategy_name" not in st.session_state:
    st.session_state["query_strategy_name"] = "Paired Retrieval (So sánh đôi)"

# ==================== SETTINGS MODAL ====================
@st.dialog("⚙️ Cài đặt Hệ thống RAG")
def show_settings():
    st.subheader("⚙️ Cấu hình")
    
    # Model selections
    new_llm = st.text_input("🧠 Mô hình LLM (Ollama)", value=st.session_state["llm_model"])
    new_embed = st.text_input("📦 Mô hình Embedding (Ollama)", value=st.session_state["embedding_model"])
    
    # Strategy selections
    selected_idx_strat = st.selectbox(
        "🗂️ Chiến lược Lập chỉ mục (Indexing Strategy)",
        options=list(INDEXING_STRATEGIES.keys()),
        index=list(INDEXING_STRATEGIES.keys()).index(st.session_state["indexing_strategy_name"])
    )
    
    selected_query_strat = st.selectbox(
        "🔍 Chiến lược Truy vấn (Query Strategy)",
        options=list(STRATEGIES.keys()),
        index=list(STRATEGIES.keys()).index(st.session_state["query_strategy_name"])
    )

    st.divider()
    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("💾 Lưu cài đặt", type="primary", use_container_width=True):
            st.session_state["llm_model"] = new_llm
            st.session_state["embedding_model"] = new_embed
            st.session_state["indexing_strategy_name"] = selected_idx_strat
            st.session_state["query_strategy_name"] = selected_query_strat
            st.success("Đã lưu cấu hình!")
            time.sleep(0.5)
            st.rerun()
    with col_cancel:
        if st.button("Huỷ", use_container_width=True):
            st.rerun()


# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📥 Nạp Dữ Liệu")
    st.markdown("Hệ thống chỉ lưu tập trung 2 tài liệu vào bộ nhớ để so sánh chính xác nhất.")
    
    file_old = st.file_uploader("1️⃣ Tài liệu Bản gốc", type=['docx', 'pdf', 'txt'])
    file_new = st.file_uploader("2️⃣ Tài liệu Bản sửa đổi/Bổ sung", type=['docx', 'pdf', 'txt'])
    
    if st.button("🚀 Khởi tạo Hệ thống RAG", type="primary", use_container_width=True):
        if file_old and file_new:
            st.session_state["old_law_name"] = file_old.name
            st.session_state["new_law_name"] = file_new.name
            
            init_start = time.time()
            with st.spinner("Đang cấu trúc và nạp văn bản (theo Indexing Strategy)... Xin đợi..."):
                try:
                    # Parse document chunks
                    all_chunks = []
                    for f_obj in [file_old, file_new]:
                        ext = os.path.splitext(f_obj.name)[1].lower()
                        chunks = process_document(file_source=f_obj, filename=f_obj.name, ext=ext)
                        for chunk in chunks:
                             chunk.metadata["source"] = f_obj.name
                        all_chunks.extend(chunks)
                    
                    if all_chunks:
                        # Dùng Indexing Strategy đã chọn
                        idx_class = INDEXING_STRATEGIES[st.session_state["indexing_strategy_name"]]
                        indexer = idx_class(embedding_model=st.session_state["embedding_model"])
                            
                        # Keep it globally so we can retrieve exactly what was just parsed 
                        # This is especially true for NoEmbed which holds data in RAM.
                        st.session_state["active_indexer"] = indexer
                        
                        success = indexer.index(all_chunks)
                        
                        init_elapsed = time.time() - init_start
                        if success:
                            st.session_state["db_ready"] = True
                            st.success(
                                f"✅ Đã nạp thành công **{len(all_chunks)} đoạn văn bản** từ 2 tài liệu!\n\n"
                                f"🕒 Thời gian khởi tạo: **{init_elapsed:.1f} giây**"
                            )
                        else:
                            st.error("❌ Quá trình lưu thất bại. Kiểm tra lại file tài liệu.")
                    else:
                        st.error("Không thể rút trích văn bản từ 2 file này.")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            st.warning("Vui lòng tải lên ĐỦ 2 văn bản để bắt đầu!")

    st.divider()

    # Hiển thị trạng thái setting hiện tại + nút mở modal
    st.markdown("**⚙️ Cài đặt hiện tại:**")
    st.info(
        f"🧠 **LLM:** {st.session_state['llm_model']}\n\n"
        f"📦 **Embedding:** {st.session_state['embedding_model']}\n\n"
        f"🗂️ **Index Strat:** {st.session_state['indexing_strategy_name']}\n\n"
        f"🔍 **Query Strat:** {st.session_state['query_strategy_name']}"
    )
    if st.button("✏️ Thay đổi cài đặt", use_container_width=True):
        show_settings()

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
            if "active_indexer" not in st.session_state:
                st.error("Không tìm thấy bộ nhớ. Vui lòng nhấn 'Khởi tạo Hệ thống RAG' bên trái trước.")
                st.stop()
                
            indexer = st.session_state["active_indexer"]
            llm_client = LLMClient(model_name=st.session_state.get("llm_model", "qwen3:8b"))
            rag_engine = LegalRAGEngine(
                indexing_strategy=indexer,
                llm_client=llm_client,
                old_law_source=st.session_state["old_law_name"],
                new_law_source=st.session_state["new_law_name"]
            )
            
            start_time = time.time()
            
            with st.spinner("Đang suy nghĩ..."):
                full_text = ""
                for chunk_text in rag_engine.stream_ask(
                    query=prompt, 
                    top_k=12,
                    strategy_name=st.session_state.get("query_strategy_name")
                ):
                    full_text += chunk_text
                
                st.markdown(full_text)
                end_time = time.time()
                st.caption(f"🕒 Thời gian phản hồi: {end_time - start_time:.2f} giây")
                full_response = full_text + f"\n\n🕒 Thời gian phản hồi: {end_time - start_time:.2f} giây"
                
        # Lưu câu trả lời của Trợ lý vào session
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
