import streamlit as st
import os
import time
from src.embedding.chroma_manager import ChromaManager
from src.generation.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine
from src.ingestion.document_processor import process_document
from src.indexing_strategies import INDEXING_STRATEGIES
from src.query_strategies import STRATEGIES

st.set_page_config(page_title="Hệ thống Trợ lý Pháp lý RAG", layout="wide")

# ==================== CAU HINH MODEL OLLAMA ====================
LLM_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "qwen3-embedding:8b"

# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Xin chào! Vui lòng nạp 2 tài liệu pháp lý ở menu bên trái để tôi có thể hỗ trợ bạn so sánh và phân tích."}]
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
@st.dialog("Cài đặt Hệ thống RAG")
def show_settings():
    st.subheader("Cấu hình")
    
    new_llm = st.text_input("Mô hình LLM (Ollama)", value=st.session_state["llm_model"])
    new_embed = st.text_input("Mô hình Embedding (Ollama)", value=st.session_state["embedding_model"])
    
    selected_idx_strat = st.selectbox(
        "Chiến lược Lập chỉ mục (Indexing Strategy)",
        options=list(INDEXING_STRATEGIES.keys()),
        index=list(INDEXING_STRATEGIES.keys()).index(st.session_state["indexing_strategy_name"])
    )
    
    selected_query_strat = st.selectbox(
        "Chiến lược Truy vấn (Query Strategy)",
        options=list(STRATEGIES.keys()),
        index=list(STRATEGIES.keys()).index(st.session_state["query_strategy_name"])
    )

    st.divider()
    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Lưu cài đặt", type="primary", use_container_width=True):
            st.session_state["llm_model"] = new_llm
            st.session_state["embedding_model"] = new_embed
            st.session_state["indexing_strategy_name"] = selected_idx_strat
            st.session_state["query_strategy_name"] = selected_query_strat
            st.success("Đã lưu cấu hình!")
            time.sleep(0.5)
            st.rerun()
    with col_cancel:
        if st.button("Hủy", use_container_width=True):
            st.rerun()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Nạp Dữ Liệu")
    st.markdown("Hệ thống chỉ lưu tập trung 2 tài liệu vào bộ nhớ để so sánh chính xác nhất.")
    
    file_old = st.file_uploader("1. Tài liệu Bản gốc", type=['docx', 'pdf', 'txt'])
    file_new = st.file_uploader("2. Tài liệu Bản sửa đổi/Bổ sung", type=['docx', 'pdf', 'txt'])
    
    if st.button("Khởi tạo Hệ thống RAG", type="primary", use_container_width=True):
        if file_old and file_new:
            st.session_state["old_law_name"] = file_old.name
            st.session_state["new_law_name"] = file_new.name
            
            init_start = time.time()
            with st.spinner("Đang cấu trúc và nạp văn bản (theo Indexing Strategy)... Xin đợi..."):
                try:
                    ext_old = os.path.splitext(file_old.name)[1].lower()
                    chunks_old = process_document(file_source=file_old, filename=file_old.name, ext=ext_old)
                    for chunk in chunks_old:
                        chunk.metadata["source"] = file_old.name
                    
                    ext_new = os.path.splitext(file_new.name)[1].lower()
                    chunks_new = process_document(file_source=file_new, filename=file_new.name, ext=ext_new)
                    for chunk in chunks_new:
                        chunk.metadata["source"] = file_new.name
                    
                    st.session_state["chunks_old"] = chunks_old
                    st.session_state["chunks_new"] = chunks_new
                    all_chunks = chunks_old + chunks_new
                    
                    if all_chunks:
                        idx_class = INDEXING_STRATEGIES[st.session_state["indexing_strategy_name"]]
                        indexer = idx_class(embedding_model=st.session_state["embedding_model"])
                            
                        st.session_state["active_indexer"] = indexer
                        
                        success = indexer.index(all_chunks)
                        
                        init_elapsed = time.time() - init_start
                        if success:
                            st.session_state["db_ready"] = True
                            st.success(
                                f"Đã nạp thành công {len(all_chunks)} đoạn văn bản từ 2 tài liệu!\n\n"
                                f"Thời gian khởi tạo: {init_elapsed:.1f} giây"
                            )
                        else:
                            st.error("Quá trình lưu thất bại. Kiểm tra lại file tài liệu.")
                    else:
                        st.error("Không thể rút trích văn bản từ 2 file này.")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            st.warning("Vui lòng tải lên ĐỦ 2 văn bản để bắt đầu!")

    st.divider()

    st.markdown("**Cài đặt hiện tại:**")
    st.info(
        f"LLM: {st.session_state['llm_model']}\n\n"
        f"Embedding: {st.session_state['embedding_model']}\n\n"
        f"Index Strat: {st.session_state['indexing_strategy_name']}\n\n"
        f"Query Strat: {st.session_state['query_strategy_name']}"
    )
    if st.button("Thay đổi cài đặt", use_container_width=True):
        show_settings()

# ==================== MAIN INTERFACE ====================
tab_chat, tab_diff = st.tabs(["Hỏi Đáp", "Phân tích Thay đổi"])

with tab_chat:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Hỏi gì đó (Ví dụ: So sánh hạn sử dụng thẻ căn cước...)"):
        if not st.session_state["db_ready"]:
            st.error("Bạn cần tải lên văn bản và nhấn 'Khởi tạo Hệ thống RAG' trước khi đặt câu hỏi!")
        else:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
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
                    st.caption(f"Thời gian phản hồi: {end_time - start_time:.2f} giây")
                    full_response = full_text + f"\n\nThời gian phản hồi: {end_time - start_time:.2f} giây"
                    
            st.session_state["messages"].append({"role": "assistant", "content": full_response})
            st.rerun()

with tab_diff:
    st.subheader("So sánh Song song các Điều khoản")
    if not st.session_state["db_ready"] or "chunks_old" not in st.session_state or "chunks_new" not in st.session_state:
        st.warning("Vui lòng tải lên tài liệu và Khởi tạo Hệ thống RAG để sử dụng tính năng này.")
    else:
        from src.diff.text_diff_engine import TextDiffEngine
        diff_engine = TextDiffEngine()
        
        chunks_old = st.session_state["chunks_old"]
        chunks_new = st.session_state["chunks_new"]
        
        metas_old = [c.metadata for c in chunks_old]
        texts_old = [c.text for c in chunks_old]
        metas_new = [c.metadata for c in chunks_new]
        texts_new = [c.text for c in chunks_new]
        
        with st.spinner("Đang phân tích sự khác biệt..."):
            diff_results = diff_engine.get_structured_diff(metas_old, texts_old, metas_new, texts_new)
            
        if not diff_results:
            st.info("Không tìm thấy sự thay đổi nào giữa hai bản tài liệu.")
        else:
            num_added = sum(1 for r in diff_results if r["status"] == "added")
            num_deleted = sum(1 for r in diff_results if r["status"] == "deleted")
            num_modified = sum(1 for r in diff_results if r["status"] == "modified")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Số Điều thêm mới", num_added)
            with col_stat2:
                st.metric("Số Điều bị xóa", num_deleted)
            with col_stat3:
                st.metric("Số Điều sửa đổi", num_modified)
                
            st.divider()
            
            for item in diff_results:
                label = item["label"]
                status = item["status"]
                
                if status == "unchanged":
                    continue
                    
                status_text = ""
                if status == "modified":
                    status_text = "[SỬA ĐỔI]"
                elif status == "added":
                    status_text = "[THÊM MỚI]"
                elif status == "deleted":
                    status_text = "[ĐÃ XÓA]"
                    
                with st.expander(f"{status_text} {label}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Bản cũ:**")
                        st.markdown(item["old_text"])
                    with c2:
                        st.markdown("**Bản mới:**")
                        st.markdown(item["new_text"])
