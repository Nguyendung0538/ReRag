import streamlit as st
import os
import time
from src.embedding.chroma_manager import ChromaManager
from src.generation.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine
from src.ingestion.document_processor import process_document
from src.query_strategies import STRATEGIES
from src.indexing_strategies import INDEXING_STRATEGIES
from src.diff import ClauseDiffer

st.set_page_config(page_title="Hệ thống Trợ lý Pháp lý RAG", page_icon="⚖️", layout="wide")

# ==================== CẤU HÌNH MODEL OLLAMA ====================
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
if "strategy_choice" not in st.session_state:
    st.session_state["strategy_choice"] = list(STRATEGIES.keys())[0]
if "indexing_strategy" not in st.session_state:
    st.session_state["indexing_strategy"] = list(INDEXING_STRATEGIES.keys())[0]

if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = LLM_MODEL
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = EMBEDDING_MODEL

# ==================== SETTINGS MODAL ====================
@st.dialog("Cài đặt Hệ thống RAG")
def show_settings():
    # ── Query Strategy ───────────────────────────────────────────────────────
    st.subheader("Kiến trúc Truy vấn")
    st.caption("Chọn phương pháp RAG để gửi câu hỏi vào cơ sở dữ liệu vector.")
    chosen_strategy = st.selectbox(
        label="Query Strategy:",
        options=list(STRATEGIES.keys()),
        index=list(STRATEGIES.keys()).index(st.session_state["strategy_choice"])
    )

    st.divider()


    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Lưu cài đặt", type="primary", use_container_width=True):
            st.session_state["strategy_choice"] = chosen_strategy
            st.rerun()
    with col_cancel:
        if st.button("Huỷ", use_container_width=True):
            st.rerun()


# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Nạp Dữ Liệu")
    st.markdown("Hệ thống chỉ lưu tập trung 2 tài liệu vào bộ nhớ để so sánh chính xác nhất.")
    
    file_old = st.file_uploader("Tài liệu Bản gốc", type=['docx', 'pdf', 'txt'])
    file_new = st.file_uploader("Tài liệu Bản sửa đổi/Bổ sung", type=['docx', 'pdf', 'txt'])
    
    if st.button("Khởi tạo Hệ thống RAG", type="primary", use_container_width=True):
        if file_old and file_new:
            st.session_state["old_law_name"] = file_old.name
            st.session_state["new_law_name"] = file_new.name
            
            init_start = time.time()
            with st.spinner("Đang cấu trúc và nạp văn bản"):
                try:
                    # Parse document chunks — tách riêng từng tài liệu để chạy diff
                    ext_old = os.path.splitext(file_old.name)[1].lower()
                    old_chunks = process_document(file_source=file_old, filename=file_old.name, ext=ext_old)
                    for chunk in old_chunks:
                        chunk.metadata["source"] = file_old.name
                    
                    ext_new = os.path.splitext(file_new.name)[1].lower()
                    new_chunks = process_document(file_source=file_new, filename=file_new.name, ext=ext_new)
                    for chunk in new_chunks:
                        chunk.metadata["source"] = file_new.name
                    
                    all_chunks = old_chunks + new_chunks
                    
                    # Chạy Text Diff thuần túy (1 lần duy nhất)
                    differ = ClauseDiffer()
                    clause_diffs = differ.compare(old_chunks, new_chunks)
                    st.session_state["clause_diffs"] = clause_diffs
                    
                    if all_chunks:
                        # Init strategy (chỉ còn TradiRAG)
                        idx_strat_name = st.session_state["indexing_strategy"]
                        strat_class = INDEXING_STRATEGIES[idx_strat_name]
                        indexer = strat_class(embedding_model=st.session_state["embedding_model"])
                        
                        st.session_state["active_indexer"] = indexer
                        
                        success = indexer.index(all_chunks)
                        
                        init_elapsed = time.time() - init_start
                        if success:
                            st.session_state["db_ready"] = True
                            diff_count = len(clause_diffs)
                            st.success(
                                f"Đã nạp thành công **{len(all_chunks)} đoạn văn bản** từ 2 tài liệu!\n\n"
                                f"📊 Diff tự động phát hiện **{diff_count} điều khoản có thay đổi**.\n\n"
                                f"Thời gian khởi tạo: **{init_elapsed:.1f} giây**"
                            )
                        else:
                            st.error(f"Quá trình lưu thất bại (có thể tổng dung lượng quá giới hạn của Indexing Strategy hiện tại). Hãy thử đổi Indexing Strategy.")
                    else:
                        st.error("Không thể rút trích văn bản từ 2 file này.")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            st.warning("Vui lòng tải lên ĐỦ 2 văn bản để bắt đầu!")

    st.divider()

    # Hiển thị trạng thái setting hiện tại + nút mở modal
    st.markdown("**Cài đặt hiện tại:**")
    st.info(
        f" **LLM:** {st.session_state['llm_model']}\n\n"
        f" **Indexing:** {st.session_state['indexing_strategy']}\n\n"
        f" **Query:** {st.session_state['strategy_choice']}"
    )
    if st.button("Thay đổi cài đặt", use_container_width=True):
        show_settings()

# ==================== MAIN CHAT ====================

# Render các tin nhắn cũ
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
# Xử lý nhập chat
if prompt := st.chat_input("Hỏi gì đó (Ví dụ: So sánh hạn sử dụng thẻ căn cước...)"):
    if not st.session_state["db_ready"]:
        st.error("Bạn cần tải lên văn bản và nhấn 'Khởi tạo Hệ thống RAG' trước khi đặt câu hỏi!")
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
                new_law_source=st.session_state["new_law_name"],
                clause_diffs=st.session_state.get("clause_diffs", [])
            )
            
            chosen_strategy = st.session_state.get("strategy_choice", "Normal_v1 (Raw Query)")
            start_time = time.time()
            
            with st.spinner("Đang suy nghĩ..."):
                full_text = ""
                for chunk_text in rag_engine.stream_ask(query=prompt, strategy_name=chosen_strategy, top_k=6):
                    full_text += chunk_text
                
                st.markdown(full_text)
                
                # Tính Grounding Score
                score = rag_engine.compute_grounding_score(full_text)
                end_time = time.time()
                elapsed = end_time - start_time
                
                st.caption(f"Thời gian phản hồi: {elapsed:.2f}s | Grounding: {score:.0f}%")
                
                # Hiển thị nguồn trích dẫn
                ctx = rag_engine.last_retrieved_context
                src_docs = ctx.get("documents", [[]])[0]
                src_metas = ctx.get("metadatas", [[]])[0]
                
                if src_docs:
                    with st.expander(f"Nguồn trích dẫn ({len(src_docs)} đoạn)", expanded=False):
                        for i, (doc, meta) in enumerate(zip(src_docs, src_metas)):
                            source = meta.get("source", "?")
                            dieu = meta.get("dieu", "?")
                            st.markdown(f"**[{i+1}] {dieu}** - `{source}`")
                            st.text(doc[:300] + ("..." if len(doc) > 300 else ""))
                            if i < len(src_docs) - 1:
                                st.divider()
                
                full_response = full_text + f"\n\nThời gian: {elapsed:.2f}s | Grounding: {score:.0f}%"
                
        # Lưu câu trả lời của Trợ lý vào session
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
