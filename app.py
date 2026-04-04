import streamlit as st
import os
import time
from src.embedding.chroma_manager import ChromaManager
from src.generation.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine
from src.ingestion.document_processor import process_document
from src.query_strategies import STRATEGIES

st.set_page_config(page_title="Hệ thống Trợ lý Pháp lý RAG", page_icon="⚖️", layout="wide")

# ==================== HELPER: LẤY DANH SÁCH MODEL OLLAMA ====================
@st.cache_data(ttl=60)  # Cache 60 giây, tránh gọi API liên tục mỗi lần re-render
def get_ollama_models() -> list[str]:
    """Lấy danh sách tất cả model đang có trong Ollama cục bộ."""
    try:
        import ollama
        models = ollama.list()
        # ollama.list() trả về dict với key "models", mỗi item có key "model"
        return sorted([m["model"] for m in models.get("models", [])])
    except Exception:
        return ["qwen3:8b", "qwen3-embedding:8b"]  # Fallback nếu Ollama không phản hồi

# Lấy danh sách model 1 lần khi app khởi động
_all_models = get_ollama_models()

# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "👋 Xin chào! Vui lòng nạp 2 tài liệu pháp lý ở menu bên trái để tôi có thể hỗ trợ bạn so sánh và phân tích."}]
if "db_ready" not in st.session_state:
    st.session_state["db_ready"] = False
if "old_law_name" not in st.session_state:
    st.session_state["old_law_name"] = ""
if "new_law_name" not in st.session_state:
    st.session_state["new_law_name"] = ""
if "strategy_choice" not in st.session_state:
    st.session_state["strategy_choice"] = list(STRATEGIES.keys())[0]
if "display_mode" not in st.session_state:
    st.session_state["display_mode"] = "Hiển thị Nhanh (Buffered Stream)"
# Mặc định chọn model phù hợp nhất nếu có, không thì lấy model đầu tiên
if "llm_model" not in st.session_state:
    preferred_llm = next((m for m in _all_models if "qwen3" in m and "embedding" not in m), _all_models[0] if _all_models else "qwen3:8b")
    st.session_state["llm_model"] = preferred_llm
if "embedding_model" not in st.session_state:
    preferred_emb = next((m for m in _all_models if "embedding" in m), _all_models[0] if _all_models else "qwen3-embedding:8b")
    st.session_state["embedding_model"] = preferred_emb

# ==================== SETTINGS MODAL ====================
@st.dialog("⚙️ Cài đặt Hệ thống RAG")
def show_settings():
    # ── Model Selection ──────────────────────────────────────────────────────
    st.subheader("🤖 Lựa chọn Model Ollama")

    all_models = get_ollama_models()
    if not all_models:
        st.warning("Không kết nối được Ollama. Hãy đảm bảo `ollama serve` đang chạy.")

    col_llm, col_emb = st.columns(2)
    with col_llm:
        llm_idx = all_models.index(st.session_state["llm_model"]) if st.session_state["llm_model"] in all_models else 0
        chosen_llm = st.selectbox(
            "🧠 LLM (sinh câu trả lời):",
            options=all_models,
            index=llm_idx,
            help="Model dùng để lập luận và sinh văn bản. Khuyến nghị: qwen3:8b"
        )
    with col_emb:
        emb_idx = all_models.index(st.session_state["embedding_model"]) if st.session_state["embedding_model"] in all_models else 0
        chosen_emb = st.selectbox(
            "📐 Embedding (nhúng văn bản):",
            options=all_models,
            index=emb_idx,
            help="Model dùng để tạo vector. Khuyến nghị: qwen3-embedding:8b"
        )

    if chosen_llm == chosen_emb:
        st.warning("⚠️ LLM và Embedding đang dùng cùng 1 model — sẽ tranh VRAM, có thể chậm hơn.")

    st.divider()

    # ── Query Strategy ───────────────────────────────────────────────────────
    st.subheader("📐 Kiến trúc Truy vấn")
    st.caption("Chọn phương pháp RAG để gửi câu hỏi vào cơ sở dữ liệu vector.")
    chosen_strategy = st.selectbox(
        label="Query Strategy:",
        options=list(STRATEGIES.keys()),
        index=list(STRATEGIES.keys()).index(st.session_state["strategy_choice"])
    )

    st.divider()

    # ── Display Mode ─────────────────────────────────────────────────────────
    st.subheader("🖥️ Tốc độ Hiển thị")
    st.caption("Buffered Stream cho chữ chạy liên tục. Instant gom hết rồi hiện 1 lần.")
    chosen_mode = st.radio(
        "Chế độ:",
        options=["Hiển thị Nhanh (Buffered Stream)", "Hiển thị Ngay (Instant)"],
        index=0 if st.session_state["display_mode"] == "Hiển thị Nhanh (Buffered Stream)" else 1
    )

    st.divider()
    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("💾 Lưu cài đặt", type="primary", use_container_width=True):
            st.session_state["llm_model"] = chosen_llm
            st.session_state["embedding_model"] = chosen_emb
            st.session_state["strategy_choice"] = chosen_strategy
            st.session_state["display_mode"] = chosen_mode
            # Reset ChromaDB nếu embedding model thay đổi (vector cũ sẽ không tương thích)
            if chosen_emb != st.session_state.get("embedding_model", chosen_emb):
                st.session_state["db_ready"] = False
                st.info("ℹ️ Embedding model thay đổi — cần Khởi tạo lại RAG.")
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
            with st.spinner("Đang cấu trúc lại văn bản và nhúng (Embed) lên ChromaDB... Xin đợi..."):
                try:
                    db_manager = ChromaManager(
                        collection_name="legal_compare",
                        embedding_model=st.session_state["embedding_model"]
                    )
                    db_manager.reset_collection()
                    
                    all_chunks = []
                    for f_obj in [file_old, file_new]:
                        ext = os.path.splitext(f_obj.name)[1].lower()
                        chunks = process_document(file_source=f_obj, filename=f_obj.name, ext=ext)
                        for chunk in chunks:
                             chunk.metadata["source"] = f_obj.name
                        all_chunks.extend(chunks)
                    
                    if all_chunks:
                        db_manager.add_documents(all_chunks)
                        # Giải phóng VRAM của embedding model → nhường tài nguyên cho LLM
                        db_manager.embedder.unload()
                        init_elapsed = time.time() - init_start
                        st.session_state["db_ready"] = True
                        st.success(
                            f"✅ Đã nạp thành công **{len(all_chunks)} đoạn văn bản** từ 2 tài liệu!\n\n"
                            f"🕒 Thời gian khởi tạo: **{init_elapsed:.1f} giây**"
                        )
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
        f"📐 **Embedding:** {st.session_state['embedding_model']}\n\n"
        f"💼 **Strategy:** {st.session_state['strategy_choice']}\n\n"
        f"🖥 **Hiển thị:** {st.session_state['display_mode']}"
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
            db_manager = ChromaManager(
                collection_name="legal_compare",
                embedding_model=st.session_state.get("embedding_model", "qwen3-embedding:8b")
            )
            llm_client = LLMClient(model_name=st.session_state.get("llm_model", "qwen3:8b"))
            rag_engine = LegalRAGEngine(
                db_manager=db_manager,
                llm_client=llm_client,
                old_law_source=st.session_state["old_law_name"],
                new_law_source=st.session_state["new_law_name"]
            )
            
            chosen_strategy = st.session_state.get("strategy_choice", "Normal_v1 (Raw Query)")
            display_mode = st.session_state.get("display_mode", "Hiển thị Nhanh (Buffered Stream)")
            
            start_time = time.time()
            
            if display_mode == "Hiển thị Ngay (Instant)":
                with st.spinner(""):
                    full_text = ""
                    for chunk_text in rag_engine.stream_ask(query=prompt, strategy_name=chosen_strategy, top_k=6):
                        full_text += chunk_text
                    
                    st.markdown(full_text)
                    end_time = time.time()
                    st.caption(f"🕒 Thời gian phản hồi: {end_time - start_time:.2f} giây")
                    full_response = full_text + f"\n\n🕒 Thời gian phản hồi: {end_time - start_time:.2f} giây"
            else:
                def generate_reply():
                    buffer = ""
                    for chunk_text in rag_engine.stream_ask(query=prompt, strategy_name=chosen_strategy, top_k=6):
                        buffer += chunk_text
                        if len(buffer) >= 20:
                            yield buffer
                            buffer = ""
                    if buffer:
                        yield buffer
                        
                full_response_text = st.write_stream(generate_reply)
                end_time = time.time()
                st.caption(f"🕒 Thời gian phản hồi: {end_time - start_time:.2f} giây")
                full_response = full_response_text + f"\n\n🕒 Thời gian phản hồi: {end_time - start_time:.2f} giây"
                
        # Lưu câu trả lời của Trợ lý vào session
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
