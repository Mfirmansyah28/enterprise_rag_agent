# app.py - Streamlit UI untuk Enterprise RAG System
import os
import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

import rag_core

# ============================================
# ENVIRONMENT & LANGSMITH TRACING
# ============================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LangSmith tracing: cukup isi LANGCHAIN_API_KEY di file .env untuk aktifkan.
# Semua run (retrieval, prompt, LLM call, latency, token usage) otomatis
# ter-trace di https://smith.langchain.com tanpa perlu ubah kode lain,
# karena LangChain runnables membaca env var ini secara otomatis.
LANGSMITH_ENABLED = bool(os.getenv("LANGCHAIN_API_KEY"))
if LANGSMITH_ENABLED:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "enterprise-rag-agent")

# ============================================
# SETUP STREAMLIT PAGE
# ============================================
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enterprise RAG System")
st.markdown("Upload dokumen dan tanyakan apa saja tentang isinya!")


# ============================================
# FUNGSI RAG (wrapper cache di atas rag_core.py)
# ============================================
@st.cache_resource
def load_and_process_documents(uploaded_files):
    """Menulis uploaded_files ke temp file, lalu bangun vector store lewat rag_core."""
    pdf_sources = []
    tmp_paths = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        pdf_sources.append((tmp_path, uploaded_file.name))
        tmp_paths.append(tmp_path)

    persist_dir = os.path.join(
        tempfile.gettempdir(), f"chroma_db_{uuid.uuid4().hex}"
    )

    try:
        vector_store = rag_core.build_vector_store(pdf_sources, persist_directory=persist_dir)
    finally:
        for path in tmp_paths:
            os.unlink(path)

    return vector_store


@st.cache_resource
def create_rag_chain(_vector_store):
    return rag_core.create_rag_chain(_vector_store, groq_api_key=GROQ_API_KEY)


def build_chat_history(messages, max_turns=4):
    """Ubah st.session_state.messages (list of dict) jadi list of LangChain
    message objects, dibatasi `max_turns` pertukaran terakhir supaya prompt
    tidak membengkak (hemat token & latency)."""
    history = []
    for m in messages[-(max_turns * 2):]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


# ============================================
# SIDEBAR: UPLOAD DOKUMEN
# ============================================
with st.sidebar:
    st.header("📤 Upload Dokumen")

    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY belum diset di file .env. Sistem tidak akan bisa menjawab.")

    uploaded_files = st.file_uploader(
        "Pilih file PDF",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload file PDF untuk dianalisis"
    )

    if uploaded_files:
        if not GROQ_API_KEY:
            st.warning("⚠️ Set GROQ_API_KEY dulu di .env sebelum upload dokumen.")
        else:
            with st.spinner("🔄 Memproses dokumen..."):
                try:
                    for file in uploaded_files:
                        st.info(f"📄 {file.name}")
                    vector_store = load_and_process_documents(uploaded_files)
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_chain = create_rag_chain(vector_store)
                    st.success(f"✅ {len(uploaded_files)} Dokumen berhasil diproses!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.caption("Dibangun dengan LangChain + Groq + ChromaDB")
    st.caption(f"🔍 LangSmith Tracing: {'🟢 ON' if LANGSMITH_ENABLED else '⚪ OFF'}")

    selected_file = "All Documents"

    if "vector_store" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Statistics")

        st.metric("PDF Files", len(uploaded_files) if uploaded_files else 0)
        st.metric("Chunks", st.session_state.vector_store.chunk_count)
        st.metric("Embedding", "MiniLM")
        st.metric("LLM", "Llama 3.3 70B")

        st.markdown("---")
        st.subheader("📂 Document Filter")

        selected_file = st.selectbox(
            "Search only in:",
            ["All Documents"] + st.session_state.vector_store.file_names
        )

# ============================================
# MAIN CHAT INTERFACE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input chat
if prompt := st.chat_input("Tanyakan tentang dokumen..."):
    # Cek apakah ada dokumen yang sudah diupload
    if "rag_chain" not in st.session_state:
        st.warning("⚠️ Silakan upload dokumen terlebih dahulu di sidebar!")
        st.stop()

    # Ambil riwayat SEBELUM menambahkan pertanyaan saat ini, supaya
    # chat_history yang dikirim ke rag_chain tidak duplikat dengan `prompt`.
    chat_history = build_chat_history(st.session_state.messages)

    # Tampilkan pesan user
    with st.chat_message("user"):
        st.write(prompt)

    # Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Proses dengan RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari jawaban..."):
            try:
                chain_result = st.session_state.rag_chain.invoke({
                    "question": prompt,
                    "chat_history": chat_history,
                })
                answer = chain_result["answer"]

                if selected_file == "All Documents":
                    retrieved = st.session_state.vector_store.similarity_search_with_score(
                        prompt,
                        k=3
                    )
                else:
                    retrieved = st.session_state.vector_store.similarity_search_with_score(
                        prompt,
                        k=3,
                        filter={"source": selected_file}
                    )

                docs = [doc for doc, score in retrieved]
                num_docs = len(docs)

                if num_docs >= 3:
                    confidence = "🟢 High"
                elif num_docs == 2:
                    confidence = "🟡 Medium"
                else:
                    confidence = "🔴 Low"

                st.write(answer)
                st.caption(f"Confidence : {confidence}")
                st.divider()
                st.info(f"📂 Search Scope : {selected_file}")

                if docs:
                    st.subheader("Source Documents")
                    best_page = docs[0].metadata.get("page", 0) + 1
                    st.success(
                        f"⭐ Best Match : {os.path.basename(docs[0].metadata.get('source', 'Unknown'))} "
                        f"(Page {best_page})"
                    )

                    for i, (doc, score) in enumerate(retrieved, start=1):
                        page = doc.metadata.get("page", 0)
                        source = doc.metadata.get("source", "Unknown file")

                        with st.container(border=True):
                            if i == 1:
                                st.markdown("### ⭐ Best Match")
                            else:
                                st.markdown(f"### Source {i}")
                                st.metric("Similarity Score", f"{score:.4f}")

                            col1, col2 = st.columns([3, 1])

                            with col1:
                                if i == 1:
                                    st.success(f"🏆 {os.path.basename(source)}")
                                else:
                                    st.info(f"{os.path.basename(source)}")

                            with col2:
                                st.caption("Page")
                                st.info(f"Page {page + 1}")

                            st.divider()

                            preview = doc.page_content.strip()
                            if len(preview) > 300:
                                preview = preview[:300] + "..."

                            st.markdown("**Preview**")
                            st.write(preview)
                            with st.expander("Show Full Content"):
                                st.write(doc.page_content)
                else:
                    st.warning("Tidak ada dokumen sumber yang cocok ditemukan.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })

                st.session_state["last_docs"] = docs

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================
# TOMBOL RESET
# ============================================
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🗑️ Hapus Dokumen"):
    if "vector_store" in st.session_state:
        del st.session_state.vector_store
    if "rag_chain" in st.session_state:
        del st.session_state.rag_chain
    st.session_state.messages = []
    st.rerun()
