# app.py - Streamlit UI untuk RAG System
import streamlit as st
import os
import json
import datetime
from dotenv import load_dotenv
import tempfile

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
# FUNGSI RAG
# ============================================
@st.cache_resource
def load_and_process_documents(uploaded_files):
    """Membaca dan memproses semua dokumen yang diupload."""

    all_chunks = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        finally:
            os.unlink(tmp_path)

        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name

        all_chunks.extend(chunks)

    # Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_temp"
    )
    vector_store.chunk_count = len(all_chunks)
    vector_store.file_names = list(
        {chunk.metadata["source"] for chunk in all_chunks}
    )
    return vector_store


def format_docs(docs):
    """Format dokumen dengan metadata source dan page."""
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", 0)
        page_display = page + 1 if isinstance(page, int) else page
        content = doc.page_content.strip()
        formatted.append(
            f"""
==========================
SOURCE : {source}
PAGE   : {page_display}
==========================

{content}
"""
        )
    return "\n\n".join(formatted)


def format_chat_history(messages):
    """Konversi messages session_state ke LangChain message objects."""
    history = []
    for msg in messages[:-1]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


@st.cache_resource
def create_llm():
    """Buat LLM instance."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )


@st.cache_resource
def create_retriever(_vector_store):
    """Buat retriever dari vector store."""
    return _vector_store.as_retriever(search_kwargs={"k": 3})


def build_rag_prompt():
    """Buat prompt dengan conversation memory."""
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """Anda adalah AI Assistant untuk Enterprise RAG System.

Jawablah pertanyaan HANYA berdasarkan konteks yang diberikan.

Setiap kali menjawab, WAJIB sebutkan:
- Nama file
- Nomor halaman

Contoh:
"Berdasarkan Employee_Handbook.pdf halaman 12, ..."

Jika informasi berasal dari beberapa dokumen, sebutkan semuanya.

Jika jawaban tidak ditemukan di dalam konteks, katakan:
"Saya tidak menemukan informasi tersebut di dalam dokumen."

KONTEKS:
{context}"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])


# ============================================
# FUNGSI EXPORT CHAT HISTORY
# ============================================
def export_chat_as_txt(messages):
    """Export chat history sebagai plain text."""
    lines = []
    lines.append("=" * 60)
    lines.append("ENTERPRISE RAG SYSTEM - CHAT HISTORY")
    lines.append(f"Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}]")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def export_chat_as_json(messages):
    """Export chat history sebagai JSON."""
    export_data = {
        "exported_at": datetime.datetime.now().isoformat(),
        "total_messages": len(messages),
        "messages": messages
    }
    return json.dumps(export_data, ensure_ascii=False, indent=2)


# ============================================
# SIDEBAR: UPLOAD DOKUMEN
# ============================================
selected_file = "All Documents"

with st.sidebar:
    st.header("📤 Upload Dokumen")

    uploaded_files = st.file_uploader(
        "Pilih file PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload file PDF untuk dianalisis"
    )

    if uploaded_files:
        with st.spinner("🔄 Memproses dokumen..."):
            try:
                for file in uploaded_files:
                    st.info(f"📄 {file.name}")
                vector_store = load_and_process_documents(uploaded_files)
                st.session_state.vector_store = vector_store
                st.session_state.retriever = create_retriever(vector_store)
                st.session_state.llm = create_llm()
                st.success(f"✅ {len(uploaded_files)} Dokumen berhasil diproses!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.caption("Dibangun dengan LangChain + Groq + ChromaDB")

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

    # Export Chat
    if "messages" in st.session_state and st.session_state.messages:
        st.markdown("---")
        st.subheader("💾 Export Chat")

        col_txt, col_json = st.columns(2)
        with col_txt:
            st.download_button(
                label="📄 TXT",
                data=export_chat_as_txt(st.session_state.messages).encode("utf-8"),
                file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_json:
            st.download_button(
                label="📋 JSON",
                data=export_chat_as_json(st.session_state.messages).encode("utf-8"),
                file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    st.markdown("---")
    col_reset, col_hapus = st.columns(2)
    with col_reset:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_hapus:
        if st.button("🗑️ Hapus Dok", use_container_width=True):
            for key in ["vector_store", "retriever", "llm"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.messages = []
            st.rerun()

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
if user_input := st.chat_input("Tanyakan tentang dokumen..."):
    if "retriever" not in st.session_state:
        st.warning("⚠️ Silakan upload dokumen terlebih dahulu di sidebar!")
        st.stop()

    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        try:
            llm = st.session_state.llm
            rag_prompt = build_rag_prompt()

            # Similarity search
            if selected_file == "All Documents":
                search_result = st.session_state.vector_store.similarity_search_with_score(
                    user_input, k=3
                )
            else:
                search_result = st.session_state.vector_store.similarity_search_with_score(
                    user_input, k=3, filter={"source": selected_file}
                )

            docs = [doc for doc, score in search_result]
            context = format_docs(docs)
            chat_history = format_chat_history(st.session_state.messages)

            # Chain
            chain = rag_prompt | llm | StrOutputParser()

            # Streaming response
            answer_placeholder = st.empty()
            full_answer = ""

            for chunk in chain.stream({
                "context": context,
                "question": user_input,
                "chat_history": chat_history,
            }):
                full_answer += chunk
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

            # Confidence
            num_docs = len(docs)
            if num_docs >= 3:
                confidence = "🟢 High"
            elif num_docs == 2:
                confidence = "🟡 Medium"
            else:
                confidence = "🔴 Low"

            st.caption(f"Confidence: {confidence}")
            st.divider()
            st.info(f"📂 Search Scope: {selected_file}")

            # Source Documents
            if docs:
                st.subheader("Source Documents")
                st.success(
                    f"⭐ Best Match: {os.path.basename(docs[0].metadata.get('source', 'Unknown'))}"
                    f" (Page {docs[0].metadata.get('page', 0) + 1})"
                )

                for i, (doc, score) in enumerate(search_result, start=1):
                    page = doc.metadata.get("page", "Unknown")
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
                            st.info(f"Page {page + 1 if isinstance(page, int) else page}")

                        st.divider()

                        preview = doc.page_content.strip()
                        if len(preview) > 300:
                            preview = preview[:300] + "..."

                        st.markdown("**Preview**")
                        st.write(preview)

                        with st.expander("Show Full Content"):
                            st.write(doc.page_content)
            else:
                st.warning("Tidak ada dokumen yang ditemukan untuk query ini.")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
            })
            st.session_state["last_docs"] = docs

        except Exception as e:
            st.error(f"❌ Error: {e}")
