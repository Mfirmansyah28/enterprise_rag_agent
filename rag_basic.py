# app.py - Streamlit UI untuk RAG System
import streamlit as st
import os
from dotenv import load_dotenv
import tempfile

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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
def load_and_process_documents(uploaded_file):
    """Membaca dan memproses dokumen yang diupload."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Load dokumen
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Buat embeddings & vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_temp"
    )
    
    # Cleanup
    os.unlink(tmp_path)
    
    return vector_store

@st.cache_resource
def create_rag_chain(_vector_store):
    """Membuat RAG chain dari vector store yang sudah ada."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )
    
    retriever = _vector_store.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang disediakan.
        
        Gunakan KONTEKS berikut untuk menjawab pertanyaan. Jika tidak tahu, katakan "Saya tidak menemukan informasi itu dalam dokumen."
        
        KONTEKS:
        {context}
        """),
        ("human", "{question}")
    ])
    
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ============================================
# SIDEBAR: UPLOAD DOKUMEN
# ============================================
with st.sidebar:
    st.header("📤 Upload Dokumen")
    
    uploaded_file = st.file_uploader(
        "Pilih file PDF",
        type=['pdf'],
        help="Upload file PDF untuk dianalisis"
    )
    
    if uploaded_file:
        with st.spinner("🔄 Memproses dokumen..."):
            try:
                vector_store = load_and_process_documents(uploaded_file)
                st.session_state.vector_store = vector_store
                st.session_state.rag_chain = create_rag_chain(vector_store)
                st.success(f"✅ Dokumen berhasil diproses!")
                st.info(f"📄 Nama file: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    st.caption("Dibangun dengan LangChain + Groq + ChromaDB")

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
    
    # Tampilkan pesan user
    with st.chat_message("user"):
        st.write(prompt)
    
    # Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Proses dengan RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari jawaban..."):
            try:
                answer = st.session_state.rag_chain.invoke(prompt)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
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