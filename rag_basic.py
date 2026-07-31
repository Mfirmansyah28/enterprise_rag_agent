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

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        # Simpan nama file ke metadata
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name

        all_chunks.extend(chunks)

        # Hapus file sementara
        os.unlink(tmp_path)

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

    vector_store.file_names = list (
        {
            chunk.metadata["source"]
            for chunk in all_chunks
        }
    )
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

    def format_docs(docs):
        formatted=[]
        for doc in docs:
            source=os.path.basename(doc.metadata.get("source","Unknown"))
            page=doc.metadata.get("page",0)+1
            formatted.append(f"Source: {source}\nPage: {page}\n\nContent:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain=(
        RunnableParallel(
            docs=retriever,
            question=RunnablePassthrough(),
        )
        | RunnableParallel(
            answer=(
                {
                    "context": lambda x: format_docs(x["docs"]),
                    "question": lambda x: x["question"],
                }
                | prompt
                | llm
                | StrOutputParser()
            ),
            docs=lambda x: x["docs"],
        )
    )
    return rag_chain

# ============================================
# SIDEBAR: UPLOAD DOKUMEN
# ============================================
with st.sidebar:
    st.header("📤 Upload Dokumen")
    
    uploaded_files = st.file_uploader(
        "Pilih file PDF",
        type=['pdf'],
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
                st.session_state.rag_chain = create_rag_chain(vector_store)
                st.success(f"✅ {len(uploaded_files)} Dokumen berhasil diproses!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    st.caption("Dibangun dengan LangChain + Groq + ChromaDB")

    if "vector_store" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Statistics")

        st.metric(
            "PDF Files",
            len(uploaded_files)
        )

        st.metric(
            "Chunks",
            st.session_state.vector_store.chunk_count
        )

        st.metric(
            "Embedding",
            "MiniLM"
        )

        st.metric(
            "LLM",
            "Llama 3.3 70B"
        )

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
    
    # Tampilkan pesan user
    with st.chat_message("user"):
        st.write(prompt)
    
    # Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Proses dengan RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari jawaban..."):
            try:
                result = st.session_state.rag_chain.invoke(prompt)

                answer = result["answer"]

                if selected_file == "All Documents":
                    result = st.session_state.vector_store.similarity_search_with_score (
                        prompt,
                        k=3
                    )

                else:
                    result = st.session_state.vector_store.similarity_search_with_score(
                        prompt,
                        k=3,
                        filter={
                            "source": selected_file
                        }
                    )

                    docs = [doc for doc, score in result]
                
                num_docs = len(docs)

                if num_docs >=3:
                    confidence =  "🟢 High"

                elif num_docs == 2:
                    confidence = "🟡 Medium"

                else:
                    confidence = "🔴 Low"

                st.write(answer)
                st.caption(f"Confidence : {confidence}")
                st.divider()
                st.info(f"📂 Search Scope : {selected_file}")
                st.subheader("Source Documents")
                st.success(
                    f"⭐ Best Match : {os.path.basename(docs[0].metadata.get('source', 'Unknown'))}"
                    f"(Page {docs[0].metadata.get('page', 0) + 1})"
                )

                for i, (doc, score) in enumerate (result, start=1):
                    page = doc.metadata.get("page", "Unknown")

                    source = doc.metadata.get(
                        "source",
                        "Unknown file"
                    ) 

                    with st.container(border=True):
                        if i == 1:
                            st.markdown("### ⭐ Best Match")

                        else:
                            st.markdown(f"###  Source {i}")

                            st.metric(
                                "Similarity Score",
                                f"{score: .4f}"
                            )
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            if i == 1:
                                st.success(f"🏆 {os.path.basename(source)}")
                            else:
                                st.info(f"{os.path.basename(source)}")
                        
                        with col2:
                            st.caption("Page")
                            st.info(f" Page {page + 1}")

                        st.divider()

                        preview = doc.page_content.strip()
                        if len(preview) > 300:
                            preview = preview[:300] + "..."

                        st.markdown("**Preview**")

                        st.write(preview)
                        with st.expander("Show Full Content"):
                            st.write(doc.page_content)

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