# evaluate_rag.py - Evaluasi kualitas RAG pipeline pakai RAGAS
#
# Cara pakai:
#   1. Taruh beberapa PDF contoh di folder ./eval_docs/
#   2. Isi EVAL_DATASET di bawah dengan pertanyaan + jawaban benar (ground truth)
#      yang kamu tahu jawabannya berdasarkan isi PDF tersebut.
#   3. Jalankan:  python evaluate_rag.py
#
# Ini SENGAJA dipisah dari app.py (bukan tombol di UI Streamlit) karena:
#   - Evaluasi ini bersifat batch/offline, bukan interaktif
#   - Idealnya dijalankan di CI/CD tiap kali prompt/chunking/model diganti,
#     supaya kamu bisa lihat apakah perubahan bikin kualitas naik/turun
#
# Install dependency tambahan dulu:
#   pip install ragas datasets

import glob
import os

from dotenv import load_dotenv

import rag_core

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EVAL_DOCS_FOLDER = "./eval_docs"

# ------------------------------------------------------------------
# GANTI dengan pertanyaan + ground truth sesuai isi PDF kamu sendiri.
# Semakin representatif dataset ini terhadap dokumen asli, semakin
# berguna hasil evaluasinya.
# ------------------------------------------------------------------
EVAL_DATASET = [
    {
        "question": "Apa topik utama dari dokumen ini?",
        "ground_truth": "GANTI: jawaban benar sesuai isi dokumen kamu",
    },
    {
        "question": "Ganti dengan pertanyaan spesifik dari isi dokumenmu",
        "ground_truth": "GANTI: jawaban benar sesuai isi dokumen kamu",
    },
    # Tambahkan minimal 5-10 pasang Q&A untuk hasil evaluasi yang lebih stabil.
]


def main():
    if not GROQ_API_KEY:
        raise SystemExit("❌ GROQ_API_KEY belum diset di .env")

    pdf_paths = glob.glob(os.path.join(EVAL_DOCS_FOLDER, "*.pdf"))
    if not pdf_paths:
        raise SystemExit(
            f"❌ Tidak ada PDF ditemukan di {EVAL_DOCS_FOLDER}/. "
            "Taruh beberapa file PDF contoh di sana dulu."
        )

    pdf_sources = [(path, os.path.basename(path)) for path in pdf_paths]

    print(f"📚 Membangun vector store dari {len(pdf_sources)} PDF...")
    vector_store = rag_core.build_vector_store(pdf_sources)

    print("🔗 Membuat RAG chain...")
    rag_chain = rag_core.create_rag_chain(vector_store, groq_api_key=GROQ_API_KEY)

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"❓ Menjalankan {len(EVAL_DATASET)} pertanyaan evaluasi...")
    for item in EVAL_DATASET:
        result = rag_chain.invoke({"question": item["question"], "chat_history": []})

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result["docs"]])
        ground_truths.append(item["ground_truth"])

    print("📊 Menjalankan evaluasi RAGAS (faithfulness, relevancy, precision, recall)...")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        raise SystemExit(
            "❌ Package RAGAS belum terinstall. Jalankan dulu:\n"
            "   pip install ragas datasets"
        )

    # RAGAS secara default pakai OpenAI sebagai "judge" LLM. Kita ganti supaya
    # pakai LLM & embedding yang SAMA dengan yang dipakai RAG kita (Groq +
    # HuggingFace), supaya tidak butuh OPENAI_API_KEY tambahan.
    from langchain_groq import ChatGroq
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
    judge_embeddings = HuggingFaceEmbeddings(model_name=rag_core.EMBEDDING_MODEL_NAME)

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
    )

    df = results.to_pandas()
    print("\n=== HASIL EVALUASI ===")
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

    output_path = "ragas_eval_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Hasil lengkap disimpan ke {output_path}")

    print("\n=== RATA-RATA SKOR ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"{metric:20s}: {df[metric].mean():.3f}")


if __name__ == "__main__":
    main()
