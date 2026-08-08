# Enterprise RAG System

Aplikasi tanya-jawab berbasis dokumen yang dibangun menggunakan arsitektur Retrieval-Augmented Generation (RAG). Pengguna dapat mengunggah dokumen PDF lalu mengajukan pertanyaan dalam bahasa natural, dan sistem akan menjawab berdasarkan isi dokumen tersebut secara akurat menggunakan model bahasa besar dari Groq.

---

## Fitur Utama

- **Upload Dokumen PDF** — Unggah file PDF langsung dari antarmuka web, tanpa konfigurasi tambahan.
- **Pemrosesan Otomatis** — Dokumen dipotong menjadi chunk dengan ukuran optimal, lalu diindeks ke dalam vector store secara otomatis setelah diunggah.
- **Pencarian Semantik** — Menggunakan model embedding `sentence-transformers/all-MiniLM-L6-v2` untuk menemukan konteks yang paling relevan dari dokumen.
- **Jawaban Berbasis Konteks** — Pertanyaan dijawab oleh model `llama-3.3-70b-versatile` via Groq API, dibatasi pada isi dokumen yang diunggah.
- **Antarmuka Chat** — Riwayat percakapan ditampilkan secara interaktif dalam satu sesi.
- **Manajemen Sesi** — Tombol reset chat dan hapus dokumen tersedia untuk memulai sesi baru dengan mudah.
- **Caching Sumber Daya** — Proses embedding dan pembuatan chain di-cache untuk performa optimal selama sesi berlangsung.

---

## Instalasi dan Menjalankan Proyek

### Prasyarat

- Python 3.10 atau lebih baru
- `pip` tersedia di lingkungan Python Anda

### Langkah Instalasi

**1. Clone repositori**

```bash
git clone <url-repositori>
cd enterprise-rag-agent
```

**2. Buat dan aktifkan virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependensi**

```bash
pip install -r requirements.txt
```

**4. Konfigurasi environment variable**

Buat file `.env` di root direktori proyek dengan isi berikut:

```
GROQ_API_KEY=your_groq_api_key_here
```

Dapatkan API key dari [https://console.groq.com](https://console.groq.com).

### Menjalankan Aplikasi

```bash
streamlit run rag_basic.py
```

Aplikasi akan tersedia di `http://localhost:8501` pada browser Anda.

---

## Struktur Folder

```
enterprise-rag-agent/
|
|-- rag_basic.py          # Entry point aplikasi Streamlit dan logika RAG
|-- requirements.txt      # Daftar dependensi Python
|-- .env                  # Konfigurasi environment variable (tidak di-commit)
|-- .gitignore            # Daftar file yang diabaikan oleh Git
|-- README.md             # Dokumentasi proyek
|
|-- chroma_db_temp/       # Direktori penyimpanan vector store lokal (auto-generated)
|-- venv/                 # Virtual environment Python (tidak di-commit)
```

---

## Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| Antarmuka pengguna | Streamlit |
| Orkestrasi RAG | LangChain |
| Model bahasa (LLM) | Llama 3.3 70B via Groq API |
| Model embedding | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| Loader dokumen | PyPDFLoader (LangChain Community) |
| Text splitting | RecursiveCharacterTextSplitter |
| Manajemen environment | python-dotenv |

---

## Catatan Pengembangan

- Direktori `chroma_db_temp/` dibuat secara otomatis saat dokumen pertama diproses. Direktori ini dapat dihapus untuk mereset seluruh data vektor.
- File `.env` tidak boleh di-commit ke repositori publik. Pastikan sudah terdaftar di `.gitignore`.
- Pada penggunaan pertama, model embedding `all-MiniLM-L6-v2` akan diunduh secara otomatis oleh `sentence-transformers`. Pastikan koneksi internet tersedia.

Link Demo:[https://enterpriseragagent-nyeclrndfwz7cvo5y3a43w.streamlit.app/]