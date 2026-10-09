# 🛡️ Multimodal & Cross-Lingual Semantic Plagiarism Detection Agent

A high-precision, multimodal plagiarism detection system capable of semantic verification across text documents, cross-lingual translations, transcribed speech audio, and refactored source code.

## 🌟 Key Features

* **Multilingual Semantic Embeddings:** Detects conceptual theft and paraphrasing across languages (English, Hindi, Telugu) using sentence transformers (`paraphrase-multilingual-MiniLM-L12-v2`).
* **Multi-Format Document Intake:** Ingests and processes `.pdf`, `.docx`, `.txt`, and `.md` files with NFKC Unicode normalization and invisible-character stripping.
* **Sliding-Window Chunking:** Slices text into overlapping 3-sentence windows ($W=3, S=1$) to preserve sentence context across paragraph boundaries.
* **Vector Search & Persistence:** Uses FAISS `IndexFlatIP` for rapid cosine similarity matching coupled with an SQLite metadata store.
* **Two-Stage Re-Ranking:** Deep cross-encoder verification (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with language-aware guardrails to suppress false positives.
* **Cross-Modal Audio Detection:** Transcribes speech recordings (`.wav`, `.mp3`) via OpenAI Whisper with timestamp citations mapped directly against indexed text.
* **Code Logic & AST Normalizer:** Detects variable renaming, syntax canonicalization, and structural theft in Python files using AST parsing and UniXcoder embeddings.
* **Modern Web Dashboard:** Full React + Vite + Tailwind CSS audit console alongside a lightweight Streamlit demonstration interface.

---

## 🏗️ Architecture

```
                          ┌────────────────────────┐
                          │   Frontend Interfaces  │
                          │ React (Vite) / Streamlit│
                          └───────────┬────────────┘
                                      │ REST API
                                      ▼
                          ┌────────────────────────┐
                          │  FastAPI Backend Core  │
                          └───────────┬────────────┘
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Text Pipeline   │        │  Audio Pipeline  │        │  Code Pipeline   │
│  • Normalizer    │        │  • FFmpeg Decode │        │  • AST Parser    │
│  • Lang Detect   │        │  • Local Whisper │        │  • Renaming Norm │
│  • Window Chunk  │        │  • Timestamping  │        │  • UniXcoder     │
└────────┬─────────┘        └────────┬─────────┘        └────────┬─────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │   Embedding & Scoring Core    │
                     │  • MiniLM-L12-v2 (384-dim)    │
                     │  • Jaccard Lexical Overlap    │
                     │  • MS-MARCO Cross-Encoder     │
                     └───────────────┬───────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │       Persistence Layer       │
                     │  • FAISS IndexFlatIP          │
                     │  • SQLite Metadata DB         │
                     └───────────────────────────────┘

```

---

## 📋 System Prerequisites

1. **Python 3.10+** (Tested on Python 3.11 and 3.12).
2. **Node.js 18+** & **npm** (for the React frontend).
3. **FFmpeg** installed on your system PATH (required by Whisper for audio decoding).
* **Windows:** `winget install "FFmpeg (Essentials Build)"`
* **macOS:** `brew install ffmpeg`
* **Linux:** `sudo apt install ffmpeg`



---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Backend

```bash
# Clone the repository
git clone https://github.com/your-username/semantic-plagiarism-agent.git
cd semantic-plagiarism-agent

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: .\venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r backend/requirements.txt

```

### 2. Configure Environment

Copy the example environment configuration:

```bash
cp .env.example .env           # On Windows PowerShell: Copy-Item .env.example .env

```

### 3. Run the Automated Tests

Verify that all ingestion, embedding, FAISS, audio, code, and reranking pipelines pass:

```bash
# On Linux/macOS
PYTHONPATH=backend pytest backend/tests/ -v

# On Windows PowerShell
$env:PYTHONPATH="backend"; pytest backend/tests/ -v

```

### 4. Start the FastAPI Backend

```bash
# On Linux/macOS
PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# On Windows PowerShell
$env:PYTHONPATH="backend"; uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

```

* **API Documentation (Swagger UI):** `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`
* **System Health Probe:** `[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)`

---

## 💻 Running the Frontend

### Option A: Modern React + Vite Web UI (Recommended)

In a separate terminal window:

```bash
cd frontend
npm install
npm run dev

```

Open **`http://localhost:3000`** in your browser.

### Option B: Streamlit Console

In a separate terminal window:

```bash
source venv/bin/activate       # On Windows: .\venv\Scripts\Activate.ps1
streamlit run streamlit_demo.py

```

Open **`http://localhost:8501`** in your browser.

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Root system and environment diagnostic probe. |
| `POST` | `/api/v1/corpus/ingest` | Multipart upload to chunk, embed, and index reference files into FAISS. |
| `POST` | `/api/v1/analyze` | Unified submission audit for Text, Audio (`.wav`/`.mp3`), or Python Code (`.py`). |
| `GET` | `/api/v1/analyze/{id}` | Retrieve previously cached analysis reports by ID. |

---

## 📂 Project Directory Structure

```text
semantic-plagiarism-agent/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/  # Health, corpus ingestion, and analysis routes
│   │   ├── core/              # Extractors, chunkers, embedders, AST normalizer, Whisper, reranker
│   │   └── db/                # FAISS IndexFlatIP manager and SQLite metadata models
│   └── tests/                 # Full unit & integration test suite (Phases 1–9)
├── frontend/                  # React + TypeScript + Tailwind web application
├── data/
│   ├── corpus/                # Reference documents library
│   ├── uploads/               # Uploaded suspect submissions
│   └── indexes/               # Persistent FAISS binary indexes and SQLite database
├── streamlit_demo.py          # Quick-probe demonstration console
└── requirements.txt           # Python library specifications

```
