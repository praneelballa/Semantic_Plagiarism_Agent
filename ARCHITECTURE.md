# System Architecture

```text
========================================================================================================
                      MULTIMODAL & CROSS-LINGUAL SEMANTIC PLAGIARISM AGENT
========================================================================================================

 [ CLIENT / PRESENTATION LAYER ]
   ├── Streamlit Multi-Modal Dashboard (Port 8501)
   └── Production React + Vite + Tailwind SPA (Port 3000)
                               │
                               ▼ HTTP / REST (JSON + Multipart Form-Data)
 [ API GATEWAY / FASTAPI ENGINE (Port 8000) ]
   ├── POST /api/v1/analyze                <-- Document, Audio & Code Semantic Pipeline
   ├── POST /api/v1/corpus/ingest          <-- Reference Material Batch Embedding & Storage
   ├── GET  /api/v1/analyze/{analysis_id}  <-- In-Memory Pollable Job Store
   └── GET  /api/v1/health                 <-- Liveness & Environment Diagnostic Probe
                               │
 ┌─────────────────────────────┴─────────────────────────────┐
 │                                                           │
 ▼ (Text & Transcribed Audio Pipeline)                       ▼ (Source Code Pipeline)
[ DOCUMENT / AUDIO INGESTION ]                              [ PYTHON AST PIPELINE ]
 ├── PyMuPDF: PDF BBox & Page Offsets                        ├── Python AST Parser (ast.parse)
 ├── python-docx: Word Paragraphs & Tables                   ├── ast.NodeTransformer Identifier De-aliasing
 ├── Normalizer: Unicode NFKC + Invisible Char Stripping     │   └── Sequential Mapping (var_1, func_1)
 ├── Sliding-Window Chunker (W=3, S=1 + Punctuation)         ├── Cosmetic Stripping: Comments & Docstrings
 ├── Language Detector: Langdetect per Chunk                 └── Structural Code Serialization (astor)
 └── Whisper ASR: Audio to Timestamped Intervals                             │
                 │                                                           │
                 ▼                                                           ▼
[ EMBEDDING STAGE 1 (BI-ENCODER) ]                          [ CODE EMBEDDINGS & STRUCTURAL DIFF ]
 ├── Model: paraphrase-multilingual-MiniLM-L12-v2            ├── Model: microsoft/unixcoder-base (768-d)
 ├── 384-dimensional Dense Embeddings                       ├── AST Node Hierarchy Edit Distance (difflib)
 └── Geometric L2 Normalization (||v|| = 1.0)                └── Variable Renaming & Refactoring Detector
                 │                                                           │
                 ▼                                                           │
[ VECTOR RETRIEVAL LAYER ]                                                   │
 ├── FAISS IndexFlatIP (Inner Product = Cosine Sim)                          │
 └── SQLite Metadata Store (chunk_id, doc_id, offsets, page)                 │
                 │                                                           │
                 ▼ (Top-K Candidate Pairs)                                   │
[ OPTIONAL STAGE 2 VERIFICATION (CROSS-ENCODER) ]                             │
 ├── Model: cross-encoder/ms-marco-MiniLM-L-6-v2                             │
 ├── Sigmoid Logistic Normalization (0.0 to 1.0)                              │
 └── Indic Script Guardrail (Protects Bi-Encoder Recall)                     │
                 │                                                           │
                 └─────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
[ AGGREGATION, REPORTING & RISK CLASSIFICATION ]
 ├── Lexical Jaccard Overlap Matrix: |A ∩ B| / |A ∪ B|
 ├── Bounded Word-Token Coverage %: (Flagged Tokens / Total Document Tokens) * 100
 ├── Risk Band Evaluation:
 │     ├── 0%  - 25%: Incidental conceptual similarity
 │     ├── 26% - 50%: Moderate semantic overlap / potential paraphrasing
 │     └── > 50%    : Severe conceptual/structural overlap
 └── Output Generation: Detailed Match Coordinates, Confidence Scores & Explainable Disclosures
========================================================================================================---

#### 4. `README.md`
Update `README.md` in the project root:

```markdown
# Multimodal & Cross-Lingual Semantic Plagiarism Detection Agent

A production-grade, CPU-optimized AI system capable of detecting semantic, paraphrased, and cross-lingual plagiarism across rich text documents, spoken audio transcripts, and Python source code.

---

## 1. Problem Statement
Traditional plagiarism detection engines (e.g., Turnitin, MOSS) depend primarily on exact keyword matches, string winnowing, and n-gram hashing. Consequently, they fail when users:
- Paraphrase content using synonyms while preserving sentence structure.
- Translate text across languages (e.g., English translated into Hindi or Telugu).
- Transcribe unindexed spoken audio (podcasts, lectures) into text submissions.
- Refactor source code by renaming variables, reordering functions, and inverting control-flow statements.

## 2. Solution Overview
This system implements a dense semantic retrieval architecture:
- **Zero-Shot Cross-Lingual Alignment:** Maps multilingual text into a unified 384-dimensional latent metric space where translations project to proximate coordinates.
- **Word-Token Coverage Engine:** Replaces misleading whole-document average similarities with non-duplicate token coverage percentages.
- **Cross-Modal Speech Citations:** Transcribes audio via OpenAI Whisper and tracks playback timestamps against source texts.
- **Abstract Syntax Tree (AST) Canonicalization:** Normalizes Python syntax hierarchies to expose identifier renaming and structural cloning.
- **Optional Two-Stage Verification:** Employs deep cross-attention via a Cross-Encoder to minimize false-positive conceptual drift.

## 3. Technology Stack
- **Backend API:** FastAPI, Uvicorn, Pydantic v2
- **NLP & Embeddings:** PyTorch, Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector Database & Persistence:** FAISS (IndexFlatIP), SQLite (`chunk_metadata`)
- **Speech-to-Text (ASR):** OpenAI Whisper (`tiny`/`base`)
- **Code Intelligence:** Python `ast`, `astor`, HuggingFace (`microsoft/unixcoder-base`)
- **Document Extractors:** PyMuPDF (`fitz`), `python-docx`
- **Frontend Presentation:** Streamlit (Instant Demo Console) & React + Vite + Tailwind CSS

---

## 4. Local Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Node.js v18+ & npm (Optional, for React frontend)
- FFmpeg (Required on system PATH for audio decoding)

### Setup Virtual Environment
```bash
# Clone and enter project
cd semantic-plagiarism-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt