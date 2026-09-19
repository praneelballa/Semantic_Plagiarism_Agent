# Release Notes — v1.0.0-demo

### Version
`v1.0.0-demo` (Hackathon / Presentation Ready)

### Implemented & Validated Features
- **Zero-Shot Cross-Lingual Retrieval**: Multilingual MiniLM L12-v2 embedding space (English, Hindi, Telugu).
- **Word-Token Coverage Engine**: Non-duplicate token span calculation replacing document-wide averages.
- **Python AST Semantic Normalizer**: De-aliasing of variables and functions using `ast.NodeTransformer` and `astor`.
- **Cross-Modal Speech Citations**: Whisper ASR ingestion with playback timestamp alignment.
- **Hybrid Vector Persistence**: FAISS IndexFlatIP inner-product indexing mapped to SQLite chunk metadata.
- **Two-Stage Verification**: Optional Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) with Indic recall guardrails.
- **One-Click Live Demo Engine**: Sidebar runner executing all 5 scenarios automatically from `/demo`.

### Deployment Status
- **GitHub**: Pending push to user repository.
- **Live Demo**: Deployment pending — local release validated.