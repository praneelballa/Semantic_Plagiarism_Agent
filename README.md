# Multimodal & Cross-Lingual Semantic Plagiarism Detection Agent

A high-accuracy, multimodal plagiarism detection system capable of semantic verification across text, cross-lingual sources, paraphrased content, transcribed audio, and source code.

## Phase 1 Status
- Standard package layout and configuration setup
- FastAPI backend configured with CORS, custom logging, and environment settings
- Health probes at `/health` and `/api/v1/health`
- Streamlit connectivity probe
- Automated tests passing via `pytest`

---

## Getting Started Locally

### 1. Environment Setup
Create and activate a virtual environment (Python 3.10 or 3.11 recommended):

```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate