# Pre-Flight Deployment Checklist

- [x] Repository cleaned of virtual environments (`venv`, `.venv`)
- [x] Secrets removed and `.env` added to `.gitignore`
- [x] `.env.example` created with template variables
- [x] Root `requirements.txt` configured with pinned dependencies
- [x] `.streamlit/config.toml` configured with theme and upload caps
- [x] Absolute machine paths replaced with `Path(__file__)`
- [x] Model cached via `@st.cache_resource` in `streamlit_demo.py`
- [x] Fallback handlers added for FAISS and Whisper
- [x] Demo files generated and verified via `demo/run_demo_check.py`
- [x] Local test completed via `streamlit run streamlit_demo.py`
- [x] `QUICKSTART.md`, `RELEASE_NOTES.md`, and `README.md` updated
- [ ] Push to GitHub `main` branch
- [ ] Deploy repository on Streamlit Community Cloud
- [ ] Insert verified live URL into `README.md` and `RELEASE_NOTES.md`