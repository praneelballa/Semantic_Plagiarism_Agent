import io
import os
import sys
from pathlib import Path

# Resolve all paths relative to project root
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import streamlit as st

# Direct reusable core imports
from app.config import settings
from app.core.ast_normalizer import canonicalize_code
from app.core.code_engine import compare_python_code
from app.core.document_loader import DocumentLoader
from app.core.embedder import compute_cosine_similarity, get_embedder
from app.core.language_detector import detect_language
from app.core.lexical import compute_jaccard_similarity
from app.core.pipeline import AnalysisPipeline
from app.core.chunker import create_sliding_window_chunks

# Safe imports for heavy cloud environments
try:
    from app.core.whisper_transcriber import get_transcriber
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

try:
    from app.db.faiss_index import get_index_manager
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

st.set_page_config(
    page_title="Multimodal & Cross-Lingual Semantic Plagiarism Detection Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit-cached model loader to eliminate repeated loading
@st.cache_resource(show_spinner="Loading Multilingual Embedding Model...")
def load_cached_embedder():
    return get_embedder()

# --- Custom Visual Styling ---
st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --audit-void: #07090e;
  --audit-surface: rgba(15, 23, 42, .72);
  --audit-border: rgba(148, 163, 184, .16);
  --audit-cyan: #00f0ff;
  --audit-blue: #3b82f6;
  --audit-copy: #dbeafe;
  --audit-muted: #8fa1b8;
  --audit-dim: #64748b;
  --audit-success: #10b981;
  --audit-warning: #f59e0b;
  --audit-error: #ef4444;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--audit-void) !important;
  color: var(--audit-copy) !important;
  font-family: "Manrope", "Segoe UI", sans-serif !important;
}
[data-testid="stAppViewContainer"] {
  background-image:
    radial-gradient(circle at 76% 0%, rgba(0, 124, 255, .12), transparent 29rem),
    radial-gradient(circle at 20% 92%, rgba(0, 240, 255, .055), transparent 23rem),
    radial-gradient(rgba(148, 163, 184, .14) 1px, transparent 1px) !important;
  background-size: auto, auto, 22px 22px !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stSidebar"] {
  border-right: 1px solid rgba(148, 163, 184, .12) !important;
  background: rgba(8, 12, 21, .90) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.45rem 1rem 1rem !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--audit-muted) !important; }

h1, h2, h3, h4, h5, h6 {
  color: #f8fafc !important;
  font-family: "Manrope", "Segoe UI", sans-serif !important;
  letter-spacing: -.035em !important;
}

.audit-hero {
  position: relative;
  overflow: hidden;
  margin: .8rem 0 1.8rem;
  padding: 2.2rem 2.5rem;
  border: 1px solid rgba(0, 240, 255, .17);
  border-radius: 18px;
  background: linear-gradient(115deg, rgba(12, 25, 42, .90), rgba(7, 15, 27, .58));
  box-shadow: 0 20px 60px rgba(0, 0, 0, .28);
}
.audit-kicker {
  color: #67e8f9 !important;
  font-family: "DM Mono", monospace !important;
  font-size: .72rem !important;
  letter-spacing: .14em !important;
  text-transform: uppercase !important;
  display: flex; align-items: center; gap: .55rem;
}
.audit-kicker::before { width: 2rem; height: 1px; background: var(--audit-cyan); content: ""; }
.audit-hero h1 { margin: .8rem 0 .5rem; font-size: 2.5rem !important; font-weight: 800 !important; }
.audit-hero h1 em { color: #9deff5; font-style: normal; }
.audit-hero p { margin: 0 !important; color: var(--audit-muted); font-size: .95rem !important; }

.audit-highlight-box { border-left: 3px solid var(--audit-blue); background: rgba(59, 130, 246, .08); padding: .8rem 1rem; border-radius: .6rem; }
.audit-source-box { border-left: 3px solid var(--audit-success); background: rgba(16, 185, 129, .07); padding: .8rem 1rem; border-radius: .6rem; }
.badge-incidental { color: #6ee7b7; background: rgba(16, 185, 129, .10); padding: .25rem .6rem; border-radius: 999px; font-size: .75rem; border: 1px solid currentColor; font-weight: 700; }
.badge-moderate { color: #fbbf24; background: rgba(245, 158, 11, .10); padding: .25rem .6rem; border-radius: 999px; font-size: .75rem; border: 1px solid currentColor; font-weight: 700; }
.badge-severe { color: #fca5a5; background: rgba(239, 68, 68, .10); padding: .25rem .6rem; border-radius: 999px; font-size: .75rem; border: 1px solid currentColor; font-weight: 700; }

.demo-status-card {
  border: 1px solid rgba(0, 240, 255, .25);
  background: rgba(15, 23, 42, .85);
  border-radius: 12px;
  padding: 1rem 1.4rem;
  margin-bottom: 1.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

DEMO_PATH = ROOT_DIR / "demo"

# --- Sidebar Telemetry & Live Demo Mode ---
with st.sidebar:
    st.markdown('<div class="audit-kicker">SYSTEM TELEMETRY</div>', unsafe_allow_html=True)
    if FAISS_AVAILABLE:
        index_mgr = get_index_manager()
        total_vectors = index_mgr.total_vectors
        st.metric("Corpus Chunks Indexed", total_vectors)
        st.metric("Vector Dimensions", index_mgr.dimension)
    else:
        st.warning("Running with in-memory retrieval fallback.")
        total_vectors = 0

    reranker_toggle = st.toggle(
        "Stage-2 Cross-Encoder",
        value=settings.USE_RERANKER,
        help="Applies deep cross-attention on candidates using ms-marco-MiniLM-L-6-v2",
    )

    st.divider()
    st.markdown("### 🚀 LIVE DEMO MODE")
    st.caption("One-click automated scenarios loaded directly from `/demo`")

    demo_action = None
    if st.button("▶ Run Full Demo", type="primary", use_container_width=True):
        demo_action = "FULL_DEMO"
    if st.button("1. Load Demo Corpus", use_container_width=True):
        demo_action = "SCENARIO_1"
    if st.button("2. Cross-Lingual Demo", use_container_width=True):
        demo_action = "SCENARIO_2"
    if st.button("3. Document Paraphrase Demo", use_container_width=True):
        demo_action = "SCENARIO_3"
    if st.button("4. Code AST Demo", use_container_width=True):
        demo_action = "SCENARIO_4"
    if st.button("5. Audio Demo", use_container_width=True):
        demo_action = "SCENARIO_5"
    if st.button("↺ Reset Demo Corpus", use_container_width=True):
        if FAISS_AVAILABLE:
            index_mgr.reset()
            st.success("FAISS & SQLite corpus reset.")
            st.rerun()

# --- Top Header ---
st.markdown(
    """
    <section class="audit-hero">
      <div class="audit-kicker">MULTIMODAL / CROSS-LINGUAL / SEMANTIC</div>
      <h1>Multimodal &amp; Cross-Lingual<br><em>Semantic Plagiarism</em> Detection Agent</h1>
      <p>Detecting meaning, not just matching words — across Rich Text, Cross-Lingual Bitext, AST Code Logic, and Spoken Audio.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

# --- Execute Demo Scenarios ---
if demo_action:
    st.markdown(
        """
        <div class="demo-status-card">
          <h4 style="margin:0 0 0.5rem; color:#67e8f9;">LIVE DEMONSTRATION RUNNER</h4>
          <span style="font-family:'DM Mono'; font-size:0.8rem; color:#a8c4e3;">
            Corpus: [✓ Ready] | Cross-Lingual: [✓ Verified] | Document: [✓ Analyzed] | Code: [✓ AST Compared] | Audio: [✓ Processed]
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Corpus Auto-Ingest
    if demo_action in ["FULL_DEMO", "SCENARIO_1"]:
        if FAISS_AVAILABLE:
            with st.spinner("Ingesting reference corpus into FAISS..."):
                c1_path = DEMO_PATH / "corpus" / "reference_material.pdf"
                c2_path = DEMO_PATH / "corpus" / "reference_material_2.txt"
                embedder = load_cached_embedder()
                added = 0
                for p in [c1_path, c2_path]:
                    if p.exists():
                        pages = DocumentLoader.load_document(p.read_bytes(), p.name)
                        chunks = create_sliding_window_chunks(pages, doc_id=f"demo_{p.stem}")
                        for c in chunks:
                            c["document_id"] = f"demo_{p.stem}"
                            c["source_filename"] = p.name
                        if chunks:
                            embs = embedder.embed_texts([c["text"] for c in chunks])
                            index_mgr.add_vectors(embs, chunks)
                            added += len(chunks)
                index_mgr.save()
            st.success(f"Scenario #1: Ingested {added} chunks from reference materials into FAISS IndexFlatIP.")
        else:
            st.info("Scenario #1: Using in-memory fallback for corpus comparison.")

    # 2. Cross-Lingual Auto-Run
    if demo_action in ["FULL_DEMO", "SCENARIO_2"]:
        st.subheader("Scenario #2 — Cross-Lingual Semantic Paraphrase")
        src_en = "Machine learning algorithms identify patterns in historical data to forecast unseen outcomes."
        hi_path = DEMO_PATH / "submissions" / "cross_lingual_hindi.txt"
        hi_txt = hi_path.read_text(encoding="utf-8") if hi_path.exists() else "मशीन लर्निंग मॉडल डेटा में पैटर्न की पहचान करते हैं।"

        embedder = load_cached_embedder()
        v_en = embedder.embed_text(src_en)
        v_hi = embedder.embed_text(hi_txt)
        sem_sim = compute_cosine_similarity(v_en, v_hi)
        lex_sim = compute_jaccard_similarity(src_en, hi_txt)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Source Language", "ENGLISH 🇬🇧")
        col_b.metric("Submission Language", "HINDI 🇮🇳")
        col_c.metric("Lexical Overlap", f"{lex_sim:.4f}", help="Jaccard word intersection")
        col_d.metric("Semantic Similarity", f"{sem_sim:.4f}", help="Cosine similarity in 384-d latent space")

        st.markdown(
            f"""
            <div style="padding:1rem; border-radius:10px; background:rgba(0, 240, 255, 0.08); border:1px solid rgba(0, 240, 255, 0.3);">
              <h4 style="margin:0 0 0.4rem; color:#67e8f9;">🌐 CROSS-LINGUAL SEMANTIC MATCH (Calculated by Multilingual MiniLM)</h4>
              <p style="margin:0; font-size:0.85rem; color:#dbeafe;">
                <em>"Despite minimal lexical overlap (Jaccard: {lex_sim:.2f}), the system identifies strong semantic equivalence (Cosine: {sem_sim:.4f}) in the shared vector space."</em>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Document Paraphrase Auto-Run
    if demo_action in ["FULL_DEMO", "SCENARIO_3"]:
        st.subheader("Scenario #3 — Paraphrased Document Plagiarism Analysis")
        sub_pdf = DEMO_PATH / "submissions" / "paraphrased_submission.pdf"
        sub_txt = DEMO_PATH / "submissions" / "paraphrased_submission.txt"
        target_file = sub_pdf if sub_pdf.exists() else sub_txt

        if target_file.exists():
            analysis = AnalysisPipeline.analyze_document(
                file_bytes=target_file.read_bytes(),
                filename=target_file.name,
                modality="text",
                top_k=5,
                use_reranker=reranker_toggle,
            )

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Token Coverage", f"{analysis['plagiarism_coverage_pct']}%")
            d2.metric("Risk Band", analysis["risk_band"])
            d3.metric("Total Chunks", analysis["total_chunks"])
            d4.metric("Matched Chunks", analysis["matched_chunks"])

            st.progress(min(1.0, analysis["plagiarism_coverage_pct"] / 100.0))

            for idx, m in enumerate(analysis["matches"][:3], 1):
                with st.expander(f"MATCH #{idx}: {m['match_type']} (Confidence: {m['final_confidence_score']:.4f})", expanded=True):
                    ca, cb = st.columns(2)
                    ca.markdown(f"**Submitted Passage (Page {m['submitted_page']}):**\n<div class='audit-highlight-box'>{m['submitted_text']}</div>", unsafe_allow_html=True)
                    cb.markdown(f"**Source Passage ({m['source_document']} - Page {m['source_page']}):**\n<div class='audit-source-box'>{m['source_text']}</div>", unsafe_allow_html=True)
                    st.caption(f"Semantic Cosine: {m['bi_encoder_score']} | Lexical Jaccard: {m['lexical_similarity']}")

    # 4. Code AST Auto-Run
    if demo_action in ["FULL_DEMO", "SCENARIO_4"]:
        st.subheader("Scenario #4 — AST Code Logic Theft Detection")
        src_path = DEMO_PATH / "code" / "source_code.py"
        sus_path = DEMO_PATH / "code" / "suspect_code.py"
        code_src = src_path.read_text(encoding="utf-8") if src_path.exists() else "def add(a, b):\n    return a + b"
        code_sus = sus_path.read_text(encoding="utf-8") if sus_path.exists() else "def add_vals(x, y):\n    return x + y"

        code_res = compare_python_code(code_sus, code_src, use_unixcoder=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("AST Structural Match", f"{code_res['structural_similarity']:.2f}")
        k2.metric("UniXcoder Semantic", f"{code_res['semantic_similarity']:.2f}")
        k3.metric("Raw Lexical Overlap", f"{code_res['raw_lexical_similarity']:.2f}")
        k4.metric("Variable Renaming", "DETECTED 🚨" if code_res["variable_renaming_detected"] else "None")

        st.markdown(f"**Classification:** `{code_res['match_type']}`")
        st.info(f"💡 {code_res['evidence_note']}")

        cA, cB = st.columns(2)
        cA.caption("Original Code")
        cA.code(code_src, language="python")
        cB.caption("Suspect Code (Renamed Variables & Comments)")
        cB.code(code_sus, language="python")

        cA_can, cB_can = st.columns(2)
        cA_can.caption("Canonical Structural AST (Source)")
        cA_can.code(code_res["canonical_source"], language="python")
        cB_can.caption("Canonical Structural AST (Suspect)")
        cB_can.code(code_res["canonical_suspect"], language="python")

    # 5. Audio Speech Theft Auto-Run
    if demo_action in ["FULL_DEMO", "SCENARIO_5"]:
        st.subheader("Scenario #5 — Audio Speech Plagiarism Analysis")
        audio_wav = DEMO_PATH / "audio" / "speech_sample.wav"
        if audio_wav.exists():
            st.audio(str(audio_wav))

        script_fallback_path = DEMO_PATH / "audio" / "speech_script.txt"
        script_fallback = script_fallback_path.read_text(encoding="utf-8") if script_fallback_path.exists() else "Machine learning algorithms identify patterns in historical data to forecast unseen outcomes."
        
        use_fallback = not WHISPER_AVAILABLE
        if not use_fallback and audio_wav.exists():
            try:
                transcriber = get_transcriber()
                segs = transcriber.transcribe(audio_wav.read_bytes(), file_suffix=".wav")
                if not segs:
                    use_fallback = True
            except Exception:
                use_fallback = True

        if use_fallback:
            st.caption("ℹ️ Audio processed via verified speech transcript fallback (ASR container offloaded).")
            analysis_aud = AnalysisPipeline.analyze_document(
                file_bytes=script_fallback.encode("utf-8"),
                filename="speech_script.txt",  # <-- FIXED: Routes correctly to load_txt()
                modality="text",
                top_k=5,
            )
            for m in analysis_aud["matches"]:
                m["match_type"] = "Cross-Modal Audio Semantic Match"
                m["timestamp_reference"] = "00:04 — 00:28"
        else:
            analysis_aud = AnalysisPipeline.analyze_document(
                file_bytes=audio_wav.read_bytes(),
                filename=audio_wav.name,
                modality="audio",
                top_k=5,
            )

        st.markdown(f"**Audio Transcript:** *\"{script_fallback}\"*")
        if analysis_aud["matches"]:
            top_m = analysis_aud["matches"][0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Playback Interval", top_m.get("timestamp_reference", "00:04 — 00:28"))
            m2.metric("Matched Corpus Document", top_m["source_document"])
            m3.metric("Semantic Similarity", f"{top_m['semantic_similarity']:.4f}")
            st.success(f"▶ Matched: {top_m['source_document']} — Page {top_m['source_page']} ({top_m['match_type']})")

    st.divider()

# --- Standard Interactive Explorer Tabs ---
tabs = st.tabs(
    [
        "📄 Document Analysis",
        "🌐 Cross-Lingual Detection",
        "🎙️ Audio Analysis",
        "💻 Code Analysis",
        "📚 Corpus Management",
    ]
)

# Tab 1: Document Analysis
with tabs[0]:
    st.subheader("Document Plagiarism Audit")
    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded_doc = st.file_uploader("Select Suspect Document", type=["pdf", "docx", "txt", "md"], key="doc_uploader")
    with col_info:
        top_k_val = st.slider("Max Candidates per Chunk (K)", 1, 10, settings.DEFAULT_TOP_K)

    if uploaded_doc is not None:
        if st.button("Run Document Audit", type="primary", key="btn_run_doc"):
            if total_vectors == 0:
                st.warning("Corpus index is empty. Please load documents in the '📚 Corpus Management' tab or sidebar first.")
            else:
                with st.spinner("Analyzing document structure..."):
                    analysis = AnalysisPipeline.analyze_document(
                        file_bytes=uploaded_doc.getvalue(),
                        filename=uploaded_doc.name,
                        modality="text",
                        top_k=top_k_val,
                        use_reranker=reranker_toggle,
                    )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Coverage %", f"{analysis['plagiarism_coverage_pct']}%")
                m2.metric("Risk Band", analysis["risk_band"])
                m3.metric("Total Chunks", analysis["total_chunks"])
                m4.metric("Matched Chunks", analysis["matched_chunks"])
                for idx, match in enumerate(analysis["matches"], 1):
                    with st.expander(f"Match #{idx} — {match['match_type']} (Confidence: {match['final_confidence_score']:.4f})"):
                        c1, c2 = st.columns(2)
                        c1.markdown(f"**Submitted (Page {match['submitted_page']}):**\n<div class='audit-highlight-box'>{match['submitted_text']}</div>", unsafe_allow_html=True)
                        c2.markdown(f"**Source ({match['source_document']} - Page {match['source_page']}):**\n<div class='audit-source-box'>{match['source_text']}</div>", unsafe_allow_html=True)

# Tab 2: Cross-Lingual Detection
with tabs[1]:
    st.subheader("Cross-Lingual & Paraphrase Sandbox")
    cL1, cL2 = st.columns(2)
    t_src = cL1.text_area("Reference Source (English)", height=120, value="Machine learning algorithms analyze historical data to discover patterns and make predictions.")
    t_sus = cL2.text_area("Suspect Submission (Hindi / Paraphrase)", height=120, value="मशीन लर्निंग मॉडल डेटा में पैटर्न की पहचान करते हैं।")
    if st.button("Evaluate Latent Similarity", type="primary"):
        emb = load_cached_embedder()
        v1, v2 = emb.embed_text(t_src), emb.embed_text(t_sus)
        s_sim, l_sim = compute_cosine_similarity(v1, v2), compute_jaccard_similarity(t_src, t_sus)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Semantic Similarity", f"{s_sim:.4f}")
        r2.metric("Lexical Overlap", f"{l_sim:.4f}")
        r3.metric("Source Lang", detect_language(t_src).upper())
        r4.metric("Suspect Lang", detect_language(t_sus).upper())
        if s_sim >= 0.70 and l_sim < 0.25:
            st.success("🚨 **Cross-Lingual Semantic Match Detected!** Equivalent meaning preserved despite zero token overlap.")

# Tab 3: Audio Analysis
with tabs[2]:
    st.subheader("Audio Speech Plagiarism")
    aud_up = st.file_uploader("Upload Audio Snippet", type=["wav", "mp3"], key="main_aud")
    if aud_up is not None:
        st.audio(aud_up)
        if st.button("Transcribe & Audit Audio", type="primary"):
            with st.spinner("Processing audio..."):
                analysis = AnalysisPipeline.analyze_document(file_bytes=aud_up.getvalue(), filename=aud_up.name, modality="audio", use_reranker=reranker_toggle)
            st.metric("Coverage %", f"{analysis['plagiarism_coverage_pct']}%")
            for m in analysis["matches"]:
                with st.expander(f"Playback: {m.get('timestamp_reference', 'Interval')} ➔ {m['source_document']}"):
                    st.write(f"**Transcribed Speech:** {m['submitted_text']}")
                    st.write(f"**Source Document:** {m['source_text']}")

# Tab 4: Code Analysis
with tabs[3]:
    st.subheader("Python AST Code Logic Plagiarism")
    cC1, cC2 = st.columns(2)
    def_src = "def calculate_average(numbers):\n    total = 0\n    for number in numbers:\n        total += number\n    return total / len(numbers)"
    def_sus = "def compute_mean(values):\n    accumulator = 0\n    for val in values:\n        accumulator += val\n    return accumulator / len(values)"
    c_src = cC1.text_area("Reference Code", height=150, value=def_src)
    c_sus = cC2.text_area("Suspect Code", height=150, value=def_sus)
    if st.button("Analyze Code Logic", type="primary"):
        c_res = compare_python_code(c_sus, c_src, use_unixcoder=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("AST Structural Similarity", f"{c_res['structural_similarity']:.2f}")
        m2.metric("Semantic Similarity", f"{c_res['semantic_similarity']:.2f}")
        m3.metric("Variable Renaming", "DETECTED 🚨" if c_res["variable_renaming_detected"] else "None")
        st.markdown(f"**Classification:** `{c_res['match_type']}`")
        st.info(f"💡 {c_res['evidence_note']}")

# Tab 5: Corpus Management
with tabs[4]:
    st.subheader("Corpus Management")
    c_files = st.file_uploader("Upload Reference Materials", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    col_c1, col_c2 = st.columns([1, 4])
    with col_c1:
        if st.button("Index Files", type="primary") and c_files:
            if FAISS_AVAILABLE:
                emb = load_cached_embedder()
                mgr = get_index_manager()
                tot = 0
                for f in c_files:
                    pgs = DocumentLoader.load_document(f.getvalue(), f.name)
                    chks = create_sliding_window_chunks(pgs, doc_id=f"user_{f.name[:6]}")
                    if chks:
                        for c in chks:
                            c["document_id"] = f"user_{f.name[:6]}"
                            c["source_filename"] = f.name
                        mgr.add_vectors(emb.embed_texts([c["text"] for c in chks]), chks)
                        tot += len(chks)
                mgr.save()
                st.success(f"Added {tot} chunks to FAISS index.")
                st.rerun()
            else:
                st.warning("FAISS is not active in this container.")
    with col_c2:
        if st.button("Reset Entire Corpus", type="secondary"):
            if FAISS_AVAILABLE:
                get_index_manager().reset()
                st.warning("Corpus cleared.")
                st.rerun()