import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.coverage import calculate_token_coverage, determine_risk_band
from app.db.models import MetadataStore
from app.db.faiss_index import FaissIndexManager
from app.core.embedder import get_embedder


def test_coverage_and_risk_band_allocation():
    assert determine_risk_band(15.0) == "Incidental conceptual similarity"
    assert determine_risk_band(25.0) == "Incidental conceptual similarity"
    assert determine_risk_band(35.5) == "Moderate semantic overlap / potential paraphrasing"
    assert determine_risk_band(50.0) == "Moderate semantic overlap / potential paraphrasing"
    assert determine_risk_band(78.2) == "Severe conceptual/structural overlap"

    # Verify token-span coverage calculation
    submitted_chunks = [
        {"chunk_id": "c1", "text": "Five distinct words here now.", "page": 1, "char_start": 0, "char_end": 28},
        {"chunk_id": "c2", "text": "Another sentence with five words.", "page": 1, "char_start": 29, "char_end": 62},
    ]
    # No matches -> 0.0%
    assert calculate_token_coverage(submitted_chunks, []) == 0.0

    # 1 out of 2 chunks matched -> ~50%
    matches = [{"submitted_chunk_id": "c1"}]
    coverage = calculate_token_coverage(submitted_chunks, matches)
    assert 45.0 <= coverage <= 55.0


def test_analyze_endpoint_empty_corpus(monkeypatch, tmp_path):
    # Setup clean isolated FAISS & DB
    test_db = tmp_path / "empty_meta.db"
    test_idx = tmp_path / "empty_faiss.index"
    meta_store = MetadataStore(db_path=test_db)
    index_mgr = FaissIndexManager(dimension=384, index_path=test_idx, metadata_store=meta_store)
    monkeypatch.setattr("app.core.pipeline.get_index_manager", lambda: index_mgr)

    client = TestClient(app)

    txt_content = b"This is a fresh piece of content submitted for evaluation."
    files = {"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}
    data = {"type": "text", "compare_corpus": "true"}

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["total_chunks"] >= 1
    assert res["matched_chunks"] == 0
    assert res["plagiarism_coverage_pct"] == 0.0
    assert res["risk_band"] == "Incidental conceptual similarity"
    assert res["matches"] == []


def test_analyze_endpoint_with_corpus_matches(monkeypatch, tmp_path):
    embedder = get_embedder()
    test_db = tmp_path / "active_meta.db"
    test_idx = tmp_path / "active_faiss.index"
    meta_store = MetadataStore(db_path=test_db)
    index_mgr = FaissIndexManager(dimension=embedder.dimension, index_path=test_idx, metadata_store=meta_store)

    # 1. Pre-populate corpus with an English source
    corpus_text = (
        "Machine learning models identify patterns in data. "
        "They utilize training algorithms to predict unseen outcomes. "
        "Modern deep architectures generalize across complex domains."
    )
    c_emb = embedder.embed_texts([corpus_text])
    c_meta = [{
        "chunk_id": "src_chunk_1",
        "document_id": "src_doc_ml",
        "source_filename": "ml_textbook.txt",
        "page": 1,
        "text": corpus_text,
        "lang": "en",
        "char_start": 0,
        "char_end": len(corpus_text),
    }]
    index_mgr.add_vectors(c_emb, c_meta)

    monkeypatch.setattr("app.core.pipeline.get_index_manager", lambda: index_mgr)
    client = TestClient(app)

    # 2. Submit a paraphrased / cross-lingual document
    submitted_text = (
        "ML algorithms discover relationships within datasets. "
        "Statistical methods forecast trends without manual rules. "
        "Neural networks learn intricate features effectively."
    ).encode("utf-8")

    files = {"file": ("student_submission.txt", io.BytesIO(submitted_text), "text/plain")}
    data = {"type": "text", "compare_corpus": "true"}

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["total_chunks"] >= 1
    assert res["matched_chunks"] >= 1
    assert res["plagiarism_coverage_pct"] > 0.0
    assert len(res["matches"]) >= 1

    first_match = res["matches"][0]
    assert first_match["source_document"] == "ml_textbook.txt"
    assert first_match["semantic_similarity"] >= 0.65
    assert first_match["submitted_page"] == 1
    assert first_match["source_page"] == 1