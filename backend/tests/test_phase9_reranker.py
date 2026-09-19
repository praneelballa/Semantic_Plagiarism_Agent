import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.reranker import (
    CrossEncoderVerifier,
    calculate_final_confidence,
)
from app.core.pipeline import AnalysisPipeline
from app.db.models import MetadataStore
from app.db.faiss_index import FaissIndexManager
from app.core.embedder import get_embedder


def test_confidence_calculation_reranker_disabled():
    # When cross-encoder is disabled, confidence equals the bi-encoder score exactly
    score = calculate_final_confidence(
        bi_encoder_score=0.82,
        cross_encoder_score=None,
    )
    assert score == 0.82


def test_confidence_calculation_reranker_enabled_monolingual():
    # 0.40 * 0.70 + 0.60 * 0.90 = 0.28 + 0.54 = 0.82
    score = calculate_final_confidence(
        bi_encoder_score=0.70,
        cross_encoder_score=0.90,
        submitted_lang="en",
        source_lang="en",
    )
    assert score == 0.82


def test_confidence_calculation_reranker_cross_lingual_guardrail():
    # Cross-lingual Indic pair protects bi-encoder recall:
    # 0.85 * 0.80 + 0.15 * 0.40 = 0.68 + 0.06 = 0.74
    score = calculate_final_confidence(
        bi_encoder_score=0.80,
        cross_encoder_score=0.40,
        submitted_lang="hi",
        source_lang="en",
    )
    assert score == 0.74


def test_reranker_prediction_probability_range():
    verifier = CrossEncoderVerifier()
    source = "Machine learning identifies patterns in historical data."
    candidate = "Algorithms recognize underlying structures within datasets."

    prob = verifier.predict_pair(source, candidate)
    # Sigmoid output must be bounded between 0.0 and 1.0
    assert 0.0 <= prob <= 1.0


def test_pipeline_with_reranker_disabled(monkeypatch, tmp_path):
    embedder = get_embedder()
    test_db = tmp_path / "p9_dis_meta.db"
    test_idx = tmp_path / "p9_dis_faiss.index"
    meta_store = MetadataStore(db_path=test_db)
    index_mgr = FaissIndexManager(dimension=embedder.dimension, index_path=test_idx, metadata_store=meta_store)

    ref_text = "Machine learning algorithms learn patterns directly from data inputs."
    emb = embedder.embed_texts([ref_text])
    meta = [{
        "chunk_id": "c_ref",
        "document_id": "doc_ref",
        "source_filename": "ml.txt",
        "page": 1,
        "text": ref_text,
        "lang": "en",
        "char_start": 0,
        "char_end": len(ref_text),
    }]
    index_mgr.add_vectors(emb, meta)
    monkeypatch.setattr("app.core.pipeline.get_index_manager", lambda: index_mgr)

    sub_bytes = b"Machine learning algorithms learn patterns directly from data inputs."
    result = AnalysisPipeline.analyze_document(
        file_bytes=sub_bytes,
        filename="sub.txt",
        use_reranker=False,
    )

    assert result["reranker_enabled"] is False
    assert result["matched_chunks"] >= 1
    match = result["matches"][0]
    assert match["bi_encoder_score"] >= 0.88
    assert match["cross_encoder_score"] is None
    assert match["final_confidence_score"] == match["bi_encoder_score"]


def test_pipeline_with_reranker_enabled(monkeypatch, tmp_path):
    embedder = get_embedder()
    test_db = tmp_path / "p9_en_meta.db"
    test_idx = tmp_path / "p9_en_faiss.index"
    meta_store = MetadataStore(db_path=test_db)
    index_mgr = FaissIndexManager(dimension=embedder.dimension, index_path=test_idx, metadata_store=meta_store)

    ref_text = "Photosynthesis converts solar light energy into biological chemical sugars."
    emb = embedder.embed_texts([ref_text])
    meta = [{
        "chunk_id": "c_bio",
        "document_id": "doc_bio",
        "source_filename": "biology.txt",
        "page": 1,
        "text": ref_text,
        "lang": "en",
        "char_start": 0,
        "char_end": len(ref_text),
    }]
    index_mgr.add_vectors(emb, meta)
    monkeypatch.setattr("app.core.pipeline.get_index_manager", lambda: index_mgr)

    sub_bytes = b"Photosynthesis transforms solar light energy into biological chemical sugars."
    result = AnalysisPipeline.analyze_document(
        file_bytes=sub_bytes,
        filename="sub_bio.txt",
        use_reranker=True,
    )

    assert result["reranker_enabled"] is True
    assert result["matched_chunks"] >= 1
    match = result["matches"][0]
    assert match["bi_encoder_score"] is not None
    assert match["cross_encoder_score"] is not None
    assert 0.0 <= match["cross_encoder_score"] <= 1.0
    assert match["final_confidence_score"] > 0.65
    