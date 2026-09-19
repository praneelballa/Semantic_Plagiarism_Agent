import io
import pytest
import numpy as np
from fastapi.testclient import TestClient
import fitz

from app.main import app
from app.db.models import MetadataStore
from app.db.faiss_index import FaissIndexManager
from app.core.embedder import get_embedder


@pytest.fixture
def clean_db_and_index(tmp_path):
    embedder = get_embedder()
    db_path = tmp_path / "test_metadata.db"
    index_path = tmp_path / "test_faiss.index"
    meta_store = MetadataStore(db_path=db_path)
    index_mgr = FaissIndexManager(
        dimension=embedder.dimension, index_path=index_path, metadata_store=meta_store
    )
    return index_mgr, meta_store


def test_empty_index_search(clean_db_and_index):
    index_mgr, _ = clean_db_and_index
    assert index_mgr.total_vectors == 0

    dummy_query = np.zeros(index_mgr.dimension, dtype=np.float32)
    results = index_mgr.search(dummy_query, top_k=5)
    assert results == []


def test_add_and_search_top_k(clean_db_and_index):
    index_mgr, meta_store = clean_db_and_index
    embedder = get_embedder()

    corpus_texts = [
        "Machine learning models identify patterns in data.",
        "Photosynthesis allows plants to convert light into food.",
        "Deep neural networks require large computational resources.",
    ]
    embeddings = embedder.embed_texts(corpus_texts)

    metadata = [
        {
            "chunk_id": f"chunk_{i}",
            "document_id": "doc_test",
            "source_filename": "test.txt",
            "page": 1,
            "text": text,
            "lang": "en",
            "char_start": 0,
            "char_end": len(text),
        }
        for i, text in enumerate(corpus_texts)
    ]

    added = index_mgr.add_vectors(embeddings, metadata)
    assert added == 3
    assert index_mgr.total_vectors == 3
    assert meta_store.count() == 3

    # Query for ML pattern matching (paraphrase with low lexical overlap)
    query = embedder.embed_text("ML algorithms discover relationships within datasets.")
    hits = index_mgr.search(query, top_k=2)

    assert len(hits) == 2
    # Best match should be chunk_0 (Machine learning)
    assert hits[0]["chunk_id"] == "chunk_0"
    # Calibrated threshold for MiniLM paraphrase semantic score
    assert hits[0]["similarity_score"] > 0.65
    assert "Machine learning" in hits[0]["text"]


def test_dimension_validation(clean_db_and_index):
    index_mgr, _ = clean_db_and_index
    # Intentionally mismatched dimension (128 vs 384)
    invalid_query = np.ones(128, dtype=np.float32)

    with pytest.raises(ValueError, match="dimension"):
        index_mgr.search(invalid_query)


def test_disk_save_and_reload(tmp_path):
    embedder = get_embedder()
    db_path = tmp_path / "persist_meta.db"
    index_path = tmp_path / "persist_faiss.index"

    meta_store = MetadataStore(db_path=db_path)
    index_mgr = FaissIndexManager(
        dimension=embedder.dimension, index_path=index_path, metadata_store=meta_store
    )

    vec = embedder.embed_texts(["A sample sentence for persistence check."])
    meta = [{
        "chunk_id": "c1",
        "document_id": "d1",
        "source_filename": "s.txt",
        "page": 1,
        "text": "A sample sentence for persistence check.",
        "lang": "en",
        "char_start": 0,
        "char_end": 40,
    }]
    index_mgr.add_vectors(vec, meta)
    index_mgr.save()

    # Re-instantiate pointing to the persisted files
    reloaded_meta_store = MetadataStore(db_path=db_path)
    reloaded_index = FaissIndexManager(
        dimension=embedder.dimension, index_path=index_path, metadata_store=reloaded_meta_store
    )
    assert reloaded_index.total_vectors == 1
    assert reloaded_meta_store.count() == 1


def test_corpus_ingest_api_endpoint(monkeypatch, tmp_path):
    embedder = get_embedder()
    test_db = tmp_path / "api_meta.db"
    test_index = tmp_path / "api_faiss.index"
    meta_store = MetadataStore(db_path=test_db)
    test_manager = FaissIndexManager(
        dimension=embedder.dimension, index_path=test_index, metadata_store=meta_store
    )
    monkeypatch.setattr("app.api.v1.endpoints.corpus.get_index_manager", lambda: test_manager)

    client = TestClient(app)

    txt_content = (
        "Artificial intelligence powers semantic search. "
        "It understands context rather than raw keywords. "
        "Vector databases index representations efficiently. "
        "This aids rapid candidate retrieval."
    ).encode("utf-8")

    pdf_doc = fitz.open()
    pdf_page = pdf_doc.new_page()
    pdf_page.insert_text(
        (50, 72),
        "Photosynthesis allows green plants to produce glucose. "
        "Chlorophyll absorbs sunlight in leaves. "
        "Oxygen is released as a byproduct."
    )
    pdf_bytes = pdf_doc.write()
    pdf_doc.close()

    files = [
        ("files", ("test_ai.txt", io.BytesIO(txt_content), "text/plain")),
        ("files", ("test_biology.pdf", io.BytesIO(pdf_bytes), "application/pdf")),
    ]

    response = client.post("/api/v1/corpus/ingest", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["documents_added"] == 2
    assert data["chunks_added"] > 0
    assert data["index_size"] == data["chunks_added"]