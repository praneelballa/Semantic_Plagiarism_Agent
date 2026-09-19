import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.audio_chunker import create_audio_sliding_window_chunks
from app.core.pipeline import AnalysisPipeline
from app.db.models import MetadataStore
from app.db.faiss_index import FaissIndexManager
from app.core.embedder import get_embedder


def test_audio_sliding_window_chunking_with_timestamps():
    mock_segments = [
        {"text": "First sentence of speech. Second statement follows.", "start": 0.0, "end": 10.0},
        {"text": "Third argument articulated here. Fourth concluding point.", "start": 10.0, "end": 20.0},
    ]

    chunks = create_audio_sliding_window_chunks(
        segments=mock_segments,
        doc_id="test_aud",
        window_size=3,
        step=1,
    )

    assert len(chunks) >= 2
    c0 = chunks[0]
    assert "First sentence of speech." in c0["text"]
    assert "Third argument articulated here." in c0["text"]
    assert "timestamp_reference" in c0
    assert "s" in c0["timestamp_reference"]
    assert c0["start_time"] == 0.0
    assert c0["end_time"] > 0.0


def test_audio_to_text_cross_modal_pipeline(monkeypatch, tmp_path):
    embedder = get_embedder()
    test_db = tmp_path / "audio_meta.db"
    test_idx = tmp_path / "audio_faiss.index"
    meta_store = MetadataStore(db_path=test_db)
    index_mgr = FaissIndexManager(dimension=embedder.dimension, index_path=test_idx, metadata_store=meta_store)

    # 1. Index source textbook passage into FAISS
    book_text = (
        "Machine learning models identify patterns in data. "
        "They utilize training algorithms to predict unseen outcomes. "
        "Modern deep architectures generalize across complex domains."
    )
    b_emb = embedder.embed_texts([book_text])
    b_meta = [{
        "chunk_id": "text_book_01",
        "document_id": "book_1",
        "source_filename": "machine_learning_course.pdf",
        "page": 42,
        "text": book_text,
        "lang": "en",
        "char_start": 0,
        "char_end": len(book_text),
    }]
    index_mgr.add_vectors(b_emb, b_meta)

    monkeypatch.setattr("app.core.pipeline.get_index_manager", lambda: index_mgr)

    # 2. Mock Whisper output simulating transcribed speech from a lecture recording
    class MockWhisper:
        def transcribe(self, audio_bytes, file_suffix=".wav"):
            return [
                {
                    "text": "ML algorithms discover relationships within datasets. Statistical methods forecast trends without manual rules. Neural networks learn intricate features effectively.",
                    "start": 14.2,
                    "end": 28.1,
                }
            ]

    monkeypatch.setattr("app.core.pipeline.get_transcriber", lambda: MockWhisper())

    # 3. Analyze audio file
    result = AnalysisPipeline.analyze_document(
        file_bytes=b"RIFF_FAKE_AUDIO_BYTES",
        filename="lecture_snippet.mp3",
        modality="audio",
    )

    assert result["modality"] == "audio"
    assert result["matched_chunks"] >= 1
    assert result["plagiarism_coverage_pct"] > 0.0

    match = result["matches"][0]
    assert match["match_type"] == "Cross-Modal Audio"
    assert match["source_document"] == "machine_learning_course.pdf"
    assert match["source_page"] == 42
    assert "14.2s" in match["timestamp_reference"]
    assert "28.1s" in match["timestamp_reference"]


def test_analyze_audio_api_endpoint(monkeypatch, tmp_path):
    embedder = get_embedder()
    test_db = tmp_path / "api_audio_meta.db"
    test_idx = tmp_path / "api_audio_faiss.index"
    meta_store = MetadataStore(db_path=test_db)
    index_mgr = FaissIndexManager(dimension=embedder.dimension, index_path=test_idx, metadata_store=meta_store)

    ref_text = "Photosynthesis enables green plants to synthesize nutrients directly from sunlight."
    r_emb = embedder.embed_texts([ref_text])
    r_meta = [{
        "chunk_id": "bio_01",
        "document_id": "doc_bio",
        "source_filename": "botany.txt",
        "page": 1,
        "text": ref_text,
        "lang": "en",
        "char_start": 0,
        "char_end": len(ref_text),
    }]
    index_mgr.add_vectors(r_emb, r_meta)

    monkeypatch.setattr("app.core.pipeline.get_index_manager", lambda: index_mgr)

    class MockWhisper:
        def transcribe(self, audio_bytes, file_suffix=".wav"):
            return [
                {
                    "text": "Plants synthesize nutrients directly from solar sunlight via photosynthesis.",
                    "start": 5.0,
                    "end": 12.5,
                }
            ]

    monkeypatch.setattr("app.core.pipeline.get_transcriber", lambda: MockWhisper())

    client = TestClient(app)
    fake_audio = io.BytesIO(b"FAKE_WAV_HEADER_DATA")

    files = {"file": ("student_audio_essay.wav", fake_audio, "audio/wav")}
    data = {"type": "audio", "compare_corpus": "true"}

    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["modality"] == "audio"
    assert res["matched_chunks"] >= 1
    assert len(res["matches"]) >= 1
    assert res["matches"][0]["timestamp_reference"] == "5.0s - 12.5s"
    assert res["matches"][0]["match_type"] == "Cross-Modal Audio"
    