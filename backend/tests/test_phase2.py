import pytest
import fitz  # PyMuPDF
from app.core.normalizer import normalize_text
from app.core.language_detector import detect_language
from app.core.chunker import split_into_sentences_with_spans, create_sliding_window_chunks
from app.core.document_loader import DocumentLoader


def test_normalization_removes_invisible_characters():
    # Input containing zero-width space, non-breaking space, and arbitrary spacing
    dirty_text = "This is a\u200B test\u00A0string.\n\n\n\nAnother   line."
    clean = normalize_text(dirty_text)
    assert "\u200B" not in clean
    assert "\u00A0" not in clean
    assert "This is a test string." in clean
    assert "\n\n\n" not in clean


def test_sentence_split_preserves_punctuations_and_danda():
    text = "First statement! Second query? Third fact. चौथी पंक्ति।"
    sentences = split_into_sentences_with_spans(text)
    texts = [s["text"] for s in sentences]

    assert len(texts) == 4
    assert texts[0] == "First statement!"
    assert texts[1] == "Second query?"
    assert texts[2] == "Third fact."
    assert texts[3] == "चौथी पंक्ति।"


def test_sliding_window_chunking_metadata():
    pages_data = [
        {
            "page": 1,
            "text": (
                "Sentence one. Sentence two. Sentence three. "
                "Sentence four. Sentence five."
            ),
        }
    ]
    chunks = create_sliding_window_chunks(
        pages_data=pages_data, doc_id="docA", window_size=3, step=1
    )

    # 5 sentences with window=3, step=1 produces:
    # Chunk 0: s0-s2 (1, 2, 3)
    # Chunk 1: s1-s3 (2, 3, 4)
    # Chunk 2: s2-s4 (3, 4, 5)
    assert len(chunks) == 3

    c0 = chunks[0]
    assert c0["chunk_id"] == "docA_p1_w3_s0-2"
    assert c0["page"] == 1
    assert c0["lang"] == "en"
    assert "Sentence one." in c0["text"]
    assert "Sentence three." in c0["text"]
    assert c0["char_start"] == 0
    assert c0["char_end"] > c0["char_start"]

    c1 = chunks[1]
    assert c1["chunk_id"] == "docA_p1_w3_s1-3"
    assert "Sentence two." in c1["text"]
    assert "Sentence four." in c1["text"]


def test_language_detection():
    en_text = "This is a formal paragraph written in English for validation."
    hi_text = "यह हिंदी में लिखा गया एक परीक्षण वाक्य है।"
    assert detect_language(en_text) == "en"
    assert detect_language(hi_text) == "hi"


def test_pdf_extraction(tmp_path):
    # Programmatically create a two-page PDF using PyMuPDF to test ingestion
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((50, 72), "Page one first sentence. Page one second sentence.")

    page2 = doc.new_page()
    page2.insert_text((50, 72), "Page two content begins here. Another sentence follows.")

    doc.save(str(pdf_path))
    doc.close()

    extracted = DocumentLoader.load_pdf(pdf_path)
    assert len(extracted) == 2
    assert extracted[0]["page"] == 1
    assert "Page one first sentence." in extracted[0]["text"]
    assert extracted[1]["page"] == 2
    assert "Page two content begins here." in extracted[1]["text"]

    # Verify chunking over multi-page PDF output
    chunks = create_sliding_window_chunks(extracted, doc_id="test_pdf", window_size=3, step=1)
    assert len(chunks) >= 2
    assert any(c["page"] == 1 for c in chunks)
    assert any(c["page"] == 2 for c in chunks)