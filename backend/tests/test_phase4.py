import pytest
from app.core.comparator import compare_chunks
from app.core.lexical import compute_jaccard_similarity
from app.core.scorer import MatchType


def test_lexical_jaccard_similarity():
    # Exact text match
    text_a = "Machine learning identifies patterns."
    text_b = "Machine learning identifies patterns."
    assert compute_jaccard_similarity(text_a, text_b) == 1.0

    # Partial overlap
    text_c = "Deep learning identifies patterns in computer vision."
    score = compute_jaccard_similarity(text_a, text_c)
    assert 0.0 < score < 1.0

    # No overlap
    text_d = "Cooking pasta requires boiling water."
    assert compute_jaccard_similarity(text_a, text_d) == 0.0


def test_comparison_identical_text():
    source_chunks = [{
        "chunk_id": "src_1",
        "text": "Deep learning architectures have revolutionized natural language processing.",
        "lang": "en",
    }]
    submitted_chunks = [{
        "chunk_id": "sub_1",
        "text": "Deep learning architectures have revolutionized natural language processing.",
        "lang": "en",
    }]

    matches = compare_chunks(submitted_chunks, source_chunks)
    assert len(matches) == 1

    match = matches[0]
    assert match["semantic_similarity"] >= 0.95
    assert match["lexical_similarity"] == 1.0
    assert match["match_type"] == MatchType.DIRECT_COPY.value

def test_comparison_paraphrased_english():
    source_chunks = [{
        "chunk_id": "src_para",
        "text": "Machine learning models identify patterns in data.",
        "lang": "en",
    }]
    submitted_chunks = [{
        "chunk_id": "sub_para",
        "text": "ML algorithms discover relationships within datasets.",
        "lang": "en",
    }]

    matches = compare_chunks(submitted_chunks, source_chunks)
    assert len(matches) == 1

    match = matches[0]
    # Calibrated for MiniLM (scores ~0.67)
    assert match["semantic_similarity"] >= 0.65
    assert match["lexical_similarity"] < 0.30
    assert match["match_type"] in [
        MatchType.PARAPHRASE.value,
        MatchType.STRONG_SEMANTIC.value,
    ]

def test_comparison_cross_lingual_text():
    source_chunks = [{
        "chunk_id": "src_en",
        "text": "Machine learning models identify patterns in data.",
        "lang": "en",
    }]
    # Hindi direct equivalent
    submitted_chunks = [{
        "chunk_id": "sub_hi",
        "text": "मशीन लर्निंग मॉडल डेटा में पैटर्न की पहचान करते हैं।",
        "lang": "hi",
    }]

    matches = compare_chunks(submitted_chunks, source_chunks)
    assert len(matches) == 1

    match = matches[0]
    # Cross-lingual match has high semantic similarity and zero lexical token overlap
    assert match["semantic_similarity"] >= 0.78
    assert match["lexical_similarity"] == 0.0
    assert match["match_type"] in [
        MatchType.PARAPHRASE.value,
        MatchType.STRONG_SEMANTIC.value,
    ]


def test_comparison_unrelated_text():
    source_chunks = [{
        "chunk_id": "src_science",
        "text": "Photosynthesis is the process by which green plants produce glucose from sunlight.",
        "lang": "en",
    }]
    submitted_chunks = [{
        "chunk_id": "sub_finance",
        "text": "Central banks determine national monetary policy and manage interest rates.",
        "lang": "en",
    }]

    matches = compare_chunks(submitted_chunks, source_chunks)
    assert len(matches) == 1

    match = matches[0]
    assert match["semantic_similarity"] < 0.65
    assert match["lexical_similarity"] == 0.0
    assert match["match_type"] == MatchType.UNRELATED.value