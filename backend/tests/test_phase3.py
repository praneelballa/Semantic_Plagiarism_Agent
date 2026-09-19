import pytest
import numpy as np
from app.core.embedder import (
    MultilingualEmbedder,
    get_embedder,
    compute_cosine_similarity,
    compute_cosine_similarity_matrix,
)


def test_embedder_initialization_and_dimension():
    embedder = get_embedder()
    # paraphrase-multilingual-MiniLM-L12-v2 produces 384-dim dense vectors
    assert embedder.dimension == 384
    assert isinstance(embedder.dimension, int)


def test_embedder_caching_singleton():
    embedder_1 = get_embedder()
    embedder_2 = get_embedder()
    assert embedder_1 is embedder_2


def test_empty_input_handling():
    embedder = get_embedder()

    empty_res = embedder.embed_texts([])
    assert empty_res.shape == (0, embedder.dimension)

    blank_vec = embedder.embed_text("   ")
    assert blank_vec.shape == (embedder.dimension,)
    assert np.allclose(blank_vec, 0.0)


def test_normalization_properties():
    embedder = get_embedder()
    text = "Artificial intelligence is transforming automated plagiarism detection."
    vec = embedder.embed_text(text, normalize_embeddings=True)

    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_english_paraphrase_semantic_similarity():
    embedder = get_embedder()

    base_text = "Machine learning models identify patterns in data."
    paraphrase_text = "ML algorithms discover relationships within datasets."
    unrelated_text = "Cooking Italian pasta requires boiling salted water in a large pot."

    emb_base = embedder.embed_text(base_text)
    emb_para = embedder.embed_text(paraphrase_text)
    emb_unrelated = embedder.embed_text(unrelated_text)

    sim_paraphrase = compute_cosine_similarity(emb_base, emb_para)
    sim_unrelated = compute_cosine_similarity(emb_base, emb_unrelated)

    # Calibrated for paraphrase-multilingual-MiniLM-L12-v2
    assert sim_paraphrase > 0.65, f"Expected > 0.65, got {sim_paraphrase}"
    assert sim_unrelated < 0.35, f"Expected < 0.35, got {sim_unrelated}"
    assert sim_paraphrase > (sim_unrelated + 0.30)


def test_cross_lingual_semantic_similarity():
    embedder = get_embedder()

    base_text = "Machine learning models identify patterns in data."
    hindi_text = "मशीन लर्निंग मॉडल डेटा में पैटर्न की पहचान करते हैं।"
    unrelated_text = "The quick brown fox jumps over the lazy dog."

    emb_base = embedder.embed_text(base_text)
    emb_hi = embedder.embed_text(hindi_text)
    emb_unrelated = embedder.embed_text(unrelated_text)

    sim_hi = compute_cosine_similarity(emb_base, emb_hi)
    sim_unrelated = compute_cosine_similarity(emb_base, emb_unrelated)

    # Hindi alignment is strong in MiniLM-L12-v2
    assert sim_hi > 0.70, f"Expected Hindi similarity > 0.70, got {sim_hi}"
    assert sim_unrelated < 0.35, f"Expected unrelated similarity < 0.35, got {sim_unrelated}"


def test_batch_matrix_similarity():
    embedder = get_embedder()
    set_a = [
        "Machine learning models identify patterns in data.",
        "The weather is sunny today.",
    ]
    set_b = [
        "ML algorithms discover relationships within datasets.",
        "It is a bright and clear day outside.",
    ]

    embs_a = embedder.embed_texts(set_a)
    embs_b = embedder.embed_texts(set_b)

    sim_matrix = compute_cosine_similarity_matrix(embs_a, embs_b)

    assert sim_matrix.shape == (2, 2)
    # Diagonal semantic alignment
    assert sim_matrix[0, 0] > 0.65  # ML <-> ML
    assert sim_matrix[1, 1] > 0.65  # Weather <-> Weather
    # Off-diagonal cross-topic divergence
    assert sim_matrix[0, 1] < 0.35
    assert sim_matrix[1, 0] < 0.35