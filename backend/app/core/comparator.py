from typing import Any, Dict, List
import numpy as np

from app.core.embedder import get_embedder, compute_cosine_similarity_matrix
from app.core.lexical import compute_jaccard_similarity
from app.core.scorer import classify_match


def compare_chunks(
    submitted_chunks: List[Dict[str, Any]],
    source_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compares submitted chunks against a pool of source chunks:
    1. Generates normalized dense multilingual embeddings.
    2. Computes the cosine similarity matrix.
    3. Finds the highest scoring source candidate for every submitted chunk.
    4. Computes token-level lexical Jaccard similarity.
    5. Returns classified match records.
    """
    if not submitted_chunks or not source_chunks:
        return []

    embedder = get_embedder()

    submitted_texts = [c.get("text", "") for c in submitted_chunks]
    source_texts = [c.get("text", "") for c in source_chunks]

    # Generate L2-normalized embeddings
    submitted_embs = embedder.embed_texts(submitted_texts, normalize_embeddings=True)
    source_embs = embedder.embed_texts(source_texts, normalize_embeddings=True)

    # Compute full pairwise similarity matrix: shape (N_submitted, M_source)
    similarity_matrix = compute_cosine_similarity_matrix(submitted_embs, source_embs)

    results: List[Dict[str, Any]] = []

    for i, sub_chunk in enumerate(submitted_chunks):
        scores_for_chunk = similarity_matrix[i]
        best_source_idx = int(np.argmax(scores_for_chunk))
        best_semantic_score = round(float(scores_for_chunk[best_source_idx]), 4)

        src_chunk = source_chunks[best_source_idx]
        lexical_score = compute_jaccard_similarity(sub_chunk.get("text", ""), src_chunk.get("text", ""))

        match_type = classify_match(best_semantic_score, lexical_score)

        results.append({
            "submitted_chunk_id": sub_chunk.get("chunk_id", f"sub_{i}"),
            "source_chunk_id": src_chunk.get("chunk_id", f"src_{best_source_idx}"),
            "submitted_text": sub_chunk.get("text", ""),
            "source_text": src_chunk.get("text", ""),
            "submitted_lang": sub_chunk.get("lang", "unknown"),
            "source_lang": src_chunk.get("lang", "unknown"),
            "semantic_similarity": best_semantic_score,
            "lexical_similarity": lexical_score,
            "match_type": match_type,
        })

    return results