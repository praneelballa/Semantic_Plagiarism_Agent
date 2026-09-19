import difflib
from functools import lru_cache
from typing import Any, Dict, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from app.config import settings
from app.core.ast_normalizer import canonicalize_code
from app.core.lexical import compute_jaccard_similarity
from app.core.logging import logger


class UniXcoderEngine:
    """CPU-friendly code embedding pipeline using microsoft/unixcoder-base."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.CODE_EMBEDDING_MODEL
        logger.info(f"Loading Code Model: '{self.model_name}' on CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        logger.info("UniXcoder initialized successfully.")

    def embed_code(self, code_str: str) -> np.ndarray:
        if not code_str or not code_str.strip():
            return np.zeros((768,), dtype=np.float32)

        inputs = self.tokenizer(
            code_str,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Pool using sentence representation ([CLS] token or mean pool)
            embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()

        norm = np.linalg.norm(embeddings)
        if norm > 0:
            embeddings = embeddings / norm
        return embeddings.astype(np.float32)


@lru_cache(maxsize=1)
def get_code_embedder() -> UniXcoderEngine:
    """Singleton cached UniXcoder instance."""
    return UniXcoderEngine()


def compute_ast_structural_similarity(seq_a: str, seq_b: str) -> float:
    """Computes sequence similarity ratio over AST node types."""
    if not seq_a or not seq_b:
        return 0.0
    nodes_a = seq_a.split("-")
    nodes_b = seq_b.split("-")
    return round(float(difflib.SequenceMatcher(None, nodes_a, nodes_b).ratio()), 4)


def compare_python_code(
    suspect_code: str,
    source_code: str,
    use_unixcoder: bool = True,
) -> Dict[str, Any]:
    """
    Compares two Python source scripts:
    1. Extracts and canonicalizes AST structures.
    2. Measures raw lexical similarity vs. AST structural similarity.
    3. Detects deliberate variable/function renaming evasion.
    4. Computes semantic vector similarity via UniXcoder.
    5. Returns classified result with evidence disclosure.
    """
    raw_lexical = compute_jaccard_similarity(suspect_code, source_code)

    try:
        canon_suspect, nodes_suspect = canonicalize_code(suspect_code)
        canon_source, nodes_source = canonicalize_code(source_code)
    except ValueError as exc:
        return {
            "syntax_valid": False,
            "error": str(exc),
            "semantic_similarity": 0.0,
            "structural_similarity": 0.0,
            "variable_renaming_detected": False,
            "match_type": "Syntax Compilation Failure",
            "evidence_note": "Code could not be parsed into an Abstract Syntax Tree.",
        }

    # AST structural edit ratio
    structural_sim = compute_ast_structural_similarity(nodes_suspect, nodes_source)

    # Canonical source overlap
    canon_code_sim = round(
        float(
            difflib.SequenceMatcher(
                None, canon_suspect.splitlines(), canon_source.splitlines()
            ).ratio()
        ),
        4,
    )

    # Semantic embedding similarity
    semantic_sim = canon_code_sim  # Baseline heuristic fallback
    if use_unixcoder:
        try:
            embedder = get_code_embedder()
            vec_a = embedder.embed_code(canon_suspect)
            vec_b = embedder.embed_code(canon_source)
            norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
            if norm_a > 0 and norm_b > 0:
                semantic_sim = round(float(np.dot(vec_a, vec_b) / (norm_a * norm_b)), 4)
        except Exception as e:
            logger.warning(f"UniXcoder inference fallback to canonical sequence similarity: {e}")
            semantic_sim = canon_code_sim

    # Variable renaming check: High structural match despite low raw lexical tokens
    renaming_detected = bool(structural_sim >= 0.85 and raw_lexical < 0.65)

    # Match classification
    if structural_sim >= 0.95 and canon_code_sim >= 0.95:
        match_type = "Semantic Code Logic Match (High Confidence Refactor)"
    elif structural_sim >= 0.80 or semantic_sim >= 0.80:
        match_type = "Structural Logic Clone"
    elif structural_sim >= 0.60:
        match_type = "Partial Algorithmic Overlap"
    else:
        match_type = "Distinct Implementation"

    evidence_note = (
        "Structural and semantic metrics indicate strong architectural alignment; "
        "structural similarity constitutes evidence of shared logic patterns, "
        "not deterministic proof of unauthorized copying."
    )

    return {
        "syntax_valid": True,
        "semantic_similarity": semantic_sim,
        "structural_similarity": structural_sim,
        "canonical_code_similarity": canon_code_sim,
        "raw_lexical_similarity": raw_lexical,
        "variable_renaming_detected": renaming_detected,
        "match_type": match_type,
        "evidence_note": evidence_note,
        "canonical_suspect": canon_suspect,
        "canonical_source": canon_source,
    }