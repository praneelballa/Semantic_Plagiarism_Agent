from functools import lru_cache
from typing import List, Tuple, Optional
import math
from sentence_transformers import CrossEncoder

from app.config import settings
from app.core.logging import logger


class CrossEncoderVerifier:
    """
    Second-stage deep verification using full cross-attention.
    Operates over candidate pairs [source_text, submitted_text].
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Loading Cross-Encoder verification model: '{self.model_name}' on CPU...")
        self.model = CrossEncoder(self.model_name, max_length=512)
        logger.info("Cross-Encoder initialized successfully.")

    def predict_pair(self, source_text: str, submitted_text: str) -> float:
        """Computes cross-encoder score normalized into [0.0, 1.0] using sigmoid."""
        if not source_text or not submitted_text:
            return 0.0

        raw_score = float(self.model.predict([(source_text, submitted_text)])[0])
        # Logistic sigmoid normalization
        probability = 1.0 / (1.0 + math.exp(-raw_score))
        return round(float(probability), 4)

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Computes batch cross-encoder scores."""
        if not pairs:
            return []

        raw_scores = self.model.predict(pairs)
        probabilities = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]
        return [round(p, 4) for p in probabilities]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderVerifier:
    """Singleton cached provider for the CrossEncoderVerifier."""
    return CrossEncoderVerifier()


def calculate_final_confidence(
    bi_encoder_score: float,
    cross_encoder_score: Optional[float],
    submitted_lang: str = "en",
    source_lang: str = "en",
) -> float:
    """
    Calculates final composite decision confidence:
    - If cross_encoder is disabled (None): returns bi_encoder_score.
    - If cross_lingual pair involving Indic/non-Latin scripts (e.g. 'hi', 'te'):
      relies heavily on the multilingual bi-encoder to protect cross-lingual recall.
    - If mono-lingual / Latin: applies 40/60 weighted ensemble.
    """
    if cross_encoder_score is None:
        return round(bi_encoder_score, 4)

    is_cross_lingual = submitted_lang != source_lang
    has_indic = any(lang in ["hi", "te", "ta", "bn"] for lang in [submitted_lang, source_lang])

    if is_cross_lingual and has_indic:
        # Cross-encoder fallback guardrail: weight bi-encoder higher
        final_score = (0.85 * bi_encoder_score) + (0.15 * cross_encoder_score)
    else:
        # Standard mono-lingual precision refinement
        final_score = (0.40 * bi_encoder_score) + (0.60 * cross_encoder_score)

    return round(float(final_score), 4)