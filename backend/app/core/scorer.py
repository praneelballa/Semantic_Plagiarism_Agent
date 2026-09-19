from enum import Enum


class MatchType(str, Enum):
    DIRECT_COPY = "Identical or Direct Copy"
    STRONG_SEMANTIC = "Near-Direct Translation / Semantic Theft"
    PARAPHRASE = "Paraphrased Semantic Theft"
    WEAK_SIMILARITY = "Weak/Questionable Similarity"
    UNRELATED = "Unrelated"


def classify_match(semantic_score: float, lexical_score: float) -> str:
    if semantic_score >= 0.88:
        if lexical_score >= 0.85:
            return MatchType.DIRECT_COPY.value
        return MatchType.STRONG_SEMANTIC.value

    # Calibrated: Sentences with low lexical overlap and >0.65 semantic match are paraphrases
    if semantic_score >= 0.65:
        if lexical_score < 0.30:
            return MatchType.PARAPHRASE.value
        return MatchType.WEAK_SIMILARITY.value

    if semantic_score >= 0.50:
        return MatchType.WEAK_SIMILARITY.value

    return MatchType.UNRELATED.value