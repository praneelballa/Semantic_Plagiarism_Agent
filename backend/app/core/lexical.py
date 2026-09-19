import re
from typing import Set

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> Set[str]:
    """Tokenizes text into a set of normalized, lowercase alphanumeric tokens."""
    if not text:
        return set()
    return set(TOKEN_PATTERN.findall(text.lower()))


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Computes lexical Jaccard similarity: |A ∩ B| / |A ∪ B|.
    Returns 0.0 if either set or both sets are empty.
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)

    if not union:
        return 0.0

    return round(float(len(intersection) / len(union)), 4)