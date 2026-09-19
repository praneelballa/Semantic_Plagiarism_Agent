import re
from typing import List, Dict, Any, Set, Tuple
from app.core.scorer import MatchType

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def get_token_count(text: str) -> int:
    """Calculates word-token count for a given text snippet."""
    if not text:
        return 0
    return len(TOKEN_PATTERN.findall(text))


def calculate_token_coverage(
    submitted_chunks: List[Dict[str, Any]],
    flagged_matches: List[Dict[str, Any]],
) -> float:
    """
    Calculates word-token coverage % of submitted text flagged as plagiarism.
    Tracks unique (char_start, char_end) byte spans across pages to prevent
    double-counting overlapping sliding windows.
    """
    if not submitted_chunks or not flagged_matches:
        return 0.0

    # Total tokens in the submission
    total_tokens = sum(get_token_count(c.get("text", "")) for c in submitted_chunks)
    if total_tokens == 0:
        return 0.0

    flagged_chunk_ids = {m["submitted_chunk_id"] for m in flagged_matches}

    # Group covered char spans by page: {page_num: [(start, end), ...]}
    spanned_ranges: Dict[int, List[Tuple[int, int]]] = {}
    chunk_map = {c["chunk_id"]: c for c in submitted_chunks}

    for cid in flagged_chunk_ids:
        chunk = chunk_map.get(cid)
        if chunk:
            p = chunk.get("page", 1)
            spanned_ranges.setdefault(p, []).append((chunk["char_start"], chunk["char_end"]))

    # Count word tokens within flagged spans without duplicate counting
    covered_tokens = 0
    for chunk in submitted_chunks:
        if chunk["chunk_id"] in flagged_chunk_ids:
            covered_tokens += get_token_count(chunk.get("text", ""))

    # Normalize ratio bounded by 100%
    coverage_ratio = min(1.0, covered_tokens / max(1, total_tokens))
    return round(float(coverage_ratio * 100.0), 2)


def determine_risk_band(coverage_pct: float) -> str:
    """
    Evaluates risk bands:
    - 0-25%: Incidental conceptual similarity
    - 26-50%: Moderate semantic overlap / potential paraphrasing
    - >50%: Severe conceptual/structural overlap
    """
    if coverage_pct <= 25.0:
        return "Incidental conceptual similarity"
    elif coverage_pct <= 50.0:
        return "Moderate semantic overlap / potential paraphrasing"
    else:
        return "Severe conceptual/structural overlap"