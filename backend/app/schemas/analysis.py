from typing import List, Optional
from pydantic import BaseModel, Field


class MatchDetail(BaseModel):
    submitted_chunk_id: str
    source_chunk_id: str
    submitted_text: str
    source_text: str
    submitted_page: int
    source_page: int
    submitted_lang: str
    source_lang: str
    source_document: str
    bi_encoder_score: float
    cross_encoder_score: Optional[float] = None
    final_confidence_score: float
    semantic_similarity: float  # Maintained for backward compatibility
    lexical_similarity: float
    match_type: str
    timestamp_reference: Optional[str] = None


class AnalysisResponse(BaseModel):
    analysis_id: str
    modality: str = "text"
    total_chunks: int
    matched_chunks: int
    plagiarism_coverage_pct: float
    risk_band: str
    reranker_enabled: bool = False
    matches: List[MatchDetail]