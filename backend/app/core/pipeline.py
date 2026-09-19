import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

from app.config import settings
from app.core.document_loader import DocumentLoader
from app.core.chunker import create_sliding_window_chunks
from app.core.audio_chunker import create_audio_sliding_window_chunks
from app.core.whisper_transcriber import get_transcriber
from app.core.embedder import get_embedder
from app.core.reranker import get_reranker, calculate_final_confidence
from app.core.lexical import compute_jaccard_similarity
from app.core.scorer import classify_match, MatchType
from app.core.coverage import calculate_token_coverage, determine_risk_band
from app.db.faiss_index import get_index_manager
from app.core.logging import logger


class AnalysisPipeline:
    """Multimodal plagiarism pipeline with two-stage verification."""

    @classmethod
    def analyze_document(
        cls,
        file_bytes: bytes,
        filename: str,
        modality: str = "text",
        top_k: int = 5,
        use_reranker: Optional[bool] = None,
    ) -> Dict[str, Any]:
        analysis_id = f"job_{uuid.uuid4().hex[:8]}"
        suffix = Path(filename).suffix.lower()

        reranker_enabled = settings.USE_RERANKER if use_reranker is None else use_reranker

        if suffix in [".wav", ".mp3"] or modality == "audio":
            modality = "audio"

        submitted_chunks: List[Dict[str, Any]] = []

        # 1. Extraction & Chunking
        if modality == "audio":
            transcriber = get_transcriber()
            segments = transcriber.transcribe(file_bytes, file_suffix=suffix or ".wav")
            submitted_chunks = create_audio_sliding_window_chunks(segments, doc_id="sub_audio")
        else:
            pages_data = DocumentLoader.load_document(file_bytes, filename)
            if pages_data:
                submitted_chunks = create_sliding_window_chunks(pages_data, doc_id="sub")

        total_chunks = len(submitted_chunks)
        if total_chunks == 0:
            return {
                "analysis_id": analysis_id,
                "modality": modality,
                "total_chunks": 0,
                "matched_chunks": 0,
                "plagiarism_coverage_pct": 0.0,
                "risk_band": determine_risk_band(0.0),
                "reranker_enabled": reranker_enabled,
                "matches": [],
            }

        # 2. Stage 1: Bi-Encoder Embeddings
        embedder = get_embedder()
        submitted_texts = [c["text"] for c in submitted_chunks]
        submitted_embeddings = embedder.embed_texts(submitted_texts, normalize_embeddings=True)

        # 3. Stage 1: Vector Retrieval via FAISS
        index_mgr = get_index_manager()
        matches: List[Dict[str, Any]] = []

        if index_mgr.total_vectors > 0:
            # Prepare reranker if enabled
            reranker = get_reranker() if reranker_enabled else None

            for chunk, query_vec in zip(submitted_chunks, submitted_embeddings):
                candidates = index_mgr.search(query_vec, top_k=top_k)
                if not candidates:
                    continue

                best_candidate = candidates[0]
                bi_score = float(best_candidate["similarity_score"])
                source_text = best_candidate["text"]
                sub_text = chunk["text"]
                sub_lang = chunk.get("lang", "en")
                src_lang = best_candidate.get("language", "en")

                # Stage 2: Cross-Encoder Verification (Optional)
                ce_score = None
                if reranker_enabled and reranker is not None:
                    try:
                        ce_score = reranker.predict_pair(source_text, sub_text)
                    except Exception as e:
                        logger.warning(f"Cross-encoder evaluation failed: {e}")
                        ce_score = None

                final_confidence = calculate_final_confidence(
                    bi_encoder_score=bi_score,
                    cross_encoder_score=ce_score,
                    submitted_lang=sub_lang,
                    source_lang=src_lang,
                )

                lexical_score = compute_jaccard_similarity(sub_text, source_text)
                match_category = classify_match(final_confidence, lexical_score)

                # Flag condition based on calibrated final confidence score
                if match_category != MatchType.UNRELATED.value and final_confidence >= 0.65:
                    final_match_type = "Cross-Modal Audio" if modality == "audio" else match_category

                    matches.append({
                        "submitted_chunk_id": chunk["chunk_id"],
                        "source_chunk_id": best_candidate["chunk_id"],
                        "submitted_text": sub_text,
                        "source_text": source_text,
                        "submitted_page": chunk.get("page", 1),
                        "source_page": best_candidate["page"],
                        "submitted_lang": sub_lang,
                        "source_lang": src_lang,
                        "source_document": best_candidate["source_filename"],
                        "bi_encoder_score": round(bi_score, 4),
                        "cross_encoder_score": ce_score,
                        "final_confidence_score": round(final_confidence, 4),
                        "semantic_similarity": round(final_confidence, 4),
                        "lexical_similarity": round(lexical_score, 4),
                        "match_type": final_match_type,
                        "timestamp_reference": chunk.get("timestamp_reference"),
                    })

        # 4. Token Coverage & Risk Calculation
        coverage_pct = calculate_token_coverage(submitted_chunks, matches)
        risk_band = determine_risk_band(coverage_pct)

        return {
            "analysis_id": analysis_id,
            "modality": modality,
            "total_chunks": total_chunks,
            "matched_chunks": len(matches),
            "plagiarism_coverage_pct": coverage_pct,
            "risk_band": risk_band,
            "reranker_enabled": reranker_enabled,
            "matches": matches,
        }