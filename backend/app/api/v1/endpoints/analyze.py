from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.schemas.analysis import AnalysisResponse
from app.core.pipeline import AnalysisPipeline
from app.core.code_engine import compare_python_code
from app.core.logging import logger

router = APIRouter()

# In-memory storage cache for production-style pollable analysis results
ANALYSIS_RESULTS_STORE: Dict[str, Dict[str, Any]] = {}


@router.post(
    "",
    status_code=status.HTTP_200_OK,
    summary="Analyze submitted document, audio, or source code for semantic plagiarism",
)
async def analyze_document(
    file: UploadFile = File(..., description="File (.pdf, .docx, .txt, .wav, .mp3, .py)"),
    type: str = Form("text", description="Modality type ('text', 'audio', 'code')"),
    compare_corpus: bool = Form(True, description="Whether to compare against indexed FAISS corpus"),
    source_code_reference: str = Form(None, description="Reference code string when type='code'"),
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    suffix = Path(file.filename).suffix.lower()

    if type == "code" or suffix == ".py":
        try:
            contents = await file.read()
            suspect_code = contents.decode("utf-8", errors="replace")
            reference_code = source_code_reference or ""

            result = compare_python_code(suspect_code, reference_code)
            response_payload = {
                "analysis_id": f"code_{Path(file.filename).stem}",
                "modality": "code",
                "result": result,
            }
            ANALYSIS_RESULTS_STORE[response_payload["analysis_id"]] = response_payload
            return response_payload
        except Exception as exc:
            logger.error(f"Code analysis failure: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Code pipeline error: {str(exc)}",
            )

    allowed_audio = [".wav", ".mp3"]
    allowed_text = [".txt", ".md", ".pdf", ".docx"]

    if suffix in allowed_audio or type == "audio":
        effective_modality = "audio"
    elif suffix in allowed_text or type == "text":
        effective_modality = "text"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'.",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        report = AnalysisPipeline.analyze_document(
            file_bytes=contents,
            filename=file.filename,
            modality=effective_modality,
        )
        # Store for GET queries
        ANALYSIS_RESULTS_STORE[report["analysis_id"]] = report
        return report
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get(
    "/{analysis_id}",
    status_code=status.HTTP_200_OK,
    summary="Retrieve completed analysis report by ID",
)
async def get_analysis_by_id(analysis_id: str):
    """Retrieves an existing analysis report by its unique job identifier."""
    if analysis_id not in ANALYSIS_RESULTS_STORE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis report '{analysis_id}' not found.",
        )
    return ANALYSIS_RESULTS_STORE[analysis_id]