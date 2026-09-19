import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel

from app.core.document_loader import DocumentLoader
from app.core.chunker import create_sliding_window_chunks
from app.core.embedder import get_embedder
from app.db.faiss_index import get_index_manager
from app.core.logging import logger

router = APIRouter()


class IngestResponse(BaseModel):
    documents_added: int
    chunks_added: int
    index_size: int


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest reference corpus documents",
)
async def ingest_corpus(files: List[UploadFile] = File(...)):
    """
    Ingests multiple reference corpus files (.txt, .md, .docx, .pdf):
    1. Extracts raw text and page positions.
    2. Normalizes text via NFKC and character cleanups.
    3. Performs sliding-window chunking (W=3, S=1) with language tagging.
    4. Computes normalized dense multilingual embeddings.
    5. Stores vectors into FAISS and metadata into SQLite.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for ingestion.",
        )

    embedder = get_embedder()
    index_mgr = get_index_manager()

    docs_count = 0
    all_chunks_to_index = []

    for file in files:
        filename = file.filename or "unknown_doc"
        contents = await file.read()

        if not contents:
            continue

        try:
            pages_data = DocumentLoader.load_document(contents, filename)
        except ValueError as exc:
            logger.warning(f"Skipping {filename}: {exc}")
            continue
        except Exception as exc:
            logger.error(f"Error reading {filename}: {exc}")
            continue

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        chunks = create_sliding_window_chunks(pages_data, doc_id=doc_id)

        for chunk in chunks:
            chunk["document_id"] = doc_id
            chunk["source_filename"] = filename
            all_chunks_to_index.append(chunk)

        docs_count += 1

    if not all_chunks_to_index:
        return IngestResponse(
            documents_added=docs_count,
            chunks_added=0,
            index_size=index_mgr.total_vectors,
        )

    # Generate embeddings in batch
    texts = [c["text"] for c in all_chunks_to_index]
    embeddings = embedder.embed_texts(texts, normalize_embeddings=True)

    # Insert into FAISS and SQLite
    chunks_added = index_mgr.add_vectors(embeddings, all_chunks_to_index)
    index_mgr.save()

    return IngestResponse(
        documents_added=docs_count,
        chunks_added=chunks_added,
        index_size=index_mgr.total_vectors,
    )
