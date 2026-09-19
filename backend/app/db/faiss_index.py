from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from app.config import settings
from app.core.logging import logger
from app.core.embedder import get_embedder
from app.db.models import MetadataStore


class FaissIndexManager:
    """
    Coordinates FAISS IndexFlatIP (Inner Product) vector searches
    with SQLite chunk metadata tracking.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        index_path: Optional[Path] = None,
        metadata_store: Optional[MetadataStore] = None,
    ):
        # Dynamically discover dimension from embedder if not explicitly specified
        self.dimension = dimension or get_embedder().dimension
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.metadata_store = metadata_store or MetadataStore()
        self.index: faiss.IndexFlatIP = self._initialize_index()

    def _initialize_index(self) -> faiss.IndexFlatIP:
        """Loads index from disk if present; otherwise creates a new IndexFlatIP."""
        if self.index_path.exists():
            try:
                logger.info(f"Loading persistent FAISS index from {self.index_path}")
                index = faiss.read_index(str(self.index_path))
                if index.d != self.dimension:
                    raise ValueError(
                        f"Dimension mismatch: Index has {index.d}, expected {self.dimension}"
                    )
                return index
            except Exception as e:
                logger.error(f"Error reading index file: {e}. Instantiating fresh index.")

        logger.info(f"Creating new FAISS IndexFlatIP with dimension {self.dimension}")
        return faiss.IndexFlatIP(self.dimension)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def add_vectors(
        self, embeddings: np.ndarray, chunk_metadata: List[Dict[str, Any]]
    ) -> int:
        """
        Validates dimensions, adds L2-normalized embeddings into FAISS,
        and saves chunk metadata into SQLite.
        """
        if embeddings.size == 0 or len(chunk_metadata) == 0:
            return 0

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}"
            )

        if embeddings.shape[0] != len(chunk_metadata):
            raise ValueError("Number of embeddings does not match chunk metadata length")

        start_id = self.index.ntotal
        self.index.add(embeddings)
        self.metadata_store.insert_batch(start_id, chunk_metadata)

        logger.info(f"Added {len(chunk_metadata)} vectors. New index total: {self.index.ntotal}")
        return len(chunk_metadata)

    def search(
        self, query_vector: np.ndarray, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches top-K nearest candidates by inner product (exact cosine similarity).
        Returns matching SQLite metadata alongside similarity scores.
        """
        # 1. Format and validate dimension FIRST, regardless of index size
        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match index dimension {self.dimension}"
            )

        # 2. Check for empty index after validation
        if self.index.ntotal == 0:
            return []

        k = top_k or settings.DEFAULT_TOP_K
        k = min(k, self.index.ntotal)

        # L2-normalize query vector for exact cosine equivalence
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-10, norm)
        normalized_query = query / norm

        distances, indices = self.index.search(normalized_query, k)

        top_indices = indices[0].tolist()
        top_scores = distances[0].tolist()

        metadata_records = self.metadata_store.get_by_faiss_ids(top_indices)

        results = []
        for faiss_id, score, meta in zip(top_indices, top_scores, metadata_records):
            if faiss_id != -1 and meta is not None:
                results.append({
                    "faiss_id": faiss_id,
                    "similarity_score": round(float(score), 4),
                    "chunk_id": meta["chunk_id"],
                    "document_id": meta["document_id"],
                    "source_filename": meta["source_filename"],
                    "page": meta["page"],
                    "text": meta["text"],
                    "language": meta["language"],
                    "char_start": meta["char_start"],
                    "char_end": meta["char_end"],
                })

        return results

    def save(self, path: Optional[Path] = None) -> None:
        """Persists FAISS index to disk."""
        target_path = Path(path or self.index_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target_path))
        logger.info(f"Persisted FAISS index to {target_path}")

    def reset(self) -> None:
        """Clears FAISS in-memory index, SQLite store, and disk files."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store.clear()
        if self.index_path.exists():
            self.index_path.unlink()
        logger.info("FAISS index and SQLite metadata store reset.")


_index_manager_instance: Optional[FaissIndexManager] = None


def get_index_manager() -> FaissIndexManager:
    """Singleton accessor for FaissIndexManager."""
    global _index_manager_instance
    if _index_manager_instance is None:
        _index_manager_instance = FaissIndexManager()
    return _index_manager_instance