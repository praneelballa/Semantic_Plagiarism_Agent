from functools import lru_cache
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.core.logging import logger


class MultilingualEmbedder:
    """
    Multilingual semantic embedding manager.
    Encapsulates sentence-transformers with caching, normalization, and NumPy transformations.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name or settings.TEXT_EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded '{self.model_name}'. Dynamic vector dimension: {self._dimension}")

    @property
    def dimension(self) -> int:
        """Returns the dynamic vector dimension of the loaded embedding model."""
        return self._dimension

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Generates 2D NumPy embeddings for a list of strings with L2 normalization.
        Handles empty or whitespace-only inputs gracefully.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Replace empty/whitespace strings with a fallback space to keep aligned array indices
        sanitized_texts = [t if (t and t.strip()) else " " for t in texts]
        actual_batch_size = batch_size or self.batch_size

        embeddings = self.model.encode(
            sanitized_texts,
            batch_size=actual_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )

        return embeddings.astype(np.float32)

    def embed_text(
        self,
        text: str,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Generates a 1D NumPy vector for a single string."""
        if not text or not text.strip():
            return np.zeros((self.dimension,), dtype=np.float32)

        res = self.embed_texts([text], normalize_embeddings=normalize_embeddings)
        return res[0]


@lru_cache(maxsize=1)
def get_embedder() -> MultilingualEmbedder:
    """Cached singleton provider for the embedder instance."""
    return MultilingualEmbedder()


def compute_cosine_similarity(
    u: np.ndarray,
    v: np.ndarray,
) -> float:
    """
    Computes cosine similarity between two 1D vectors or between single 1D and 2D.
    Cosine similarity: (u · v) / (||u|| * ||v||)
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    return float(np.dot(u, v) / (norm_u * norm_v))


def compute_cosine_similarity_matrix(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Computes pair-wise cosine similarity between two 2D embedding matrices.
    If vectors are already L2 normalized, this is a fast matrix multiplication.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)

    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)

    norm_a = np.where(norm_a == 0, 1e-10, norm_a)
    norm_b = np.where(norm_b == 0, 1e-10, norm_b)

    a_normalized = a / norm_a
    b_normalized = b / norm_b

    return np.dot(a_normalized, b_normalized.T)