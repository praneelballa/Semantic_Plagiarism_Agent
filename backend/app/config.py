import os
from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Multimodal & Cross-Lingual Semantic Plagiarism Detection Agent"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    CORPUS_DIR: Path = DATA_DIR / "corpus"
    INDEXES_DIR: Path = DATA_DIR / "indexes"

    # Persistence Paths
    SQLITE_DB_PATH: Path = INDEXES_DIR / "metadata.db"
    FAISS_INDEX_PATH: Path = INDEXES_DIR / "corpus_flat_ip.index"

    # Security & File Limits
    MAX_UPLOAD_SIZE_MB: int = 25

    # Models Selection
    TEXT_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CODE_EMBEDDING_MODEL: str = "microsoft/unixcoder-base"
    WHISPER_MODEL: str = "tiny"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Hardware & Batching
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"

    # Retrieval & Reranker Configuration
    DEFAULT_TOP_K: int = 5
    USE_RERANKER: bool = False

    # Detection Thresholds
    THRESHOLD_DIRECT_COPY: float = 0.88
    THRESHOLD_PARAPHRASE: float = 0.65
    THRESHOLD_WEAK_SIMILARITY: float = 0.50
    THRESHOLD_LEXICAL_DIVERGENCE: float = 0.35

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

for folder in [settings.DATA_DIR, settings.UPLOADS_DIR, settings.CORPUS_DIR, settings.INDEXES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)