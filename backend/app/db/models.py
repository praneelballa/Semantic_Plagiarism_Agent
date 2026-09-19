import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import settings
from app.core.logging import logger


class MetadataStore:
    """SQLite metadata store mapping FAISS row IDs to chunk attributes."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or settings.SQLITE_DB_PATH)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initializes the chunk_metadata schema."""
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_metadata (
                    faiss_id INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunk_metadata(chunk_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_id ON chunk_metadata(document_id)"
            )
            conn.commit()

    def insert_chunk_metadata(self, faiss_id: int, meta: Dict[str, Any]) -> None:
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunk_metadata (
                    faiss_id, chunk_id, document_id, source_filename,
                    page, text, language, char_start, char_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    faiss_id,
                    meta["chunk_id"],
                    meta["document_id"],
                    meta["source_filename"],
                    meta["page"],
                    meta["text"],
                    meta.get("lang") or meta.get("language", "en"),
                    meta["char_start"],
                    meta["char_end"],
                ),
            )
            conn.commit()

    def insert_batch(self, start_faiss_id: int, chunks: List[Dict[str, Any]]) -> None:
        with self.get_connection() as conn:
            records = [
                (
                    start_faiss_id + i,
                    chunk["chunk_id"],
                    chunk["document_id"],
                    chunk["source_filename"],
                    chunk["page"],
                    chunk["text"],
                    chunk.get("lang") or chunk.get("language", "en"),
                    chunk["char_start"],
                    chunk["char_end"],
                )
                for i, chunk in enumerate(chunks)
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_metadata (
                    faiss_id, chunk_id, document_id, source_filename,
                    page, text, language, char_start, char_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def get_by_faiss_id(self, faiss_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM chunk_metadata WHERE faiss_id = ?", (faiss_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_faiss_ids(self, faiss_ids: List[int]) -> List[Optional[Dict[str, Any]]]:
        if not faiss_ids:
            return []
        with self.get_connection() as conn:
            placeholders = ",".join("?" for _ in faiss_ids)
            cursor = conn.execute(
                f"SELECT * FROM chunk_metadata WHERE faiss_id IN ({placeholders})",
                faiss_ids,
            )
            rows = {row["faiss_id"]: dict(row) for row in cursor.fetchall()}
            return [rows.get(fid) for fid in faiss_ids]

    def count(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunk_metadata")
            return cursor.fetchone()[0]

    def clear(self) -> None:
        with self.get_connection() as conn:
            conn.execute("DELETE FROM chunk_metadata")
            conn.commit()