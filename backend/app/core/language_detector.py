from typing import Optional
from langdetect import DetectorFactory, detect
from app.core.logging import logger

# Ensure deterministic language detection across runs
DetectorFactory.seed = 0


def detect_language(text: str, default: str = "en") -> str:
    """
    Detects language using a lightweight probability model.
    Falls back to `default` for short or non-linguistic inputs.
    """
    if not text or not text.strip():
        return default

    # Ignore numbers/punctuation-only strings
    clean_sample = "".join(c for c in text if c.isalpha())
    if len(clean_sample) < 3:
        return default

    try:
        lang = detect(text)
        return lang
    except Exception as exc:
        logger.debug(f"Language detection fallback for input sample: {exc}")
        return default