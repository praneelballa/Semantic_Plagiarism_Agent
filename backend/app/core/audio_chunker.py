import re
from typing import List, Dict, Any
from app.core.language_detector import detect_language
from app.core.chunker import SENTENCE_SPLIT_REGEX


def create_audio_sliding_window_chunks(
    segments: List[Dict[str, Any]],
    doc_id: str,
    window_size: int = 3,
    step: int = 1,
) -> List[Dict[str, Any]]:
    """
    Transforms transcript segments into 3-sentence sliding windows with
    timestamp references (e.g., '14.2s - 28.1s').
    """
    # 1. Flatten transcript segments into discrete timestamped sentences
    atomic_sentences: List[Dict[str, Any]] = []

    for seg in segments:
        text = seg["text"]
        start_t = seg["start"]
        end_t = seg["end"]

        last_end = 0
        matches = list(SENTENCE_SPLIT_REGEX.finditer(text))

        if not matches:
            if text.strip():
                atomic_sentences.append({"text": text.strip(), "start": start_t, "end": end_t})
        else:
            total_matches = len(matches)
            for idx, m in enumerate(matches):
                sentence = text[last_end:m.start()].strip()
                if sentence:
                    # Estimate proportional timestamps across segmented sentences within the segment
                    seg_duration = max(0.1, end_t - start_t)
                    s_ratio = last_end / max(1, len(text))
                    e_ratio = m.start() / max(1, len(text))
                    atomic_sentences.append({
                        "text": sentence,
                        "start": round(start_t + (s_ratio * seg_duration), 2),
                        "end": round(start_t + (e_ratio * seg_duration), 2),
                    })
                last_end = m.end()

            if last_end < len(text):
                remaining = text[last_end:].strip()
                if remaining:
                    atomic_sentences.append({
                        "text": remaining,
                        "start": round(start_t + ((last_end / len(text)) * (end_t - start_t)), 2),
                        "end": end_t,
                    })

    if not atomic_sentences:
        return []

    # 2. Build 3-sentence sliding windows (W=3, S=1)
    chunks: List[Dict[str, Any]] = []
    num_sents = len(atomic_sentences)

    for i in range(0, num_sents, step):
        window = atomic_sentences[i : i + window_size]
        if not window:
            continue

        chunk_text = " ".join(s["text"] for s in window)
        start_time = window[0]["start"]
        end_time = window[-1]["end"]
        timestamp_str = f"{start_time:.1f}s - {end_time:.1f}s"

        chunk_id = f"{doc_id}_aud_w{window_size}_s{i}-{i + len(window) - 1}"
        lang = detect_language(chunk_text)

        chunks.append({
            "chunk_id": chunk_id,
            "page": 1,
            "char_start": 0,
            "char_end": len(chunk_text),
            "text": chunk_text,
            "lang": lang,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp_reference": timestamp_str,
        })

        if i + window_size >= num_sents:
            break

    return chunks