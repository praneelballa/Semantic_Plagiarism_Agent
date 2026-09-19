import re
from typing import Any, Dict, List
from app.core.language_detector import detect_language

# Regex splitting on '.', '!', '?', and '।' followed by whitespace or string end
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?।])(?:\s+|\n+)")


def split_into_sentences_with_spans(text: str) -> List[Dict[str, Any]]:
    """
    Splits text into sentences while tracking accurate (char_start, char_end) offsets.
    Handles standard terminal punctuation (. ! ?) and Devanagari Danda (।).
    """
    sentences = []
    if not text:
        return sentences

    last_end = 0
    for match in SENTENCE_SPLIT_REGEX.finditer(text):
        sentence_raw = text[last_end:match.start()].strip()
        if sentence_raw:
            # Locate precise start/end in the original slice
            start = text.find(sentence_raw, last_end)
            end = start + len(sentence_raw)
            sentences.append({"text": sentence_raw, "start": start, "end": end})
        last_end = match.end()

    # Capture remaining trailing sentence
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            start = text.find(remaining, last_end)
            end = start + len(remaining)
            sentences.append({"text": remaining, "start": start, "end": end})

    return sentences


def create_sliding_window_chunks(
    pages_data: List[Dict[str, Any]],
    doc_id: str,
    window_size: int = 3,
    step: int = 1,
) -> List[Dict[str, Any]]:
    """
    Creates overlapping sliding-window sentence chunks.
    Window size = 3 sentences, Step = 1 sentence.

    Produces items matching:
    {
        "chunk_id": "docA_w3_s0-2",
        "page": 1,
        "char_start": 0,
        "char_end": 284,
        "text": "...",
        "lang": "en"
    }
    """
    chunks: List[Dict[str, Any]] = []

    for page_item in pages_data:
        page_num = page_item["page"]
        raw_text = page_item["text"]

        if not raw_text.strip():
            continue

        sentences = split_into_sentences_with_spans(raw_text)
        if not sentences:
            continue

        num_sentences = len(sentences)

        # Slide over sentence indices
        for i in range(0, num_sentences, step):
            window_slice = sentences[i : i + window_size]
            if not window_slice:
                continue

            start_idx = i
            end_idx = i + len(window_slice) - 1

            chunk_text = " ".join(s["text"] for s in window_slice)
            char_start = window_slice[0]["start"]
            char_end = window_slice[-1]["end"]

            chunk_id = f"{doc_id}_p{page_num}_w{window_size}_s{start_idx}-{end_idx}"
            lang = detect_language(chunk_text)

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "char_start": char_start,
                "char_end": char_end,
                "text": chunk_text,
                "lang": lang,
            })

            # If the current window reached the end of the sentence list, break early
            if i + window_size >= num_sentences:
                break

    return chunks