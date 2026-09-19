import re
import unicodedata

# Matches zero-width spaces, joiners, directional isolates, and other invisible characters
INVISIBLE_CHARS_REGEX = re.compile(
    r"[\u200B-\u200D\uFEFF\u00A0\u200E\u200F\u202A-\u202E\u2060-\u206F]"
)

# Standardizes whitespace without stripping sentence markers
MULTIPLE_SPACES_REGEX = re.compile(r"[^\S\r\n]+")
MULTIPLE_NEWLINES_REGEX = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Normalizes raw extracted text:
    1. Unicode decomposition/composition using NFKC.
    2. Removal of invisible/unprintable characters.
    3. Normalization of varied whitespace while preserving structural delimiters (. ! ? ।).
    """
    if not text:
        return ""

    # 1. NFKC Canonical Decomposition followed by Canonical Composition
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip non-printable and invisible control characters
    text = INVISIBLE_CHARS_REGEX.sub(" ", text)

    # 3. Canonicalize newline carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Collapse consecutive horizontal whitespace (preserving single spaces)
    text = MULTIPLE_SPACES_REGEX.sub(" ", text)

    # 5. Limit consecutive line breaks to at most two
    text = MULTIPLE_NEWLINES_REGEX.sub("\n\n", text)

    # 6. Strip leading/trailing whitespaces per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    return text