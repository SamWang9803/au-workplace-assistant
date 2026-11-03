"""Pure text helpers: tidy text extracted from a PDF page and split it into overlapping chunks."""
import re

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s")
_WHITESPACE = re.compile(r"\s")


def clean_text(text: str) -> str:
    """Drop NUL bytes, turn Symbol-font bullets into •, squeeze spaces, trim lines and collapse blank-line runs."""
    text = text.replace("\x00", "").replace("", "•")  # U+F0B7 is a private-use bullet some PDFs use
    lines = (" ".join(line.split()) for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _last_break(pattern: re.Pattern, text: str, lo: int, hi: int) -> int | None:
    """Return the position just after the last match of the pattern inside text[lo:hi], or None."""
    end = None
    for match in pattern.finditer(text, lo, hi):
        end = match.end()
    return end


def chunk_page(text: str, size: int = 1800, overlap: int = 200) -> list[str]:
    """Split page text into chunks of at most `size` characters, each restarting up to `overlap` characters before the previous one ended."""
    if not text.strip():
        return []
    chunks, start = [], 0
    while len(text) - start > size:
        lo, hi = start + size // 2, start + size
        end = (
            _last_break(_PARAGRAPH_BREAK, text, lo, hi)
            or _last_break(_SENTENCE_END, text, lo, hi)
            or _last_break(_WHITESPACE, text, lo, hi)
            or hi
        )
        chunks.append(text[start:end])
        word = _WHITESPACE.search(text, end - overlap, end)  # begin the overlap on a word boundary
        start = word.end() if word else end - overlap
    chunks.append(text[start:])
    return chunks
