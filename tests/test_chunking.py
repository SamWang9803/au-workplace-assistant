"""Tests for cleaning extracted text and chunking a page."""
from rag.chunking import chunk_page, clean_text

SIZE, OVERLAP = 1800, 200


def sample_text(paragraphs: int, sentences: int = 6, separator: str = "\n\n") -> str:
    """Build text from unique words, so every chunk can be located in it unambiguously."""
    counter = iter(range(100_000))
    paras = []
    for _ in range(paragraphs):
        sents = [" ".join(f"w{next(counter)}" for _ in range(8)) + "." for _ in range(sentences)]
        paras.append(" ".join(sents))
    return separator.join(paras)


def offsets(text: str, chunks: list[str]) -> list[int]:
    """Find where each chunk starts in the text."""
    found, pos = [], 0
    for chunk in chunks:
        pos = text.index(chunk, pos)
        found.append(pos)
    return found


def test_blank_text_gives_no_chunks():
    """Whitespace-only pages produce no chunks."""
    assert chunk_page(" \n\n ") == []


def test_short_text_is_one_chunk():
    """Text within the size limit comes back unchanged as a single chunk."""
    assert chunk_page("Annual leave is 4 weeks.") == ["Annual leave is 4 weeks."]


def test_chunks_respect_size_overlap_and_lose_no_text():
    """Chunks stay within the limit, overlap the previous chunk by at most OVERLAP characters, and cover the text."""
    text = sample_text(paragraphs=25)
    chunks = chunk_page(text, SIZE, OVERLAP)
    assert len(chunks) > 2
    assert all(len(c) <= SIZE for c in chunks)
    starts = offsets(text, chunks)
    assert starts[0] == 0
    assert starts[-1] + len(chunks[-1]) == len(text)
    for prev_start, prev, start in zip(starts, chunks, starts[1:]):
        overlap = prev_start + len(prev) - start
        assert 0 < overlap <= OVERLAP


def test_prefers_paragraph_breaks():
    """When a paragraph break is available, every chunk but the last ends on one."""
    chunks = chunk_page(sample_text(paragraphs=25), SIZE, OVERLAP)
    assert all(c.endswith("\n\n") for c in chunks[:-1])


def test_falls_back_to_sentence_ends():
    """Without paragraph breaks, every chunk but the last ends after a full stop."""
    chunks = chunk_page(sample_text(paragraphs=25, separator=" "), SIZE, OVERLAP)
    assert len(chunks) > 2
    assert all(c.endswith(". ") for c in chunks[:-1])


def test_hard_cut_without_any_break_keeps_all_text():
    """A string with no whitespace is cut at the size limit with an exact overlap, and nothing is lost."""
    text = "".join(chr(65 + (i * 7) % 26) for i in range(5000))
    chunks = chunk_page(text, SIZE, OVERLAP)
    assert all(len(c) <= SIZE for c in chunks)
    assert chunks[0] + "".join(c[OVERLAP:] for c in chunks[1:]) == text


def test_clean_text_normalises_whitespace():
    """NUL bytes go, Symbol-font bullets become •, spaces are squeezed, lines are trimmed, blank-line runs collapse."""
    assert clean_text("  Annual   leave \x00\n\n\n\n  4 weeks  \n") == "Annual leave\n\n4 weeks"
    assert clean_text(" paid leave") == "• paid leave"
