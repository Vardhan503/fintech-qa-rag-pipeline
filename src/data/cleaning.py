"""Text and table cleaning for FinQA pages.

The important function here is index_preserving_clean. FinQA's ground-truth
labels (ann_text_rows) are positional indices into the raw pre_text + post_text
lists, so any cleaning step that drops, merges, or reorders lines silently
repoints every gold label at the wrong sentence. Cleaning therefore has to be
one-line-in, one-line-out, with unusable lines flagged rather than removed.
"""

import re

NOISE_LINES = {".", ",", ";", ":", "-", ""}


def clean_line(line: str) -> str:
    """Undo FinQA's tokenizer spacing: '$ 12' -> '$12', '( 1 )' -> '(1)', 'inc .' -> 'inc.'"""
    line = re.sub(r"\$\s+(\d)", r"$\1", line)
    line = re.sub(r"\(\s+(\d)", r"(\1", line)
    line = re.sub(r"(\d)\s+\)", r"\1)", line)
    line = re.sub(r"\s+([.,;:%])", r"\1", line)
    line = re.sub(r"\s{2,}", " ", line).strip()
    return line


def is_noise_line(line: str) -> bool:
    """Lone punctuation and blanks: kept for index alignment, excluded from the index."""
    return line.strip() in NOISE_LINES


def index_preserving_clean(lines: list[str]) -> list[dict]:
    """Same length and order as the input, so positional gold labels stay valid.

    Returns [{"text": cleaned, "is_noise": bool}, ...]. Never drops or dedupes.
    """
    return [{"text": clean_line(line), "is_noise": is_noise_line(line)} for line in lines]


def strip_html(cell: str) -> str:
    """table_ori cells carry markup like <td>; table cells do not."""
    cell = re.sub(r"<[^>]+>", "", cell)
    return re.sub(r"\s{2,}", " ", cell).strip()


def clean_table(table_ori: list[list[str]]) -> list[list[str]]:
    return [[strip_html(cell) for cell in row] for row in table_ori]
