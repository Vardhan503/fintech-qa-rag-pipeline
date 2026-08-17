"""Table chunking at a tunable granularity: N table rows per chunk.

Strategies 1 and 3 are the two ends of one dial -- one row per chunk versus the
whole table per chunk -- and they disagree about which is better. This module
makes the dial itself the parameter so the middle can be measured, with
rows_per_group=1 reproducing row-level table chunks and a large value
reproducing whole-table chunks.

Row sentences come from row_level.linearize_table_from_clean, so grouping never
diverges from the baseline's wording or its ragged-row fallback.
"""

from src.chunking import attach_chunk_ids
from src.chunking.row_level import linearize_table_from_clean, linearize_text

WHOLE_TABLE = 999   # any value >= max rows per table collapses to one chunk


def group_table_rows(table: list[list[str]], rows_per_group: int) -> list[str]:
    """Linearized row sentences, concatenated rows_per_group at a time."""
    if rows_per_group < 1:
        raise ValueError("rows_per_group must be >= 1")

    sentences = [c["text"] for c in linearize_table_from_clean(table)]
    return [
        " ".join(sentences[start:start + rows_per_group])
        for start in range(0, len(sentences), rows_per_group)
    ]


def build_row_group_chunks(doc: dict, rows_per_group: int) -> list[dict]:
    """Grouped table chunks plus unchanged text-line chunks, so the only variable
    against the baseline is table granularity."""
    table_chunks = [
        {"chunk_type": "row_group", "row_index": i, "text": text, "is_noise": False}
        for i, text in enumerate(group_table_rows(doc["table"], rows_per_group))
    ]
    chunks = table_chunks + linearize_text(doc["pre_text"], doc["post_text"])
    return attach_chunk_ids(doc["doc_id"], chunks)
