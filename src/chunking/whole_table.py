"""Strategy 3: the whole table as a single chunk, text still line-level.

Best end-to-end answer accuracy (45.5% vs 40.9% for row-level). Text chunking
is left identical to strategy 1 so the only variable is table granularity.
Most FinQA questions need two cells from the same table; a whole-table chunk
delivers both in one hit — this is the production RAG index.
"""

from src.chunking import attach_chunk_ids
from src.chunking.row_level import linearize_text


def whole_table_chunk(doc: dict) -> list[dict]:
    """Pipe-delimited rows, header first. Empty tables yield no chunk."""
    table = doc["table"]
    if not table:
        return []

    header, *rows = table
    lines = [" | ".join(header)] + [" | ".join(row) for row in rows]
    chunks = [{
        "chunk_type": "whole_table",
        "row_index": 0,
        "text": "\n".join(lines),
        "is_noise": False,
    }]
    return attach_chunk_ids(doc["doc_id"], chunks)


def build_whole_table_chunks(doc: dict) -> list[dict]:
    text_chunks = attach_chunk_ids(doc["doc_id"], linearize_text(doc["pre_text"], doc["post_text"]))
    return whole_table_chunk(doc) + text_chunks
