"""Strategy 5: parent-child / small-to-big retrieval.

Problem it solves: row-level retrieval is precise (a single row is a tight
semantic match), but most FinQA questions need two cells from the same table, so
handing the LLM only the retrieved row leaves it short of information. Whole-table
chunks fix the context problem but dilute retrieval precision.

Solution: index and retrieve at row-level (child), then expand the context handed
to the LLM to the whole table the row came from (parent). Text lines keep their
sentence-window context.

The retrieval index is identical to the row-level baseline's, so:
  - Gold chunk_id matching in eval_dataset.jsonl still works unchanged.
  - Retrieval precision/recall numbers are directly comparable to strategy 1.
  - The only thing that changes is what goes into the LLM's prompt.
"""

from src.chunking.row_level import build_row_level_chunks, linearize_table_from_clean
from src.chunking.sentence_window import sentence_window_text
from src.chunking.whole_table import whole_table_chunk


def build_parent_text_map(documents: list[dict]) -> dict[str, str]:
    """child chunk_id -> parent context text.

    For a table_row child the parent is the whole table rendered as pipe-delimited
    rows. For a text_line child the parent is the sentence-window text (center
    line +/- 1 neighbour) -- same as strategy 4.

    Returns a flat dict so the caller just does parent_map[child_id] at query time
    without touching the document store again.
    """
    parent_map = {}

    for doc in documents:
        doc_id = doc["doc_id"]

        # --- table rows: parent = the whole table chunk text ---
        whole = whole_table_chunk(doc)
        parent_text = whole[0]["text"] if whole else ""
        for child in linearize_table_from_clean(doc["table"]):
            child_id = f"{doc_id}::table_row::{child['row_index']}"
            parent_map[child_id] = parent_text

        # --- text lines: parent = the sentence-window text for that line ---
        for window_chunk in sentence_window_text(doc, window=1):
            child_id = f"{doc_id}::text_line::{window_chunk['row_index']}"
            parent_map[child_id] = window_chunk["text"]

    return parent_map


def build_index_chunks(documents: list[dict]) -> list[dict]:
    """Row-level chunks, identical to strategy 1. These go into the vector index."""
    chunks = []
    for doc in documents:
        chunks.extend(build_row_level_chunks(doc))
    return chunks
