"""Chunking strategies, one module per strategy.

Every strategy returns chunk dicts with the same keys, so the eval harness can
consume any of them interchangeably:

    {chunk_type, row_index, text, chunk_id, doc_id, is_noise}

chunk_id is "{doc_id}::{chunk_type}::{row_index}". For the row-level strategy
row_index is deliberately the same index FinQA's ann_table_rows/ann_text_rows
use, which is what makes exact-id gold matching possible at all.
"""


def attach_chunk_ids(doc_id: str, chunks: list[dict]) -> list[dict]:
    """Stamp doc_id and the derived chunk_id onto in-place-built chunks."""
    for chunk in chunks:
        chunk["doc_id"] = doc_id
        chunk["chunk_id"] = f"{doc_id}::{chunk['chunk_type']}::{chunk['row_index']}"
    return chunks


def build_all(documents: list[dict], builder) -> list[dict]:
    """Apply a per-document builder across the corpus."""
    chunks = []
    for doc in documents:
        chunks.extend(builder(doc))
    return chunks
