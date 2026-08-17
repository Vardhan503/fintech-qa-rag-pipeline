"""Strategy 4: index one text line, but embed it together with its neighbours.

Each chunk still corresponds to a single line (row_index is the centre line, so
gold alignment survives), while the text handed to the embedder and the LLM also
carries the +/- window lines around it. Table rows are untouched from strategy 1
so the only variable is how much surrounding context a text line gets.
"""

from src.chunking import attach_chunk_ids
from src.chunking.row_level import linearize_table_from_clean


def sentence_window_text(doc: dict, window: int = 1) -> list[dict]:
    items = doc["pre_text"] + doc["post_text"]
    lines = [item["text"] for item in items]
    is_noise_flags = [item["is_noise"] for item in items]

    chunks = []
    for i in range(len(lines)):
        start = max(0, i - window)
        end = min(len(lines), i + window + 1)
        chunks.append({
            "chunk_type": "text_window",
            "row_index": i,
            "text": " ".join(lines[start:end]),
            "is_noise": is_noise_flags[i],
        })
    return attach_chunk_ids(doc["doc_id"], chunks)


def build_sentence_window_chunks(doc: dict, window: int = 1) -> list[dict]:
    table_chunks = attach_chunk_ids(doc["doc_id"], linearize_table_from_clean(doc["table"]))
    return table_chunks + sentence_window_text(doc, window=window)
