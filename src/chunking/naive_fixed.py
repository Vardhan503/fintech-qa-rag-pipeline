"""Strategy 2 (control group): fixed-size character windows over flat text.

Deliberately structure-blind -- it concatenates pre_text, the table, and
post_text into one blob and slices it, the way a RecursiveCharacterTextSplitter
would if pointed at a financial filing. This is the baseline the
structure-aware strategies have to beat to justify their complexity.
"""

from src.chunking import attach_chunk_ids


def flatten_document(doc: dict) -> str:
    pre = " ".join(item["text"] for item in doc["pre_text"])
    table_flat = " ".join(" ".join(row) for row in doc["table"])
    post = " ".join(item["text"] for item in doc["post_text"])
    return f"{pre} {table_flat} {post}"


def naive_fixed_size_chunks(doc: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    full_text = flatten_document(doc)

    chunks = []
    start = 0
    idx = 0
    while start < len(full_text):
        chunks.append({
            "chunk_type": "fixed_size",
            "row_index": idx,
            "text": full_text[start:start + chunk_size],
            "is_noise": False,
        })
        start += chunk_size - overlap
        idx += 1
    return attach_chunk_ids(doc["doc_id"], chunks)
