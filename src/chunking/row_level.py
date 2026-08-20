"""One chunk per table row and per text line.

row_index matches FinQA's own indexing: table rows count from 1 because row 0
is the header, and text lines are indexed across concatenated pre_text + post_text
exactly as ann_text_rows does. That alignment is what lets gold_mapping build
eval_dataset.jsonl by exact chunk_id.
"""

from src.chunking import attach_chunk_ids


def linearize_table_from_clean(table: list[list[str]]) -> list[dict]:
    """'Visa Inc.(1): Payments Volume is $2,457; Cards is 1,592.'

    Ragged rows (a row longer than the header, which happens with multi-level
    headers) can't be zipped against column names, so they fall back to a
    positional sentence instead of raising IndexError or dropping the values.
    """
    if not table:
        return []

    header, *rows = table
    out = []
    for row_idx, row in enumerate(rows, start=1):
        row_label = row[0]
        if len(row) == len(header):
            facts = [f"{header[j]} is {row[j]}" for j in range(1, len(row)) if row[j].strip()]
            sentence = f"{row_label}: " + "; ".join(facts) + "."
        else:
            values = [v for v in row[1:] if v.strip()]
            sentence = f"{row_label}: values are " + ", ".join(values) + "."
        out.append({"chunk_type": "table_row", "row_index": row_idx, "text": sentence, "is_noise": False})
    return out


def linearize_text(pre: list[dict], post: list[dict]) -> list[dict]:
    """pre and post are index_preserving_clean output, so enumerate() indices
    line up with ann_text_rows, which counts across pre_text then post_text."""
    combined = pre + post
    return [
        {"chunk_type": "text_line", "row_index": i, "text": item["text"], "is_noise": item["is_noise"]}
        for i, item in enumerate(combined)
    ]


def build_row_level_chunks(doc: dict) -> list[dict]:
    chunks = linearize_table_from_clean(doc["table"]) + linearize_text(doc["pre_text"], doc["post_text"])
    return attach_chunk_ids(doc["doc_id"], chunks)
