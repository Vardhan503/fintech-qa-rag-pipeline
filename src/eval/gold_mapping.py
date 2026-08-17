"""Map FinQA's human annotations onto row-level chunk ids.

ann_table_rows and ann_text_rows are positional indices into the raw page, and
the row-level chunker preserves those indices, so a gold reference is just a
formatted chunk_id. The integrity check matters more than it looks: if cleaning
ever starts dropping lines again, `missing` goes non-zero here instead of
quietly turning the whole evaluation into noise.
"""


def get_gold_chunk_ids(ex: dict, doc: dict) -> list[str]:
    qa = ex["qa"]
    ids = [f"{doc['doc_id']}::table_row::{r}" for r in qa.get("ann_table_rows", [])]
    ids += [f"{doc['doc_id']}::text_line::{r}" for r in qa.get("ann_text_rows", [])]
    return ids


def build_eval_dataset(data: list[dict], documents: list[dict], chunks: list[dict]) -> tuple[list[dict], int]:
    """Returns (eval_examples, n_missing). n_missing must be 0."""
    doc_lookup = {d["doc_id"]: d for d in documents}
    known_chunk_ids = {c["chunk_id"] for c in chunks}

    eval_examples, missing = [], 0
    for ex in data:
        doc = doc_lookup[ex["filename"]]
        gold_ids = get_gold_chunk_ids(ex, doc)
        missing += sum(1 for g in gold_ids if g not in known_chunk_ids)
        qa = ex["qa"]
        eval_examples.append({
            "id": ex["id"],
            "doc_id": doc["doc_id"],
            "question": qa["question"],
            "answer": qa.get("answer"),
            "exe_ans": qa.get("exe_ans"),
            "program": qa.get("program"),
            "gold_chunk_ids": gold_ids,
        })
    return eval_examples, missing


def count_missing_gold_ids(eval_examples: list[dict], chunks: list[dict]) -> int:
    known_chunk_ids = {c["chunk_id"] for c in chunks}
    return sum(1 for ex in eval_examples for g in ex["gold_chunk_ids"] if g not in known_chunk_ids)
