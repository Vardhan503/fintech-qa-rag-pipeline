"""Read the raw FinQA dev split and summarise it for EDA."""

import re

from src.utils.io import RAW_DIR, load_json

DEFAULT_RAW_PATH = RAW_DIR / "dev.json"


def load_raw_finqa(path=DEFAULT_RAW_PATH) -> list[dict]:
    """One entry per QA example. Page content (pre_text/table/post_text) is
    duplicated across every question asked about the same page."""
    return load_json(path)


def program_ops(program: str) -> list[str]:
    """Operation names in a FinQA program: divide, subtract, table_sum, ..."""
    return re.findall(r"([a-z_]+)\(", program or "")


def flatten_example(ex: dict) -> dict:
    """Flatten one raw example into a single row for tabular EDA."""
    qa = ex["qa"]
    program = qa.get("program", "")
    ops = program_ops(program)

    return {
        "id": ex["id"],
        "filename": ex.get("filename", ""),
        "n_pre_text_lines": len(ex["pre_text"]),
        "n_post_text_lines": len(ex["post_text"]),
        "n_table_rows": len(ex["table"]),
        "n_table_cols": len(ex["table"][0]) if ex["table"] else 0,
        "question": qa["question"],
        "question_word_len": len(qa["question"].split()),
        "answer": qa.get("answer", ""),
        "exe_ans": qa.get("exe_ans", None),
        "program": program,
        "n_ops": len(ops),
        "ops": ops,
        "n_gold_table_rows": len(qa.get("ann_table_rows", [])),
        "n_gold_text_rows": len(qa.get("ann_text_rows", [])),
    }


def flatten_examples(data: list[dict]) -> list[dict]:
    return [flatten_example(ex) for ex in data]
