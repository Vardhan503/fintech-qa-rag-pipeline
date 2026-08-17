"""Turn question-centric raw examples into page-centric documents.

dev.json stores one entry per question, so the 883 dev questions carry only 299
distinct pages between them, each page repeated once per question asked about it.
Embedding that as-is would put the same text in the index several times over, so
examples are grouped by filename and the page is kept once with a qa_ids backlink.
"""

from collections import defaultdict

from src.data.cleaning import clean_table, index_preserving_clean


def group_by_filename(data: list[dict]) -> dict[str, list[dict]]:
    by_file = defaultdict(list)
    for ex in data:
        by_file[ex["filename"]].append(ex)
    return dict(by_file)


def build_document(filename: str, examples: list[dict]) -> dict:
    """One unified document dict. Page content is identical across `examples`,
    so the first one supplies the text and the rest only contribute their ids."""
    ex = examples[0]
    return {
        "doc_id": filename,
        "pre_text": index_preserving_clean(ex["pre_text"]),
        "post_text": index_preserving_clean(ex["post_text"]),
        "table": clean_table(ex["table_ori"]),
        "qa_ids": [e["id"] for e in examples],
    }


def build_documents(data: list[dict]) -> list[dict]:
    return [build_document(fname, exs) for fname, exs in group_by_filename(data).items()]
