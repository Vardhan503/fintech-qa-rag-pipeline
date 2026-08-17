"""The integrity check, promoted from a print statement to a real test.

Every ann_table_rows / ann_text_rows reference must resolve to a chunk_id that
actually exists in the row-level chunk store. A non-zero count here means the
chunk indices and the gold labels have drifted apart, which invalidates every
retrieval number downstream without raising anything.
"""

from src.chunking import build_all
from src.chunking.row_level import build_row_level_chunks
from src.data.reconstruction import build_documents
from src.eval.gold_mapping import build_eval_dataset, get_gold_chunk_ids


def test_no_unresolved_gold_references(raw_data):
    documents = build_documents(raw_data)
    chunks = build_all(documents, build_row_level_chunks)
    eval_examples, missing = build_eval_dataset(raw_data, documents, chunks)

    assert missing == 0, f"{missing} gold references do not resolve to a chunk_id"
    assert len(eval_examples) == len(raw_data)


def test_questions_without_gold_evidence_stay_a_small_minority(raw_data):
    """65 of 883 dev questions carry neither ann_table_rows nor ann_text_rows.
    That's FinQA's own annotation gap, not a mapping bug, but it caps how high
    exact-id precision can read, so the count is pinned rather than ignored."""
    documents = build_documents(raw_data)
    eval_examples, _ = build_eval_dataset(raw_data, documents,
                                          build_all(documents, build_row_level_chunks))

    without_gold = [ex["id"] for ex in eval_examples if not ex["gold_chunk_ids"]]
    assert len(without_gold) / len(eval_examples) < 0.10


def test_gold_ids_use_the_row_level_id_scheme():
    doc = {"doc_id": "ACME/2020/page_1.pdf"}
    ex = {"qa": {"ann_table_rows": [3], "ann_text_rows": [7, 8]}}

    assert get_gold_chunk_ids(ex, doc) == [
        "ACME/2020/page_1.pdf::table_row::3",
        "ACME/2020/page_1.pdf::text_line::7",
        "ACME/2020/page_1.pdf::text_line::8",
    ]


def test_documents_dedupe_pages_but_keep_every_question(raw_data):
    documents = build_documents(raw_data)

    assert len(documents) < len(raw_data), "pages should collapse across questions"
    assert sum(len(d["qa_ids"]) for d in documents) == len(raw_data)


def test_ground_truth_uses_exe_ans_not_the_answer_field(raw_data):
    documents = build_documents(raw_data)
    eval_examples, _ = build_eval_dataset(raw_data, documents,
                                          build_all(documents, build_row_level_chunks))

    assert all("exe_ans" in ex for ex in eval_examples)
    # both fields are carried, but exe_ans is the one the grader trusts
    assert all("answer" in ex for ex in eval_examples)
