"""The ragged-row regression.

Some FinQA tables have multi-level headers, which makes the header row shorter
than its data rows. Zipping row values against header[j] then raised
IndexError: list index out of range. The fallback path has to survive that
without crashing and without silently discarding the row's values.
"""

import pytest

from src.chunking.row_group import WHOLE_TABLE, build_row_group_chunks, group_table_rows
from src.chunking.row_level import linearize_table_from_clean, linearize_text
from src.data.cleaning import clean_table
from src.data.reconstruction import build_documents


def test_ragged_row_does_not_crash_and_keeps_all_values():
    table = [
        ["", "2008", "2007"],                          # header shorter than the rows below
        ["net revenue", "100", "90", "80", "70"],
    ]
    chunks = linearize_table_from_clean(table)

    assert len(chunks) == 1
    text = chunks[0]["text"]
    for value in ["100", "90", "80", "70"]:
        assert value in text, f"ragged fallback dropped {value}"


def test_well_formed_row_uses_column_names():
    table = [
        ["company", "payments volume", "cards"],
        ["visa inc.", "$2,457", "1,592"],
    ]
    text = linearize_table_from_clean(table)[0]["text"]

    assert text.startswith("visa inc.:")
    assert "payments volume is $2,457" in text
    assert "cards is 1,592" in text


def test_row_index_starts_at_one_because_row_zero_is_the_header():
    table = [["h", "a"], ["first", "1"], ["second", "2"]]
    chunks = linearize_table_from_clean(table)

    assert [c["row_index"] for c in chunks] == [1, 2]
    assert chunks[0]["text"].startswith("first")


def test_empty_table_yields_no_chunks():
    assert linearize_table_from_clean([]) == []


def test_header_only_table_yields_no_chunks():
    assert linearize_table_from_clean([["h", "a"]]) == []


def test_blank_cells_are_skipped_not_rendered_as_empty_facts():
    table = [["h", "2008", "2007"], ["row", "5", "  "]]
    text = linearize_table_from_clean(table)[0]["text"]

    assert "2008 is 5" in text
    assert "2007 is" not in text


def test_text_row_index_spans_pre_then_post():
    pre = [{"text": "p0", "is_noise": False}, {"text": "p1", "is_noise": False}]
    post = [{"text": "q0", "is_noise": False}]
    chunks = linearize_text(pre, post)

    assert [c["row_index"] for c in chunks] == [0, 1, 2]
    assert chunks[2]["text"] == "q0"


def test_noise_flag_survives_linearization():
    pre = [{"text": ".", "is_noise": True}, {"text": "real", "is_noise": False}]
    chunks = linearize_text(pre, [])

    assert [c["is_noise"] for c in chunks] == [True, False]


def test_no_ragged_table_crashes_across_real_corpus(raw_data):
    documents = build_documents(raw_data)
    ragged_seen = 0

    for doc in documents:
        table = doc["table"]
        if table and any(len(row) != len(table[0]) for row in table[1:]):
            ragged_seen += 1
        try:
            linearize_table_from_clean(table)
        except IndexError as exc:                      # the exact original failure
            pytest.fail(f"{doc['doc_id']} raised IndexError: {exc}")

    assert ragged_seen > 0, "expected the corpus to contain ragged tables to exercise the fallback"


def test_row_group_of_one_matches_the_row_level_baseline():
    table = [["h", "2008"], ["a", "1"], ["b", "2"], ["c", "3"]]

    assert group_table_rows(table, 1) == [c["text"] for c in linearize_table_from_clean(table)]


def test_row_group_concatenates_n_rows_and_keeps_the_remainder():
    table = [["h", "2008"], ["a", "1"], ["b", "2"], ["c", "3"]]
    groups = group_table_rows(table, 2)

    assert len(groups) == 2                      # 3 rows at 2 per group -> 2 + 1
    assert "a:" in groups[0] and "b:" in groups[0]
    assert "c:" in groups[1]


def test_large_group_collapses_to_a_single_chunk():
    table = [["h", "2008"], ["a", "1"], ["b", "2"], ["c", "3"]]

    assert len(group_table_rows(table, WHOLE_TABLE)) == 1


def test_row_group_rejects_a_zero_group_size():
    with pytest.raises(ValueError):
        group_table_rows([["h", "a"], ["r", "1"]], 0)


def test_row_group_chunks_keep_text_lines_unchanged():
    doc = {
        "doc_id": "ACME/2020/page_1.pdf",
        "table": [["h", "2008"], ["a", "1"], ["b", "2"]],
        "pre_text": [{"text": "intro", "is_noise": False}],
        "post_text": [{"text": "outro", "is_noise": False}],
    }
    chunks = build_row_group_chunks(doc, 2)

    text_chunks = [c for c in chunks if c["chunk_type"] == "text_line"]
    assert [c["text"] for c in text_chunks] == ["intro", "outro"]
    assert all(c["chunk_id"].startswith("ACME/2020/page_1.pdf::") for c in chunks)


def test_clean_table_output_is_what_the_linearizer_consumes(raw_data):
    ex = raw_data[0]
    chunks = linearize_table_from_clean(clean_table(ex["table_ori"]))
    assert chunks and all("<" not in c["text"] for c in chunks)
