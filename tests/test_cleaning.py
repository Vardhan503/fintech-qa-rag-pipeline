"""The regression that motivated index_preserving_clean.

An earlier clean_text_lines() dropped lone-punctuation lines. That shifted every
subsequent line up by one, so FinQA's ann_text_rows indices pointed at the wrong
sentence and the gold labels became silently wrong. These tests fail if anything
ever starts dropping, merging, or reordering lines again.
"""

from src.data.cleaning import clean_line, clean_table, index_preserving_clean, is_noise_line, strip_html


def test_length_and_order_preserved_with_noise_lines():
    lines = ["first line .", ".", "second line", "", "third line", "-"]
    cleaned = index_preserving_clean(lines)

    assert len(cleaned) == len(lines)
    assert cleaned[0]["text"] == "first line."
    assert cleaned[2]["text"] == "second line"
    assert cleaned[4]["text"] == "third line"


def test_noise_lines_are_flagged_not_removed():
    cleaned = index_preserving_clean(["real content here", ".", "more content"])

    assert [c["is_noise"] for c in cleaned] == [False, True, False]
    assert len(cleaned) == 3


def test_duplicate_lines_are_not_deduped():
    lines = ["same line", "same line", "same line"]
    assert len(index_preserving_clean(lines)) == 3


def test_gold_index_still_points_at_the_same_line():
    lines = ["intro .", ".", "the answer lives here", "outro"]
    gold_index = 2   # what ann_text_rows would carry

    cleaned = index_preserving_clean(lines)
    assert "answer lives here" in cleaned[gold_index]["text"]


def test_length_preserved_across_real_corpus(raw_data):
    for ex in raw_data[:200]:
        assert len(index_preserving_clean(ex["pre_text"])) == len(ex["pre_text"])
        assert len(index_preserving_clean(ex["post_text"])) == len(ex["post_text"])


def test_clean_line_fixes_tokenizer_spacing():
    assert clean_line("( 1 ) footnote") == "(1) footnote"
    assert clean_line("revenue of $ 2457 million") == "revenue of $2457 million"
    assert clean_line("margin of 12 %") == "margin of 12%"
    assert clean_line("visa inc .") == "visa inc."
    assert clean_line("too    many     spaces") == "too many spaces"


def test_known_gap_parens_around_words_stay_spaced():
    """clean_line only tightens parentheses around digits, so '( billions )'
    survives on ~2,100 corpus lines. Pinned deliberately: changing it rewrites
    documents.jsonl and every chunk store derived from it."""
    assert clean_line("cards ( billions )") == "cards ( billions )"


def test_is_noise_line():
    assert all(is_noise_line(x) for x in [".", ",", ";", ":", "-", "", "   "])
    assert not is_noise_line("actual sentence.")
    assert not is_noise_line("2007")


def test_strip_html_and_clean_table():
    assert strip_html("<td>total revenue</td>") == "total revenue"
    assert clean_table([["<td>a</td>", "<b>1</b>"]]) == [["a", "1"]]
