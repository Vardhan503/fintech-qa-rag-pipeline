"""The grading regressions.

Two distinct bugs are pinned here:

  1. exe_ans is a float for numeric questions but "yes"/"no" for boolean ones.
     Comparing them with abs(gold - pred) raised
     TypeError: unsupported operand type(s) for -: 'float' and 'str'.
  2. Percent scale is ambiguous: exe_ans stores 0.84882 while the model answers
     "85%". A literal tolerance check scores those as wrong.
"""

import pytest

from src.eval.generation_harness import answers_match, extract_final_answer, parse_number


@pytest.mark.parametrize("gold,pred", [
    ("yes", "Yes, it increased."),
    ("no", "No."),
    ("yes", "yes"),
])
def test_boolean_gold_does_not_raise_and_matches(gold, pred):
    assert answers_match(gold, pred) is True


@pytest.mark.parametrize("gold,pred", [
    ("yes", "no"),
    ("no", "Yes, revenue grew."),
])
def test_boolean_gold_rejects_the_wrong_polarity(gold, pred):
    assert answers_match(gold, pred) is False


def test_percent_scaling_is_accepted_in_either_form():
    assert answers_match(0.84882, "85%")
    assert answers_match(0.84882, "0.849")
    assert answers_match(0.14336, "14.3%")


def test_currency_and_thousands_separators():
    assert answers_match(127.4, "$127.40")
    assert answers_match(2560.0, "2,560")


def test_sign_difference_still_matches_on_magnitude():
    assert answers_match(-0.42327, "42.32%")


def test_unparseable_prediction_is_wrong_not_an_error():
    assert answers_match(127.4, "I don't know") is False
    assert answers_match(127.4, "") is False
    assert answers_match(127.4, None) is False


def test_zero_gold_is_handled():
    assert answers_match(0.0, "0")
    assert not answers_match(0.0, "5")


def test_clearly_wrong_number_is_rejected():
    assert answers_match(0.84882, "57%") is False


def test_parse_number_takes_the_last_number_after_the_reasoning():
    assert parse_number("46.6 / 54.9 = 0.849") == pytest.approx(0.849)
    assert parse_number("14.3%") == pytest.approx(0.143)
    assert parse_number("no numbers here") is None


def test_extract_final_answer_prefers_the_answer_line():
    raw = "46.6 / 54.9 = 0.849\nANSWER: 84.9%"
    assert extract_final_answer(raw) == "84.9%"
    assert extract_final_answer("just 42") == "just 42"
    assert extract_final_answer("") == ""


def test_the_noisy_answer_field_is_why_exe_ans_is_ground_truth():
    """dev.json row with exe_ans=0.015 carries answer='1%'. Grading against the
    typed `answer` field would call a correct 1.5% response wrong."""
    assert answers_match(0.015, "1.5%")
    assert not answers_match(0.015, "1%")
