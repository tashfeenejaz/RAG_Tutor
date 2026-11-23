import math

from eval_retrieval import is_relevant, ndcg_at_k, reciprocal_rank, norm


def test_is_relevant_ignores_case_and_whitespace():
    assert is_relevant("The  Gradient\nis the MEASURE of slope", ["gradient is the measure of slope"])
    assert not is_relevant("something else", ["gradient"])
    assert norm("Newton’s law") == "newton's law"


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True]) == 1 / 3
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg():
    assert ndcg_at_k([True, False, False], 5) == 1.0
    assert math.isclose(ndcg_at_k([False, True, False], 5), 1 / math.log2(3))
    assert ndcg_at_k([False] * 5, 5) == 0.0


def test_parse_judge_json_tolerates_fences_and_noise():
    from eval_generation import parse_judge_json

    assert parse_judge_json('```json\n{"faithful": true, "answers_question": false}\n```') == {
        "faithful": True, "answers_question": False}
    assert parse_judge_json("no json here") is None
    assert parse_judge_json('{"broken": ') is None
