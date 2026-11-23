import json

import pytest

import config
from eval_retrieval import is_relevant, load_gold

needs_data = pytest.mark.skipif(
    not (config.TEXTBOOK_CHUNKS.exists() and config.PAST_PAPER_CHUNKS.exists()),
    reason="chunk files not present",
)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


@needs_data
def test_chunk_ids_are_unique_and_text_not_empty():
    chunks = load(config.TEXTBOOK_CHUNKS) + load(config.PAST_PAPER_CHUNKS)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c["text"].strip() for c in chunks)


@needs_data
def test_every_answerable_gold_query_has_a_relevant_chunk():
    """If this fails after re-chunking, the gold set no longer matches the data."""
    chunks = load(config.TEXTBOOK_CHUNKS) + load(config.PAST_PAPER_CHUNKS)
    missing = [
        g["id"] for g in load_gold()
        if not g.get("should_refuse")
        and not any(is_relevant(c["text"], g["answer_phrases"]) for c in chunks)
    ]
    assert not missing, f"no chunk contains the answer phrase for: {missing}"


def test_gold_set_shape():
    gold = load_gold()
    assert len({g["id"] for g in gold}) == len(gold)
    assert any(g.get("should_refuse") for g in gold) and any(not g.get("should_refuse") for g in gold)
    assert all(g["answer_phrases"] for g in gold if not g.get("should_refuse"))


def test_no_secrets_in_source_files():
    import pathlib, re

    pattern = re.compile(r"AIza[0-9A-Za-z_-]{20,}|AQ\.[A-Za-z0-9_-]{20,}")
    for path in pathlib.Path(".").glob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), f"possible API key in {path}"


HELDOUT = "eval/heldout_set.jsonl"


@needs_data
def test_heldout_set_is_valid_and_does_not_overlap_the_tuning_set():
    held, dev = load_gold(HELDOUT), load_gold()
    chunks = load(config.TEXTBOOK_CHUNKS) + load(config.PAST_PAPER_CHUNKS)
    assert len({g["id"] for g in held}) == len(held) and not {g["id"] for g in held} & {g["id"] for g in dev}
    assert any(g.get("should_refuse") for g in held)
    answerable = [g for g in held if not g.get("should_refuse")]
    assert all(any(is_relevant(c["text"], g["answer_phrases"]) for c in chunks) for g in answerable)
    dev_phrases = {p.lower() for g in dev for p in g["answer_phrases"]}
    assert not [g["id"] for g in answerable if any(p.lower() in dev_phrases for p in g["answer_phrases"])]
    assert not {g["query"].lower() for g in held} & {g["query"].lower() for g in dev}
