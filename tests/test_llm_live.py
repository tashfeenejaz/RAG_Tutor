"""Live tests against Gemini. Skipped unless RUN_LLM_TESTS=1 (they use API quota)."""
import os

import pytest

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(os.getenv("RUN_LLM_TESTS") != "1", reason="set RUN_LLM_TESTS=1 to run live LLM tests"),
]


def test_in_scope_answer_is_grounded_and_cited():
    import qa

    answer, chunks = qa.ask("What is Pascal's law?")
    assert chunks, "expected retrieved context"
    assert "[Source" in answer


@pytest.mark.parametrize("q", ["What is the theory of general relativity?", "Explain black holes", "What is dark matter?"])
def test_out_of_scope_is_refused(q):
    import qa

    answer, chunks = qa.ask(q)
    refusal_markers = ("not found", "nahi mila", "could not find", "not cover", "does not", "do not", "doesn't", "not mentioned", "not available")
    assert not chunks or any(m in answer.lower() for m in refusal_markers)
