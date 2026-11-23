GROUNDING_TESTS = [
    # In-scope: should answer confidently and correctly
    {"query": "What is Newton's first law of motion?", "type": "in_scope"},
    {"query": "Define density", "type": "in_scope"},

    # Out-of-scope: should honestly say "not found", not hallucinate
    {"query": "What is the theory of general relativity?", "type": "out_of_scope"},
    {"query": "Explain black holes", "type": "out_of_scope"},
    {"query": "What is the periodic table?", "type": "out_of_scope"},

    # Adjacent/tricky: related terms exist in book but full answer might not
    {"query": "Who won the Nobel Prize for Physics from Pakistan and for what?", "type": "adjacent"},
    {"query": "What is dark matter?", "type": "out_of_scope"},
]


# Manual smoke test. For automated checks use:  python eval_generation.py  /  pytest -m llm
def run_grounding_check():
    from qa import ask, format_sources

    for test in GROUNDING_TESTS:
        print("=" * 60)
        print(f"[{test['type'].upper()}] {test['query']}")
        print("=" * 60)
        answer, chunks = ask(test["query"])
        print(answer)
        print(format_sources(chunks) or "\n(no chunks passed the relevance gate)")
        print()


if __name__ == "__main__":
    run_grounding_check()