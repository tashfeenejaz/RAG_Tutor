"""
End-to-end generation evaluation with an LLM judge (uses Gemini API quota).

For answerable gold queries it checks FAITHFULNESS (is every claim supported by
the retrieved sources?) and ANSWER RELEVANCE; for out-of-scope queries it checks
whether the tutor correctly REFUSES.

    python eval_generation.py --limit 20 --sleep 4

The judge is the same model family as the generator, so treat the numbers as a
quick regression signal, not ground truth - spot-check a few by hand.
"""
import argparse
import json
import re
import time
from pathlib import Path

import config
import qa
from eval_retrieval import load_gold

JUDGE_PROMPT = """You are a strict grader of a physics tutor's answer.

SOURCES GIVEN TO THE TUTOR:
{context}

QUESTION: {question}

TUTOR'S ANSWER:
{answer}

Grade and reply with ONLY a JSON object:
{{"faithful": true/false,  "answers_question": true/false, "unsupported_claims": ["..."]}}
- faithful: every factual claim is supported by the sources (MCQ answers explicitly labelled "solved by tutor" are allowed).
- answers_question: the answer actually addresses the question."""

REFUSAL_PROMPT = """A physics tutor was asked a question that is NOT covered by its textbook.

QUESTION: {question}
TUTOR'S ANSWER:
{answer}

Did the tutor decline / say the topic is not covered, WITHOUT inventing an answer from outside knowledge?
Reply with ONLY a JSON object: {{"refused": true/false}}"""


def parse_judge_json(text):
    """Extract the first JSON object from an LLM reply (tolerates ```json fences)."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge(prompt):
    return parse_judge_json(qa._generate(prompt, retries=3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=20, help="max answerable queries to grade (evenly sampled)")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between API calls (free-tier rate limit)")
    ap.add_argument("--out", default=str(config.EVAL_RESULTS_DIR))
    args = ap.parse_args()

    gold = load_gold()
    answerable = [g for g in gold if not g.get("should_refuse")]
    step = max(1, len(answerable) // args.limit)
    answerable = answerable[::step][: args.limit]
    refusals = [g for g in gold if g.get("should_refuse")]

    faithful = relevant = graded = 0
    details = []
    for g in answerable:
        answer, chunks = qa.ask(g["query"], where_filter=g.get("filter"))
        time.sleep(args.sleep)
        verdict = judge(JUDGE_PROMPT.format(context=qa.format_context(chunks), question=g["query"], answer=answer)) if chunks else None
        time.sleep(args.sleep)
        if verdict:
            graded += 1
            faithful += bool(verdict.get("faithful"))
            relevant += bool(verdict.get("answers_question"))
        details.append({"id": g["id"], "query": g["query"], "answer": answer, "verdict": verdict})
        print(f"{g['id']}: {verdict}")

    refused = 0
    for g in refusals:
        answer, chunks = qa.ask(g["query"])
        time.sleep(args.sleep)
        if not chunks:
            ok = True  # rejected at retrieval
        else:
            v = judge(REFUSAL_PROMPT.format(question=g["query"], answer=answer)) or {}
            ok = bool(v.get("refused"))
            time.sleep(args.sleep)
        refused += ok
        details.append({"id": g["id"], "query": g["query"], "answer": answer, "refused": ok})
        print(f"{g['id']}: refused={ok}")

    summary = {
        "graded_answerable": graded,
        "faithfulness": faithful / graded if graded else None,
        "answer_relevance": relevant / graded if graded else None,
        "refusal_accuracy": refused / len(refusals) if refusals else None,
    }
    print("\n", json.dumps(summary, indent=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generation.json").write_text(json.dumps({"summary": summary, "details": details}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
