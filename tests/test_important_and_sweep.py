import qa
import important as im
import eval_sweep as sw
from conftest import make_chunk
from eval_retrieval import bootstrap_ci, relevant_ids


def pp(cid, text, chapter=1, qtype="short", year=2023, group="I"):
    return make_chunk(cid, text, source_type="past_paper", chapter=chapter, question_type=qtype,
                      year=year, group=group, paper_type="subjective")


PAPERS = [
    pp("a1", "Define centre of mass.", year=2023, group="I"),
    pp("a2", "Define centre of mass.", year=2024, group="II"),
    pp("a3", "Differentiate between vectors and scalars.", year=2023, group="I"),
    pp("a4", "Differentiate scalars and vectors quantities.", year=2024, group="I"),
    pp("a5", "What are prefixes and give two examples.", year=2024, group="II"),
    pp("a6", "Write two characteristics of physical quantities.", year=2023, group="II"),
    pp("b1", "Define plasma physics.", chapter=9),
    make_chunk("t1", "Textbook text about centre of mass", chapter=1),
]


# ---------------------------------------------------------------- intent
def test_intent_detection():
    assert im.detect_important_intent("chp 9 k imp short questions")
    assert im.detect_important_intent("which questions are expected in chapter 3?")
    assert not im.detect_important_intent("What is Pascal's law?")
    assert not im.detect_important_intent("explain pressure")


def test_followup_inherits_intent_only_for_question_type_messages():
    history = [("chp 9 k imp short questions", "...")]
    assert im.detect_important_intent("or is chp k long questions", history)
    assert not im.detect_important_intent("explain torque", history)
    assert not im.detect_important_intent("or is chp k long questions", [("what is torque", "...")])


# -------------------------------------------------------------- grouping
def test_groups_repeats_across_papers_and_counts_distinct_papers():
    groups = im.group_questions(im.select_chunks(PAPERS, chapter=1, question_type="short"))
    top = {g["text"]: len(g["papers"]) for g in groups}
    assert top["Define centre of mass."] == 2
    assert sorted(top.values(), reverse=True)[:2] == [2, 2]       # vectors/scalars paraphrase merged
    assert groups[0]["papers"] and len(groups[0]["papers"]) >= len(groups[-1]["papers"])


def test_answer_lists_repeated_before_once_and_states_the_data_limit():
    text = im.answer_important(PAPERS, chapter=1, question_type="short")
    assert text.index("Repeated") < text.index("Appeared once")
    assert "Define centre of mass. (2023 Gp I, 2024 Gp II)" in text
    assert "nothing is predicted" in text and "Textbook exercise" in text


def test_thin_chapter_refuses_to_judge_importance():
    text = im.answer_important(PAPERS, chapter=9, question_type="short")
    assert "cannot judge importance" in text and "Define plasma physics." in text
    assert "Repeated" not in text


def test_no_data_and_hinglish():
    assert "no past-paper questions" in im.answer_important(PAPERS, chapter=7)
    assert "importance ka andaza" in im.answer_important(PAPERS, chapter=9, language="hinglish")


def test_textbook_chunks_never_count():
    assert all(c["metadata"]["source_type"] == "past_paper" for c in im.select_chunks(PAPERS, chapter=1))


# ------------------------------------------------------------- qa routing
class Searcher:
    all_chunks = PAPERS

    def search(self, *a, **k):
        raise AssertionError("the important-questions route must not call the retriever")


def test_prepare_routes_important_questions_without_retrieval_or_llm():
    p = qa.prepare("chapter 1 imp short questions", searcher=Searcher())
    assert p.direct_answer and "Repeated" in p.direct_answer and p.prompt == ""
    assert p.chunks == []


def test_chapter_is_taken_from_earlier_turn_when_followup_omits_it(monkeypatch):
    monkeypatch.setattr(qa, "rewrite_query", lambda m, qa_pairs=None, language="english": m)
    hist = [("chapter 1 imp short questions", "answer")]
    p = qa.prepare("imp long questions", qa_pairs=hist, searcher=Searcher())
    assert p.where_filter == {"chapter": 1, "question_type": "long"}


def test_dropdown_filter_wins(monkeypatch):
    p = qa.prepare("chapter 1 imp questions", where_filter={"chapter": 9}, searcher=Searcher())
    assert p.where_filter["chapter"] == 9


# ------------------------------------------------------------ eval helpers
def test_bootstrap_ci_brackets_the_mean_and_handles_edges():
    lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert lo <= 0.75 <= hi
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)
    assert bootstrap_ci([])[0] != bootstrap_ci([])[0]    # nan


def test_relevant_ids_finds_every_chunk_containing_the_phrase():
    assert relevant_ids(PAPERS, ["centre of mass"]) == {"a1", "a2", "t1"}


# ------------------------------------------------------------------ sweep
GOLD = [
    {"id": "g1", "type": "definition", "query": "What is torque moment arm", "answer_phrases": ["moment arm"],
     "filter": None, "should_refuse": False},
    {"id": "g2", "type": "definition", "query": "Define pressure", "answer_phrases": ["force exerted normally"],
     "filter": None, "should_refuse": False},
    {"id": "o1", "type": "out_of_scope", "query": "zxqvwidget history", "answer_phrases": [],
     "filter": None, "should_refuse": True},
]


def test_sweep_solo_gate_trades_false_retrieval_for_nothing_here(searcher):
    # dense gate closed so only BM25 can retrieve; the nonsense token "zxqvwidget" exists in t5
    res = sw.run_sweep(searcher, GOLD, distances=[1e-9], solo_scores=[0.0, 1e9], rerank_cutoffs=None)
    low, never = res
    assert low["false_n"] == 1 and never["false_n"] == 0
    best, ok = sw.pick_best(res, max_false_rate=0.0)
    assert ok and best["bm25_solo"] == 1e9


def test_pick_best_falls_back_to_lowest_false_rate_when_target_unreachable():
    rs = [{"mrr@5": 0.9, "hit@5": 1, "false_rate": 0.5}, {"mrr@5": 0.1, "hit@5": 1, "false_rate": 0.25}]
    best, ok = sw.pick_best(rs, max_false_rate=0.0)
    assert not ok and best["false_rate"] == 0.25


def test_sweep_with_rerank_cutoffs_and_caches(searcher):
    calls = []

    class Fake:
        def score(self, query, texts):
            calls.append(len(texts))
            return [float(len(t)) for t in texts]

    searcher._reranker = Fake()
    sw.add_caches(searcher)
    res = sw.run_sweep(searcher, GOLD, distances=[None], solo_scores=[0.0], rerank_cutoffs=[None, 1e9])
    assert res[0]["reranker"] and res[1]["false_n"] == 0      # a huge cutoff rejects everything
    n = len(calls)
    sw.run_sweep(searcher, GOLD, distances=[None], solo_scores=[0.0], rerank_cutoffs=[None, 1e9])
    assert len(calls) == n                                    # second pass served from the cache


def test_env_lines():
    assert sw.env_lines({"bm25_solo": 12.0, "reranker": True, "rerank_min": -2.5}) == [
        "BM25_SOLO_MIN_SCORE=12", "RERANK_MIN_SCORE=-2.5"]
    assert sw.env_lines({"bm25_solo": 8.0, "reranker": False, "rerank_min": None}) == ["BM25_SOLO_MIN_SCORE=8"]


def test_clean_question_strips_number_prefix_and_trailing_marks():
    assert im.clean_question("1.1 Define torque. 4") == "Define torque."
    assert im.clean_question("5 The work done is zero") == "The work done is zero"
