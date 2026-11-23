import qa
from conftest import make_chunk


class FakeSearcher:
    """Returns canned results and records how it was called."""

    def __init__(self, filtered=None, unfiltered=None):
        self.filtered, self.unfiltered, self.calls = filtered or [], unfiltered or [], []

    def search(self, query, n_results=5, where_filter=None, max_distance=None, **kw):
        self.calls.append({"q": query, "n": n_results, "filter": where_filter, "gate": max_distance})
        return self.filtered if where_filter else self.unfiltered


CH = [make_chunk("t1", "Pressure is force per unit area", chapter=6, page=132)]


def test_source_labels_and_footer():
    pp = make_chunk("p1", "Define torque.", source_type="past_paper", year=2024, group="I",
                    paper_type="subjective", question_type="short")
    footer = qa.format_sources(CH + [pp])
    assert "1. Textbook, Chapter 6, Page 132" in footer
    assert "2. Past Paper 2024 Group I (subjective, short)" in footer
    assert qa.format_sources([]) == ""


def test_prompt_demands_citations_and_mcq_label():
    p = qa.build_prompt("q", qa.format_context(CH))
    assert "[Source n]" in p and "solved by tutor" in p and "[Source 1" in p


def test_prepare_not_found_has_no_prompt():
    p = qa.prepare("What is dark matter?", searcher=FakeSearcher())
    assert p.chunks == [] and p.prompt == ""


def test_distance_gate_depends_on_the_kind_of_filter():
    s = FakeSearcher(unfiltered=CH, filtered=CH)
    qa.prepare("What is pressure?", searcher=s)
    assert s.calls[-1]["gate"] == qa.config.MAX_DISTANCE                # open question: strict
    qa.prepare("explain pressure", where_filter={"chapter": 6}, searcher=s)
    assert s.calls[-1]["gate"] == qa.config.RELAXED_MAX_DISTANCE        # chapter filter: relaxed, NOT removed
    qa.prepare("Show MCQs from chapter 6", searcher=s)
    assert s.calls[-1]["filter"] == {"chapter": 6, "question_type": "MCQ"}
    assert s.calls[-1]["gate"] is None                                  # browse request: ranking only


def test_prepare_falls_back_to_unfiltered_search():
    s = FakeSearcher(filtered=[], unfiltered=CH)
    p = qa.prepare("explain pressure", where_filter={"chapter": 2}, searcher=s)
    assert p.chunks == CH
    assert [c["filter"] for c in s.calls] == [{"chapter": 2}, None]
    assert s.calls[-1]["gate"] == qa.config.MAX_DISTANCE


def test_list_all_requests_widen_retrieval():
    s = FakeSearcher(unfiltered=CH, filtered=CH)
    qa.prepare("give me all mcqs of 2024", searcher=s)
    assert s.calls[0]["n"] == 25


def test_not_found_message_follows_the_students_language():
    assert qa.NOT_FOUND["hinglish"].startswith("Ye topic")
    assert qa.NOT_FOUND["english"].startswith("I could not find")


def test_ask_with_monkeypatched_llm(monkeypatch):
    monkeypatch.setattr(qa, "get_searcher", lambda: FakeSearcher(unfiltered=CH))
    monkeypatch.setattr(qa, "_generate", lambda prompt, retries=3: "Pressure is force per area [Source 1].")
    answer, chunks = qa.ask("What is pressure?")
    assert "[Source 1]" in answer and chunks == CH


def test_rewrite_is_skipped_for_standalone_english(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(qa, "_generate", boom)
    assert qa.rewrite_query("What is pressure?", None, "english") == "What is pressure?"


def test_rewrite_falls_back_to_original_on_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("quota")

    monkeypatch.setattr(qa, "_generate", boom)
    assert qa.rewrite_query("pressure kya hai", None, "hinglish") == "pressure kya hai"
