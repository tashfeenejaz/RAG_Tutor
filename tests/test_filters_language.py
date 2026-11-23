from qa import detect_language, extract_implicit_filters
from hybrid_search import build_where_clause, tokenize


def test_chapter_filter_variants():
    assert extract_implicit_filters("explain chapter 4")["chapter"] == 4
    assert extract_implicit_filters("ch. 7 notes")["chapter"] == 7
    assert extract_implicit_filters("chp 2")["chapter"] == 2


def test_question_type_filters():
    assert extract_implicit_filters("give me MCQs on friction")["question_type"] == "MCQ"
    assert extract_implicit_filters("show long questions")["question_type"] == "long"
    assert extract_implicit_filters("show short questions")["question_type"] == "short"


def test_plain_word_long_or_short_is_not_a_filter():
    # regression: "long" / "short" inside a normal question used to force a filter
    assert extract_implicit_filters("how long does light take to reach earth") is None
    assert extract_implicit_filters("what is a short circuit") is None


def test_question_type_only_from_current_message():
    f = extract_implicit_filters("chapter 3 mcqs", current_query="explain it simply")
    assert f == {"chapter": 3}


def test_language_detection():
    assert detect_language("What is Pascal's law?") == "english"
    assert detect_language("Pascal's law kya hai?") == "hinglish"
    assert detect_language("mujhe force ke bare mein samjhao aur examples do") == "hinglish"
    # a single marker inside a long English sentence must not flip the language
    assert detect_language("Explain why the force on the first body ka is equal and opposite to the second body") == "english"


def test_where_clause():
    assert build_where_clause(None) is None
    assert build_where_clause({"chapter": 4}) == {"chapter": 4}
    assert build_where_clause({"chapter": 4, "source_type": "textbook"}) == {
        "$and": [{"chapter": 4}, {"source_type": "textbook"}]
    }


def test_tokenize_drops_stopwords():
    assert tokenize("What is the SI unit of force?") == ["si", "unit", "force"]
