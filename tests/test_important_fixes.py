import important as im
from test_important_and_sweep import PAPERS, pp


def test_empty_type_reports_what_the_chapter_does_have():
    en = im.answer_important(PAPERS, chapter=9, question_type="MCQ")
    assert "No MCQs from chapter 9 appeared" in en and "MCQ=0, short=1, long=0" in en
    hi = im.answer_important(PAPERS, chapter=9, question_type="MCQ", language="hinglish")
    assert "koi MCQs nahi aaya" in hi and "MCQ=0, short=1, long=0" in hi


def test_chapter_with_no_questions_at_all_keeps_the_generic_message():
    assert "no past-paper questions" in im.answer_important(PAPERS, chapter=7, question_type="MCQ")


def test_thin_chapter_merges_the_same_question_asked_in_two_papers():
    data = PAPERS + [pp("b2", "Define plasma physics.", chapter=9, year=2023, group="II")]
    text = im.answer_important(data, chapter=9, question_type="short")
    assert text.count("Define plasma physics.") == 1
    assert "Define plasma physics. (2023 Gp I, 2023 Gp II)" in text


def test_topic_asked_as_short_and_long_counts_as_one_repeat():
    data = [
        pp("s1", "Define latent heat of fusion.", chapter=7, qtype="short", year=2024, group="II"),
        pp("l1", "Define and explain latent heat of fusion. 4", chapter=7, qtype="long", year=2023, group="I"),
    ] + [pp(f"x{i}", f"Filler question number {i} about heat.", chapter=7, year=2023, group="I") for i in range(5)]
    text = im.answer_important(data, chapter=7)   # all types of the chapter
    assert "Repeated" in text
    assert "(2023 Gp I [long], 2024 Gp II [short])" in text
