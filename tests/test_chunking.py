from chunk_and_tag import chunk_text, extract_chapter_number


def test_short_text_is_one_chunk():
    assert len(chunk_text("word " * 100)) == 1


def test_no_redundant_tail_chunk():
    # 340 words fit in ONE 350-word chunk; the old loop emitted a second chunk
    # made only of words already covered by the first.
    assert len(chunk_text("w " * 340)) == 1
    assert len(chunk_text("w " * 350)) == 1


def test_overlap_and_coverage():
    words = [f"w{i}" for i in range(700)]
    chunks = chunk_text(" ".join(words), chunk_size=350, overlap=50)
    assert len(chunks) == 3
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]  # 50-word overlap
    assert chunks[-1].split()[-1] == "w699"                  # nothing is lost at the end


def test_empty_text():
    assert chunk_text("   ") == []


def test_chapter_number_extraction():
    assert extract_chapter_number("CHAPTER 4 Turning Effects") == 4
    assert extract_chapter_number("no heading here") is None
