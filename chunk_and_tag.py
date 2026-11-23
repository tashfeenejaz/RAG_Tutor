import re
import json
from pathlib import Path


def extract_chapter_number(text):
    """
    Looks for a 'Chapter N' pattern in the text.
    Returns the chapter number as int if found, else None.
    """
    match = re.search(r'chapter\s+(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def load_textbook_pages(ocr_dir):
    """
    Loads all page_*.txt files in order, and returns a list of
    (page_number, chapter_number, text) tuples.
    Chapter number is forward-filled: if a page doesn't explicitly
    mention a chapter, it inherits the last known chapter number.
    """
    ocr_dir = Path(ocr_dir)
    page_files = sorted(ocr_dir.glob("page_*.txt"))

    pages = []
    current_chapter = None  # will hold the "last seen" chapter number

    for page_file in page_files:
        # extract page number from filename, e.g. "page_045.txt" -> 45
        page_num = int(page_file.stem.split("_")[1])
        text = page_file.read_text(encoding="utf-8").strip()

        detected_chapter = extract_chapter_number(text)
        if detected_chapter is not None:
            if current_chapter is None or detected_chapter >= current_chapter:
                current_chapter = detected_chapter  # update forward-fill value

        pages.append({
            "page_num": page_num,
            "chapter": current_chapter,
            "text": text
        })

    return pages


def chunk_text(text, chunk_size=350, overlap=50):
    """
    Splits text into word-based chunks with overlap.
    chunk_size and overlap are in words, not characters.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            # Stop here: the next window would only repeat the overlap of this
            # chunk (a redundant "tail" chunk made of already-covered words).
            break
        start += (chunk_size - overlap)  # move forward, but overlap a bit

    return chunks


def build_textbook_chunks(ocr_dir, subject, board, class_level):
    """
    Takes all OCR'd textbook pages, tags them with chapter numbers,
    and splits into chunks with full metadata attached.
    Returns a list of chunk dictionaries.
    """
    pages = load_textbook_pages(ocr_dir)
    all_chunks = []
    chunk_id = 0

    for page in pages:
        text_chunks = chunk_text(page["text"])
        for tc in text_chunks:
            chunk_id += 1
            all_chunks.append({
                "id": f"textbook_{chunk_id:05d}",
                "text": tc,
                "metadata": {
                    "class": class_level,
                    "subject": subject,
                    "board": board,
                    "chapter": page["chapter"],
                    "page": page["page_num"],
                    "source_type": "textbook"
                }
            })

    return all_chunks


if __name__ == "__main__":
    chunks = build_textbook_chunks(
        ocr_dir="data/class_9/Lahore_board/physics/textbook_ocr",
        subject="Physics",
        board="Punjab",
        class_level=9
    )

    print(f"Total textbook chunks created: {len(chunks)}")
    print("\nSample chunk:")
    print(json.dumps(chunks[0], indent=2, ensure_ascii=False))

    # save all chunks to a JSONL file (one JSON object per line)
    output_path = Path("data/class_9/Lahore_board/physics/textbook_chunks.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nSaved to {output_path}")