import re
import json
from pathlib import Path


def extract_paper_metadata(filename):
    """
    Extracts year, group, and paper_type from filenames like:
    '2023-Group-II-Objective.txt' or '2024-Group-I-Subjective.txt'
    Returns a dict, or None if pattern doesn't match.
    """
    pattern = r'(\d{4})-Group-(I{1,2})-(Objective|Subjective)'
    match = re.search(pattern, filename, re.IGNORECASE)
    if not match:
        return None
    year, group, paper_type = match.groups()
    return {
        "year": int(year),
        "group": group.upper(),
        "paper_type": paper_type.lower()
    }


def clean_page_markers(text):
    """Removes '--- Page N ---' markers left over from OCR script"""
    return re.sub(r'---\s*Page\s*\d+\s*---', '', text).strip()


# ---------- OBJECTIVE (MCQ) PARSING ----------
STANDARD_OPTS = re.compile(r'\bA\)?\b.*\bB\)?\b.*\bC\)?\b.*\bD\)?\b', re.DOTALL)
CIRCLED_OPTS = re.compile(r'\u24B6.*\u24B7.*\u24B8.*\u24B9', re.DOTALL)


def has_four_options(text):
    """Checks if text contains all 4 option markers, in either standard (A/B/C/D or A)) or circled (Ⓐ Ⓑ Ⓒ Ⓓ) style"""
    return bool(STANDARD_OPTS.search(text)) or bool(CIRCLED_OPTS.search(text))


def parse_objective(text):
    """
    Splits an objective paper into individual MCQ chunks.
    Each chunk = one question + its 4 options, kept together.
    Handles multiple numbering styles ("1.1", "1.", "**1**") and
    multiple option-marker styles (A/B/C/D, "A)", circled Ⓐ Ⓑ Ⓒ Ⓓ).
    """
    text = clean_page_markers(text)
    text = text.replace('*', '')

    pattern = r'(?=^\s*\d+(?:\.\d+)?\.?\s+\S)'
    parts = re.split(pattern, text, flags=re.MULTILINE)

    questions = []
    for part in parts:
        part = part.strip()
        if part and has_four_options(part):
            questions.append(part)

    return questions


# ---------- SUBJECTIVE PARSING ----------

ROMAN_ITEMS = r'(?:i|ii|iii|iv|v|vi|vii|viii)'

def split_parts(text):
    """Splits subjective paper text into Part I and Part II sections"""
    text = clean_page_markers(text)
    # Flexible match for "PART – I" / "PART-I" / "PART - I" etc.
    match = re.search(r'PART\s*[-–]\s*I{2}\b', text, re.IGNORECASE)
    if match:
        part1 = text[:match.start()]
        part2 = text[match.start():]
    else:
        part1 = text
        part2 = ""
    return part1, part2


def parse_short_questions(part1_text):
    """
    Extracts short-answer sub-questions (i, ii, iii...) from Part I.
    Each sub-item becomes its own chunk, tagged with its parent question number.
    """
    results = []
    # Find each "Q.N Write short answers..." block
    q_blocks = re.split(r'(?=Q\.\d+\s+Write short answers)', part1_text)

    for block in q_blocks:
        q_num_match = re.search(r'Q\.(\d+)', block)
        if not q_num_match:
            continue
        q_num = q_num_match.group(1)

        # Find each roman-numeral sub-item within this block
        sub_pattern = rf'(?:^|\n)\s*({ROMAN_ITEMS})\s+(.*?)(?=(?:\n\s*{ROMAN_ITEMS}\s+)|\Z)'
        for m in re.finditer(sub_pattern, block, re.DOTALL):
            item_text = m.group(2).strip()
            if item_text:
                results.append({
                    "text": item_text,
                    "parent_question": q_num,
                    "question_type": "short"
                })

    return results


def parse_long_questions(part2_text):
    """
    Extracts long-answer sub-questions (a, b) from Part II.
    Each (a)/(b) part becomes its own chunk, tagged with its parent question number.
    """
    results = []
    q_blocks = re.split(r'(?=Q\.\d+\(a\))', part2_text)

    for block in q_blocks:
        q_num_match = re.search(r'Q\.(\d+)\(a\)', block)
        if not q_num_match:
            continue
        q_num = q_num_match.group(1)

        sub_pattern = r'\(([ab])\)\s*(.*?)(?=\(\s*[ab]\s*\)|\Z)'
        for m in re.finditer(sub_pattern, block, re.DOTALL):
            item_text = m.group(2).strip()
            if item_text:
                results.append({
                    "text": item_text,
                    "parent_question": q_num,
                    "question_type": "long"
                })

    return results


def parse_subjective(text):
    """Parses a full subjective paper into short + long question chunks"""
    part1, part2 = split_parts(text)
    short_qs = parse_short_questions(part1)
    long_qs = parse_long_questions(part2)
    return short_qs + long_qs


# ---------- BUILD CHUNKS FOR ALL PAST PAPERS ----------

def build_past_paper_chunks(ocr_dir, subject, board, class_level):
    """
    Reads all .txt files in past_papers_ocr folder, parses each based on
    its type (objective/subjective), and returns a list of tagged chunks.
    """
    ocr_dir = Path(ocr_dir)
    all_chunks = []
    chunk_id = 0

    for txt_file in sorted(ocr_dir.glob("*.txt")):
        meta = extract_paper_metadata(txt_file.name)
        if meta is None:
            print(f"Warning: couldn't parse metadata from filename '{txt_file.name}', skipping.")
            continue

        text = txt_file.read_text(encoding="utf-8")

        if meta["paper_type"] == "objective":
            questions = parse_objective(text)
            for q_text in questions:
                chunk_id += 1
                all_chunks.append({
                    "id": f"pastpaper_{chunk_id:05d}",
                    "text": q_text,
                    "metadata": {
                        "class": class_level,
                        "subject": subject,
                        "board": board,
                        "year": meta["year"],
                        "group": meta["group"],
                        "paper_type": "objective",
                        "question_type": "MCQ",
                        "source_type": "past_paper"
                    }
                })

        else:  # subjective
            questions = parse_subjective(text)
            for q in questions:
                chunk_id += 1
                all_chunks.append({
                    "id": f"pastpaper_{chunk_id:05d}",
                    "text": q["text"],
                    "metadata": {
                        "class": class_level,
                        "subject": subject,
                        "board": board,
                        "year": meta["year"],
                        "group": meta["group"],
                        "paper_type": "subjective",
                        "question_type": q["question_type"],
                        "parent_question": q["parent_question"],
                        "source_type": "past_paper"
                    }
                })

    return all_chunks


if __name__ == "__main__":
    chunks = build_past_paper_chunks(
        ocr_dir="data/class_9/Lahore_board/physics/past_papers_ocr",
        subject="Physics",
        board="Lahore",
        class_level=9
    )

    print(f"Total past paper chunks created: {len(chunks)}\n")

    # Show a few samples of each type to eyeball correctness
    seen_types = set()
    for c in chunks:
        qtype = c["metadata"]["question_type"]
        if qtype not in seen_types:
            seen_types.add(qtype)
            print(f"--- Sample ({qtype}) ---")
            print(json.dumps(c, indent=2, ensure_ascii=False))
            print()

    output_path = Path("data/class_9/Lahore_board/physics/past_paper_chunks.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")