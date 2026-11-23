import json
import time
from config import get_client, LLM_MODEL

client = get_client()
MODEL = LLM_MODEL

CHAPTER_LIST = """
1. Physical Quantities and Measurements
2. Kinematics
3. Dynamics
4. Turning Effects of Force
5. Work, Energy and Power
6. Mechanical Properties of Matter
7. Thermal Properties of Matter
8. Magnetism
9. Nature of Science
"""


def classify_chapter(question_text, max_retries=5):
    prompt = f"""Here is the list of Class 9 Physics chapters:
{CHAPTER_LIST}

Question: "{question_text}"

Which chapter number (1-9) does this question belong to? Reply with ONLY the number, nothing else."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            text = response.text.strip()
            digits = ''.join(c for c in text if c.isdigit())
            if digits:
                chapter_num = int(digits)
                if 1 <= chapter_num <= 9:
                    return chapter_num
        except Exception as e:
            wait = 15  # wait long enough to clear the per-minute quota window
            print(f"    Error: {e}. Waiting {wait}s before retry...")
            time.sleep(wait)

    return None


def fix_none_chapters(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    none_chunks = [c for c in chunks if c["metadata"].get("chapter") is None]
    print(f"Found {len(none_chunks)} chunks needing reclassification.\n")

    for i, chunk in enumerate(none_chunks):
        print(f"[{i+1}/{len(none_chunks)}] {chunk['id']}...")
        chapter = classify_chapter(chunk["text"])
        chunk["metadata"]["chapter"] = chapter
        print(f"    -> Chapter {chapter}")
        time.sleep(4)  # stay comfortably under 15 requests/minute

    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nSaved to {path}")


if __name__ == "__main__":
    fix_none_chapters("data/class_9/Lahore_board/physics/past_paper_chunks.jsonl")