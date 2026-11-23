import json
import time
from pathlib import Path
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


def classify_chapter(question_text, max_retries=3):
    """Asks Gemini which chapter (1-9) a past paper question belongs to"""
    prompt = f"""Here is the list of Class 9 Physics chapters:
{CHAPTER_LIST}

Question: "{question_text}"

Which chapter number (1-9) does this question belong to? Reply with ONLY the number, nothing else."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            text = response.text.strip()
            # Extract just the digit(s)
            digits = ''.join(c for c in text if c.isdigit())
            if digits:
                chapter_num = int(digits)
                if 1 <= chapter_num <= 9:
                    return chapter_num
            print(f"    Unexpected response: '{text}', retrying...")
        except Exception as e:
            print(f"    Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    return None  # couldn't classify after retries


def tag_past_papers_with_chapters(input_path, output_path):
    chunks = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Classifying {len(chunks)} past paper chunks by chapter...\n")

    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] {chunk['id']}...")
        chapter = classify_chapter(chunk["text"])
        chunk["metadata"]["chapter"] = chapter
        print(f"    -> Chapter {chapter}")

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    tag_past_papers_with_chapters(
        input_path="data/class_9/Lahore_board/physics/past_paper_chunks.jsonl",
        output_path="data/class_9/Lahore_board/physics/past_paper_chunks.jsonl"
    )