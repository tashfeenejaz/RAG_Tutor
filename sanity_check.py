import json
from pathlib import Path

def load_jsonl(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def sanity_check(chunks, label):
    print(f"\n=== {label} ===")
    print(f"Total chunks: {len(chunks)}")

    # Check for duplicate IDs
    ids = [c["id"] for c in chunks]
    duplicates = len(ids) - len(set(ids))
    print(f"Duplicate IDs: {duplicates}")

    # Check for empty text
    empty_text = sum(1 for c in chunks if not c["text"].strip())
    print(f"Empty text chunks: {empty_text}")

    # Check for missing metadata keys (find union of all keys used)
    all_keys = set()
    for c in chunks:
        all_keys.update(c["metadata"].keys())
    print(f"Metadata keys used across chunks: {all_keys}")

    # Check average/min/max text length
    lengths = [len(c["text"].split()) for c in chunks]
    print(f"Text length (words) — min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)//len(lengths)}")

    # Flag suspiciously short chunks (likely parsing errors)
    short_chunks = [c for c in chunks if len(c["text"].split()) < 3]
    print(f"Suspiciously short chunks (<3 words): {len(short_chunks)}")
    for c in short_chunks[:5]:
        print(f"  -> {c['id']}: {repr(c['text'])}")


if __name__ == "__main__":
    textbook_chunks = load_jsonl("data/class_9/Lahore_board/physics/textbook_chunks.jsonl")
    past_paper_chunks = load_jsonl("data/class_9/Lahore_board/physics/past_paper_chunks.jsonl")

    sanity_check(textbook_chunks, "Textbook Chunks")
    sanity_check(past_paper_chunks, "Past Paper Chunks")

    # Check combined ID uniqueness (important since both will go into same vector DB)
    all_ids = [c["id"] for c in textbook_chunks] + [c["id"] for c in past_paper_chunks]
    combined_dupes = len(all_ids) - len(set(all_ids))
    print(f"\n=== Combined ===")
    print(f"Total combined chunks: {len(all_ids)}")
    print(f"Duplicate IDs across both files: {combined_dupes}")