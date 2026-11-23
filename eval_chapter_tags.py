"""Chapter-tag accuracy for past-paper chunks.

  python eval_chapter_tags.py sample --n 60   # writes eval/chapter_tag_sample.csv
  # fill the true_chapter column by hand (look at the textbook), save the CSV
  python eval_chapter_tags.py score           # accuracy + per-chapter P/R/F1 + wrong rows
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import config

CSV_PATH = Path("eval/chapter_tag_sample.csv")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sample(n, seed=7):
    chunks = load(config.PAST_PAPER_CHUNKS)
    # stratify by predicted chapter so rare chapters (8, 9) are included
    by_ch = defaultdict(list)
    for c in chunks:
        by_ch[c["metadata"].get("chapter")].append(c)
    rng = random.Random(seed)
    for v in by_ch.values():
        rng.shuffle(v)
    picked, keys = [], sorted(by_ch, key=lambda k: (k is None, k))
    while len(picked) < n and any(by_ch.values()):
        for k in keys:
            if by_ch[k] and len(picked) < n:
                picked.append(by_ch[k].pop())
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "predicted_chapter", "true_chapter", "text"])
        for c in picked:
            w.writerow([c["id"], c["metadata"].get("chapter"), "",
                        c["text"].replace("\n", " ")[:300]])
    print(f"Wrote {len(picked)} rows to {CSV_PATH}. Fill true_chapter, then run: score")


def score():
    rows = [r for r in csv.DictReader(open(CSV_PATH, encoding="utf-8"))
            if r["true_chapter"].strip()]
    if not rows:
        print("true_chapter column is empty.")
        return
    ok = sum(r["predicted_chapter"].strip() == r["true_chapter"].strip() for r in rows)
    print(f"Accuracy: {ok}/{len(rows)} = {ok/len(rows):.3f}\n")
    labels = sorted({r["true_chapter"].strip() for r in rows} |
                    {r["predicted_chapter"].strip() for r in rows})
    tp, pc, tc = Counter(), Counter(), Counter()
    for r in rows:
        p, t = r["predicted_chapter"].strip(), r["true_chapter"].strip()
        pc[p] += 1
        tc[t] += 1
        if p == t:
            tp[t] += 1
    print("chapter  precision  recall  f1   (support)")
    for l in labels:
        pr = tp[l] / pc[l] if pc[l] else 0.0
        rc = tp[l] / tc[l] if tc[l] else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        print(f"{l:>7}  {pr:9.2f}  {rc:6.2f}  {f1:4.2f}   ({tc[l]})")
    print("\nWrong rows:")
    for r in rows:
        if r["predicted_chapter"].strip() != r["true_chapter"].strip():
            print(f"{r['id']} pred={r['predicted_chapter']} true={r['true_chapter']} | {r['text'][:100]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    sample(a.n) if a.cmd == "sample" else score()
