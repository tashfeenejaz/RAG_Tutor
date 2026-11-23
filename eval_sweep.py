"""
Threshold sweep: trade answerable hits against false retrievals on out-of-scope queries.

Sweeps the three gates that decide whether a chunk is returned at all:
    * dense distance gate   (config.MAX_DISTANCE)
    * BM25 "solo" score     (config.BM25_SOLO_MIN_SCORE): a chunk only BM25 found
    * cross-encoder cutoff  (config.RERANK_MIN_SCORE)
and prints the setting with the best MRR whose false-retrieval rate stays at or below
``--max-false-rate``. Embeddings, Chroma lookups and re-ranker scores are cached, so the
whole grid costs little more than one evaluation run.

Usage
-----
    python eval_sweep.py                        # full grid incl. re-ranker
    python eval_sweep.py --no-rerank            # dense + BM25 gates only
    python eval_sweep.py --max-false-rate 0.08  # stricter refusal target

CAUTION: the gold set is both the tuning set and the report set, so the chosen numbers
are optimistic. Before quoting them, add new queries that were not used here.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import config
from eval_retrieval import evaluate, load_gold
from hybrid_search import HybridSearcher

DEFAULT_DISTANCES = [0.50, 0.55, 0.60, 0.65, 0.70]
DEFAULT_SOLO = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 1e9]   # 1e9 = BM25 may never retrieve on its own


# ------------------------------------------------------------------ caching
def add_caches(searcher):
    """Cache dense lookups (gate applied afterwards) and re-ranker scores per (query, text)."""
    orig_dense = searcher.dense_candidates
    dense_cache = {}

    def cached_dense(query, n_results=None, where_filter=None, max_distance=None):
        key = (query, json.dumps(where_filter, sort_keys=True), n_results)
        if key not in dense_cache:
            dense_cache[key] = orig_dense(query, n_results, where_filter, None)
        return [(c, d) for c, d in dense_cache[key] if max_distance is None or d <= max_distance]

    searcher.dense_candidates = cached_dense

    rr = searcher.reranker
    if rr is not None:
        orig_score, memo = rr.score, {}

        def cached_score(query, texts):
            missing = [t for t in texts if (query, t) not in memo]
            if missing:
                for t, s in zip(missing, orig_score(query, missing)):
                    memo[(query, t)] = s
            return [memo[(query, t)] for t in texts]

        rr.score = cached_score
    return searcher


def auto_cutoffs(searcher, gold, k=5):
    """Cross-encoder cutoffs on the model's own scale: percentiles of the top-1 scores."""
    top = []
    for g in gold:
        res = searcher.search(g["query"], n_results=k, where_filter=g.get("filter"),
                              max_distance=None, bm25_solo_min_score=1e9, rerank=True)
        if res and res[0]["retrieval"]["rerank_score"] is not None:
            top.append(res[0]["retrieval"]["rerank_score"])
    if not top:
        return []
    top.sort()
    cuts = {round(top[min(len(top) - 1, int(p / 100 * len(top)))], 1) for p in range(0, 101, 10)}
    return sorted(cuts)


# -------------------------------------------------------------------- sweep
def run_sweep(searcher, gold, distances, solo_scores, rerank_cutoffs, k=5):
    """Evaluate every grid point. ``rerank_cutoffs`` of None = no re-ranker at all."""
    results = []
    saved = config.MAX_DISTANCE
    try:
        for dist, solo, cut in itertools.product(distances, solo_scores, rerank_cutoffs or [None]):
            config.MAX_DISTANCE = dist   # eval_retrieval.effective_gate reads this
            use_rr = rerank_cutoffs is not None and rerank_cutoffs != [None]
            summ, _ = evaluate(searcher, gold, k=k, mode="hybrid", rerank=use_rr,
                               bm25_solo_min_score=solo, rerank_min_score=cut)
            results.append({
                "max_distance": dist, "bm25_solo": solo, "rerank_min": cut, "reranker": use_rr,
                "hit@1": summ["hit@1"], f"hit@{k}": summ[f"hit@{k}"], f"mrr@{k}": summ[f"mrr@{k}"],
                "false_rate": summ["false_retrieval_rate"], "false_n": summ["false_retrieval_count"],
                "n_oos": summ["n_out_of_scope"],
            })
    finally:
        config.MAX_DISTANCE = saved
    return results


def pick_best(results, max_false_rate, k=5):
    """Best MRR (then Hit@k) among settings meeting the false-retrieval target.
    If none meets it, the lowest false rate wins. Returns (best, feasible_flag)."""
    feasible = [r for r in results if r["false_rate"] <= max_false_rate]
    pool = feasible or results
    key = (lambda r: (r[f"mrr@{k}"], r[f"hit@{k}"])) if feasible else (
        lambda r: (-r["false_rate"], r[f"mrr@{k}"]))
    return max(pool, key=key), bool(feasible)


def fmt_cut(v):
    return "off" if v is None else f"{v:g}"


def to_markdown(results, k=5, top=15, max_false_rate=0.17):
    best, ok = pick_best(results, max_false_rate, k)
    ranked = sorted(results, key=lambda r: (r["false_rate"] > max_false_rate, -r[f"mrr@{k}"], -r[f"hit@{k}"]))
    lines = [f"| max_distance | bm25_solo | rerank_min | Hit@1 | Hit@{k} | MRR@{k} | false retrieval |",
             "|---|---|---|---|---|---|---|"]
    for r in ranked[:top]:
        mark = " **<- pick**" if r is best else ""
        solo = "never" if r["bm25_solo"] >= 1e8 else f"{r['bm25_solo']:g}"
        lines.append(f"| {r['max_distance']:g} | {solo} | {fmt_cut(r['rerank_min'])} | {r['hit@1']:.2f} | "
                     f"{r[f'hit@{k}']:.2f} | {r[f'mrr@{k}']:.3f} | {r['false_rate']:.2f} "
                     f"({r['false_n']}/{r['n_oos']}){mark} |")
    return "\n".join(lines), best, ok


def env_lines(best):
    out = [f"BM25_SOLO_MIN_SCORE={best['bm25_solo']:g}"]
    if best["reranker"] and best["rerank_min"] is not None:
        out.append(f"RERANK_MIN_SCORE={best['rerank_min']:g}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=str(config.GOLD_SET_PATH))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-false-rate", type=float, default=0.17)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--out", default=str(config.EVAL_RESULTS_DIR))
    args = ap.parse_args()

    gold = load_gold(args.gold)
    searcher = HybridSearcher(use_reranker=not args.no_rerank)
    add_caches(searcher)

    cutoffs = None
    if not args.no_rerank:
        if searcher.reranker is None:
            print("Re-ranker could not be loaded; sweeping dense + BM25 gates only.")
        else:
            cutoffs = [None] + auto_cutoffs(searcher, gold, args.k)
            print("Re-ranker cutoffs tried:", [fmt_cut(c) for c in cutoffs])

    results = run_sweep(searcher, gold, DEFAULT_DISTANCES, DEFAULT_SOLO, cutoffs, args.k)
    table, best, ok = to_markdown(results, args.k, max_false_rate=args.max_false_rate)
    print("\n" + table)
    if not ok:
        print(f"\nNo setting reaches false retrieval <= {args.max_false_rate}; showing the lowest.")
    print("\nPut this in .env, and set MAX_DISTANCE in config.py:")
    print(f"MAX_DISTANCE = {best['max_distance']:g}   (config.py)")
    for line in env_lines(best):
        print(line)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.md").write_text(table + "\n", encoding="utf-8")
    with open(out / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {out/'sweep.md'} and {out/'sweep.csv'}")


if __name__ == "__main__":
    main()
