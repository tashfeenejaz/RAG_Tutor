"""
Hybrid retrieval: dense (ChromaDB + BGE) + sparse (BM25), fused with
Reciprocal Rank Fusion (RRF), optionally re-ranked by a cross-encoder.

Changes vs. the first version
-----------------------------
* BM25 is now an *independent* retriever. Before, a BM25 hit only counted if the
  dense retriever had already returned it, so BM25 could re-order but never
  rescue an exact-keyword match. The old behaviour is still available with
  ``gate_bm25=True`` so it can be measured in the ablation (see eval_retrieval.py).
* BM25-only candidates must reach ``BM25_MIN_SCORE`` so that off-topic queries
  still return nothing (needed for grounded refusals).
* Optional cross-encoder re-ranking of the fused candidates.
* ``mode`` ("dense" | "bm25" | "hybrid") for ablations, injectable embedder /
  collection / reranker for offline testing.
"""
import json
import logging
import re

import config

log = logging.getLogger(__name__)

# Small English stop-word list so BM25 is driven by content words
# ("what", "is", "the" would otherwise match almost every chunk).
STOPWORDS = frozenset(
    """a an the is are was were be been being of to in on at by for with from as and or
    but if then so than that this these those it its what which who whom whose when where
    why how do does did done can could should would will shall may might must not no yes
    me my we our you your he she they them his her their i give tell define explain state
    write short note about between""".split()
)


def load_jsonl(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def tokenize(text):
    """Lower-case word tokenizer for BM25, without stop-words."""
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in STOPWORDS]


def build_where_clause(filters):
    """
    Convert a simple {key: value, ...} dict to ChromaDB's where format:
    one condition is passed as-is, several are wrapped in $and.
    """
    if not filters:
        return None
    if len(filters) == 1:
        return filters
    return {"$and": [{k: v} for k, v in filters.items()]}


def matches_filter(chunk, where_filter):
    if not where_filter:
        return True
    return all(chunk["metadata"].get(k) == v for k, v in where_filter.items())


class CrossEncoderReranker:
    """Thin wrapper around fastembed's cross-encoder (loaded lazily)."""

    def __init__(self, model_name=None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name or config.RERANK_MODEL
        self.model = TextCrossEncoder(model_name=self.model_name)

    def score(self, query, texts):
        return [float(s) for s in self.model.rerank(query, texts)]


class HybridSearcher:
    def __init__(
        self,
        chunks=None,
        collection=None,
        embed_fn=None,
        reranker=None,
        use_reranker=None,
        chroma_path=None,
        collection_name=None,
    ):
        # --- chunks (shared by BM25 and the id -> chunk lookup) ---
        if chunks is None:
            chunks = load_jsonl(config.TEXTBOOK_CHUNKS) + load_jsonl(config.PAST_PAPER_CHUNKS)
        self.all_chunks = chunks
        self.chunk_ids = [c["id"] for c in chunks]
        self.id_to_chunk = {c["id"]: c for c in chunks}

        # --- dense side (lazy so BM25-only use / tests need no model download) ---
        self._collection = collection
        self._chroma_path = chroma_path or config.CHROMA_PATH
        self._collection_name = collection_name or config.COLLECTION_NAME
        self._embed_fn = embed_fn

        # --- sparse side ---
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(c["text"]) or ["_empty_"] for c in chunks])

        # --- re-ranker (optional, fail-soft) ---
        self._reranker = reranker
        self._use_reranker = config.USE_RERANKER if use_reranker is None else use_reranker
        self._reranker_failed = False

    # ------------------------------------------------------------ lazy parts
    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=self._chroma_path)
            self._collection = client.get_collection(name=self._collection_name)
        return self._collection

    @property
    def embed_fn(self):
        if self._embed_fn is None:
            from fastembed import TextEmbedding

            model = TextEmbedding(model_name=config.EMBED_MODEL)
            self._embed_fn = lambda q: list(model.embed([q]))[0].tolist()
        return self._embed_fn

    @property
    def reranker(self):
        if self._reranker is None and self._use_reranker and not self._reranker_failed:
            try:
                self._reranker = CrossEncoderReranker()
            except Exception as exc:  # no internet, unsupported model, ...
                log.warning("Re-ranker unavailable (%s); continuing without it.", exc)
                self._reranker_failed = True
        return self._reranker

    # ------------------------------------------------------------ retrievers
    def dense_candidates(self, query, n_results=None, where_filter=None, max_distance=None):
        """[(chunk_id, distance)] best first; distance above max_distance is dropped."""
        n_results = n_results or config.CANDIDATES
        res = self.collection.query(
            query_embeddings=[self.embed_fn(query)],
            n_results=n_results,
            where=build_where_clause(where_filter),
        )
        pairs = list(zip(res["ids"][0], res["distances"][0]))
        if max_distance is not None:
            pairs = [(cid, d) for cid, d in pairs if d <= max_distance]
        return pairs

    def bm25_candidates(self, query, n_results=None, where_filter=None, min_score=None):
        """[(chunk_id, bm25_score)] best first; only positive scores >= min_score."""
        n_results = n_results or config.CANDIDATES
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        scored = [
            (cid, float(s))
            for cid, s in zip(self.chunk_ids, scores)
            if s > 0 and matches_filter(self.id_to_chunk[cid], where_filter)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        if min_score:
            scored = [(cid, s) for cid, s in scored if s >= min_score]
        return scored[:n_results]

    # kept for backwards compatibility with the old API (ids only)
    def vector_search(self, query, n_results=20, where_filter=None, max_distance=None):
        return [c for c, _ in self.dense_candidates(query, n_results, where_filter, max_distance)]

    def bm25_search(self, query, n_results=20, where_filter=None):
        return [c for c, _ in self.bm25_candidates(query, n_results, where_filter)]

    # ------------------------------------------------------------ main entry
    def search(
        self,
        query,
        n_results=5,
        where_filter=None,
        max_distance=None,
        mode="hybrid",
        rerank=None,
        gate_bm25=False,
        bm25_min_score=None,
        bm25_solo_min_score=None,
        rerank_min_score=None,
        k=None,
    ):
        """
        Return up to ``n_results`` chunks (copies) with a ``retrieval`` dict that
        explains where each came from.

        mode        "hybrid" (dense + BM25 via RRF), "dense", or "bm25".
        rerank      None = use the searcher default; True/False to force.
        gate_bm25   reproduce the OLD behaviour: BM25 ranks only count for chunks
                    the dense retriever already returned (used for the ablation).
        max_distance dense distance gate (None = no gate).
        bm25_min_score       floor for BM25-only mode and for the old gated hybrid.
        bm25_solo_min_score  hybrid mode: a chunk found ONLY by BM25 needs this score;
                             chunks the dense retriever also returned need any score > 0.
                             None = config.BM25_SOLO_MIN_SCORE when the re-ranker runs,
                             config.BM25_SOLO_MIN_SCORE_NO_RERANK when it does not.
        rerank_min_score     drop re-ranked chunks below this cross-encoder score
                             (only when the re-ranker actually ran; None = no cutoff).
        """
        k = k or config.RRF_K
        bm25_min = config.BM25_MIN_SCORE if bm25_min_score is None else bm25_min_score
        dense, sparse = [], []

        if mode in ("hybrid", "dense"):
            dense = self.dense_candidates(query, where_filter=where_filter, max_distance=max_distance)
        use_rerank = self._use_reranker if rerank is None else rerank
        rerank_active = bool(use_rerank and mode != "bm25" and self.reranker is not None)
        dense_dist = dict(dense)
        if mode == "bm25" or (mode == "hybrid" and gate_bm25):
            sparse = self.bm25_candidates(query, where_filter=where_filter, min_score=bm25_min)
        elif mode == "hybrid":
            solo_min = bm25_solo_min_score
            if solo_min is None:  # the re-ranker is what filters weak BM25-only chunks; without it be stricter
                solo_min = config.BM25_SOLO_MIN_SCORE if rerank_active else config.BM25_SOLO_MIN_SCORE_NO_RERANK
            sparse = self.bm25_candidates(query, where_filter=where_filter)
            sparse = [(cid, s) for cid, s in sparse if cid in dense_dist or s >= solo_min]
        bm25_score = dict(sparse)
        if gate_bm25:
            sparse = [(cid, s) for cid, s in sparse if cid in dense_dist]

        rrf = {}
        for rank, (cid, _) in enumerate(dense):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)
        for rank, (cid, _) in enumerate(sparse):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

        rerank_scores = {}
        if use_rerank and ranked:
            reranker = self.reranker
            if reranker is not None:
                pool = ranked[: max(config.RERANK_CANDIDATES, n_results)]
                texts = [self.id_to_chunk[cid]["text"] for cid, _ in pool]
                for (cid, _), s in zip(pool, reranker.score(query, texts)):
                    rerank_scores[cid] = s
                ranked = sorted(
                    pool, key=lambda x: rerank_scores[x[0]], reverse=True
                )
                if rerank_min_score is not None:
                    ranked = [r for r in ranked if rerank_scores[r[0]] >= rerank_min_score]

        out = []
        for rank, (cid, score) in enumerate(ranked[:n_results], 1):
            chunk = dict(self.id_to_chunk[cid])
            chunk["retrieval"] = {
                "rank": rank,
                "rrf": round(score, 5),
                "dense_distance": dense_dist.get(cid),
                "bm25_score": bm25_score.get(cid),
                "rerank_score": rerank_scores.get(cid),
            }
            out.append(chunk)
        return out

    def hybrid_search(self, query, n_results=5, where_filter=None, k=60, max_distance=0.65):
        """Backwards-compatible wrapper around :meth:`search`."""
        return self.search(
            query, n_results=n_results, where_filter=where_filter, max_distance=max_distance, k=k
        )


if __name__ == "__main__":
    searcher = HybridSearcher()
    for q, f in [("Newton's law", {"question_type": "MCQ"}), ("centre of gravity", {"chapter": 4})]:
        print("=" * 60, f"\n{q!r} filter={f}\n" + "=" * 60)
        for r in searcher.search(q, where_filter=f, max_distance=config.MAX_DISTANCE):
            print(r["id"], r["metadata"], r["retrieval"])
            print("   ", r["text"][:160].replace("\n", " "))
