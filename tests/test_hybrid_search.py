import config


def ids(results):
    return [r["id"] for r in results]


def test_bm25_rescues_exact_keyword_when_dense_gate_rejects_everything(searcher):
    # dense gate so strict that nothing passes -> only BM25 can find the chunk
    fixed = searcher.search("zxqvwidget", max_distance=1e-9, bm25_solo_min_score=0)
    assert ids(fixed) == ["t5"]
    # the OLD behaviour (BM25 only counts if dense already returned it) finds nothing
    old = searcher.search("zxqvwidget", max_distance=1e-9, bm25_min_score=0, gate_bm25=True)
    assert old == []


def test_bm25_min_score_blocks_weak_keyword_matches(searcher):
    assert searcher.search("pressure", max_distance=1e-9, bm25_solo_min_score=1e6) == []


def test_modes(searcher):
    dense = searcher.search("moment arm axis rotation", mode="dense", max_distance=None)
    bm25 = searcher.search("moment arm axis rotation", mode="bm25", bm25_min_score=0)
    assert dense[0]["retrieval"]["dense_distance"] is not None and dense[0]["retrieval"]["bm25_score"] is None
    assert bm25[0]["retrieval"]["bm25_score"] is not None and bm25[0]["retrieval"]["dense_distance"] is None
    assert dense[0]["id"] == bm25[0]["id"] == "t3"


def test_filters_apply_to_both_retrievers(searcher):
    res = searcher.search("pressure", where_filter={"source_type": "past_paper"}, max_distance=None, bm25_min_score=0)
    assert res and all(r["metadata"]["source_type"] == "past_paper" for r in res)
    res = searcher.search("torque", where_filter={"chapter": 4, "source_type": "textbook"}, max_distance=None, bm25_min_score=0)
    assert ids(res) == ["t3"]


def test_results_are_copies_with_retrieval_info(searcher):
    res = searcher.search("pressure liquid", max_distance=None, bm25_min_score=0)
    assert "retrieval" in res[0]
    assert "retrieval" not in searcher.id_to_chunk[res[0]["id"]]  # original untouched


def test_out_of_scope_query_returns_nothing_with_default_gates(searcher):
    assert searcher.search("how to bake a chocolate cake", max_distance=config.MAX_DISTANCE) == []


def test_reranker_reorders_and_is_optional(searcher):
    class ReverseByLength:  # fake cross-encoder: prefers longer text
        def score(self, query, texts):
            return [float(len(t)) for t in texts]

    base = searcher.search("pressure", max_distance=None, bm25_min_score=0, n_results=4)
    searcher._reranker = ReverseByLength()
    reranked = searcher.search("pressure", max_distance=None, bm25_min_score=0, n_results=4, rerank=True)
    lengths = [len(r["text"]) for r in reranked]
    assert lengths == sorted(lengths, reverse=True)
    assert reranked[0]["retrieval"]["rerank_score"] is not None
    # the re-ranker looks at a wider candidate pool than the final top-n
    assert len(reranked) == len(base) == 4


def test_old_api_still_works(searcher):
    assert searcher.hybrid_search("torque moment arm", n_results=2, max_distance=None)
    assert isinstance(searcher.vector_search("torque", max_distance=None)[0], str)


def test_solo_bm25_needs_a_higher_score_than_chunks_dense_also_found(searcher):
    # dense gate off -> every BM25 hit is "supported" by dense, so any positive score passes
    assert ids(searcher.search("zxqvwidget", max_distance=None, bm25_solo_min_score=1e6))[:1] == ["t5"]
    # dense gate closed -> BM25 is on its own and must clear the solo bar
    assert searcher.search("zxqvwidget", max_distance=1e-9, bm25_solo_min_score=1e6) == []
    assert ids(searcher.search("zxqvwidget", max_distance=1e-9, bm25_solo_min_score=0)) == ["t5"]


def test_rerank_min_score_drops_weak_chunks_only_when_reranker_ran(searcher):
    class ByLength:
        def score(self, query, texts):
            return [float(len(t)) for t in texts]

    searcher._reranker = ByLength()
    allr = searcher.search("pressure", max_distance=None, n_results=5, rerank=True)
    cut = sorted(r["retrieval"]["rerank_score"] for r in allr)[len(allr) // 2]
    kept = searcher.search("pressure", max_distance=None, n_results=5, rerank=True, rerank_min_score=cut)
    assert kept and all(r["retrieval"]["rerank_score"] >= cut for r in kept) and len(kept) < len(allr)
    assert searcher.search("pressure", max_distance=None, rerank=True, rerank_min_score=1e9) == []
    # reranker off -> the cutoff is ignored (there are no cross-encoder scores to compare)
    assert searcher.search("pressure", max_distance=None, rerank=False, rerank_min_score=1e9)


def test_default_solo_bar_is_stricter_without_the_reranker(searcher, monkeypatch):
    monkeypatch.setattr(config, "BM25_SOLO_MIN_SCORE", 0.0)
    monkeypatch.setattr(config, "BM25_SOLO_MIN_SCORE_NO_RERANK", 1e9)

    class Fake:
        def score(self, query, texts):
            return [1.0] * len(texts)

    # no re-ranker: strict bar, BM25 cannot rescue anything when the dense gate is closed
    assert searcher.search("zxqvwidget", max_distance=1e-9, rerank=False) == []
    # re-ranker available: the lenient bar applies (the re-ranker is the safety net)
    searcher._reranker = Fake()
    assert ids(searcher.search("zxqvwidget", max_distance=1e-9, rerank=True)) == ["t5"]
