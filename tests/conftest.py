import hashlib
import math
import re
import uuid

import pytest


def fake_embed(text, dim=64):
    """Deterministic bag-of-words hashing embedder - no model download needed."""
    vec = [0.0] * dim
    for w in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def make_chunk(cid, text, **meta):
    base = {"source_type": "textbook", "chapter": 1, "page": 1}
    base.update(meta)
    return {"id": cid, "text": text, "metadata": base}


@pytest.fixture
def corpus():
    return [
        make_chunk("t1", "Pressure is the force exerted normally on unit area of a surface", chapter=6, page=132),
        make_chunk("t2", "Pascal law states that pressure applied to a liquid is transmitted equally", chapter=6, page=140),
        make_chunk("t3", "Torque is the product of force and moment arm about the axis of rotation", chapter=4, page=83),
        make_chunk("t4", "Kinetic energy is the energy possessed by a body due to its motion", chapter=5, page=105),
        make_chunk("t5", "A zxqvwidget is an extremely rare apparatus described only here", chapter=9, page=190),
        make_chunk("p1", "Define torque. What is its SI unit?", source_type="past_paper", chapter=4, question_type="short", year=2024, group="I", paper_type="subjective"),
        make_chunk("p2", "8 The SI unit of pressure is: A) Pascal B) Newton C) Joule D) Watt", source_type="past_paper", chapter=6, question_type="MCQ", year=2023, group="II", paper_type="objective"),
    ]


@pytest.fixture
def searcher(corpus):
    import chromadb
    from hybrid_search import HybridSearcher

    client = chromadb.EphemeralClient()
    col = client.create_collection(name=f"test_{uuid.uuid4().hex[:8]}")
    col.add(
        ids=[c["id"] for c in corpus],
        documents=[c["text"] for c in corpus],
        embeddings=[fake_embed(c["text"]) for c in corpus],
        metadatas=[c["metadata"] for c in corpus],
    )
    return HybridSearcher(chunks=corpus, collection=col, embed_fn=fake_embed, use_reranker=False)
