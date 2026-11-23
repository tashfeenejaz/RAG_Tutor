"""
Central configuration for Pak_Edu_RAG.

Secrets (the Gemini API key) are read from the environment / a local .env file.
They must never be written in source code.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()  # reads a .env file in the working directory, if present
except ImportError:  # python-dotenv is optional at import time
    pass

# ---------------------------------------------------------------- paths
DATA_DIR = Path(os.getenv("PAKEDU_DATA_DIR", "data/class_9/Lahore_board/physics"))
TEXTBOOK_OCR_DIR = DATA_DIR / "textbook_ocr"
PAST_PAPERS_OCR_DIR = DATA_DIR / "past_papers_ocr"
TEXTBOOK_CHUNKS = DATA_DIR / "textbook_chunks.jsonl"
PAST_PAPER_CHUNKS = DATA_DIR / "past_paper_chunks.jsonl"
CHROMA_PATH = os.getenv("PAKEDU_CHROMA_PATH", "chroma_db")
COLLECTION_NAME = "physics_class9"
GOLD_SET_PATH = Path("eval/gold_set.jsonl")
EVAL_RESULTS_DIR = Path("eval/results")

# ---------------------------------------------------------------- models
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "1") not in ("0", "false", "False", "")

# ---------------------------------------------------------------- retrieval
# Chroma distance here is squared-L2 on normalised BGE vectors (Chroma default),
# i.e. cosine_distance = distance / 2. Lower is better.
MAX_DISTANCE = 0.55            # strict gate for open questions (swept: 0.50-0.70 within noise, 0.55 picked)
RELAXED_MAX_DISTANCE = 0.85    # used when the user filters by question type ("list all MCQs ...")
BM25_MIN_SCORE = float(os.getenv("BM25_MIN_SCORE", "6.0"))  # BM25-only candidates must reach this score
# In hybrid mode a chunk that ONLY BM25 found (dense did not return it) must reach this
# higher score. Chunks both retrievers agree on are accepted at any positive BM25 score.
# Raw BM25 scores of out-of-scope questions overlap with real hits (6-11), so the bar alone
# does not separate them; with dense-support chunks exempt, eval_sweep.py picked 6. Re-tune
# with eval_sweep.py whenever the gold set or the models change.
BM25_SOLO_MIN_SCORE = float(os.getenv("BM25_SOLO_MIN_SCORE", "6.0"))   # used together with the re-ranker
# Without the re-ranker nothing else filters weak BM25-only chunks (solo 6 gave 50% false
# retrieval in the ablation), so a stricter bar applies.
BM25_SOLO_MIN_SCORE_NO_RERANK = float(os.getenv("BM25_SOLO_MIN_SCORE_NO_RERANK", "12.0"))
# Cross-encoder cutoff: re-ranked chunks scoring below this are dropped (None = off).
# The default (-8.6) was picked by eval_sweep.py for Xenova/ms-marco-MiniLM-L-6-v2 and is
# meaningless for any other model, so it only applies when RERANK_MODEL is not overridden
# (set RERANK_MIN_SCORE explicitly if you change the model). Not applied to "browse" requests.
_rr = os.getenv("RERANK_MIN_SCORE", "").strip()
if _rr:
    RERANK_MIN_SCORE = float(_rr)
else:
    RERANK_MIN_SCORE = None if os.getenv("RERANK_MODEL") else -8.6
CANDIDATES = 20                # candidates taken from each retriever before fusion
RRF_K = 60
RERANK_CANDIDATES = 15         # how many fused candidates go to the cross-encoder


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and put your key there."
        )
    return key


def get_client():
    """Create a Gemini client from the environment (imported lazily)."""
    from google import genai

    return genai.Client(api_key=get_api_key())
