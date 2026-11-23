"""
Question answering on top of HybridSearcher + Gemini.

One shared pipeline (`prepare`) feeds both `ask` (blocking) and `ask_stream`
(streaming), so the retrieval logic exists exactly once.
"""
import logging
import re
import time
from dataclasses import dataclass, field

import config
import important
from hybrid_search import HybridSearcher

log = logging.getLogger(__name__)

MODEL = config.LLM_MODEL

_client = None
_searcher = None  # module-level caches, created once


def get_client():
    """Gemini client, created lazily so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = config.get_client()
    return _client


def get_searcher():
    """HybridSearcher, created lazily and reused across calls."""
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher


# ------------------------------------------------------------------ language
HINGLISH_MARKERS = {
    "hai", "hain", "kya", "kyun", "kyu", "kaise", "mein", "ka", "ki", "ke", "wala",
    "batao", "samjhao", "kar", "karo", "kro", "nahi", "nai", "acha", "theek",
    "yeh", "woh", "aap", "tum", "mujhe", "humein", "kahan", "kab", "kitna",
    "asaan", "matlab", "chahiye", "raha", "rahi", "rha", "ko", "se", "bhi", "aur",
}


def detect_language(text):
    """
    Heuristic: Roman-Urdu/Hinglish if the text has >=2 marker words, or 1 marker
    in a short message ("Pascal's law kya hai?"). Otherwise English.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    hits = [w for w in words if w in HINGLISH_MARKERS]
    if len(hits) >= 2 or (len(hits) == 1 and len(words) <= 6):
        return "hinglish"
    return "english"


NOT_FOUND = {
    "english": "I could not find this topic in the textbook or the past papers.",
    "hinglish": "Ye topic textbook ya past papers mein nahi mila.",
}


# ------------------------------------------------------------------- filters
def extract_implicit_filters(broad_query, current_query=None):
    """
    Detect chapter / question-type filters from the query text.
    Chapter comes from the broader (history-inclusive) query since it tends to
    stay constant over a conversation. Question type is read ONLY from the
    current message so it does not "stick" to later turns.
    """
    if current_query is None:
        current_query = broad_query

    filters = {}
    chapter_match = re.search(
        r"chapter\s+(\d+)|chp\.?\s+(\d+)|\bch\.?\s+(\d+)", broad_query, re.IGNORECASE
    )
    if chapter_match:
        filters["chapter"] = int(next(g for g in chapter_match.groups() if g is not None))

    q = current_query.lower()
    if re.search(r"\bmcqs?\b", q):
        filters["question_type"] = "MCQ"
    elif re.search(r"\blong\s+(questions?|q)\b", q):
        filters["question_type"] = "long"
    elif re.search(r"\bshort\s+(questions?|q)\b", q):
        filters["question_type"] = "short"

    return filters or None


# ------------------------------------------------------------------ prompting
def source_label(meta):
    if meta.get("source_type") == "textbook":
        return f"Textbook, Chapter {meta.get('chapter')}, Page {meta.get('page')}"
    return (
        f"Past Paper {meta.get('year')} Group {meta.get('group')} "
        f"({meta.get('paper_type')}, {meta.get('question_type')})"
    )


def format_context(chunks):
    return "\n\n".join(
        f"[Source {i} — {source_label(c['metadata'])}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


def format_sources(chunks):
    """Markdown list shown under an answer; numbering matches the [Source n] tags."""
    if not chunks:
        return ""
    lines = [f"{i}. {source_label(c['metadata'])}" for i, c in enumerate(chunks, 1)]
    return "\n\n---\n**Sources used**\n" + "\n".join(lines)


def build_prompt(query, context, instruction_extra="", language="english"):
    lang_instruction = (
        "Respond in Hinglish (Roman Urdu/Hindi mixed with English), matching how the student wrote their question."
        if language == "hinglish"
        else "Respond in plain English."
    )
    return f"""You are a study tutor for Class 9 Physics (Lahore Board / Punjab Textbook Board), helping a student prepare for their board exams.

STRICT RULES:
1. For factual content (definitions, explanations, concepts), answer ONLY using the information in the sources below. Do not add outside facts, even if you know them.
2. Cite the supporting source after each factual statement as [Source n], using the numbers in the sources below.
3. For MCQ correct answers: FIRST check whether the sources state the correct option directly. If they do, give it and start the line with "Answer (from sources):". If they do NOT (Lahore Board past papers publish no answer key), solve it yourself with your Physics knowledge, give a definite answer, and start the line with "Answer (solved by tutor, not stated in sources):" so the student knows the difference.
4. If the sources do not cover the topic at all (a missing answer key is not the same as a missing topic), say so honestly instead of guessing.
5. You MAY reorganize, summarize, or add headings/structure for readability.
6. {lang_instruction}

{instruction_extra}

SOURCES:
{context}

STUDENT'S QUESTION: {query}

Answer:"""


def _generate(prompt, retries=3):
    """Blocking Gemini call with exponential back-off."""
    last = None
    for attempt in range(retries):
        try:
            return get_client().models.generate_content(model=MODEL, contents=prompt).text
        except Exception as exc:  # rate limit, network, ...
            last = exc
            log.warning("Gemini call failed (%s); retry %d/%d", exc, attempt + 1, retries)
            time.sleep(2 ** attempt)
    raise last


def rewrite_query(message, qa_pairs=None, language="english"):
    """
    Turn a follow-up ("explain them too") into a standalone ENGLISH question.
    The embedding model (bge-small-en) is English-only, so Roman-Urdu questions
    are also normalised to English here. Falls back to the original message on
    any failure.
    """
    if not qa_pairs and language == "english":
        return message  # nothing to resolve, nothing to translate

    history_text = "".join(f"Student: {u}\nTutor: {a}\n\n" for u, a in (qa_pairs or []))
    prompt = f"""Rewrite the new student message as ONE fully standalone English question that makes sense without the conversation history.
- Keep chapter / topic / question-type references explicit.
- If the message is in Roman Urdu or Hinglish, translate it to English.
- If it is already standalone English, return it unchanged.
Reply with ONLY the rewritten question.

CONVERSATION HISTORY:
{history_text or '(none)'}

NEW MESSAGE: {message}

Standalone English question:"""
    try:
        rewritten = (_generate(prompt, retries=2) or "").strip()
        return rewritten or message
    except Exception:
        return message


# ------------------------------------------------------------------- pipeline
@dataclass
class Prepared:
    effective_query: str
    language: str
    where_filter: dict = None
    chunks: list = field(default_factory=list)
    prompt: str = ""
    direct_answer: str = None   # set by the "important questions" route: answered from data, no LLM


LIST_ALL_HINTS = ("all mcqs", "saare mcq", "sare mcq", "all questions", "saare question", "complete list")


def prepare(query, n_results=5, where_filter=None, instruction_extra="", max_distance=None,
            lang_query=None, qa_pairs=None, searcher=None):
    """Language detection -> query rewriting -> filters -> retrieval -> prompt."""
    searcher = searcher or get_searcher()
    max_distance = config.MAX_DISTANCE if max_distance is None else max_distance

    language = detect_language(lang_query if lang_query else query)
    effective_query = rewrite_query(query, qa_pairs, language)

    # --- "important / expected questions": answer from past-paper frequency, not from the LLM
    if important.detect_important_intent(query, qa_pairs):
        flt = dict(where_filter or {})
        implicit = extract_implicit_filters(effective_query, current_query=effective_query) or {}
        if "chapter" not in flt and "chapter" not in implicit:
            for u, _ in reversed(qa_pairs or []):  # chapter stays constant over a conversation
                implicit_prev = extract_implicit_filters(u) or {}
                if "chapter" in implicit_prev:
                    implicit["chapter"] = implicit_prev["chapter"]
                    break
        flt = {**implicit, **flt}
        direct = important.answer_important(
            searcher.all_chunks, chapter=flt.get("chapter"), question_type=flt.get("question_type"),
            language=language,
        )
        return Prepared(effective_query, language, flt or None, [], "", direct_answer=direct)

    if any(h in effective_query.lower() for h in LIST_ALL_HINTS):
        n_results = 25  # complete sets (e.g. all 12 MCQs of a paper) must not be cut short
    elif len(effective_query.split()) > 15:
        n_results = min(n_results * 2, 25)  # multi-topic follow-ups need more chunks

    if where_filter is None:
        where_filter = extract_implicit_filters(effective_query, current_query=effective_query)

    # Distance gate:
    #  * no filter            -> strict gate (out-of-scope questions retrieve nothing)
    #  * chapter / source     -> relaxed gate (still rejects unrelated questions)
    #  * question_type filter -> no gate: "show MCQs from chapter 4" is a browse request
    #    with little semantic overlap, so ranking decides, as in the first version.
    if not where_filter:
        gate = max_distance
    elif "question_type" in where_filter:
        gate = None
    else:
        gate = config.RELAXED_MAX_DISTANCE
    # Cross-encoder cutoff only for real questions, not for "browse" requests (no gate).
    rr_min = config.RERANK_MIN_SCORE if gate is not None else None
    chunks = searcher.search(effective_query, n_results=n_results, where_filter=where_filter,
                             max_distance=gate, rerank_min_score=rr_min)

    if not chunks and where_filter:  # filter too narrow: retry across everything, strict gate
        chunks = searcher.search(effective_query, n_results=n_results, max_distance=max_distance,
                                 rerank_min_score=config.RERANK_MIN_SCORE)

    prompt = ""
    if chunks:
        prompt = build_prompt(effective_query, format_context(chunks), instruction_extra, language)
    return Prepared(effective_query, language, where_filter, chunks, prompt)


def ask(query, n_results=5, where_filter=None, instruction_extra="", max_distance=None,
        lang_query=None, display_query=None, qa_pairs=None):
    """Blocking version. Returns (answer_text, chunks)."""
    p = prepare(query, n_results, where_filter, instruction_extra, max_distance, lang_query, qa_pairs)
    if p.direct_answer:
        return p.direct_answer, []
    if not p.chunks:
        return NOT_FOUND[p.language], []
    return _generate(p.prompt), p.chunks


def ask_stream(query, n_results=5, where_filter=None, instruction_extra="", max_distance=None,
               lang_query=None, display_query=None, qa_pairs=None):
    """
    Streaming version for the UI: yields the answer-so-far, and finally the full
    answer followed by the list of sources that were used.
    """
    p = prepare(query, n_results, where_filter, instruction_extra, max_distance, lang_query, qa_pairs)
    if p.direct_answer:
        yield p.direct_answer
        return
    if not p.chunks:
        yield NOT_FOUND[p.language]
        return

    partial = ""
    for piece in get_client().models.generate_content_stream(model=MODEL, contents=p.prompt):
        if piece.text:
            partial += piece.text
            yield partial
    yield partial + format_sources(p.chunks)


if __name__ == "__main__":
    answer, chunks = ask("What is Pascal's law?")
    print(answer + format_sources(chunks))
