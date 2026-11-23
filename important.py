"""
"Important questions" route.

When a student asks for *important / expected / repeated* questions, the honest answer
is a frequency count over the past papers we actually have - not something the LLM
guesses. This module therefore answers from the data directly (no LLM call, so nothing
can be hallucinated):

  1. detect the intent (``detect_important_intent``),
  2. take every past-paper chunk matching the chapter / question type,
  3. group near-duplicate questions (token Jaccard) across different papers,
  4. report what repeated, what came once, and how thin the data is.
"""
import re

IMPORTANT_RE = re.compile(
    r"\b(imp|imp\.|important|importance|expected|expect|guess|guess\s*paper|likely|repeat(?:ed|s)?|"
    r"baar\s+baar|bar\s+bar|frequently|frequent)\b",
    re.IGNORECASE,
)
QTYPE_RE = re.compile(r"\b(mcqs?|short|long)\b", re.IGNORECASE)

JACCARD_THRESHOLD = 0.6
MIN_PAPERS_FOR_PATTERN = 2     # a question must appear in >=2 different papers to count as "repeated"
THIN_CHAPTER_CHUNKS = 5        # fewer past-paper questions than this -> say the data is too thin
MAX_ONCE_SHOWN = 12

_STOP = frozenset(
    "what define the and write with for are its from that this state explain give "
    "differentiate between which following is of in to a an by on as or".split()
)


def detect_important_intent(query, qa_pairs=None):
    """True if the message asks for important/expected questions.

    A short follow-up that only names a question type ("or is chp k long questions")
    inherits the intent when the previous student message had it.
    """
    if IMPORTANT_RE.search(query or ""):
        return True
    if qa_pairs and QTYPE_RE.search(query or ""):
        last_user = qa_pairs[-1][0] if qa_pairs[-1] else ""
        return bool(IMPORTANT_RE.search(last_user or ""))
    return False


def _tokens(text):
    return {w for w in re.findall(r"[a-z]{3,}", text.lower())} - _STOP


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def clean_question(text, max_len=260):
    """Drop the leading question number ('5 ', '1.1 ', 'Q.3') and tidy whitespace."""
    t = re.sub(r"^\s*(?:Q\.?\s*)?\d+(?:\.\d+)?\s*[).:-]?\s*", "", text.strip())
    t = re.sub(r"\s+", " ", t.replace("\n", " ")).strip()
    t = re.sub(r"(?<=[.?])\s+\d{1,2}$", "", t)   # trailing marks column ("... lamina. 4")
    return t if len(t) <= max_len else t[: max_len - 1].rstrip() + "…"


def paper_key(meta):
    return (meta.get("year"), meta.get("group"))


def paper_label(key):
    return f"{key[0]} Gp {key[1]}"


def select_chunks(chunks, chapter=None, question_type=None):
    out = []
    for c in chunks:
        m = c["metadata"]
        if m.get("source_type") != "past_paper":
            continue
        if chapter is not None and m.get("chapter") != chapter:
            continue
        if question_type is not None and m.get("question_type") != question_type:
            continue
        out.append(c)
    return out


def group_questions(chunks, threshold=JACCARD_THRESHOLD):
    """Greedy clustering of near-identical questions (question types are NOT separated:
    "Define latent heat of fusion" as a short question in one paper and as a long question
    in another is one repeated topic).

    Returns a list of dicts {text, question_type, papers:set, chunk_ids} sorted by the
    number of distinct papers (most repeated first).
    """
    groups = []
    for c in chunks:
        m = c["metadata"]
        toks = _tokens(c["text"])
        for g in groups:
            if jaccard(toks, g["tokens"]) >= threshold:
                g["papers"].add(paper_key(m))
                g["chunk_ids"].append(c["id"])
                g["types"].add(m.get("question_type"))
                g["by_paper"].setdefault(paper_key(m), set()).add(m.get("question_type"))
                break
        else:
            groups.append({
                "text": clean_question(c["text"]), "question_type": m.get("question_type"),
                "tokens": toks, "papers": {paper_key(m)}, "chunk_ids": [c["id"]],
                "types": {m.get("question_type")}, "by_paper": {paper_key(m): {m.get("question_type")}},
            })
    groups.sort(key=lambda g: (-len(g["papers"]), g["text"]))
    return groups


def papers_available(chunks):
    return sorted({paper_key(c["metadata"]) for c in chunks})


MSG = {
    "english": {
        "title": "Important questions{ch}{qt} - based on past-paper frequency",
        "repeated": "Repeated in more than one paper",
        "once": "Appeared once",
        "in": "in",
        "more": "...and {n} more that appeared once.",
        "scope": "Data: {n_papers} papers ({papers}) of the Lahore Board, {n_q} questions in this selection. "
                 "\"Important\" here only means \"repeated in these papers\" - nothing is predicted.",
        "none_repeated": "No question in this selection repeated across papers.",
        "thin": "Only {n} past-paper question(s) exist for this selection, so I cannot judge importance from them.",
        "no_data": "I have no past-paper questions for this selection.",
        "none_of_type": "No {what} from chapter {ch} appeared in the {n_papers} past papers ({papers}). "
                        "What this chapter does have: MCQ={mcq}, short={short}, long={long} "
                        "({n} of {total} past-paper questions, {pct:.1f}%).",
        "mcq_note": "MCQ answer keys are not available, so these are listed without answers.",
        "book": "Textbook exercise questions are not counted here - ask for them separately; they are not past-paper evidence.",
    },
    "hinglish": {
        "title": "Important questions{ch}{qt} - past papers mein kitni baar aaye",
        "repeated": "Ek se zyada papers mein repeat hue",
        "once": "Sirf ek baar aaye",
        "in": "mein",
        "more": "...aur {n} sawal jo sirf ek baar aaye.",
        "scope": "Data: Lahore Board ke {n_papers} papers ({papers}), is selection mein {n_q} sawal. "
                 "Yahan \"important\" ka matlab sirf \"in papers mein repeat hua\" hai - koi andaza nahi lagaya.",
        "none_repeated": "Is selection mein koi sawal papers mein repeat nahi hua.",
        "thin": "Is selection ke sirf {n} past-paper sawal mere paas hain, is liye importance ka andaza nahi laga sakta.",
        "no_data": "Is selection ke past-paper sawal mere paas nahi hain.",
        "none_of_type": "{n_papers} past papers ({papers}) mein chapter {ch} ka koi {what} nahi aaya. "
                        "Is chapter mein jo aaya: MCQ={mcq}, short={short}, long={long} "
                        "({n} of {total} past-paper sawal, {pct:.1f}%).",
        "mcq_note": "MCQs ki answer key maujood nahi, is liye jawab ke baghair likhe hain.",
        "book": "Textbook ke exercise sawal yahan count nahi hue - woh alag se pooch lo; woh past-paper ka saboot nahi hain.",
    },
}
QTYPE_NAME = {"MCQ": "MCQs", "short": "short questions", "long": "long questions"}


def _fmt(g):
    papers = []
    for p in sorted(g["papers"]):
        tag = ""
        if len(g["types"]) > 1:   # a topic asked in different forms: say which form each paper used
            tag = " [" + "/".join(sorted(g["by_paper"][p])) + "]"
        papers.append(paper_label(p) + tag)
    return f"- {g['text']} ({', '.join(papers)})"


def answer_important(chunks, chapter=None, question_type=None, language="english"):
    """Build the full answer text (deterministic, no LLM)."""
    m = MSG["hinglish" if language == "hinglish" else "english"]
    ch = f" - Chapter {chapter}" if chapter is not None else ""
    qt = f" ({QTYPE_NAME.get(question_type, question_type)})" if question_type else ""
    title = "**" + m["title"].format(ch=ch, qt=qt) + "**"

    sel = select_chunks(chunks, chapter, question_type)
    if not sel:
        in_chapter = select_chunks(chunks, chapter, None) if chapter is not None else []
        if in_chapter and question_type:
            every = select_chunks(chunks, None, None)
            pk = papers_available(every)
            count = lambda t: sum(c["metadata"].get("question_type") == t for c in in_chapter)
            return title + "\n\n" + m["none_of_type"].format(
                what=QTYPE_NAME[question_type], ch=chapter, n_papers=len(pk),
                papers=", ".join(paper_label(p) for p in pk), mcq=count("MCQ"), short=count("short"),
                long=count("long"), n=len(in_chapter), total=len(every), pct=100 * len(in_chapter) / len(every))
        return title + "\n\n" + m["no_data"]

    # group across ALL question types of the selection's chapter, then keep the groups that
    # contain the requested type
    pool = select_chunks(chunks, chapter, None)
    groups = group_questions(pool)
    if question_type:
        groups = [g for g in groups if question_type in g["types"]]

    if len(sel) < THIN_CHAPTER_CHUNKS:
        lines = [title, "", m["thin"].format(n=len(sel)), ""]
        lines += [_fmt(g) for g in groups]
        return "\n".join(lines)

    repeated = [g for g in groups if len(g["papers"]) >= MIN_PAPERS_FOR_PATTERN]
    once = [g for g in groups if len(g["papers"]) < MIN_PAPERS_FOR_PATTERN]
    fmt = _fmt

    lines = [title, "", f"**{m['repeated']}**"]
    lines += [fmt(g) for g in repeated] or [m["none_repeated"]]
    lines += ["", f"**{m['once']}**"]
    lines += [fmt(g) for g in once[:MAX_ONCE_SHOWN]]
    if len(once) > MAX_ONCE_SHOWN:
        lines.append(m["more"].format(n=len(once) - MAX_ONCE_SHOWN))
    pk = papers_available(sel)
    lines += ["", m["scope"].format(n_papers=len(pk), papers=", ".join(paper_label(p) for p in pk), n_q=len(sel))]
    if question_type in (None, "MCQ"):
        lines.append(m["mcq_note"])
    lines.append(m["book"])
    return "\n".join(lines)
