import gradio as gr
from qa import ask_stream

CHAPTERS = {
    "All Chapters": None,
    "1. Physical Quantities and Measurements": 1,
    "2. Kinematics": 2,
    "3. Dynamics": 3,
    "4. Turning Effects of Force": 4,
    "5. Work, Energy and Power": 5,
    "6. Mechanical Properties of Matter": 6,
    "7. Thermal Properties of Matter": 7,
    "8. Magnetism": 8,
    "9. Nature of Science": 9,
} 

SOURCE_TYPES = {
    "Textbook + Past Papers": None,
    "Textbook Only": "textbook",
    "Past Papers Only": "past_paper",
}

QUESTION_TYPES = {
    "All Types": None,
    "MCQs Only": "MCQ",
    "Short Questions Only": "short",
    "Long Questions Only": "long",
}


def build_filter(chapter_label, source_label, qtype_label):
    filters = {}
    chapter = CHAPTERS[chapter_label]
    if chapter is not None:
        filters["chapter"] = chapter
    source_type = SOURCE_TYPES[source_label]
    if source_type is not None:
        filters["source_type"] = source_type
    qtype = QUESTION_TYPES[qtype_label]
    if qtype is not None:
        filters["question_type"] = qtype
    return filters


def extract_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return extract_text(value.get("text", value.get("content", "")))
    if isinstance(value, (list, tuple)):
        parts = [extract_text(item) for item in value]
        return " ".join(p for p in parts if p)
    return str(value)


def get_user_messages_from_history(history):
    user_msgs = []
    if not history:
        return user_msgs

    if history and isinstance(history[0], dict):
        for turn in history:
            if turn.get("role") == "user":
                user_msgs.append(extract_text(turn.get("content", "")))
        return user_msgs

    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) >= 1:
            user_msgs.append(extract_text(turn[0]))

    return user_msgs


def get_last_qa_pairs(history, n=2):
    pairs = []
    if not history:
        return pairs

    if history and isinstance(history[0], dict):
        pending_user = None
        for turn in history:
            role = turn.get("role")
            content = extract_text(turn.get("content", ""))
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                pairs.append((pending_user, content))
                pending_user = None
        return pairs[-n:]

    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            pairs.append((extract_text(turn[0]), extract_text(turn[1])))

    return pairs[-n:]


def chat_fn(message, history, chapter_label, source_label, qtype_label):
    dropdown_filter = build_filter(chapter_label, source_label, qtype_label)
    where_filter = dropdown_filter if dropdown_filter else None

    # drop the "Sources used" footer so it is not fed back into the prompts
    qa_pairs = [(u, a.split("\n\n---\n**Sources used**")[0]) for u, a in get_last_qa_pairs(history, n=2)]

    instruction_extra = ""
    if qa_pairs:
        conv_text = ""
        for u, a in qa_pairs:
            conv_text += "Student asked: " + u + "\n"
            conv_text += "You answered: " + a + "\n\n"
        instruction_extra = "CONVERSATION SO FAR (for context):\n" + conv_text

    try:
        yield from ask_stream(
            message,
            where_filter=where_filter,
            instruction_extra=instruction_extra,
            lang_query=message,
            display_query=message,
            qa_pairs=qa_pairs,
        )
    except Exception as e:
        yield "An error occurred: " + str(e)

with gr.Blocks(title="Pak Edu RAG — Class 9 Physics") as demo:
    gr.Markdown("""
    # 📘 Pak Edu RAG — Class 9 Physics (Lahore Board)
    Ask questions in English or Hinglish — answers are grounded in the textbook and past papers, with the sources listed under every answer.
    """)

    with gr.Row():
        chapter_dropdown = gr.Dropdown(
            choices=list(CHAPTERS.keys()),
            value="All Chapters",
            label="Filter by Chapter"
        )
        source_dropdown = gr.Dropdown(
            choices=list(SOURCE_TYPES.keys()),
            value="Textbook + Past Papers",
            label="Filter by Source"
        )
        qtype_dropdown = gr.Dropdown(
            choices=list(QUESTION_TYPES.keys()),
            value="All Types",
            label="Filter by Question Type (past papers only)"
        )

    chatbot = gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[chapter_dropdown, source_dropdown, qtype_dropdown],
        examples=[
            ["What is Pascal's law?"],
            ["Pascal's law kya hai?"],
            ["Show me short questions from Chapter 4 past papers"],
            ["Explain Newton's first law in easy words"],
        ],
        title=None,
    )

if __name__ == "__main__":
    demo.launch()