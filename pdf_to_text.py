import os
import re
import time
from pathlib import Path
from pdf2image import convert_from_path
from io import BytesIO
from config import get_client, LLM_MODEL
from google.genai import types

client = get_client()
MODEL = LLM_MODEL


def pdf_to_images(pdf_path, dpi=150):
    """Converts each page of a PDF into a PIL image"""
    return convert_from_path(pdf_path, dpi=dpi)


def image_to_bytes(pil_image):
    """Converts a PIL image to raw PNG bytes (needed to send to the API)"""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return buffered.getvalue()


def clean_ocr_output(text):
    """Removes any <think>...</think> reasoning block, keeps only the final answer"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def call_gemini(prompt, image_bytes, max_retries=3):
    """Sends a prompt + image to Gemini and returns cleaned text, with retries"""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ],
            )
            return clean_ocr_output(response.text)
        except Exception as e:
            print(f"    Error: {e}. Retrying in 10 seconds... (attempt {attempt+1}/{max_retries})")
            time.sleep(10)

    print("    Failed after retries, skipping this page.")
    return ""


def ocr_image(pil_image, is_objective=False):
    """Extracts text from a past-paper image (MCQ or subjective)"""
    img_bytes = image_to_bytes(pil_image)

    if is_objective:
        prompt = (
            "Extract all MCQ questions from this image exactly as written. "
            "Include question numbers, full question text, and all 4 options (A/B/C/D). "
            "Ignore watermarks or background text. Output as plain text."
        )
    else:
        prompt = (
            "Extract all text from this exam paper image exactly as written, "
            "including question numbers, marks, and formulas. "
            "Ignore watermarks or background text. Output as plain text, "
            "preserving the original structure (Part I, Part II, question numbers)."
        )

    return call_gemini(prompt, img_bytes)


def ocr_textbook_page(pil_image):
    """Extracts text from a single textbook page, preserving chapter/heading structure"""
    img_bytes = image_to_bytes(pil_image)

    prompt = (
        "Extract all text from this textbook page exactly as written, including "
        "headings, chapter titles, chapter numbers, formulas, and body text. "
        "If this page shows a chapter title or number (e.g. 'Chapter 12: ...'), "
        "clearly include it at the top of your output. "
        "Ignore page numbers, headers/footers, and any watermarks. "
        "Output as plain text, preserving paragraph and heading structure."
    )

    return call_gemini(prompt, img_bytes)


def process_pdf(pdf_path, output_dir):
    """Processes a single past-paper PDF: converts to images, runs OCR, saves clean text"""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_output = output_dir / f"{pdf_path.stem}.txt"

    if txt_output.exists() and txt_output.stat().st_size > 0:
        print(f"Skipping {pdf_path.name} — already OCR'd")
        return

    is_objective = "objective" in pdf_path.stem.lower()

    print(f"Processing {pdf_path.name}...")
    images = pdf_to_images(pdf_path)

    full_text = ""
    for i, img in enumerate(images):
        print(f"  OCR page {i+1}/{len(images)}...")
        page_text = ocr_image(img, is_objective=is_objective)
        full_text += f"\n--- Page {i+1} ---\n{page_text}\n"

    txt_output.write_text(full_text, encoding="utf-8")
    print(f"  Saved to {txt_output}")


def find_past_paper_pdfs(data_root="data"):
    """Finds all PDFs inside any folder named 'past_papers' (case-insensitive)"""
    data_root = Path(data_root)
    return [p for p in data_root.rglob("*.pdf") if p.parent.name.lower() == "past_papers"]


def find_textbook_pdfs(data_root="data"):
    """Finds all PDFs inside any folder named 'textbook' (case-insensitive)"""
    data_root = Path(data_root)
    return [p for p in data_root.rglob("*.pdf") if p.parent.name.lower() == "textbook"]


def process_textbook(pdf_path, output_dir):
    """Processes a textbook PDF: saves one .txt file per page (easier for chapter tagging later)"""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing textbook: {pdf_path.name}...")
    images = pdf_to_images(pdf_path)
    print(f"  Total pages: {len(images)}")

    for i, img in enumerate(images):
        page_num = i + 1
        txt_output = output_dir / f"page_{page_num:03d}.txt"

        if txt_output.exists() and txt_output.stat().st_size > 0:
            print(f"  Skipping page {page_num} — already OCR'd")
            continue

        print(f"  OCR page {page_num}/{len(images)}...")
        page_text = ocr_textbook_page(img)
        txt_output.write_text(page_text, encoding="utf-8")

    print(f"  Done. Saved pages to {output_dir}")


if __name__ == "__main__":
    # --- Past papers ---
    pdf_files = find_past_paper_pdfs("data")
    print(f"Found {len(pdf_files)} PDF(s) inside past_papers folder(s):\n")
    for p in pdf_files:
        print(f"  {p}")
    print()

    for pdf_path in pdf_files:
        output_dir = pdf_path.parent.parent / f"{pdf_path.parent.name}_ocr"
        process_pdf(pdf_path, output_dir)

    # --- Textbook ---
    textbook_files = find_textbook_pdfs("data")
    print(f"\nFound {len(textbook_files)} textbook PDF(s):\n")
    for pdf_path in textbook_files:
        output_dir = pdf_path.parent.parent / f"{pdf_path.parent.name}_ocr"
        process_textbook(pdf_path, output_dir)

    print("\nAll files processed (or already had cached OCR output).")