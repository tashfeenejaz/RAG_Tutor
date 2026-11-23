from pathlib import Path

def find_and_delete_corrupted_files(data_root="data"):
    data_root = Path(data_root)
    deleted = []

    for txt_file in data_root.rglob("*.txt"):
        content = txt_file.read_text(encoding="utf-8")
        if "<think" in content.lower():
            txt_file.unlink()  # delete the file
            deleted.append(txt_file)
            print(f"Deleted (corrupted): {txt_file}")

    print(f"\nTotal files deleted: {len(deleted)}")
    print("Re-run the OCR script (pdf_to_text.py) to regenerate these pages.")


if __name__ == "__main__":
    find_and_delete_corrupted_files("data")