from pathlib import Path
import json
from backend.src.pdf_loader import DATA_DIR, load_pdf
from backend.src.chunker import chunk_pages
from backend.src.embedding_store import add_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


def main():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF found in the data folder.")
        return

    all_pages = []

    for pdf_file in pdf_files:
        pages = load_pdf(pdf_file)
        all_pages.extend(pages)
        print(f"Loaded {len(pages)} pages from {pdf_file.name}")

    chunks = chunk_pages(all_pages)

    OUTPUT_DIR.mkdir(exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
