
from pathlib import Path
import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def extract_page_text(page):
    """
    Extract text from a PDF page while attempting to preserve
    the reading order of two-column documents.
    """
    blocks = page.get_text("blocks", sort=True)
    text_blocks = []

    for block in blocks:
        # PyMuPDF text blocks contain coordinates and text.
        if len(block) < 5:
            continue

        x0, y0, x1, y1, block_text = block[:5]

        if not isinstance(block_text, str):
            continue

        block_text = block_text.strip()

        if not block_text:
            continue

        text_blocks.append({
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "text": block_text,
        })

    if not text_blocks:
        return ""

    page_width = page.rect.width

    # Detect a possible two-column layout using block positions.
    left_blocks = [
        block for block in text_blocks
        if block["x1"] <= page_width * 0.55
    ]

    right_blocks = [
        block for block in text_blocks
        if block["x0"] >= page_width * 0.45
    ]

    # Use column-by-column ordering when there is evidence of
    # two distinct columns. Otherwise, preserve normal page order.
    if (
        len(left_blocks) >= 3
        and len(right_blocks) >= 3
        and len(left_blocks) + len(right_blocks) >= len(text_blocks)
    ):
        ordered_blocks = sorted(
            left_blocks, key=lambda block: (block["y0"], block["x0"])
        ) + sorted(
            right_blocks, key=lambda block: (block["y0"], block["x0"])
        )
    else:
        ordered_blocks = sorted(
            text_blocks, key=lambda block: (block["y0"], block["x0"])
        )

    return "\n\n".join(
        block["text"] for block in ordered_blocks
    ).strip()


def load_pdf(pdf_path):
    """
    Extract non-empty text from a PDF while preserving
    the original PDF page numbers.
    """
    pdf_path = Path(pdf_path)
    pages = []

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The provided file is not a PDF.")

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = extract_page_text(page)

            if text:
                pages.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                })

    return pages


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF found in: {DATA_DIR}")
    else:
        pdf_path = pdf_files[0]

        try:
            documents = load_pdf(pdf_path)

            print(f"PDF: {pdf_path.name}")
            print(f"Pages containing text: {len(documents)}")

            for document in documents:
                text = document["text"]
                lower_text = text.casefold()

                if any(
                    term in lower_text
                    for term in (
                        "conclusion",
                        "future work",
                        "future research",
                        "broader data support",
                    )
                ):
                    print(
                        f"\n--- Page {document['page']} ---"
                    )
                    print(text[:2000])

            empty_count = 0

            with fitz.open(pdf_path) as pdf:
                for page in pdf:
                    if not extract_page_text(page):
                        empty_count += 1

            if empty_count:
                print(
                    f"\nWarning: {empty_count} pages have no "
                    "extractable text."
                )

        except Exception as error:
            print(f"PDF extraction failed: {error}")
