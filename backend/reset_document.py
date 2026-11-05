
from pathlib import Path

from backend.src.answer import collection
from backend.src.pdf_loader import DATA_DIR

filename = "research_paper.pdf"

# Remove the PDF's indexed chunks from ChromaDB.
collection.delete(
    where={"filename": filename}
)

# Remove the PDF file.
pdf_path = DATA_DIR / filename

if pdf_path.exists():
    pdf_path.unlink()
    print(f"Deleted PDF: {pdf_path}")
else:
    print("PDF file was not found.")

print("Remaining indexed chunks:", collection.count())
print("Reset complete.")
