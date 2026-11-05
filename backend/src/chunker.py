
import re


def chunk_pages(pages, chunk_size=1200, overlap=200):
    """Split PDF pages into overlapping text chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page.get("text", "").strip()

        if not text:
            continue

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            if end < len(text):
                search_start = min(start + step // 2, end)

                boundary = text.rfind("\n\n", search_start, end)

                if boundary == -1:
                    boundary = max(
                        text.rfind(". ", search_start, end),
                        text.rfind("? ", search_start, end),
                        text.rfind("! ", search_start, end),
                    )

                if boundary != -1:
                    end = boundary + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks) + 1,
                    "source": page.get("source", "Unknown"),
                    "page": page.get("page", "Unknown"),
                    "text": chunk_text,
                })

            if end >= len(text):
                break

            start = max(start + 1, end - overlap)

    print(f"[CHUNKER] Created {len(chunks)} chunks.")
    return chunks
