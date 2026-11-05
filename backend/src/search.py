
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


# -------------------- Configuration --------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / ".chroma"

COLLECTION_NAME = "tracerag_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"

QUERY = "What future work does Section 6 propose for OpenML?"
TOP_K = 12
MAX_PASSAGE_CHARS = 1600

SECTION_PATTERN = re.compile(
    r"\b(conclusion|conclusions|future work|future research|"
    r"limitations|discussion|improvements|further work|"
    r"directions for future research|6\.1|6\.2|6\.3|6\.4)\b",
    re.IGNORECASE,
)

FUTURE_TERMS = (
    "future work",
    "future research",
    "improve openml",
    "improvements",
    "propose",
    "suggestions",
    "implement",
    "should be able",
    "in the future",
)


# -------------------- Search Helpers --------------------

def contains_relevant_heading(text: str) -> bool:
    """Check whether a passage contains a relevant heading."""
    return bool(SECTION_PATTERN.search(text))


def keyword_match_count(text: str, query: str) -> int:
    """Count distinct meaningful query words in a passage."""
    stop_words = {
        "what", "does", "the", "for", "and", "are", "is",
        "of", "in", "to", "a", "an", "on", "with", "by",
        "about", "section",
    }

    query_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z]{3,}\b", query)
        if word.lower() not in stop_words
    }

    text_lower = text.lower()
    return sum(word in text_lower for word in query_words)


def future_term_matches(text: str) -> int:
    """Count future-work terms found in a passage."""
    text_lower = text.lower()
    return sum(term in text_lower for term in FUTURE_TERMS)


# -------------------- Main --------------------

def main():
    if not DB_DIR.exists():
        print(f"Vector database not found: {DB_DIR}")
        print("Upload and index a PDF before searching.")
        return

    print(f"Database: {DB_DIR}")
    print(f"Query: {QUERY}")
    print("-" * 70)

    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        print(f"Could not open collection: {exc}")
        print("Check whether the PDF has been indexed.")
        return

    total = collection.count()
    print(f"Collection size: {total}")

    if total == 0:
        print("The collection is empty. Index a PDF first.")
        return

    try:
        print(f"Loading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)

        query_embedding = model.encode(
            QUERY,
            normalize_embeddings=True,
        ).tolist()
    except Exception as exc:
        print(f"Could not load the embedding model: {exc}")
        return

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K, total),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        print(f"Search failed: {exc}")
        return

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No passages were retrieved.")
        return

    ranked_results = []

    for index, document in enumerate(documents):
        metadata = metadatas[index] or {}
        distance = distances[index] if distances else None

        heading_found = contains_relevant_heading(document)
        keyword_matches = keyword_match_count(document, QUERY)
        future_matches = future_term_matches(document)

        # This is a diagnostic ranking, not a probability.
        # Lower scores rank higher.
        diagnostic_score = (
            float(distance)
            - min(keyword_matches, 3) * 0.015
            - min(future_matches, 4) * 0.025
            - (0.05 if heading_found else 0.0)
            if distance is not None
            else float("inf")
        )

        ranked_results.append({
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "heading_found": heading_found,
            "keyword_matches": keyword_matches,
            "future_matches": future_matches,
            "diagnostic_score": diagnostic_score,
        })

    ranked_results.sort(key=lambda item: item["diagnostic_score"])

    for rank, item in enumerate(ranked_results, start=1):
        metadata = item["metadata"]

        print("\n" + "=" * 70)
        print(f"Result: {rank}")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Page: {metadata.get('page', 'Unknown')}")

        if item["distance"] is not None:
            print(f"Cosine distance: {item['distance']:.4f}")

        print(f"Diagnostic score: {item['diagnostic_score']:.4f}")
        print(f"Relevant heading found: {item['heading_found']}")
        print(f"Query keyword matches: {item['keyword_matches']}")
        print(f"Future-work term matches: {item['future_matches']}")
        print("-" * 70)

        passage = item["document"].strip()
        if len(passage) > MAX_PASSAGE_CHARS:
            passage = passage[:MAX_PASSAGE_CHARS] + "..."

        print(passage)

    print("\n" + "=" * 70)
    print("Search diagnostics complete.")
    print("Verify that the top passages contain evidence for the question.")


if __name__ == "__main__":
    main()
