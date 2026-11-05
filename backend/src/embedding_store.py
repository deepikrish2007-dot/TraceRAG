
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "output" / "chunks.json"
DB_DIR = PROJECT_ROOT / ".chroma"

COLLECTION_NAME = "tracerag_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def build_chunk_id(item):
    """Create a stable, unique ID for each source page and chunk."""
    return (
        f"{item['source']}_"
        f"{item['page']}_"
        f"{item['chunk_id']}"
    )


def add_chunks(chunks, collection=None, model=None):
    """Embed and upsert chunks into the ChromaDB collection."""
    if not chunks:
        print("No chunks available to index.")
        return 0

    DB_DIR.mkdir(parents=True, exist_ok=True)

    if model is None:
        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)

    if collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    added = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [item["text"].strip() for item in batch]
        valid_items = [
            (item, text)
            for item, text in zip(batch, texts)
            if text
        ]

        if not valid_items:
            continue

        batch_items, texts = zip(*valid_items)

        embeddings = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=[build_chunk_id(item) for item in batch_items],
            documents=list(texts),
            embeddings=embeddings,
            metadatas=[
                {
                    "source": str(item["source"]),
                    "page": int(item["page"]),
                    "chunk_id": str(item["chunk_id"]),
                }
                for item in batch_items
            ],
        )

        added += len(batch_items)
        print(f"Indexed {added}/{len(chunks)} chunks...")

    return added


def main():
    if not CHUNKS_FILE.exists():
        print(f"Chunks file not found: {CHUNKS_FILE}")
        print("Run the chunking pipeline first.")
        return

    try:
        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        print(f"Could not read chunks.json: {error}")
        return

    if not isinstance(chunks, list) or not chunks:
        print("No valid chunks found in chunks.json.")
        return

    # Validate the fields before modifying the database.
    required_fields = {"source", "page", "chunk_id", "text"}

    for index, item in enumerate(chunks):
        if not isinstance(item, dict) or not required_fields.issubset(item):
            print(f"Invalid chunk at position {index}.")
            print(f"Required fields: {sorted(required_fields)}")
            return

    # Use one model instance and one collection.
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Rebuild the collection to remove obsolete chunks from old indexing.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    try:
        count = add_chunks(chunks, collection=collection, model=model)
    except Exception as error:
        print(f"Indexing failed: {error}")
        print("The collection may be incomplete. Run this script again.")
        return

    print("\nIndexing complete!")
    print(f"Chunks in database: {collection.count()}")
    print(f"Chunks processed: {count}")
    print(f"Database location: {DB_DIR}")


if __name__ == "__main__":
    main()
