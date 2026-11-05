
from backend.src.answer import collection

print("Total indexed chunks:", collection.count())

results = collection.get(
    limit=5,
    include=["documents", "metadatas"]
)

for i, document in enumerate(results.get("documents") or []):
    metadata = (results.get("metadatas") or [])[i] or {}

    print(f"\n--- Chunk {i + 1} ---")
    print("Source:", metadata.get("filename", metadata.get("source")))
    print("Page:", metadata.get("page", "Unknown"))
    print("Text:", document[:500])
