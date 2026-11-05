import chromadb

client = chromadb.PersistentClient(
    path="backend/.chroma"
)

collection = client.get_collection(
    "tracerag_chunks"
)

results = collection.get(
    include=["documents", "metadatas"]
)

target_sources = {
    "research_paper.pdf",
    "research_paper 3.pdf"
}

target_pages = {"1", "4"}

for document, metadata in zip(
    results["documents"],
    results["metadatas"]
):
    source = metadata.get("source", "")
    page = str(metadata.get("page", ""))

    if source in target_sources and page in target_pages:
        print("\n" + "=" * 70)
        print("SOURCE:", source)
        print("PAGE:", page)
        print("-" * 70)
        print(document[:1800])