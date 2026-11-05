
from backend.src.answer import retrieve_passages

question = "What is the main objective of this research paper?"

print(f"\nQuestion: {question}")
print("\n--- Retrieved Passages ---")

passages = retrieve_passages(question, top_k=6)

for i, passage in enumerate(passages, start=1):
    print(f"\nPassage {i}")
    print(f"Source: {passage['source']}")
    print(f"Page: {passage['page']}")
    print(f"Distance: {passage['distance']}")
    print(f"Text:\n{passage['text'][:800]}")
