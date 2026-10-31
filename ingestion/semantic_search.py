from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"

# Load the same model used in Phase 3
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing ChromaDB
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="scientific_papers"
)


def search_research(question, top_k=5):

    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count())
    )

    if not results["documents"] or not results["documents"][0]:
        print("No relevant research passages found.")
        return

    print("\n========== RESEARCH SEARCH RESULTS ==========\n")

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"Result {i + 1}")
        print(f"Title: {metadata.get('title', 'Unknown')}")
        print(f"Year: {metadata.get('year', 'Unknown')}")
        print(f"Page: {metadata.get('page', 'Unknown')}")
        print(f"Distance: {distance:.4f}")
        print(f"Content: {document}")
        print("-" * 60)


if __name__ == "__main__":

    question = input("Enter your research question: ")

    search_research(question, top_k=5)