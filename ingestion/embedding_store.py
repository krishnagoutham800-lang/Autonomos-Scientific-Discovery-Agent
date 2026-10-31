import json
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingestion.text_processor import create_chunks


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "storage" / "extracted_text"
CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent ChromaDB storage
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="scientific_papers"
)


def process_paper(json_file):
    """Read one extracted paper and store its text embeddings."""

    with open(json_file, "r", encoding="utf-8") as file:
        paper = json.load(file)

    metadata = paper.get("metadata", {})

    title = metadata.get("title", json_file.stem)
    year = metadata.get("year", 0)

    pages = paper.get("pages", [])

    documents = []
    ids = []
    metadatas = []

    for page in pages:

        page_number = page.get("page_number", 0)
        page_text = page.get("text", "")

        chunks = create_chunks(page_text)

        for index, chunk in enumerate(chunks):

            unique_text = (
                f"{json_file.stem}_{page_number}_{index}"
            )

            chunk_id = hashlib.md5(
                unique_text.encode("utf-8")
            ).hexdigest()

            documents.append(chunk)
            ids.append(chunk_id)

            metadatas.append({
                "paper_id": json_file.stem,
                "title": str(title),
                "year": int(year) if str(year).isdigit() else 0,
                "page": int(page_number),
                "chunk_index": index
            })

    if documents:
        embeddings = model.encode( documents)
        embeddings = [
            embedding.tolist() 
            for embedding in embeddings
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return len(documents)


def process_all_papers():

    json_files = list(INPUT_DIR.glob("*.json"))

    if not json_files:
        print("No extracted JSON files found.")
        return

    total_chunks = 0

    for json_file in json_files:

        print(f"\nProcessing: {json_file.name}")

        try:
            count = process_paper(json_file)
            total_chunks += count

            print(f"Stored {count} chunks.")

        except Exception as error:
            print(f"Error processing {json_file.name}: {error}")

    print("\nPhase 3 completed!")
    print(f"Total chunks stored: {total_chunks}")


if __name__ == "__main__":
    process_all_papers()