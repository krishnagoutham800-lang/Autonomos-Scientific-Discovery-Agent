import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"

# Gemini configuration
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

gemini_client = genai.Client(api_key=api_key)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_collection(
    name="scientific_papers"
)


def retrieve_research(question, top_k=5):

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, total_chunks)
    )

    research_context = []

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        research_context.append({
            "content": document,
            "title": metadata.get("title", "Unknown"),
            "year": metadata.get("year", "Unknown"),
            "page": metadata.get("page", "Unknown")
        })

    return research_context


def analyze_research(question):

    print("\nSearching scientific papers...")

    papers = retrieve_research(question)

    if not papers:
        print("No research passages found.")
        return

    # Prepare evidence for Gemini
    context = ""

    for i, paper in enumerate(papers):

        context += f"""
        SOURCE [{i + 1}]
        Title: {paper['title']}
        Year: {paper['year']}
        Page: {paper['page']}

        Content:
        {paper['content']}

        --------------------------------
        """

    prompt = f"""
    You are an AI scientific research analysis assistant.

    Research Question:
    {question}

    Retrieved Scientific Evidence:
    {context}

    Analyze the evidence and prepare a structured report.

    Include these sections:

    1. Research Overview
    2. Key Findings
    3. Comparison of Research Findings
    4. Limitations Mentioned in the Sources
    5. Potential Research Gaps
    6. Suggested Future Research Directions
    7. Conclusion
    8. References

    IMPORTANT RULES:

    - Use only the provided scientific evidence.
    - Cite sources using their SOURCE numbers, for example [SOURCE 1].
    - Do not invent authors, titles, dates, results or references.
    - Clearly state when evidence is insufficient.
    - Treat research gaps as preliminary possibilities,
      not proven gaps in the entire scientific literature.
    - Distinguish source findings from your own suggestions.
    """

    print("\nGemini is analyzing the research...\n")

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    report = response.text

    print("\n========== RESEARCH ANALYSIS REPORT ==========\n")
    print(report)

    return report


if __name__ == "__main__":

    question = input(
        "Enter your scientific research question: "
    )

    analyze_research(question)