import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer


# Project configuration
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"
OUTPUT_DIR = BASE_DIR / "storage" / "validation_reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Gemini setup
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY missing from .env")

gemini_client = genai.Client(api_key=api_key)


# ChromaDB setup
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="scientific_papers"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Step 1: Retrieve evidence for a claim
def retrieve_evidence(claim, top_k=5):

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    query_embedding = embedding_model.encode(
        claim
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_chunks)
    )

    evidence = []

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        evidence.append({
            "source_id": f"SOURCE_{i + 1}",
            "title": metadata.get("title", "Unknown"),
            "year": metadata.get("year", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "content": document
        })

    return evidence


# Step 2: Validate the claim
def validate_claim(claim):

    evidence = retrieve_evidence(claim)

    if not evidence:
        return {
            "claim": claim,
            "status": "unsupported",
            "reason": "No evidence retrieved.",
            "sources": []
        }

    context = ""

    for item in evidence:

        context += f"""
        {item['source_id']}

        Title: {item['title']}
        Year: {item['year']}
        Page: {item['page']}

        Evidence:
        {item['content']}

        ------------------------
        """

    prompt = f"""
    You are a scientific evidence validation assistant.

    CLAIM:
    {claim}

    RETRIEVED SCIENTIFIC EVIDENCE:
    {context}

    Assess whether the evidence supports the claim.

    Return ONLY valid JSON in this format:

    {{
        "status": "supported",
        "reason": "Explanation of the assessment",
        "sources": [
            {{
                "source_id": "SOURCE_1",
                "evidence_quote": "Exact text copied from the source",
                "assessment": "How this evidence relates to the claim"
            }}
        ]
    }}

    Rules:

    - status must be supported, partially_supported,
      or unsupported.
    - Use only the supplied evidence.
    - Do not invent quotations.
    - Evidence quotes must be exact excerpts.
    - Do not treat similarity of topic as proof.
    - Distinguish correlation from causation.
    - If evidence is insufficient, use unsupported.
    - Include only relevant sources.
    - Return valid JSON only.
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw_response = (response.text or "").strip()

    if raw_response.startswith("```"):
        raw_response = raw_response.split("\n", 1)[1]
        raw_response = raw_response.rsplit("```", 1)[0]

    result = json.loads(raw_response)

    # Step 3: Check quotations against retrieved text
    evidence_map = {
        item["source_id"]: item
        for item in evidence
    }

    verified_sources = []

    for source in result.get("sources", []):

        source_id = source.get("source_id")
        quote = source.get("evidence_quote", "")

        original = evidence_map.get(source_id)

        quote_verified = (
            bool(quote)
            and original is not None
            and quote.casefold() in original["content"].casefold()
        )

        if original is None:
            continue

        verified_sources.append({
            "source_id": source_id,
            "title": original["title"],
            "year": original["year"],
            "page": original["page"],
            "evidence_quote": quote,
            "quote_verified": quote_verified,
            "assessment": source.get("assessment", "")
        })

    # Never mark a claim supported if its returned
    # quotations cannot be verified.
    status = result.get("status", "unsupported")

    if status not in [
        "supported",
        "partially_supported",
        "unsupported"
    ]:
        status = "unsupported"

    if not verified_sources or not all(
        item["quote_verified"] for item in verified_sources
    ):
        status = "unsupported"

    return {
        "claim": claim,
        "status": status,
        "reason": result.get("reason", ""),
        "sources": verified_sources
    }


# Step 4: Validate multiple claims
def validate_research(claims):

    report = {
        "total_claims": len(claims),
        "results": []
    }

    for index, claim in enumerate(claims):

        print(f"\nValidating claim {index + 1}...")

        try:
            result = validate_claim(claim)
            report["results"].append(result)

            print("Status:", result["status"])

        except Exception as error:

            print("Validation error:", error)

            report["results"].append({
                "claim": claim,
                "status": "validation_error",
                "reason": str(error),
                "sources": []
            })

    output_file = OUTPUT_DIR / "validation_report.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nValidation completed!")
    print(f"Report saved at: {output_file}")

    return report