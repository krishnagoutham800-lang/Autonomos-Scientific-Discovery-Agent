import os
from dotenv import load_dotenv
from google import genai

from ingestion.paper_fetcher import search_papers

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)


def analyse_research(topic):

    print("\nSearching scientific literature...")

    papers = search_papers(topic, limit=5)

    if not papers:
        print("No research papers found.")
        return

    print(f"Retrieved {len(papers)} papers.")

    research_context = ""

    for index, paper in enumerate(papers, 1):

        research_context += f"""
        Paper {index}
        Title: {paper['title']}
        Year: {paper['year']}
        DOI: {paper['doi']}
        Abstract: {paper['abstract'][:2000]}
        """

    prompt = f"""
    You are a scientific research analysis assistant.

    Research question: {topic}

    Analyse these retrieved scientific papers:

    {research_context}

    Provide:

    1. Overview of the research topic.
    2. Main findings across the papers.
    3. Common research methodologies.
    4. Limitations mentioned in the papers.
    5. Potential research gaps requiring further investigation.

    Only make claims supported by the supplied papers.
    Distinguish evidence from your own suggestions.
    Reference papers by their paper numbers.
    If information is missing, say so.
    """

    print("\nGemini is analysing the research...\n")

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    print(response.text)


if __name__ == "__main__":

    topic = input("Enter your scientific research question: ")

    analyse_research(topic)