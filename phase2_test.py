from ingestion.paper_fetcher import search_papers
from ingestion.paper_retriever import retrieve_papers
from ingestion.document_processor import process_documents


def run_phase2():

    topic = input(
        "Enter a scientific research topic: "
    )

    print("\n[1] Searching scientific papers...")

    papers = search_papers(topic, limit=5)

    if not papers:
        print("No papers found.")
        return

    print(f"Found {len(papers)} papers.")

    print("\n[2] Retrieving accessible PDFs...")

    downloaded_files = retrieve_papers(papers)

    if not downloaded_files:
        print("No accessible PDFs were downloaded.")
        return

    print("\n[3] Extracting scientific documents...")

    processed_documents = process_documents(
        downloaded_files
    )

    print("\n========== PHASE 2 RESULTS ==========")

    print("Papers discovered:", len(papers))

    print("PDFs downloaded:", len(downloaded_files))

    print("Documents processed:", len(processed_documents))

    print("\nPhase 2 execution completed!")


if __name__ == "__main__":

    run_phase2()