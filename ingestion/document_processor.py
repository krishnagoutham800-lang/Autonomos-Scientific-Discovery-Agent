from ingestion.pdf_extractor import save_extracted_text


def process_documents(downloaded_files):

    processed_documents = []

    for item in downloaded_files:

        paper = item["paper"]

        pdf_path = item["pdf_path"]

        try:

            document = save_extracted_text(
                paper,
                pdf_path
            )

            processed_documents.append(document)

        except Exception as error:

            print(
                f"Processing failed for "
                f"{paper.get('title')}: {error}"
            )

    print(
        f"\nSuccessfully processed "
        f"{len(processed_documents)} documents."
    )

    return processed_documents