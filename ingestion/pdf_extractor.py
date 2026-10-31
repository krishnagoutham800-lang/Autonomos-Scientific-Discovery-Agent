import fitz
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_DIR = BASE_DIR / "storage" / "extracted_text"

TEXT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf(pdf_path):

    pdf_path = Path(pdf_path)

    document = fitz.open(pdf_path)

    extracted_pages = []

    for page_number, page in enumerate(document, 1):

        text = page.get_text("text")

        extracted_pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    document.close()

    return extracted_pages


def save_extracted_text(paper, pdf_path):

    pages = extract_pdf(pdf_path)

    extracted_data = {
        "paper_id": paper.get("id"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "doi": paper.get("doi"),
        "source_pdf": str(pdf_path),
        "pages": pages
    }

    output_name = Path(pdf_path).stem + ".json"

    output_path = TEXT_DIR / output_name

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            extracted_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Extracted text saved: {output_path}")

    return extracted_data