import fitz
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

PAPERS_DIR = BASE_DIR / "storage" / "papers"
IMAGE_DIR = BASE_DIR / "storage" / "extracted_images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def extract_page_images(pdf_path, max_pages=20):

    pdf_path = Path(pdf_path)

    document = fitz.open(pdf_path)

    extracted_images = []

    for page_number, page in enumerate(document):

        if page_number >= max_pages:
            break

        # Render the complete page to capture charts,
        # tables, diagrams and embedded images.
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(1.5, 1.5),
            alpha=False
        )

        image_name = (
            f"{pdf_path.stem}_page_{page_number + 1}.png"
        )

        image_path = IMAGE_DIR / image_name

        pixmap.save(image_path)

        extracted_images.append({
            "image_path": str(image_path),
            "page_number": page_number + 1,
            "paper_name": pdf_path.name
        })

    document.close()

    return extracted_images


def extract_all_paper_images():

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return []

    all_images = []

    for pdf_file in pdf_files:

        print(f"\nExtracting pages: {pdf_file.name}")

        try:
            images = extract_page_images(pdf_file)

            all_images.extend(images)

            print(f"Extracted {len(images)} page images.")

        except Exception as error:
            print(f"Error: {error}")

    print("\nImage extraction completed.")

    return all_images


if __name__ == "__main__":
    extract_all_paper_images()