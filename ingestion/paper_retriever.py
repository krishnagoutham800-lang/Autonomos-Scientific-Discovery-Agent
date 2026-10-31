import os
import requests
import hashlib

from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

PAPER_DIR = BASE_DIR / "storage" / "papers"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

MAX_PDF_SIZE = 50 * 1024 * 1024

UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL")

HEADERS = {
    "User-Agent": "ScientificDiscoveryAgent/1.0",
    "Accept": "application/pdf,*/*"
}


def get_unpaywall_urls(doi):

    if not doi or not UNPAYWALL_EMAIL:
        return []

    try:

        response = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": UNPAYWALL_EMAIL},
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        urls = []

        locations = data.get("oa_locations") or []

        for location in locations:

            url = location.get("url_for_pdf")

            if url and url not in urls:
                urls.append(url)

        return urls

    except requests.RequestException as error:

        print("Unpaywall lookup failed:", error)
        return []


def download_paper(paper):

    title = paper.get("title", "Unknown paper")

    pdf_urls = list(paper.get("pdf_urls") or [])

    # Add legally available Unpaywall locations
    pdf_urls.extend(
        get_unpaywall_urls(paper.get("doi"))
    )

    pdf_urls = list(dict.fromkeys(pdf_urls))

    if not pdf_urls:
        print("No accessible PDF locations:", title)
        return None

    paper_id = paper.get("id") or title

    file_id = hashlib.sha256(
        paper_id.encode("utf-8")
    ).hexdigest()[:16]

    pdf_path = PAPER_DIR / f"{file_id}.pdf"

    if pdf_path.exists():
        print("Already downloaded:", title)
        return pdf_path

    for url in pdf_urls:

        temp_path = PAPER_DIR / f"{file_id}.tmp"

        try:

            parsed = urlparse(url)

            if parsed.scheme not in ("https", "http"):
                continue

            print("Trying source:", url)

            with requests.get(
                url,
                headers=HEADERS,
                stream=True,
                timeout=(10, 60)
            ) as response:

                if response.status_code in (403, 404):
                    print(
                        f"HTTP {response.status_code}. "
                        "Trying another source."
                    )
                    continue

                response.raise_for_status()

                content_length = response.headers.get(
                    "Content-Length"
                )

                if (
                    content_length
                    and int(content_length) > MAX_PDF_SIZE
                ):
                    continue

                total_size = 0
                first_chunk = True

                with open(temp_path, "wb") as file:

                    for chunk in response.iter_content(
                        chunk_size=8192
                    ):

                        if not chunk:
                            continue

                        if first_chunk:

                            if not chunk.startswith(b"%PDF-"):
                                raise ValueError(
                                    "Not a valid PDF"
                                )

                            first_chunk = False

                        total_size += len(chunk)

                        if total_size > MAX_PDF_SIZE:
                            raise ValueError(
                                "PDF exceeds size limit"
                            )

                        file.write(chunk)

                if first_chunk:
                    raise ValueError("Empty PDF")

                temp_path.replace(pdf_path)

                print("Downloaded successfully:", title)

                return pdf_path

        except Exception as error:

            print("Source failed:", error)

        finally:

            if temp_path.exists():
                temp_path.unlink()

    print("No accessible PDF found:", title)

    return None


def retrieve_papers(papers):

    downloaded = []

    for paper in papers:

        path = download_paper(paper)

        if path:

            downloaded.append({
                "paper": paper,
                "pdf_path": str(path)
            })

    print(
        f"\nDownloaded {len(downloaded)} "
        f"of {len(papers)} papers."
    )

    return downloaded