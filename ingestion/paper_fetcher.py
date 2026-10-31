import requests
import xml.etree.ElementTree as ET

OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL = "https://export.arxiv.org/api/query"


def reconstruct_abstract(index):

    if not index:
        return "Abstract not available"

    words = {}

    for word, positions in index.items():
        for position in positions:
            words[position] = word

    return " ".join(words[i] for i in sorted(words))


def search_openalex(query, limit=10):

    params = {
        "search": query,
        "per_page": limit,
        "select": (
            "id,title,publication_year,doi,"
            "abstract_inverted_index,open_access,locations"
        )
    }

    response = requests.get(
        OPENALEX_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    papers = []

    for item in response.json().get("results", []):

        urls = []

        for location in item.get("locations") or []:

            url = location.get("pdf_url")

            if url and url not in urls:
                urls.append(url)

        papers.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "year": item.get("publication_year"),
            "doi": item.get("doi"),
            "abstract": reconstruct_abstract(
                item.get("abstract_inverted_index")
            ),
            "pdf_urls": urls,
            "source": "OpenAlex"
        })

    return papers


def search_arxiv(query, limit=10):

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit
    }

    response = requests.get(
        ARXIV_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    root = ET.fromstring(response.content)

    namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    papers = []

    for entry in root.findall("atom:entry", namespace):

        paper_id = entry.findtext(
            "atom:id", namespaces=namespace
        )

        title = entry.findtext(
            "atom:title", namespaces=namespace
        )

        abstract = entry.findtext(
            "atom:summary", namespaces=namespace
        )

        published = entry.findtext(
            "atom:published", namespaces=namespace
        )

        year = published[:4] if published else None

        pdf_url = None

        for link in entry.findall("atom:link", namespace):

            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")

        if not pdf_url and paper_id:
            pdf_url = paper_id.replace(
                "/abs/", "/pdf/"
            ) + ".pdf"

        papers.append({
            "id": paper_id,
            "title": title.strip() if title else None,
            "year": year,
            "doi": None,
            "abstract": abstract.strip() if abstract else "",
            "pdf_urls": [pdf_url] if pdf_url else [],
            "source": "arXiv"
        })

    return papers


def search_papers(query, limit=10):

    papers = []

    for search_function in (
        search_openalex,
        search_arxiv
    ):

        try:
            results = search_function(query, limit)

            papers.extend(results)

            print(
                f"{search_function.__name__}: "
                f"{len(results)} papers found"
            )

        except requests.RequestException as error:

            print(f"Research source unavailable: {error}")

    # Remove duplicates using DOI or normalized title
    unique = {}
    for paper in papers:

        key = (
            (paper.get("doi") or "").lower()
            or (paper.get("title") or "").strip().lower()
        )

        if key and key not in unique:
            unique[key] = paper

        elif key and key in unique:
            existing = unique[key]

            existing["pdf_urls"] = list(dict.fromkeys(
                existing["pdf_urls"] + paper["pdf_urls"]
            ))

    return list(unique.values())


if __name__ == "__main__":

    topic = input("Enter research topic: ")

    results = search_papers(topic)

    for number, paper in enumerate(results, 1):

        print(f"\nPaper {number}")
        print("Title:", paper["title"])
        print("Source:", paper["source"])
        print("PDF URLs:", paper["pdf_urls"])