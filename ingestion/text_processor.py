import re


def clean_text(text):
    """Clean extracted research paper text."""

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""

    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks