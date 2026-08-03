"""Web page scraper and HTML text extractor for MultiDocChat."""

from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from ingestion.loaders import _apply_common_metadata, _splitter


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36 MultiDocChat/1.0"
)


def scrape_url(url: str, timeout: int = 15) -> List[Document]:
    """Fetch a web page, extract clean text, and return attribution-ready chunks."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    filename = parsed.netloc + parsed.path.rstrip("/")
    if not filename:
        filename = parsed.netloc
    filename = re.sub(r"[^\w\.-]", "_", filename) + ".html"

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for element in soup(["script", "style", "nav", "footer", "header", "form", "svg"]):
        element.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else parsed.netloc
    text = soup.get_text(separator="\n")

    # Clean whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks_text = "\n".join(line for line in lines if line)

    if not chunks_text:
        return []

    doc = Document(
        page_content=chunks_text,
        metadata={
            "source_file": filename,
            "title": title,
            "url": url,
        },
    )

    chunks = _splitter().split_documents([doc])
    return _apply_common_metadata(chunks, source_file=filename)


def scrape_urls(urls: List[str]) -> List[Document]:
    """Process multiple URLs into chunks."""
    all_chunks: List[Document] = []
    for url in urls:
        if url.strip():
            try:
                chunks = scrape_url(url)
                all_chunks.extend(chunks)
            except Exception as exc:
                print(f"Failed to scrape {url}: {exc}")
    return all_chunks
