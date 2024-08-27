from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def extract_professor_links(
    directory_soup: BeautifulSoup, directory_url: str
) -> List[str]:
    """Extract and return the list of full URLs to individual professors' pages."""
    links: List[str] = [
        urljoin(directory_url, link["href"])
        for link in directory_soup.find_all("a", href=True)
        if "cs/" in link["href"]  # Adjust based on actual structure
    ]
    return links


def fetch_professor_page(prof_url: str) -> str:
    """Fetch a professor's page with SSL error handling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Exponential backoff factor (1 second, 2 seconds, 4 seconds, etc.)
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # Retry only on these methods
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    try:
        response = session.get(prof_url)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        response = session.get(prof_url, verify=False)
        response.raise_for_status()

    return response.text


def parse_bio(prof_page_html: str) -> str | None:
    """Parse the professor's bio from the HTML content."""
    prof_soup: BeautifulSoup = BeautifulSoup(prof_page_html, "html.parser")
    bio_div = prof_soup.find(
        "div",
        class_="field field-name-body field-type-text-with-summary field-label-hidden",
    )
    return bio_div.get_text(strip=True) if bio_div else None


def fetch_and_parse_professor(prof_url: str) -> str | None:
    """Fetch a professor's page and parse the bio content."""
    prof_page_html: str = fetch_professor_page(prof_url)
    return parse_bio(prof_page_html)


def fetch_all_bios(professor_links: List[str], max_workers: int = 10) -> List[str]:
    """Fetch and parse professor pages concurrently."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fetch_and_parse = partial(fetch_and_parse_professor)
        futures = {
            executor.submit(fetch_and_parse, url): url for url in professor_links
        }
        return [future.result() for future in as_completed(futures) if future.result()]
