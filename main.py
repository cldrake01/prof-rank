from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import List, Dict, Tuple
from urllib.parse import urljoin

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from transformers import AutoTokenizer, AutoModel

from VectorDatabase import VectorDatabase


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


def load_model(model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
    """Load and return the tokenizer and model for embedding texts."""
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: AutoModel = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def embed_text(
    texts: List[str], tokenizer: AutoTokenizer, model: AutoModel
) -> np.ndarray:
    """Embed a list of texts into vectors using the provided tokenizer and model."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()


def scrape(directory_url: str) -> List[str]:
    """Fetch and parse directory page, extract professor links, and fetch all bios."""
    response: requests.Response = requests.get(directory_url)
    directory_soup: BeautifulSoup = BeautifulSoup(response.text, "html.parser")
    professor_links: List[str] = extract_professor_links(directory_soup, directory_url)
    bios: List[str] = fetch_all_bios(professor_links)
    return bios


def rank(
    bios: List[str],
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    top_k: int = 5,
) -> List[Dict[str, str | float]]:
    """
    Embed bios and prompt, initialize Milvus collection, insert embeddings, query for similar bios, and collect
    results with similarity scores.
    """
    # Embed bios and prompt
    bios_vectors: np.ndarray = embed_text(bios, tokenizer, model)
    prompt_vector: np.ndarray = embed_text([prompt], tokenizer, model)

    # Initialize VectorDatabase
    vector_db = VectorDatabase()
    vector_db.create_collection(name="bios_collection", dim=bios_vectors.shape[1])

    # Insert embeddings into Milvus
    vector_db.insert_embeddings(bios_vectors)

    # Query Milvus for similar bios
    results = vector_db.query_similar_vectors(prompt_vector, top_k)

    # Collect results with similarity scores
    ranked_bios = []
    for result in results[0]:
        idx = result.id
        similarity_score = result.distance
        ranked_bios.append({"bio": bios[idx], "similarity_score": similarity_score})

    return ranked_bios


def main(
    directory_url: str,
    prompt: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> List[Dict[str, str | float]]:
    """Main function to orchestrate the bio fetching and ranking process using Milvus."""

    # Check available device
    if torch.cuda.is_available():
        torch.device("cuda")
        print("GPU is available. Using CUDA.")
    elif torch.backends.mps.is_available():
        torch.device("mps")
        print("Apple Silicon is available. Using MPS.")
    else:
        torch.device("cpu")
        print("Using CPU.")

    # Load model and tokenizer
    tokenizer, model = load_model(model_name)

    # Scrape bios
    bios: List[str] = scrape(directory_url)

    # Embed bios and prompt
    bios_vectors: np.ndarray = embed_text(bios, tokenizer, model)
    prompt_vector: np.ndarray = embed_text([prompt], tokenizer, model)

    # Initialize VectorDatabase
    vector_db = VectorDatabase()
    vector_db.create_collection(name="bios_collection", dim=bios_vectors.shape[1])

    # Insert embeddings into Milvus
    vector_db.insert_data("bios_collection", bios, bios_vectors)

    # Query Milvus for similar bios
    results = vector_db.search_data("bios_collection", prompt_vector[0], top_k)

    # Collect results with similarity scores
    ranked_bios = []
    for result in results[0]:
        idx = result["id"]
        similarity_score = result["distance"]
        ranked_bios.append({"bio": bios[idx], "similarity_score": similarity_score})

    return ranked_bios


if __name__ == "__main__":
    directory_url: str = "https://www.colorado.edu/cs/faculty-staff-directory"
    prompt: str = (
        "machine learning artificial intelligence computer vision natural language processing loss functions"
    )
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Change as needed
    top_k: int = 20

    ranked_bios = main(directory_url, prompt, model_name, top_k)

    # Output the ranked bios
    for bio_info in ranked_bios:
        print(f"Bio: {bio_info['bio']}")
        print(f"Similarity Score: {bio_info['similarity_score']}")
        print("-" * 40)
