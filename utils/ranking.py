from typing import List, Dict, Tuple

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel

from utils.VectorDatabase import VectorDatabase
from utils.scraping import extract_professor_links, fetch_all_bios


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


def rank_profs(
    directory_url: str,
    prompt: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> List[Dict[str, str | float]]:
    """Main function to orchestrate the bio fetching and ranking process using Milvus."""

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
