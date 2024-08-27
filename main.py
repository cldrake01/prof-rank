from utils.display import render
from utils.ranking import rank_profs


def main() -> None:
    directory_url: str = "https://www.colorado.edu/cs/faculty-staff-directory"
    prompt: str = (
        "machine learning artificial intelligence computer vision natural language processing loss functions"
    )
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Change as needed
    top_k: int = 20

    ranked_bios = rank_profs(directory_url, prompt, model_name, top_k)

    width: int = 80

    # Output the ranked bios
    for bio_info in ranked_bios:
        print(f"{f'Bio ({bio_info["similarity_score"]:3f})':^{width}}")
        render(bio_info["bio"], width)
        print("\n" + "=" * width)


if __name__ == "__main__":
    main()
