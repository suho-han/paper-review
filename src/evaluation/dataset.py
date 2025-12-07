import json
import os
import random
from typing import Dict, List, Tuple


def load_test_dataset(
    file_path: str,
    sample_size: int = None,
    seed: int = 42
) -> List[Dict]:
    """
    Load dataset from a JSON file and optionally sample a subset for testing.

    Args:
        file_path: Path to the JSON file containing papers and reviews.
        sample_size: Number of papers to sample. If None, return all.
        seed: Random seed for reproducibility.

    Returns:
        List of dictionaries, each containing 'forum_id', 'abstract', and 'reviews'.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter out entries with no reviews or empty abstract
    valid_data = [
        item for item in data
        if item.get('abstract') and item.get('reviews')
    ]

    if sample_size is not None and sample_size < len(valid_data):
        random.seed(seed)
        return random.sample(valid_data, sample_size)

    return valid_data


def create_test_set(
    source_files: List[str],
    output_path: str,
    sample_ratio: float = 0.1,
    seed: int = 42
):
    """
    Create a test set by sampling from source files.

    Args:
        source_files: List of paths to source JSON files (e.g., ICLR 2025).
        output_path: Path to save the test set JSON.
        sample_ratio: Ratio of data to sample for the test set (default 10%).
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    all_papers = []

    for file_path in source_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} papers from {file_path}")
        all_papers.extend(papers)

    # Filter papers that have reviews
    papers_with_reviews = [p for p in all_papers if p.get('reviews') and p.get('abstract')]
    print(f"Total papers with reviews and abstract: {len(papers_with_reviews)}")

    # Sample
    sample_size = int(len(papers_with_reviews) * sample_ratio)
    if sample_size == 0 and len(papers_with_reviews) > 0:
        sample_size = 1

    test_set = random.sample(papers_with_reviews, sample_size)

    print(f"Sampled {len(test_set)} papers for the test set.")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    print(f"Saved test set to {output_path}")


def extract_ground_truth(paper_data: Dict) -> Dict:
    """
    Extract ground truth data from a paper entry.

    Args:
        paper_data: A dictionary containing paper info and reviews.

    Returns:
        Dictionary with 'abstract' and list of 'reviews' (content and rating).
    """
    reviews = []
    for r in paper_data['reviews']:
        # Extract rating if available in content (simple parsing)
        # Note: This is a heuristic. Real parsing might need regex or LLM.
        content = r.get('content', '')
        rating = None

        # Try to find "rating: X" pattern
        import re
        match = re.search(r'rating:\s*(\d+)', content, re.IGNORECASE)
        if match:
            rating = int(match.group(1))

        reviews.append({
            'content': content,
            'rating': rating
        })

    return {
        'forum_id': paper_data['forum_id'],
        'abstract': paper_data['abstract'],
        'reviews': reviews
    }


if __name__ == "__main__":
    # Example usage: Create test set from ICLR 2025 and NeurIPS 2025 (if available)
    sources = [
        "data/ICLR.cc_2025_Conference_reviews.json",
        "data/NeurIPS.cc_2025_Conference_reviews.json"
    ]

    create_test_set(
        source_files=sources,
        output_path="data/evaluation/test_set_2025.json",
        sample_ratio=0.1
    )
