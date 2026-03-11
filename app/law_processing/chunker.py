"""
Chunker: converts LawDocument into flat, indexable chunks (one per Article/Điều).
Each chunk is a dict with full hierarchy metadata for search and knowledge graph ingestion.
"""

import json
from pathlib import Path
from typing import List

from app.law_processing.models import LawDocument


def articles_to_chunks(law_doc: LawDocument) -> List[dict]:
    """
    Flatten a LawDocument into a list of article-level chunks.
    Each chunk includes the full text plus hierarchy metadata.
    """
    chunks = []
    for article in law_doc.all_articles:
        chunk = article.to_dict()
        chunks.append(chunk)
    return chunks


def save_chunks_to_json(chunks: List[dict], output_path: str) -> None:
    """Save chunk list to a JSON file"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {len(chunks)} chunks → {path}")


def load_chunks_from_json(json_path: str) -> List[dict]:
    """Load chunks from a saved JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def chunk_both_laws(
    law_2013: LawDocument,
    law_2024: LawDocument,
    output_dir: str = "output",
) -> tuple[List[dict], List[dict]]:
    """
    Produce chunks for both laws and persist them to JSON.
    Returns (chunks_2013, chunks_2024).
    """
    chunks_2013 = articles_to_chunks(law_2013)
    chunks_2024 = articles_to_chunks(law_2024)

    save_chunks_to_json(chunks_2013, f"{output_dir}/chunks_2013.json")
    save_chunks_to_json(chunks_2024, f"{output_dir}/chunks_2024.json")

    return chunks_2013, chunks_2024


def print_chunk_stats(chunks: List[dict], law_year: int) -> None:
    """Print basic stats about chunks"""
    chapter_counts: dict = {}
    for c in chunks:
        ch = c.get("chapter_number", "?")
        chapter_counts[ch] = chapter_counts.get(ch, 0) + 1

    print(f"\n📊 Law {law_year} chunk statistics:")
    print(f"   Total articles: {len(chunks)}")
    print(f"   Articles by chapter:")
    for ch, count in sorted(chapter_counts.items(), key=lambda x: str(x[0])):
        print(f"     Chapter {ch}: {count} articles")
