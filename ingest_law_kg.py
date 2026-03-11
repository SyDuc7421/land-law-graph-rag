"""
Main ingestion script: parse both land law files, run cross-mapping, and build the Neo4j knowledge graph.

Usage:
  python ingest_law_kg.py                          # Full pipeline with LLM diff
  python ingest_law_kg.py --skip-embeddings        # Skip step 3 (no OpenAI calls for embeddings)
  python ingest_law_kg.py --skip-llm-diff          # Skip step 4 (no LLM diff)
  python ingest_law_kg.py --dry-run                # Parse + chunk + cross-map only (no Neo4j)
  python ingest_law_kg.py --no-clear               # Keep existing Neo4j data
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest Land Law into Neo4j Knowledge Graph")
    parser.add_argument("--law-2013", default="input/LandLaw2013.txt", help="Path to LandLaw 2013 text file")
    parser.add_argument("--law-2024", default="input/LandLaw2024.txt", help="Path to LandLaw 2024 text file")
    parser.add_argument("--output-dir", default="output", help="Directory for intermediate JSON output")
    parser.add_argument("--dry-run", action="store_true", help="Parse + chunk + cross-map only, skip Neo4j")
    parser.add_argument("--no-clear", action="store_true", help="Keep existing Neo4j graph data")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding fallback step (step 3)")
    parser.add_argument("--skip-llm-diff", action="store_true", help="Skip LLM diff step (step 4)")
    parser.add_argument("--load-cached", action="store_true",
                        help="Load previously saved chunks/cross-mapping from output/ instead of re-parsing")
    return parser.parse_args()


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    chunks_path_2013 = f"{output_dir}/chunks_2013.json"
    chunks_path_2024 = f"{output_dir}/chunks_2024.json"
    cross_mapping_path = f"{output_dir}/cross_mapping.json"

    print("=" * 60)
    print("  Luật Đất Đai Knowledge Graph Ingestion Pipeline")
    print("=" * 60)

    # ── Phase 1: Parse + Chunk ─────────────────────────────────────
    if args.load_cached and Path(chunks_path_2013).exists() and Path(chunks_path_2024).exists():
        print("\n📂 Loading cached chunks from disk...")
        from app.law_processing.chunker import load_chunks_from_json
        chunks_2013 = load_chunks_from_json(chunks_path_2013)
        chunks_2024 = load_chunks_from_json(chunks_path_2024)
        print(f"   ✓ Loaded {len(chunks_2013)} chunks (2013), {len(chunks_2024)} chunks (2024)")
    else:
        print("\n📖 Phase 1: Parsing law files...")
        from app.law_processing.parser import parse_law_file
        from app.law_processing.chunker import chunk_both_laws, print_chunk_stats

        if not Path(args.law_2013).exists():
            print(f"❌ Law file not found: {args.law_2013}")
            sys.exit(1)
        if not Path(args.law_2024).exists():
            print(f"❌ Law file not found: {args.law_2024}")
            sys.exit(1)

        print(f"  Parsing {args.law_2013} (LandLaw 2013)...")
        law_2013 = parse_law_file(args.law_2013, 2013)

        print(f"  Parsing {args.law_2024} (LandLaw 2024)...")
        law_2024 = parse_law_file(args.law_2024, 2024)

        chunks_2013, chunks_2024 = chunk_both_laws(law_2013, law_2024, output_dir=output_dir)
        print_chunk_stats(chunks_2013, 2013)
        print_chunk_stats(chunks_2024, 2024)

    # ── Phase 2: Cross-Mapping ────────────────────────────────────
    if args.load_cached and Path(cross_mapping_path).exists():
        print("\n📂 Loading cached cross-mapping from disk...")
        with open(cross_mapping_path, 'r', encoding='utf-8') as f:
            cross_mapping_dicts = json.load(f)
        print(f"   ✓ Loaded {len(cross_mapping_dicts)} mapping entries")
    else:
        print("\n🔗 Phase 2: Running cross-mapping...")
        from app.law_processing.cross_mapper import run_cross_mapping

        matches = run_cross_mapping(
            chunks_2013=chunks_2013,
            chunks_2024=chunks_2024,
            output_path=cross_mapping_path,
            skip_embeddings=args.skip_embeddings,
            skip_llm_diff=args.skip_llm_diff,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        cross_mapping_dicts = [m.to_dict() for m in matches]

    # ── Phase 3: Neo4j Ingestion ──────────────────────────────────
    if args.dry_run:
        print("\n⚠️  --dry-run flag set: skipping Neo4j ingestion")
        print("\n✅ Dry run complete. Output files saved to:", output_dir)
        return

    print("\n🏗️  Phase 3: Building Knowledge Graph in Neo4j...")
    from app.law_processing.kg_builder import build_knowledge_graph

    build_knowledge_graph(
        chunks_2013=chunks_2013,
        chunks_2024=chunks_2024,
        cross_mapping=cross_mapping_dicts,
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_username=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        clear_existing=not args.no_clear,
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete! Suggested Cypher queries to verify:")
    print("=" * 60)
    print("""
  // Count articles by law year
  MATCH (a:Article) RETURN a.law_year, count(a) ORDER BY a.law_year

  // Explore cross-law changes
  MATCH (a2024:Article)-[r:SUPERSEDES]->(a2013:Article)
  RETURN a2024.article_number, a2024.title, r.change_type, a2013.article_number
  ORDER BY a2024.article_number LIMIT 20

  // Show concept definitions
  MATCH (a:Article)-[:DEFINES_CONCEPT]->(c:Concept)
  RETURN a.law_year, a.article_number, c.name, c.definition LIMIT 10

  // Find articles about land recovery (thu hồi đất)
  MATCH (a:Article)
  WHERE toLower(a.content) CONTAINS 'thu hồi đất'
  RETURN a.law_year, a.article_number, a.title LIMIT 10
""")


if __name__ == "__main__":
    main()
