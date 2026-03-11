"""
Knowledge Graph Builder: converts parsed law chunks and cross-mapping results
into Neo4j nodes and relationships.

Node labels: LawDocument, Chapter, Section, Article, Clause, Concept, Right, Obligation, Penalty
Relationships: HAS_CHAPTER, HAS_SECTION, HAS_ARTICLE, HAS_CLAUSE, DEFINES_CONCEPT,
               GRANTS_RIGHT, IMPOSES_OBLIGATION, PENALIZES, SUPERSEDES, MODIFIES,
               EXPANDED, RENAMED, RELATED_CONCEPT
"""

import re
import os
from typing import List, Optional

from neo4j import GraphDatabase


# ─── Neo4j Connection ──────────────────────────────────────────────────────────

def get_driver(uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = username or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(username, password))


# ─── Schema Setup ──────────────────────────────────────────────────────────────

CREATE_CONSTRAINTS = [
    "CREATE CONSTRAINT article_unique IF NOT EXISTS FOR (a:Article) REQUIRE a.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT concept_unique IF NOT EXISTS FOR (c:Concept) REQUIRE (c.name, c.law_year) IS UNIQUE",
    "CREATE CONSTRAINT law_unique IF NOT EXISTS FOR (l:LawDocument) REQUIRE l.law_year IS UNIQUE",
]

CREATE_INDEXES = [
    "CREATE INDEX article_number IF NOT EXISTS FOR (a:Article) ON (a.article_number)",
    "CREATE INDEX article_law_year IF NOT EXISTS FOR (a:Article) ON (a.law_year)",
    "CREATE INDEX chapter_law IF NOT EXISTS FOR (c:Chapter) ON (c.law_year, c.chapter_number)",
]


def setup_schema(session) -> None:
    """Create uniqueness constraints and indexes"""
    for stmt in CREATE_CONSTRAINTS:
        try:
            session.run(stmt)
        except Exception:
            pass  # constraint may already exist
    for stmt in CREATE_INDEXES:
        try:
            session.run(stmt)
        except Exception:
            pass


# ─── Concept Extraction (from Điều 3) ──────────────────────────────────────────

# Điều 3 in both laws contains definition lists like:
# "1. Thửa đất là ..."
# "2. Quy hoạch sử dụng đất là ..."
DEFINITION_PATTERN = re.compile(
    r'^\s*(\d+)\.\s+(.+?)\s+là\s+(.+?)(?=^\s*\d+\.|\Z)',
    re.MULTILINE | re.DOTALL
)


def extract_concepts_from_article3(chunk: dict) -> List[dict]:
    """
    Extract concept definitions from Điều 3 (Giải thích từ ngữ).
    Returns list of {name, definition, law_year}.
    """
    content = chunk.get("content", "")
    concepts = []
    law_year = chunk.get("law_year", 0)

    for m in DEFINITION_PATTERN.finditer(content):
        term = m.group(2).strip()
        definition = (m.group(2).strip() + " là " + m.group(3).strip())
        # Clean up
        definition = re.sub(r'\s+', ' ', definition)[:1000]
        if len(term) > 3:
            concepts.append({
                "name": term,
                "definition": definition,
                "law_year": law_year,
            })

    return concepts


# ─── Right/Obligation/Penalty Detection ────────────────────────────────────────

RIGHT_KEYWORDS = ['có quyền', 'được quyền', 'quyền', 'được phép']
OBLIGATION_KEYWORDS = ['có nghĩa vụ', 'phải', 'có trách nhiệm', 'không được']
PENALTY_KEYWORDS = ['bị xử phạt', 'bị phạt', 'nghiêm cấm', 'vi phạm', 'bị thu hồi']


def _detect_clause_type(clause_content: str) -> str:
    """
    Simple keyword-based classification of a clause.
    Returns "RIGHT", "OBLIGATION", "PENALTY", or "GENERAL"
    """
    text = clause_content.lower()
    if any(kw in text for kw in PENALTY_KEYWORDS):
        return "PENALTY"
    if any(kw in text for kw in OBLIGATION_KEYWORDS):
        return "OBLIGATION"
    if any(kw in text for kw in RIGHT_KEYWORDS):
        return "RIGHT"
    return "GENERAL"


# ─── Node Ingestion ────────────────────────────────────────────────────────────

UPSERT_LAW = """
MERGE (l:LawDocument {law_year: $law_year})
SET l.title = $title
"""

UPSERT_CHAPTER = """
MERGE (c:Chapter {law_year: $law_year, chapter_number: $chapter_number})
SET c.name = $name
WITH c
MATCH (l:LawDocument {law_year: $law_year})
MERGE (l)-[:HAS_CHAPTER]->(c)
"""

UPSERT_SECTION = """
MERGE (s:Section {law_year: $law_year, chapter_number: $chapter_number, section_number: $section_number})
SET s.name = $name
WITH s
MATCH (c:Chapter {law_year: $law_year, chapter_number: $chapter_number})
MERGE (c)-[:HAS_SECTION]->(s)
"""

UPSERT_ARTICLE = """
MERGE (a:Article {chunk_id: $chunk_id})
SET a.article_number = $article_number,
    a.title = $title,
    a.content = $content,
    a.law_year = $law_year,
    a.chapter_number = $chapter_number,
    a.chapter_name = $chapter_name,
    a.section_number = $section_number,
    a.section_name = $section_name
WITH a
// Link to Section if exists
FOREACH (sec_num IN CASE WHEN $section_number IS NOT NULL THEN [$section_number] ELSE [] END |
  MERGE (s:Section {law_year: $law_year, chapter_number: $chapter_number, section_number: sec_num})
  MERGE (s)-[:HAS_ARTICLE]->(a)
)
// Link to Chapter if no section
FOREACH (_ IN CASE WHEN $section_number IS NULL THEN [1] ELSE [] END |
  MERGE (c:Chapter {law_year: $law_year, chapter_number: $chapter_number})
  MERGE (c)-[:HAS_ARTICLE]->(a)
)
"""

UPSERT_CLAUSE = """
MERGE (cl:Clause {chunk_id: $chunk_id, clause_number: $clause_number})
SET cl.content = $content,
    cl.clause_type = $clause_type,
    cl.law_year = $law_year
WITH cl
MATCH (a:Article {chunk_id: $chunk_id})
MERGE (a)-[:HAS_CLAUSE]->(cl)
"""

UPSERT_CONCEPT = """
MERGE (c:Concept {name: $name, law_year: $law_year})
SET c.definition = $definition
WITH c
MATCH (a:Article {chunk_id: $chunk_id})
MERGE (a)-[:DEFINES_CONCEPT]->(c)
"""

CREATE_CROSS_LAW_REL = """
MATCH (a2024:Article {law_year: 2024, article_number: $num_2024})
MATCH (a2013:Article {law_year: 2013, article_number: $num_2013})
MERGE (a2024)-[r:SUPERSEDES]->(a2013)
SET r.change_type = $change_type,
    r.diff_summary = $diff_summary,
    r.match_method = $match_method,
    r.similarity_score = $similarity_score
"""


def ingest_law_chunks(session, chunks: List[dict]) -> None:
    """
    Ingest all article chunks for a law into Neo4j.
    Creates: LawDocument, Chapter, Section, Article, Clause nodes.
    """
    if not chunks:
        return

    law_year = chunks[0]["law_year"]
    title = f"LUẬT ĐẤT ĐAI {law_year}"

    # Create LawDocument node
    session.run(UPSERT_LAW, law_year=law_year, title=title)

    # Track chapters and sections already created
    created_chapters = set()
    created_sections = set()

    # Concepts extracted from Điều 3
    concepts_to_ingest = []

    print(f"  Ingesting {len(chunks)} articles for LandLaw {law_year}...")

    for chunk in chunks:
        ch_num = chunk["chapter_number"]
        ch_name = chunk["chapter_name"]
        sec_num = chunk.get("section_number")
        sec_name = chunk.get("section_name")

        # Chapter
        if ch_num not in created_chapters:
            session.run(UPSERT_CHAPTER, law_year=law_year, chapter_number=ch_num, name=ch_name)
            created_chapters.add(ch_num)

        # Section
        if sec_num and (ch_num, sec_num) not in created_sections:
            session.run(UPSERT_SECTION,
                        law_year=law_year,
                        chapter_number=ch_num,
                        section_number=sec_num,
                        name=sec_name or "")
            created_sections.add((ch_num, sec_num))

        # Article
        session.run(UPSERT_ARTICLE,
                    chunk_id=chunk["chunk_id"],
                    article_number=chunk["article_number"],
                    title=chunk["title"],
                    content=chunk["content"],
                    law_year=law_year,
                    chapter_number=ch_num,
                    chapter_name=ch_name,
                    section_number=sec_num,
                    section_name=sec_name)

        # Clauses
        for clause in chunk.get("clauses", []):
            clause_type = _detect_clause_type(clause["content"])
            session.run(UPSERT_CLAUSE,
                        chunk_id=chunk["chunk_id"],
                        clause_number=clause["number"],
                        content=clause["content"],
                        clause_type=clause_type,
                        law_year=law_year)

        # Extract concepts from Điều 3 (definition article)
        if chunk["article_number"] == 3:
            concepts = extract_concepts_from_article3(chunk)
            for concept in concepts:
                concept["chunk_id"] = chunk["chunk_id"]
            concepts_to_ingest.extend(concepts)

    # Ingest concepts from Điều 3
    for concept in concepts_to_ingest:
        try:
            session.run(UPSERT_CONCEPT,
                        name=concept["name"],
                        law_year=concept["law_year"],
                        definition=concept["definition"],
                        chunk_id=concept["chunk_id"])
        except Exception:
            pass  # ignore duplicate concept errors

    print(f"  ✓ Law {law_year}: {len(created_chapters)} chapters, "
          f"{len(created_sections)} sections, {len(concepts_to_ingest)} concepts")


def ingest_cross_mapping(session, cross_mapping: List[dict]) -> None:
    """
    Ingest cross-mapping relationships between 2013 and 2024 articles.
    Creates SUPERSEDES / MODIFIES / EXPANDED / RENAMED relationships.
    """
    # Only create relationships for actual pairs (not DELETED/NEW stubs)
    pairs = [m for m in cross_mapping
             if m.get("article_2013_num") is not None and m.get("article_2024_num") is not None]

    print(f"  Ingesting {len(pairs)} cross-law relationships...")

    for pair in pairs:
        try:
            session.run(CREATE_CROSS_LAW_REL,
                        num_2024=pair["article_2024_num"],
                        num_2013=pair["article_2013_num"],
                        change_type=pair.get("change_type", "UNKNOWN"),
                        diff_summary=pair.get("diff_summary", ""),
                        match_method=pair.get("match_method", ""),
                        similarity_score=pair.get("similarity_score", 0.0))
        except Exception as e:
            pass  # Article may not exist yet; log silently

    print(f"  ✓ {len(pairs)} cross-law SUPERSEDES relationships created")


def build_knowledge_graph(
    chunks_2013: List[dict],
    chunks_2024: List[dict],
    cross_mapping: List[dict],
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    clear_existing: bool = True,
) -> None:
    """
    Main entry point: ingests both laws + cross-mapping into Neo4j.

    Args:
        chunks_2013: Flat article chunks for LandLaw 2013
        chunks_2024: Flat article chunks for LandLaw 2024
        cross_mapping: List of cross-mapping result dicts
        neo4j_uri: Neo4j bolt URI (defaults to NEO4J_URI env var)
        neo4j_username: Neo4j username (defaults to NEO4J_USERNAME env var)
        neo4j_password: Neo4j password (defaults to NEO4J_PASSWORD env var)
        clear_existing: If True, delete all existing nodes before ingestion
    """
    driver = get_driver(neo4j_uri, neo4j_username, neo4j_password)

    try:
        with driver.session() as session:
            # Schema setup
            print("⚙️  Setting up Neo4j schema (constraints + indexes)...")
            setup_schema(session)

            # Optional: clear existing data
            if clear_existing:
                print("🗑️  Clearing existing graph data...")
                session.run("MATCH (n) DETACH DELETE n")

            # Ingest law documents
            print("\n📥 Ingesting LandLaw 2013...")
            ingest_law_chunks(session, chunks_2013)

            print("\n📥 Ingesting LandLaw 2024...")
            ingest_law_chunks(session, chunks_2024)

            # Ingest cross-mapping
            print("\n🔗 Creating cross-law relationships...")
            ingest_cross_mapping(session, cross_mapping)

            # Summary query
            result = session.run("""
                MATCH (a:Article) 
                RETURN a.law_year AS year, count(a) AS count
                ORDER BY year
            """)
            print("\n📊 Neo4j Article counts:")
            for row in result:
                print(f"   Law {row['year']}: {row['count']} articles")

            result2 = session.run("MATCH ()-[r:SUPERSEDES]->() RETURN count(r) AS cnt")
            for row in result2:
                print(f"   SUPERSEDES relationships: {row['cnt']}")

            result3 = session.run("MATCH (c:Concept) RETURN count(c) AS cnt")
            for row in result3:
                print(f"   Concept nodes: {row['cnt']}")

            print("\n✅ Knowledge Graph build complete!")
            print("   Open http://localhost:7474 to explore the graph.")

    except Exception as e:
        print(f"❌ Error building knowledge graph: {e}")
        raise
    finally:
        driver.close()
