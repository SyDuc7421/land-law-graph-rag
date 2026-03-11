"""
Script một lần (one-time) để tạo embeddings cho Article nodes trong Neo4j
và đăng ký Vector Index.

Chạy: python -m app.scripts.create_vector_index
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from openai import OpenAI
from app.core.config import settings
from app.db.vector_store import ARTICLE_VECTOR_INDEX, EMBEDDING_MODEL, EMBEDDING_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 50  # số article mỗi batch embedding


def get_articles_without_embedding(session) -> List[Dict[str, Any]]:
    result = session.run(
        """
        MATCH (a:Article)
        WHERE a.embedding IS NULL
        RETURN a.chunk_id AS chunk_id, a.content AS content,
               a.title AS title, a.article_number AS article_number
        ORDER BY a.law_year, a.article_number
        """
    )
    return [dict(r) for r in result]


def create_embeddings_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def upsert_embeddings(session, chunk_ids: List[str], embeddings: List[List[float]]):
    session.run(
        """
        UNWIND $rows AS row
        MATCH (a:Article {chunk_id: row.chunk_id})
        SET a.embedding = row.embedding
        """,
        rows=[{"chunk_id": cid, "embedding": emb}
              for cid, emb in zip(chunk_ids, embeddings)],
    )


def create_vector_index(session):
    """Tạo Neo4j Vector Index nếu chưa có."""
    try:
        session.run(
            f"""
            CREATE VECTOR INDEX {ARTICLE_VECTOR_INDEX} IF NOT EXISTS
            FOR (a:Article) ON (a.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIM},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )
        logger.info("Vector index '%s' created (or already exists).", ARTICLE_VECTOR_INDEX)
    except Exception as e:
        logger.error("Failed to create vector index: %s", e)
        raise


def main():
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    with driver.session() as session:
        # 1. Tạo index trước
        create_vector_index(session)

        # 2. Lấy articles chưa có embedding
        articles = get_articles_without_embedding(session)
        logger.info("Found %d articles needing embeddings.", len(articles))

        if not articles:
            logger.info("All articles already have embeddings. Done.")
            return

        # 3. Tạo embedding theo batch
        total = len(articles)
        for i in range(0, total, BATCH_SIZE):
            batch = articles[i : i + BATCH_SIZE]
            texts = [
                f"{a['title']}\n{a['content'][:2000]}" for a in batch
            ]
            chunk_ids = [a["chunk_id"] for a in batch]

            try:
                embeddings = create_embeddings_batch(client, texts)
                upsert_embeddings(session, chunk_ids, embeddings)
                logger.info(
                    "Embedded batch %d-%d / %d", i + 1, min(i + BATCH_SIZE, total), total
                )
                time.sleep(0.3)  # avoid rate limiting
            except Exception as e:
                logger.error("Batch %d failed: %s", i, e)
                raise

    driver.close()
    logger.info("Vector index setup complete.")


if __name__ == "__main__":
    main()
