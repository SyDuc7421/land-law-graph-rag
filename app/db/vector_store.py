"""
Neo4j Vector Store wrapper.

Tạo / lấy Neo4j Vector index cho Article nodes để phục vụ semantic search
kết hợp với graph traversal.

"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

from app.core.config import settings

logger = logging.getLogger(__name__)

# Tên index trong Neo4j
ARTICLE_VECTOR_INDEX = "article_embedding_index"
CLAUSE_VECTOR_INDEX  = "clause_embedding_index"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )


@lru_cache(maxsize=1)
def get_article_vector_store() -> Neo4jVector:
    """
    Trả về Neo4jVector store kết nối tới index 'article_embedding_index'.
    Nếu index chưa tồn tại, tự động tạo (từ existing nodes – cần embedding sẵn).
    """
    embeddings = get_embeddings()
    try:
        store = Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            index_name=ARTICLE_VECTOR_INDEX,
            node_label="Article",
            embedding_node_property="embedding",
            text_node_property="content",
        )
        logger.info("Connected to existing Neo4j vector index '%s'.", ARTICLE_VECTOR_INDEX)
        return store
    except Exception as e:
        logger.warning(
            "Could not load existing vector index (%s). "
            "Run app/scripts/create_vector_index.py to create embeddings.",
            e,
        )
        raise


def check_vector_index_exists(index_name: str = ARTICLE_VECTOR_INDEX) -> bool:
    """Kiểm tra vector index tồn tại trong Neo4j."""
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            result = session.run(
                "SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' AND name = $name RETURN count(*) AS cnt",
                name=index_name,
            )
            return result.single()["cnt"] > 0
    finally:
        driver.close()
