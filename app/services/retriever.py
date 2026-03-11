"""
Retriever – Truy xuất điều luật theo chiến lược tương ứng với từng QueryType.

Chiến lược:
  SO_SANH  → KG: SUPERSEDES + vector similarity cho cả 2 phiên bản
  TRA_CUU  → Hybrid: vector search + keyword Cypher fallback
  CHU_THE  → KG: tìm Clause (RIGHT/OBLIGATION) theo chủ thể + vector
  THAY_DOI → KG: toàn bộ SUPERSEDES relationships có diff_summary + vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

from app.core.config import settings
from app.db.neo4j import get_graph
from app.services.query_classifier import ClassifiedQuery, QueryType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """Một mảnh ngữ cảnh đã truy xuất được."""
    source: str                    # "graph" | "vector" | "hybrid"
    node_type: str                 # "Article" | "Clause" | "Concept"
    chunk_id: str
    law_year: int
    article_number: Optional[int]
    title: Optional[str]
    content: str
    chapter_name: Optional[str]   = None
    section_name: Optional[str]   = None
    extra: Dict[str, Any]         = field(default_factory=dict)
    score: float                  = 1.0   # vector similarity score khi có


@dataclass
class RetrievalResult:
    """Kết quả truy xuất đầy đủ."""
    query_type: QueryType
    chunks: List[RetrievedChunk]
    cypher_used: List[str]        = field(default_factory=list)
    vector_used: bool             = False
    total_found: int              = 0


# ---------------------------------------------------------------------------
# Helper: chạy Cypher an toàn
# ---------------------------------------------------------------------------

def _run_cypher(graph: Neo4jGraph, cypher: str, params: Dict = {}) -> List[Dict]:
    try:
        return graph.query(cypher, params)
    except Exception as exc:
        logger.warning("Cypher failed: %s | %s", exc, cypher[:120])
        return []


# ---------------------------------------------------------------------------
# Helper: trích token tiếng Việt từ câu hỏi gốc (loại stop-words)
# ---------------------------------------------------------------------------

_VI_STOPWORDS = {
    "là", "gì", "và", "của", "có", "trong", "không", "với", "được", "theo",
    "về", "như", "thế", "nào", "các", "những", "này", "đó", "hay", "hoặc",
    "khi", "từ", "đến", "cho", "rằng", "bằng", "để", "vì", "tại", "sau",
    "trước", "cũng", "đã", "đang", "sẽ", "một", "hai", "ba", "nhiều", "ít",
    "ra", "vào", "lên", "xuống", "đây", "đó", "thì", "mà", "nhưng", "vậy",
    "phải", "cần", "do", "bởi", "qua", "lại", "đối", "hết", "chỉ", "tất",
}


def _extract_tokens(question: str, min_len: int = 4) -> List[str]:
    """Trả về list từ có nghĩa từ câu hỏi, loại stop-words và từ quá ngắn."""
    import re
    words = re.findall(r"[\w]+", question, flags=re.UNICODE)
    return [
        w for w in words
        if len(w) >= min_len and w.lower() not in _VI_STOPWORDS
    ]

def _expand_keywords(keywords: List[str]) -> List[str]:
    """
    Mở rộng danh sách từ khóa bằng cách tách cụm từ thành các bi-gram.
    Ví dụ: 'thu hồi đất' → ['thu hồi đất', 'thu hồi', 'sở hữu tư nhân'] + ['sở hữu'].
    Đảm bảo Cyử phải tìm được cả khi kiếm trú ngữ linh hoạt.
    """
    expanded: List[str] = list(keywords)
    for kw in keywords:
        parts = kw.split()
        # add 2-word sub-phrases (bigrams)
        for i in range(len(parts) - 1):
            bigram = " ".join(parts[i:i+2])
            if bigram not in expanded:
                expanded.append(bigram)
        # add individual words >= 4 chars
        for p in parts:
            if len(p) >= 4 and p not in expanded and p.lower() not in _VI_STOPWORDS:
                expanded.append(p)
    return expanded[:12]  # cap để tránh query quá dài

# ---------------------------------------------------------------------------
# Broad fallback: tìm article bằng OR trên từng token (last resort)
# ---------------------------------------------------------------------------

def _broad_fallback_search(
    graph: Neo4jGraph,
    question: str,
    years: List[int],
    top_k: int,
) -> List[RetrievedChunk]:
    """
    Last-resort: tìm Article có nội dung chứa ÍT NHẤT 1 token từ câu hỏi.
    Dùng title CONTAINS để ưu tiên kết quả liên quan hơn.
    """
    tokens = _extract_tokens(question)
    if not tokens:
        return []

    chunks: List[RetrievedChunk] = []
    for yr in years:
        # Ưu tiên tìm theo title trước
        title_cypher = """
        MATCH (a:Article {law_year: $yr})
        WHERE ANY(t IN $tokens WHERE toLower(a.title) CONTAINS toLower(t))
        RETURN a.chunk_id AS chunk_id, a.article_number AS article_number,
               a.title AS title, a.content AS content,
               a.law_year AS law_year, a.chapter_name AS chapter_name
        ORDER BY a.article_number
        LIMIT $top_k
        """
        rows = _run_cypher(graph, title_cypher, {"yr": yr, "tokens": tokens, "top_k": top_k})
        for row in rows:
            if row.get("content"):
                chunks.append(RetrievedChunk(
                    source="graph", node_type="Article",
                    chunk_id=row["chunk_id"], law_year=row["law_year"],
                    article_number=row["article_number"],
                    title=row["title"], content=row["content"],
                    chapter_name=row.get("chapter_name"),
                ))

        if len(chunks) < 3:
            # Mở rộng sang content
            content_cypher = """
            MATCH (a:Article {law_year: $yr})
            WHERE ANY(t IN $tokens WHERE toLower(a.content) CONTAINS toLower(t))
            RETURN a.chunk_id AS chunk_id, a.article_number AS article_number,
                   a.title AS title, a.content AS content,
                   a.law_year AS law_year, a.chapter_name AS chapter_name
            ORDER BY a.article_number
            LIMIT $top_k
            """
            rows2 = _run_cypher(graph, content_cypher, {"yr": yr, "tokens": tokens[:6], "top_k": top_k})
            for row in rows2:
                if row.get("content"):
                    chunks.append(RetrievedChunk(
                        source="graph", node_type="Article",
                        chunk_id=row["chunk_id"], law_year=row["law_year"],
                        article_number=row["article_number"],
                        title=row["title"], content=row["content"],
                        chapter_name=row.get("chapter_name"),
                    ))

    logger.info("Broad fallback: %d chunks for tokens=%s", len(chunks), tokens[:4])
    return chunks


# ---------------------------------------------------------------------------
# Vector Search helper (lazy import để không crash nếu chưa có index)
# ---------------------------------------------------------------------------

def _vector_search(question: str, top_k: int = 6, law_year: Optional[int] = None) -> List[RetrievedChunk]:
    """
    Semantic search trên Neo4j Vector index.
    Trả về list RetrievedChunk, rỗng nếu index chưa tồn tại.
    """
    try:
        from app.db.vector_store import get_article_vector_store
        store = get_article_vector_store()

        filter_dict = {"law_year": law_year} if law_year else None
        docs: List[Document] = store.similarity_search_with_score(
            query=question,
            k=top_k,
            filter=filter_dict,
        ) if filter_dict else store.similarity_search_with_score(question, k=top_k)

        chunks: List[RetrievedChunk] = []
        for doc, score in docs:  # type: ignore[misc]
            md = doc.metadata
            chunks.append(RetrievedChunk(
                source="vector",
                node_type="Article",
                chunk_id=md.get("chunk_id", ""),
                law_year=int(md.get("law_year", 0)),
                article_number=md.get("article_number"),
                title=md.get("title"),
                content=doc.page_content,
                chapter_name=md.get("chapter_name"),
                section_name=md.get("section_name"),
                score=float(score),
            ))
        return chunks
    except Exception as exc:
        logger.warning("Vector search unavailable: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Chiến lược 1: SO_SANH – so sánh điều khoản 2013 vs 2024
# ---------------------------------------------------------------------------

def _retrieve_so_sanh(
    graph: Neo4jGraph,
    cq: ClassifiedQuery,
    question: str,
    top_k: int,
) -> RetrievalResult:
    chunks: List[RetrievedChunk] = []
    cyphers: List[str] = []

    # --- KG: SUPERSEDES relationship ---
    if cq.article_numbers:
        for art_num in cq.article_numbers[:3]:
            cypher = """
            MATCH (a24:Article {law_year: 2024, article_number: $art_num})
            OPTIONAL MATCH (a24)-[r:SUPERSEDES]->(a13:Article {law_year: 2013, article_number: $art_num})
            RETURN
                a24.chunk_id        AS chunk_id_2024,
                a24.article_number  AS article_number,
                a24.title           AS title_2024,
                a24.content         AS content_2024,
                a24.chapter_name    AS chapter_name_2024,
                a13.chunk_id        AS chunk_id_2013,
                a13.title           AS title_2013,
                a13.content         AS content_2013,
                a13.chapter_name    AS chapter_name_2013,
                r.change_type       AS change_type,
                r.diff_summary      AS diff_summary,
                r.similarity_score  AS similarity_score
            """
            rows = _run_cypher(graph, cypher, {"art_num": art_num})
            cyphers.append(cypher)
            for row in rows:
                if row.get("content_2024"):
                    chunks.append(RetrievedChunk(
                        source="graph",
                        node_type="Article",
                        chunk_id=row.get("chunk_id_2024", ""),
                        law_year=2024,
                        article_number=row.get("article_number"),
                        title=row.get("title_2024"),
                        content=row.get("content_2024", ""),
                        chapter_name=row.get("chapter_name_2024"),
                        extra={
                            "change_type": row.get("change_type"),
                            "diff_summary": row.get("diff_summary"),
                            "similarity_score": row.get("similarity_score"),
                        },
                    ))
                if row.get("content_2013"):
                    chunks.append(RetrievedChunk(
                        source="graph",
                        node_type="Article",
                        chunk_id=row.get("chunk_id_2013", ""),
                        law_year=2013,
                        article_number=row.get("article_number"),
                        title=row.get("title_2013"),
                        content=row.get("content_2013", ""),
                        chapter_name=row.get("chapter_name_2013"),
                    ))
    else:
        # Không có số điều cụ thể → tìm theo từ khóa trong cả 2 version
        keyword_cypher = """
        MATCH (a24:Article {law_year: 2024})-[r:SUPERSEDES]->(a13:Article {law_year: 2013})
        WHERE ANY(kw IN $keywords WHERE toLower(a24.content) CONTAINS toLower(kw))
        RETURN
            a24.chunk_id AS chunk_id_2024, a24.article_number AS article_number,
            a24.title AS title_2024, a24.content AS content_2024, a24.chapter_name AS chapter_name_2024,
            a13.chunk_id AS chunk_id_2013, a13.content AS content_2013, a13.chapter_name AS chapter_name_2013,
            r.change_type AS change_type, r.diff_summary AS diff_summary
        LIMIT $top_k
        """
        raw_kws = cq.keywords if cq.keywords else _extract_tokens(question)
        kws = _expand_keywords(raw_kws)
        rows = _run_cypher(graph, keyword_cypher, {"keywords": kws, "top_k": top_k})
        cyphers.append(keyword_cypher)
        for row in rows:
            for yr, ck, tt, ct, ch in [
                (2024, row.get("chunk_id_2024"), row.get("title_2024"),
                 row.get("content_2024"), row.get("chapter_name_2024")),
                (2013, row.get("chunk_id_2013"), None,
                 row.get("content_2013"), row.get("chapter_name_2013")),
            ]:
                if ct:
                    chunks.append(RetrievedChunk(
                        source="graph", node_type="Article",
                        chunk_id=ck or "", law_year=yr,
                        article_number=row.get("article_number"),
                        title=tt, content=ct, chapter_name=ch,
                        extra={"change_type": row.get("change_type"),
                               "diff_summary": row.get("diff_summary")},
                    ))

    # --- Vector augmentation ---
    v_chunks = _vector_search(question, top_k=4)
    chunks.extend(v_chunks)

    return RetrievalResult(
        query_type=QueryType.SO_SANH,
        chunks=_deduplicate(chunks),
        cypher_used=cyphers,
        vector_used=bool(v_chunks),
        total_found=len(chunks),
    )


# ---------------------------------------------------------------------------
# Chiến lược 2: TRA_CUU – tra cứu nội dung điều khoản
# ---------------------------------------------------------------------------

def _retrieve_tra_cuu(
    graph: Neo4jGraph,
    cq: ClassifiedQuery,
    question: str,
    top_k: int,
) -> RetrievalResult:
    chunks: List[RetrievedChunk] = []
    cyphers: List[str] = []
    target_years = cq.law_years if cq.law_years else [2013, 2024]

    # --- KG: tra cứu trực tiếp theo số điều ---
    if cq.article_numbers:
        for art_num in cq.article_numbers[:5]:
            for yr in target_years:
                cypher = """
                MATCH (a:Article {article_number: $art_num, law_year: $yr})
                OPTIONAL MATCH (a)-[:HAS_CLAUSE]->(cl:Clause)
                OPTIONAL MATCH (a)-[:DEFINES_CONCEPT]->(con:Concept)
                RETURN
                    a.chunk_id AS chunk_id, a.article_number AS article_number,
                    a.title AS title, a.content AS content,
                    a.law_year AS law_year, a.chapter_name AS chapter_name,
                    a.section_name AS section_name,
                    collect(DISTINCT {number: cl.clause_number, content: cl.content,
                                      type: cl.clause_type}) AS clauses,
                    collect(DISTINCT {name: con.name, definition: con.definition}) AS concepts
                """
                rows = _run_cypher(graph, cypher, {"art_num": art_num, "yr": yr})
                cyphers.append(cypher)
                for row in rows:
                    if row.get("content"):
                        chunks.append(RetrievedChunk(
                            source="graph", node_type="Article",
                            chunk_id=row["chunk_id"],
                            law_year=row["law_year"],
                            article_number=row["article_number"],
                            title=row["title"], content=row["content"],
                            chapter_name=row.get("chapter_name"),
                            section_name=row.get("section_name"),
                            extra={
                                "clauses": row.get("clauses", []),
                                "concepts": row.get("concepts", []),
                            },
                        ))

    # --- KG: tìm theo keyword nếu không có số điều hoặc cần bổ sung ---
    if len(chunks) < top_k:
        raw_kws = cq.keywords if cq.keywords else _extract_tokens(question)
        kws = _expand_keywords(raw_kws)
        for yr in target_years:
            kw_cypher = """
            MATCH (a:Article {law_year: $yr})
            WHERE ANY(kw IN $keywords WHERE toLower(a.content) CONTAINS toLower(kw))
            WITH a,
                 CASE WHEN ANY(kw IN $keywords WHERE toLower(a.title) CONTAINS toLower(kw))
                      THEN 1 ELSE 0 END AS title_hit
            RETURN a.chunk_id AS chunk_id, a.article_number AS article_number,
                   a.title AS title, a.content AS content,
                   a.law_year AS law_year, a.chapter_name AS chapter_name,
                   a.section_name AS section_name
            ORDER BY title_hit DESC, a.article_number
            LIMIT $top_k
            """
            rows = _run_cypher(graph, kw_cypher, {"keywords": kws, "yr": yr, "top_k": top_k})
            cyphers.append(kw_cypher)
            for row in rows:
                if row.get("content"):
                    chunks.append(RetrievedChunk(
                        source="graph", node_type="Article",
                        chunk_id=row["chunk_id"], law_year=row["law_year"],
                        article_number=row["article_number"],
                        title=row["title"], content=row["content"],
                        chapter_name=row.get("chapter_name"),
                        section_name=row.get("section_name"),
                    ))

    # --- Cross-ref: nếu câu hỏi nhắc đến số điều → tìm điều đó + các điều tham chiếu ---
    if cq.article_numbers:
        for art_num in cq.article_numbers[:3]:
            ref_cypher = """
            MATCH (a:Article)
            WHERE a.law_year IN [2013, 2024]
              AND (
                toLower(a.content) CONTAINS ('điều ' + toString($art_num))
                OR a.content CONTAINS ('Điều ' + toString($art_num))
                OR a.article_number = $art_num
              )
            RETURN a.chunk_id AS chunk_id, a.article_number AS article_number,
                   a.title AS title, a.content AS content,
                   a.law_year AS law_year, a.chapter_name AS chapter_name,
                   a.section_name AS section_name
            ORDER BY a.law_year DESC, a.article_number
            LIMIT $top_k
            """
            rows = _run_cypher(graph, ref_cypher, {"art_num": art_num, "top_k": top_k})
            cyphers.append(ref_cypher)
            for row in rows:
                if row.get("content"):
                    chunks.append(RetrievedChunk(
                        source="graph", node_type="Article",
                        chunk_id=row["chunk_id"], law_year=row["law_year"],
                        article_number=row["article_number"],
                        title=row["title"], content=row["content"],
                        chapter_name=row.get("chapter_name"),
                        section_name=row.get("section_name"),
                        extra={"is_cross_ref": True, "ref_article": art_num},
                    ))

    # --- Vector search ---
    for yr in (target_years if len(target_years) == 1 else [None]):
        v_chunks = _vector_search(question, top_k=top_k, law_year=yr)  # type: ignore[arg-type]
        chunks.extend(v_chunks)

    # --- Last resort broadfallback ---
    if len(_deduplicate(chunks)) < 2:
        chunks.extend(_broad_fallback_search(graph, question, target_years, top_k))

    return RetrievalResult(
        query_type=QueryType.TRA_CUU,
        chunks=_deduplicate(chunks)[:top_k * 2],
        cypher_used=cyphers,
        vector_used=True,
        total_found=len(chunks),
    )


# ---------------------------------------------------------------------------
# Chiến lược 3: CHU_THE – thẩm quyền / quyền / nghĩa vụ chủ thể
# ---------------------------------------------------------------------------

def _retrieve_chu_the(
    graph: Neo4jGraph,
    cq: ClassifiedQuery,
    question: str,
    top_k: int,
) -> RetrievalResult:
    chunks: List[RetrievedChunk] = []
    cyphers: List[str] = []
    target_years = cq.law_years if cq.law_years else [2013, 2024]

    subject = cq.subject_entity or ""
    search_terms = [subject] + cq.keywords if subject else list(cq.keywords)
    # If no keywords at all, extract from question
    if not search_terms:
        search_terms = _extract_tokens(question)

    # --- KG: Direct article lookup by number (highest priority) ---
    if cq.article_numbers:
        for art_num in cq.article_numbers[:5]:
            for yr in target_years:
                direct_cypher = """
                MATCH (a:Article {article_number: $art_num, law_year: $yr})
                OPTIONAL MATCH (a)-[:HAS_CLAUSE]->(cl:Clause)
                RETURN a.chunk_id AS chunk_id, a.article_number AS article_number,
                       a.title AS title, a.content AS content,
                       a.law_year AS law_year, a.chapter_name AS chapter_name,
                       collect(DISTINCT {number: cl.clause_number, content: cl.content,
                                         type: cl.clause_type}) AS clauses
                """
                rows = _run_cypher(graph, direct_cypher, {"art_num": art_num, "yr": yr})
                cyphers.append(direct_cypher)
                for row in rows:
                    if row.get("content"):
                        chunks.append(RetrievedChunk(
                            source="graph", node_type="Article",
                            chunk_id=row["chunk_id"], law_year=row["law_year"],
                            article_number=row["article_number"],
                            title=row["title"], content=row["content"],
                            chapter_name=row.get("chapter_name"),
                            extra={"clauses": row.get("clauses", [])},
                        ))

    # --- KG: Clause-level search (RIGHT / OBLIGATION) ---
    for yr in target_years:
        clause_cypher = """
        MATCH (a:Article {law_year: $yr})-[:HAS_CLAUSE]->(cl:Clause)
        WHERE cl.clause_type IN ['RIGHT', 'OBLIGATION', 'GENERAL']
          AND a.article_number > 5
          AND ANY(term IN $terms WHERE toLower(cl.content) CONTAINS toLower(term))
        RETURN
            a.chunk_id AS chunk_id, a.article_number AS article_number,
            a.title AS title, a.law_year AS law_year, a.chapter_name AS chapter_name,
            a.content AS article_content,
            cl.clause_number AS clause_number, cl.content AS clause_content,
            cl.clause_type AS clause_type
        ORDER BY a.article_number, cl.clause_number
        LIMIT $top_k
        """
        terms = _expand_keywords(search_terms) if search_terms else _extract_tokens(question)
        rows = _run_cypher(graph, clause_cypher, {"yr": yr, "terms": terms, "top_k": top_k})
        cyphers.append(clause_cypher)
        for row in rows:
            if row.get("clause_content"):
                chunks.append(RetrievedChunk(
                    source="graph", node_type="Clause",
                    chunk_id=f"{row['chunk_id']}_k{row['clause_number']}",
                    law_year=row["law_year"],
                    article_number=row["article_number"],
                    title=row["title"],
                    content=row["clause_content"],
                    chapter_name=row.get("chapter_name"),
                    extra={
                        "clause_type": row.get("clause_type"),
                        "clause_number": row.get("clause_number"),
                        "article_content": row.get("article_content"),
                    },
                ))

    # --- KG: Article-level search (ALWAYS run, bao gồm title-priority) ---
    effective_terms = _expand_keywords(search_terms) if search_terms else _extract_tokens(question)
    for yr in target_years:
        art_cypher = """
        MATCH (a:Article {law_year: $yr})
        WHERE ANY(term IN $terms WHERE toLower(a.content) CONTAINS toLower(term))
        WITH a,
             CASE WHEN ANY(t IN $terms WHERE toLower(a.title) CONTAINS toLower(t))
                  THEN 1 ELSE 0 END AS title_hit
        RETURN a.chunk_id AS chunk_id, a.article_number AS article_number,
               a.title AS title, a.content AS content,
               a.law_year AS law_year, a.chapter_name AS chapter_name
        ORDER BY title_hit DESC, a.article_number
        LIMIT $top_k
        """
        rows = _run_cypher(graph, art_cypher, {"yr": yr, "terms": effective_terms, "top_k": top_k})
        for row in rows:
            if row.get("content"):
                chunks.append(RetrievedChunk(
                    source="graph", node_type="Article",
                    chunk_id=row["chunk_id"], law_year=row["law_year"],
                    article_number=row["article_number"],
                    title=row["title"], content=row["content"],
                    chapter_name=row.get("chapter_name"),
                ))

    # --- Vector augmentation ---
    v_chunks = _vector_search(question, top_k=top_k)
    chunks.extend(v_chunks)

    # --- Last resort broad fallback ---
    if len(_deduplicate(chunks)) < 2:
        chunks.extend(_broad_fallback_search(graph, question, target_years, top_k))

    return RetrievalResult(
        query_type=QueryType.CHU_THE,
        chunks=_deduplicate(chunks)[:top_k * 2],
        cypher_used=cyphers,
        vector_used=bool(v_chunks),
        total_found=len(chunks),
    )


# ---------------------------------------------------------------------------
# Chiến lược 4: THAY_DOI – thay đổi tổng thể Luật 2024 vs 2013
# ---------------------------------------------------------------------------

def _retrieve_thay_doi(
    graph: Neo4jGraph,
    cq: ClassifiedQuery,
    question: str,
    top_k: int,
) -> RetrievalResult:
    chunks: List[RetrievedChunk] = []
    cyphers: List[str] = []

    # base_cypher dùng làm fallback khi keyword search không đủ kết quả
    base_cypher = """
    MATCH (a24:Article {law_year: 2024})-[r:SUPERSEDES]->(a13:Article {law_year: 2013})
    WHERE r.diff_summary IS NOT NULL AND r.diff_summary <> ''
      AND r.change_type <> 'UNCHANGED'
    RETURN
        a24.chunk_id AS chunk_id, a24.article_number AS article_number,
        a24.title AS title, a24.content AS content_2024,
        a24.chapter_name AS chapter_name,
        a13.content AS content_2013,
        r.change_type AS change_type,
        r.diff_summary AS diff_summary,
        r.similarity_score AS similarity_score
    ORDER BY r.similarity_score ASC
    LIMIT $top_k
    """

    # --- KG: Keyword search FIRST (tìm theo từ khóa liên quan trước) ---
    raw_kws = cq.keywords if cq.keywords else _extract_tokens(question)
    kws = _expand_keywords(raw_kws)
    if kws:
        kw_cypher = """
        MATCH (a24:Article {law_year: 2024})-[r:SUPERSEDES]->(a13:Article {law_year: 2013})
        WHERE ANY(kw IN $keywords WHERE toLower(a24.content) CONTAINS toLower(kw))
        RETURN a24.chunk_id AS chunk_id, a24.article_number AS article_number,
               a24.title AS title, a24.content AS content, a24.chapter_name AS chapter_name,
               a13.content AS content_2013,
               r.diff_summary AS diff_summary, r.change_type AS change_type
        ORDER BY a24.article_number
        LIMIT $top_k
        """
        kw_rows = _run_cypher(graph, kw_cypher, {"keywords": kws, "top_k": top_k})
        cyphers.append(kw_cypher)
        for row in kw_rows:
            if row.get("content"):
                chunks.append(RetrievedChunk(
                    source="graph", node_type="Article",
                    chunk_id=row["chunk_id"], law_year=2024,
                    article_number=row["article_number"],
                    title=row["title"], content=row["content"],
                    chapter_name=row.get("chapter_name"),
                    extra={"change_type": row.get("change_type"),
                           "diff_summary": row.get("diff_summary"),
                           "content_2013": row.get("content_2013")},
                ))

        # Tìm cả articles 2024 không có SUPERSEDES (điều hoàn toàn mới)
        new_kw_cypher = """
        MATCH (a24:Article {law_year: 2024})
        WHERE NOT (a24)-[:SUPERSEDES]->(:Article {law_year: 2013})
          AND ANY(kw IN $keywords WHERE toLower(a24.content) CONTAINS toLower(kw))
        RETURN a24.chunk_id AS chunk_id, a24.article_number AS article_number,
               a24.title AS title, a24.content AS content, a24.chapter_name AS chapter_name
        ORDER BY a24.article_number
        LIMIT $top_k
        """
        new_kw_rows = _run_cypher(graph, new_kw_cypher, {"keywords": kws, "top_k": top_k})
        cyphers.append(new_kw_cypher)
        for row in new_kw_rows:
            if row.get("content"):
                chunks.append(RetrievedChunk(
                    source="graph", node_type="Article",
                    chunk_id=row["chunk_id"], law_year=2024,
                    article_number=row["article_number"],
                    title=row["title"], content=row["content"],
                    chapter_name=row.get("chapter_name"),
                    extra={"change_type": "NEW"},
                ))

        # Tìm thêm trong 2013 với cùng từ khoá (để có ngữ cảnh so sánh)
        kw_2013_cypher = """
        MATCH (a13:Article {law_year: 2013})
        WHERE ANY(kw IN $keywords WHERE toLower(a13.content) CONTAINS toLower(kw))
        RETURN a13.chunk_id AS chunk_id, a13.article_number AS article_number,
               a13.title AS title, a13.content AS content, a13.chapter_name AS chapter_name
        ORDER BY a13.article_number
        LIMIT $top_k
        """
        kw_2013_rows = _run_cypher(graph, kw_2013_cypher, {"keywords": kws, "top_k": top_k})
        cyphers.append(kw_2013_cypher)
        for row in kw_2013_rows:
            if row.get("content"):
                chunks.append(RetrievedChunk(
                    source="graph", node_type="Article",
                    chunk_id=row["chunk_id"], law_year=2013,
                    article_number=row["article_number"],
                    title=row["title"], content=row["content"],
                    chapter_name=row.get("chapter_name"),
                ))

    # --- Vector: tập trung vào nội dung liên quan đến "thay đổi" ---
    v_chunks = _vector_search(question + " 2024 thay đổi so với 2013", top_k=top_k, law_year=2024)
    chunks.extend(v_chunks)

    # --- KG: SUPERSEDES dump (nếu keyword search không đủ) ---
    if len(_deduplicate(chunks)) < 4:
        base_rows_extra = _run_cypher(graph, base_cypher, {"top_k": top_k * 2})
        for row in base_rows_extra:
            chunks.append(RetrievedChunk(
                source="graph", node_type="Article",
                chunk_id=row["chunk_id"], law_year=2024,
                article_number=row["article_number"],
                title=row["title"], content=row["content_2024"],
                chapter_name=row.get("chapter_name"),
                extra={
                    "change_type": row.get("change_type"),
                    "diff_summary": row.get("diff_summary"),
                    "content_2013": row.get("content_2013"),
                    "similarity_score": row.get("similarity_score"),
                },
            ))

    # --- KG: Articles chỉ có trong 2024 (mới hoàn toàn, broad) ---
    if len(_deduplicate(chunks)) < 5:
        new_art_cypher = """
        MATCH (a24:Article {law_year: 2024})
        WHERE NOT (a24)-[:SUPERSEDES]->(:Article {law_year: 2013})
        RETURN a24.chunk_id AS chunk_id, a24.article_number AS article_number,
               a24.title AS title, a24.content AS content,
               a24.chapter_name AS chapter_name
        ORDER BY a24.article_number
        LIMIT $top_k
        """
        cyphers.append(new_art_cypher)
        new_rows = _run_cypher(graph, new_art_cypher, {"top_k": top_k})
        for row in new_rows:
            chunks.append(RetrievedChunk(
                source="graph", node_type="Article",
                chunk_id=row["chunk_id"], law_year=2024,
                article_number=row["article_number"],
                title=row["title"], content=row["content"],
                chapter_name=row.get("chapter_name"),
                extra={"change_type": "NEW"},
            ))

    # --- Last resort ---
    if len(_deduplicate(chunks)) < 2:
        chunks.extend(_broad_fallback_search(graph, question, [2024, 2013], top_k))

    return RetrievalResult(
        query_type=QueryType.THAY_DOI,
        chunks=_deduplicate(chunks)[:top_k * 2],
        cypher_used=cyphers,
        vector_used=bool(v_chunks),
        total_found=len(chunks),
    )


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def _deduplicate(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Loại bỏ duplicate chunk dựa trên chunk_id; ưu tiên graph source."""
    seen: Dict[str, int] = {}
    result: List[RetrievedChunk] = []
    for c in sorted(chunks, key=lambda x: 0 if x.source == "graph" else 1):
        key = c.chunk_id if c.chunk_id else f"{c.law_year}_{c.article_number}_{c.content[:40]}"
        if key not in seen:
            seen[key] = 1
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class LawRetriever:
    """
    Điểm vào thống nhất: nhận ClassifiedQuery → chạy chiến lược phù hợp.
    """

    def __init__(self) -> None:
        self._graph = get_graph()

    def retrieve(
        self,
        cq: ClassifiedQuery,
        question: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        dispatch = {
            QueryType.SO_SANH:  _retrieve_so_sanh,
            QueryType.TRA_CUU:  _retrieve_tra_cuu,
            QueryType.CHU_THE:  _retrieve_chu_the,
            QueryType.THAY_DOI: _retrieve_thay_doi,
        }
        fn = dispatch[cq.query_type]
        result = fn(self._graph, cq, question, top_k)
        logger.info(
            "Retrieval [%s]: %d chunks (graph=%d, vector=%d)",
            cq.query_type,
            len(result.chunks),
            sum(1 for c in result.chunks if c.source == "graph"),
            sum(1 for c in result.chunks if c.source == "vector"),
        )
        return result


_retriever_instance: Optional[LawRetriever] = None


def get_retriever() -> LawRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = LawRetriever()
    return _retriever_instance
