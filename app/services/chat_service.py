"""
RAG Pipeline Orchestrator cho Hệ thống Tra cứu Luật Đất đai.

Luồng xử lý:
  1. QueryClassifier  → phân loại truy vấn (so_sanh / tra_cuu / chu_the / thay_doi)
  2. LawRetriever     → truy xuất điều luật (KG traversal + semantic vector search)
  3. LawReasoner      → suy luận pháp lý, trích dẫn, đánh giá độ tin cậy
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.services.query_classifier import get_classifier, ClassifiedQuery, QueryType
from app.services.retriever import get_retriever, RetrievalResult
from app.services.reasoner import get_reasoner, ReasoningResult

logger = logging.getLogger(__name__)


def chat_with_neo4j(query: str, top_k: int = 10) -> dict:
    """
    Pipeline RAG đầy đủ:
      query  – Câu hỏi người dùng.
      top_k  – Số điều luật tối đa truy xuất mỗi chiến lược.

    Trả về dict với các trường:
      success, answer, query_type, intent_summary, confidence,
      has_evidence, citations, cypher_queries, retrieval_stats, context_data
    """
    try:
        # ── Bước 1: Phân loại truy vấn ──────────────────────────────────────
        classifier = get_classifier()
        cq: ClassifiedQuery = classifier.classify(query)
        logger.info(
            "[Pipeline] query_type=%s articles=%s years=%s keywords=%s",
            cq.query_type, cq.article_numbers, cq.law_years, cq.keywords,
        )

        # ── Bước 2: Truy xuất điều luật ─────────────────────────────────────
        retriever = get_retriever()
        retrieval: RetrievalResult = retriever.retrieve(cq, query, top_k=top_k)

        # ── Bước 3: Suy luận & tổng hợp câu trả lời ─────────────────────────
        reasoner = get_reasoner()
        result: ReasoningResult = reasoner.reason(query, cq, retrieval)

        # Chuyển context_data thành list dict để trả về API
        context_data: List[Dict[str, Any]] = [
            {
                "chunk_id":       c.chunk_id,
                "law_year":       c.law_year,
                "article_number": c.article_number,
                "title":          c.title,
                "content":        c.content[:500],
                "source":         c.source,
                "score":          round(c.score, 4),
                **({"change_type":  c.extra["change_type"]}
                   if c.extra.get("change_type") else {}),
                **({"diff_summary": c.extra["diff_summary"]}
                   if c.extra.get("diff_summary") else {}),
            }
            for c in retrieval.chunks[:top_k]
        ]

        return {
            "success":         True,
            "answer":          result.answer,
            "query_type":      result.query_type,
            "intent_summary":  result.intent_summary,
            "confidence":      result.confidence,
            "has_evidence":    result.has_evidence,
            "citations":       result.citations,
            "cypher_queries":  retrieval.cypher_used,
            # keep backward-compat field names
            "cypher_query":    retrieval.cypher_used[0] if retrieval.cypher_used else "",
            "retrieval_stats": result.retrieval_stats,
            "context_data":    context_data,
        }

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        return {
            "success":         False,
            "answer":          f"Đã xảy ra lỗi trong quá trình xử lý: {exc}",
            "query_type":      "",
            "intent_summary":  "",
            "confidence":      0.0,
            "has_evidence":    False,
            "citations":       [],
            "cypher_queries":  [],
            "cypher_query":    "",
            "retrieval_stats": {},
            "context_data":    [],
        }
