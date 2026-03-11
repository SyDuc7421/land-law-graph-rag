"""
Reasoner – Suy luận pháp lý từ ngữ cảnh đã truy xuất.

Nguyên tắc:
  1. Nếu ngữ cảnh đủ căn cứ → trả lời có trích dẫn cụ thể (Điều, Khoản, Luật năm...).
  2. Nếu không đủ căn cứ (context rỗng hoặc similarity thấp) → "Không có thông tin".
  3. Theo từng loại truy vấn có prompt suy luận chuyên biệt.
  4. Luôn đánh giá độ tin cậy (confidence) và trả về cùng kết quả.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.query_classifier import ClassifiedQuery, QueryType
from app.services.retriever import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ngưỡng đánh giá đủ căn cứ
# ---------------------------------------------------------------------------

MIN_CHUNKS_REQUIRED  = 1      # ít nhất 1 chunk
MIN_CONTENT_LENGTH   = 50     # nội dung mỗi chunk ≥ 50 ký tự
LOW_CONFIDENCE_SCORE = 0.3    # vector score dưới ngưỡng này → không đủ tin cậy

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class ReasoningResult:
    answer: str
    confidence: float              # 0.0 → 1.0
    has_evidence: bool
    citations: List[Dict[str, Any]] = field(default_factory=list)
    query_type: str = ""
    intent_summary: str = ""
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Serialize context để đưa vào prompt
# ---------------------------------------------------------------------------

def _format_chunk_for_prompt(c: RetrievedChunk, idx: int) -> str:
    yr_label = f"Luật Đất đai {c.law_year}"
    art_label = f"Điều {c.article_number}" if c.article_number else "Điều (không rõ số)"
    title    = c.title or ""
    chapter  = f"({c.chapter_name})" if c.chapter_name else ""

    lines = [f"[{idx}] {yr_label} – {art_label}: {title} {chapter}".strip()]
    lines.append(f"Nội dung: {c.content[:1200]}")

    extra = c.extra
    if extra.get("diff_summary"):
        lines.append(f"Tóm tắt thay đổi: {extra['diff_summary']}")
    if extra.get("change_type"):
        lines.append(f"Loại thay đổi: {extra['change_type']}")
    if extra.get("content_2013"):
        lines.append(f"Nội dung 2013 (để so sánh): {str(extra['content_2013'])[:600]}")
    if extra.get("clause_type"):
        lines.append(f"Loại khoản: {extra['clause_type']} – Khoản {extra.get('clause_number','')}")

    return "\n".join(lines)


def _build_context_block(chunks: List[RetrievedChunk], max_chunks: int = 12) -> str:
    if not chunks:
        return "(Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu)"
    selected = chunks[:max_chunks]
    return "\n\n---\n\n".join(_format_chunk_for_prompt(c, i + 1) for i, c in enumerate(selected))


# ---------------------------------------------------------------------------
# Kiểm tra đủ căn cứ
# ---------------------------------------------------------------------------

def _has_sufficient_evidence(result: RetrievalResult) -> bool:
    """True nếu có ít nhất một chunk có nội dung đủ dài."""
    valid_chunks = [
        c for c in result.chunks
        if c.content and len(c.content.strip()) >= MIN_CONTENT_LENGTH
    ]
    if not valid_chunks:
        return False
    # Nếu chỉ có vector chunks với score thấp → không đủ tin cậy
    graph_chunks = [c for c in valid_chunks if c.source == "graph"]
    vector_high  = [c for c in valid_chunks if c.source == "vector" and c.score >= LOW_CONFIDENCE_SCORE]
    return bool(graph_chunks or vector_high)


# ---------------------------------------------------------------------------
# Prompts theo loại truy vấn
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """Bạn là chuyên gia pháp lý về Luật Đất đai Việt Nam (2013 và 2024).
Nhiệm vụ: đưa ra câu trả lời CHẮC CHẮN khi có căn cứ, KHÔNG suy đoán khi thiếu căn cứ.

QUY TẮC BẮT BUỘC:
1. Luôn trích dẫn: "theo Điều X, Luật Đất đai năm Y" hoặc "căn cứ Khoản N Điều X Luật Y".
2. Nếu context có trường `diff_summary`, trình bày tóm tắt thay đổi đó.
3. Nếu KHÔNG có thông tin trong [NGỮCẢNH], trả lời chính xác:
   "Không có thông tin về vấn đề này trong cơ sở dữ liệu Luật Đất đai 2013–2024."
4. KHÔNG bịa đặt, KHÔNG suy diễn ngoài nội dung đã được trích dẫn.
5. Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc."""

_PROMPTS: Dict[QueryType, str] = {
    QueryType.SO_SANH: _SYSTEM_BASE + """

PHONG CÁCH TRẢ LỜI (SO SÁNH):
- Trình bày song song: "Theo Luật 2013: ... | Theo Luật 2024: ..."
- Nêu rõ: thêm mới / bãi bỏ / sửa đổi nội dung gì.
- Kết luận 1-2 câu về tác động của sự thay đổi.""",

    QueryType.TRA_CUU: _SYSTEM_BASE + """

PHONG CÁCH TRẢ LỜI (TRA CỨU):
- Trích dẫn nguyên văn khi cần thiết.
- Nếu nội dung dài, liệt kê theo khoản.
- Ghi rõ: "Điều X – <tên điều>, Chương Y, Luật Đất đai năm Z".""",

    QueryType.CHU_THE: _SYSTEM_BASE + """

PHONG CÁCH TRẢ LỜI (CHỦ THỂ):
- Liệt kê QUYỀN và NGHĨA VỤ / THẨM QUYỀN tách biệt.
- Với mỗi quyền/nghĩa vụ, ghi rõ căn cứ (Điều, Khoản, Luật năm nào).
- Nếu có sự thay đổi giữa 2013 và 2024, chỉ rõ.""",

    QueryType.THAY_DOI: _SYSTEM_BASE + """

PHONG CÁCH TRẢ LỜI (THAY ĐỔI):
- Tóm tắt theo nhóm: điều khoản mới, điều khoản sửa đổi, điều khoản bãi bỏ.
- Với mỗi thay đổi quan trọng, nêu: số điều, tóm tắt thay đổi, ý nghĩa thực tiễn.
- Xếp theo mức độ quan trọng (thay đổi lớn trước).""",
}

_HUMAN_TEMPLATE = """[NGỮCẢNH TỪ CƠ SỞ DỮ LIỆU]
{context}

[CÂU HỎI]
{question}

[YÊU CẦU]
Chỉ dùng thông tin trong [NGỮCẢNH] để trả lời. Nếu [NGỮCẢNH] trống hoặc không liên quan,
trả lời: "Không có thông tin về vấn đề này trong cơ sở dữ liệu Luật Đất đai 2013–2024."

Câu trả lời:"""


# ---------------------------------------------------------------------------
# Build citations từ retrieved chunks
# ---------------------------------------------------------------------------

def _build_citations(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    seen = set()
    citations = []
    for c in chunks:
        if not c.article_number:
            continue
        key = (c.law_year, c.article_number)
        if key in seen:
            continue
        seen.add(key)
        cit: Dict[str, Any] = {
            "law_year": c.law_year,
            "article_number": c.article_number,
            "title": c.title,
            "chapter_name": c.chapter_name,
            "source_type": c.source,
        }
        if c.extra.get("change_type"):
            cit["change_type"] = c.extra["change_type"]
        if c.extra.get("diff_summary"):
            cit["diff_summary"] = c.extra["diff_summary"]
        citations.append(cit)
    return citations


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def _estimate_confidence(result: RetrievalResult, answer: str) -> float:
    """Ước tính độ tin cậy dựa trên số lượng và chất lượng chunks."""
    if not result.chunks:
        return 0.0
    graph_c   = sum(1 for c in result.chunks if c.source == "graph")
    vector_c  = sum(1 for c in result.chunks if c.source == "vector")
    no_info   = "không có thông tin" in answer.lower()

    if no_info:
        return 0.1

    base = min(0.5 + graph_c * 0.08 + vector_c * 0.04, 0.95)

    # Bonus nếu có diff_summary
    has_diff = any(c.extra.get("diff_summary") for c in result.chunks)
    if has_diff:
        base = min(base + 0.05, 0.95)

    return round(base, 2)


# ---------------------------------------------------------------------------
# Reasoner class
# ---------------------------------------------------------------------------

class LawReasoner:
    """
    Suy luận pháp lý: nhận truy vấn + retrieval result → trả lời có trích dẫn.
    """

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )

    def reason(
        self,
        question: str,
        cq: ClassifiedQuery,
        retrieval: RetrievalResult,
    ) -> ReasoningResult:
        has_evidence = _has_sufficient_evidence(retrieval)
        context_block = _build_context_block(retrieval.chunks)
        citations     = _build_citations(retrieval.chunks)

        system_prompt = _PROMPTS.get(cq.query_type, _SYSTEM_BASE)
        human_prompt  = _HUMAN_TEMPLATE.format(
            context=context_block,
            question=question,
        )

        try:
            response = self._llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ])
            answer = response.content.strip()
        except Exception as exc:
            logger.error("LLM reasoning failed: %s", exc)
            answer = (
                "Đã xảy ra lỗi khi xử lý câu hỏi. "
                "Vui lòng thử lại sau."
            )

        confidence = _estimate_confidence(retrieval, answer)

        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            has_evidence=has_evidence,
            citations=citations,
            query_type=cq.query_type.value,
            intent_summary=cq.intent_summary,
            retrieval_stats={
                "total_chunks": len(retrieval.chunks),
                "graph_chunks": sum(1 for c in retrieval.chunks if c.source == "graph"),
                "vector_chunks": sum(1 for c in retrieval.chunks if c.source == "vector"),
                "vector_used": retrieval.vector_used,
            },
        )


_reasoner_instance: Optional[LawReasoner] = None


def get_reasoner() -> LawReasoner:
    global _reasoner_instance
    if _reasoner_instance is None:
        _reasoner_instance = LawReasoner()
    return _reasoner_instance
