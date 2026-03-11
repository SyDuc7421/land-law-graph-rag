"""
Query Classifier – phân loại câu hỏi đầu vào thành 4 loại:
  SO_SANH  : "Điều X 2013 khác 2024 thế nào?"
  TRA_CUU  : "Quyền của công dân là gì?"
  CHU_THE  : "UBND tỉnh có thẩm quyền gì?"
  THAY_DOI : "Luật 2024 thay đổi gì so với 2013?"

Sử dụng LLM (structured-output) để phân loại + trích xuất thực thể từ truy vấn.
"""

from __future__ import annotations

import re
import json
import logging
from enum import Enum
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum loại truy vấn
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    SO_SANH  = "so_sanh"   # So sánh điều khoản giữa 2 phiên bản luật
    TRA_CUU  = "tra_cuu"   # Tra cứu nội dung điều khoản cụ thể
    CHU_THE  = "chu_the"   # Thẩm quyền / quyền / nghĩa vụ của một chủ thể
    THAY_DOI = "thay_doi"  # Những thay đổi tổng quát trong Luật 2024


# ---------------------------------------------------------------------------
# Schema đầu ra của Classifier
# ---------------------------------------------------------------------------

class ClassifiedQuery(BaseModel):
    """Kết quả phân loại truy vấn."""

    query_type: QueryType = Field(
        ...,
        description=(
            "Loại truy vấn: so_sanh | tra_cuu | chu_the | thay_doi"
        ),
    )
    used_llm: bool = Field(
        default=True,
        description="True nếu LLM thực hiện phân loại; False nếu dùng regex fallback.",
    )
    article_numbers: List[int] = Field(
        default_factory=list,
        description="Các số Điều được nhắc đến, ví dụ [62, 63].",
    )
    law_years: List[int] = Field(
        default_factory=list,
        description="Năm luật được nhắc đến, ví dụ [2013, 2024].",
    )
    subject_entity: Optional[str] = Field(
        None,
        description=(
            "Chủ thể pháp lý được hỏi (ví dụ 'UBND tỉnh', 'hộ gia đình', "
            "'người sử dụng đất'). Chỉ điền khi query_type = chu_the."
        ),
    )
    keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Từ khóa pháp lý chính của truy vấn, ví dụ ['thu hồi đất', 'bồi thường']."
        ),
    )
    intent_summary: str = Field(
        ...,
        description="Mô tả ngắn (<20 từ) ý định của câu hỏi bằng tiếng Việt.",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """Bạn là chuyên gia phân tích câu hỏi pháp luật đất đai Việt Nam.
Nhiệm vụ: phân loại câu hỏi của người dùng thành đúng MỘT trong 4 loại:

  so_sanh  – So sánh một điều khoản cụ thể giữa phiên bản 2013 và 2024.
             Dấu hiệu: nhắc đến hai năm, dùng từ "khác", "so với", "thay đổi ở Điều ...".
  tra_cuu  – Tra cứu nội dung / ý nghĩa của một hoặc nhiều điều khoản.
             Dấu hiệu: hỏi "là gì", "quy định gì", "nội dung Điều X", v.v.
  chu_the  – Hỏi quyền, nghĩa vụ, thẩm quyền của một chủ thể pháp lý.
             Dấu hiệu: nhắc đến một cơ quan / cá nhân + "có quyền gì", "được làm gì", "phải làm gì".
  thay_doi – Hỏi về sự thay đổi tổng quát của Luật 2024 so với 2013 (không nêu điều cụ thể).
             Dấu hiệu: "Luật 2024 thay đổi gì", "điểm mới", "bổ sung gì", "khác biệt lớn".

Trả về JSON hợp lệ theo schema đã cung cấp. KHÔNG giải thích thêm."""

_CLASSIFIER_HUMAN = "Câu hỏi: {question}"

_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _CLASSIFIER_SYSTEM), ("human", _CLASSIFIER_HUMAN)]
)


# ---------------------------------------------------------------------------
# Regex fallback – trích xuất nhanh số điều / năm
# ---------------------------------------------------------------------------

_RE_ARTICLE = re.compile(r"\bđiều\s+(\d+)\b", re.IGNORECASE)
_RE_YEAR    = re.compile(r"\b(2013|2024)\b")


def _regex_fallback(question: str) -> ClassifiedQuery:
    """Phân loại đơn giản bằng regex khi LLM không khả dụng."""
    q_lower = question.lower()
    articles = [int(m) for m in _RE_ARTICLE.findall(question)]
    years    = [int(m) for m in _RE_YEAR.findall(question)]

    if any(w in q_lower for w in ["so sánh", "khác", "so với", "điều", "thay đổi ở điều"]) \
            and len(set(years) & {2013, 2024}) == 2:
        qtype = QueryType.SO_SANH
    elif any(w in q_lower for w in ["thay đổi", "điểm mới", "bổ sung", "khác biệt"]):
        qtype = QueryType.THAY_DOI
    elif any(w in q_lower for w in ["quyền", "nghĩa vụ", "thẩm quyền", "trách nhiệm",
                                     "được làm", "phải làm", "có quyền"]):
        qtype = QueryType.CHU_THE
    else:
        qtype = QueryType.TRA_CUU

    return ClassifiedQuery(
        query_type=qtype,
        article_numbers=articles,
        law_years=years,
        keywords=[],
        intent_summary="Phân loại tự động (fallback)",
        used_llm=False,
    )


# ---------------------------------------------------------------------------
# Classifier chính
# ---------------------------------------------------------------------------

class QueryClassifier:
    """Phân loại truy vấn pháp luật bằng LLM có structured output."""

    def __init__(self) -> None:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )
        # Bind structured output schema
        self._chain = _CLASSIFIER_PROMPT | llm.with_structured_output(ClassifiedQuery)

    def classify(self, question: str) -> ClassifiedQuery:
        """
        Phân loại câu hỏi bằng LLM (structured output).
        LLM nhận prompt mô tả 4 loại, trả về JSON theo schema ClassifiedQuery.
        Nếu LLM thất bại (timeout, quota...), tự động dùng regex fallback.
        """
        try:
            result: ClassifiedQuery = self._chain.invoke({"question": question})
            # Đảm bảo flag LLM được set đúng (LLM có thể trả về field này hoặc không)
            result.used_llm = True
            logger.info(
                "[LLM] Classified '%s' → %s | articles=%s | years=%s | subject='%s'",
                question[:60],
                result.query_type,
                result.article_numbers,
                result.law_years,
                result.subject_entity,
            )
            return result
        except Exception as exc:
            logger.warning("LLM classifier failed (%s), using regex fallback.", exc)
            return _regex_fallback(question)


# Singleton để tái dùng trong suốt app lifecycle
_classifier_instance: Optional[QueryClassifier] = None


def get_classifier() -> QueryClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryClassifier()
    return _classifier_instance
