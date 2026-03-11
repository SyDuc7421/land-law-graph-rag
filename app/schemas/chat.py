from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi truy vấn của người dùng.")
    top_k: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Số lượng điều luật tối đa truy xuất mỗi chiến lược.",
    )


class CitationItem(BaseModel):
    """Trích dẫn căn cứ pháp lý."""
    law_year: int
    article_number: Optional[int] = None
    title: Optional[str] = None
    chapter_name: Optional[str] = None
    source_type: str = "graph"         # "graph" | "vector"
    change_type: Optional[str] = None  # cho SO_SANH / THAY_DOI
    diff_summary: Optional[str] = None


class ChatResponse(BaseModel):
    # Câu trả lời tổng hợp
    answer: str

    # Metadata RAG pipeline
    query_type:     str   = Field("",   description="Loại truy vấn: so_sanh|tra_cuu|chu_the|thay_doi")
    intent_summary: str   = Field("",   description="Tóm tắt ý định câu hỏi")
    confidence:     float = Field(0.0,  description="Độ tin cậy 0–1")
    has_evidence:   bool  = Field(False,description="Có căn cứ pháp lý hay không")

    # Trích dẫn căn cứ
    citations: List[CitationItem] = Field(default_factory=list)

    # Debug / transparency
    cypher_query:    str          = Field("",               description="Câu Cypher đầu tiên đã dùng (backward compat)")
    cypher_queries:  List[str]    = Field(default_factory=list, description="Tất cả câu Cypher đã dùng")
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict)
    context_data:    Any          = None
