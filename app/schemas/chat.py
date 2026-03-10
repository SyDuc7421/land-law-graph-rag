from pydantic import BaseModel, Field
from typing import Any

class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi truy vấn của người dùng.")
    top_k: int = Field(default=5, description="Giới hạn số lượng ngữ cảnh context trả về từ Neo4j")

class ChatResponse(BaseModel):
    answer: str
    cypher_query: str
    context_data: Any
