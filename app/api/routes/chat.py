from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_with_neo4j

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint chính xử lý chat. 
    Lấy câu hỏi (query) và số kết quả giới hạn lớn nhất (top_k), tìm qua Neo4j và tổng hợp câu trả lời bằng LLM.
    """
    result = chat_with_neo4j(request.query, request.top_k)
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("answer"))
        
    return ChatResponse(
        answer=result["answer"],
        cypher_query=result["cypher_query"],
        context_data=result["context_data"]
    )
