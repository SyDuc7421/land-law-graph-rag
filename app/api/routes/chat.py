from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse, CitationItem
from app.services.chat_service import chat_with_neo4j

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint Graph RAG cho Luật Đất đai 2013 & 2024.
    """
    result = chat_with_neo4j(request.query, request.top_k)

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("answer"))

    citations = [
        CitationItem(**c) for c in result.get("citations", [])
    ]

    return ChatResponse(
        answer           = result["answer"],
        query_type       = result.get("query_type", ""),
        intent_summary   = result.get("intent_summary", ""),
        confidence       = result.get("confidence", 0.0),
        has_evidence     = result.get("has_evidence", False),
        citations        = citations,
        cypher_query     = result.get("cypher_query", ""),
        cypher_queries   = result.get("cypher_queries", []),
        retrieval_stats  = result.get("retrieval_stats", {}),
        context_data     = result.get("context_data", []),
    )
