from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from app.db.neo4j import get_graph
from app.core.config import settings
from app.services.prompts import CYPHER_GENERATION_PROMPT, QA_PROMPT

def chat_with_neo4j(query: str, top_k: int = 5) -> dict:
    """
    Hàm xử lý chat logic bao gồm 2 đầu vào: 
    1. query: Câu hỏi của người dùng
    2. top_k: Giới hạn số lượng kết quả trả về từ DB
    """
    try:
        graph = get_graph()
        
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )
        
        qa_chain = GraphCypherQAChain.from_llm(
            cypher_llm=llm,
            qa_llm=llm,
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            qa_prompt=QA_PROMPT,
            top_k=top_k,
            allow_dangerous_requests=True
        )
        
        response = qa_chain.invoke({"query": query})
        
        intermediate_steps = response.get("intermediate_steps", [])
        
        cypher_query = ""
        context_data = []
        if len(intermediate_steps) >= 2:
            cypher_query = intermediate_steps[0].get("query", "")
            context_data = intermediate_steps[1].get("context", [])

        return {
            "success": True,
            "answer": response.get("result", ""),
            "cypher_query": cypher_query,
            "context_data": context_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "answer": f"Đã xảy ra lỗi trong quá trình truy xuất từ DB: {str(e)}",
            "cypher_query": "",
            "context_data": []
        }
