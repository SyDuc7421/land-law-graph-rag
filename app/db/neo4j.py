from langchain_community.graphs import Neo4jGraph
from app.core.config import settings

def get_graph() -> Neo4jGraph:
    """Khởi tạo kết nối lưu trữ tới Neo4j."""
    return Neo4jGraph(
        url=settings.NEO4J_URI,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD
    )
