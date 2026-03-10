import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    PROJECT_NAME: str = "Law GraphRAG API (Neo4j)"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API hệ thống RAG dùng Neo4j Database cho Luật đất đai 2024."
    
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    OPENAI_API_KEY: str = os.getenv("GRAPHRAG_API_KEY", "")

settings = Settings()
