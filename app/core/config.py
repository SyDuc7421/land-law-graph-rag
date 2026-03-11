import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    PROJECT_NAME: str = "Land Law GraphRAG"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Graph RAG API chuyên sâu cho Luật Đất đai 2013 & 2024. Trả lời câu hỏi pháp lý với căn cứ đầy đủ."
    
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

settings = Settings()
