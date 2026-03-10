import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Change Neo4j connection info if you changed it in docker-compose.yml
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USERNAME", "neo4j"), 
    os.getenv("NEO4J_PASSWORD", "password")
)

def get_latest_artifact_dir(base_dir="output"):
    """Get output directory of GraphRAG version 1.0.0+"""
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Please run graphrag index first.")
        return None
    return base_dir

def ingest_to_neo4j(artifact_dir):
    """Read data from parquet files and ingest into Neo4j"""
    nodes_file = os.path.join(artifact_dir, "entities.parquet")
    rels_file = os.path.join(artifact_dir, "relationships.parquet")
    
    if not os.path.exists(nodes_file) or not os.path.exists(rels_file):
        print("Could not find parquet files (Nodes / Relationships) in the artifact directory.")
        print("Ensure the `graphrag index` process has run and completed 100%.")
        return
        
    print(f"📄 Reading Nodes data from: {nodes_file}...")
    nodes_df = pd.read_parquet(nodes_file)
    print(f"📄 Reading Relationships data from: {rels_file}...")
    rels_df = pd.read_parquet(rels_file)
    
    print("\n--- Connecting to Neo4j ---")
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    try:
        with driver.session() as session:
            # Optional: Clear all existing graph data before ingesting new data
            print("Cleaning up old graph data (MATCH (n) DETACH DELETE n)...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Load Nodes
            print(f"Ingesting {len(nodes_df)} Nodes into Neo4j...")
            for _, row in nodes_df.iterrows():
                entity_type = row.get("type", "Entity")
                name = row.get("title", "")
                description = row.get("description", "")
                
                # Ingest node as (Entity {id: name, type, description})
                query = """
                MERGE (n:Entity {id: $name})
                SET n.name = $name, 
                    n.description = $description, 
                    n.entity_type = $entity_type
                """
                session.run(query, name=name, description=description, entity_type=entity_type)
                
            # Load Relationships
            print(f"Ingesting {len(rels_df)} Relationships into Neo4j...")
            for _, row in rels_df.iterrows():
                source = row.get("source", "")
                target = row.get("target", "")
                description = row.get("description", "")
                weight = row.get("weight", 1.0)
                
                query = """
                MATCH (source:Entity {id: $source})
                MATCH (target:Entity {id: $target})
                MERGE (source)-[r:RELATED]->(target)
                SET r.description = $description, r.weight = $weight
                """
                session.run(query, source=source, target=target, description=description, weight=weight)
                
        print("\nData ingestion complete! Access http://localhost:7474 to view the visual graph.")
    except Exception as e:
        print("Error occurred during data ingestion:", e)
    finally:
        driver.close()

if __name__ == "__main__":
    artifact_directory = get_latest_artifact_dir()
    if artifact_directory:
        ingest_to_neo4j(artifact_directory)
