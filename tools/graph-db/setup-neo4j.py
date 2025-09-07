#!/usr/bin/env python3
"""
RagaSense-Data: Neo4j Database Setup and Management
This script sets up the Neo4j database with the raga relationship schema.
"""

import os
import json
import logging
from typing import Dict, List, Any
from neo4j import GraphDatabase
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagaSenseNeo4j:
    """Neo4j database manager for RagaSense-Data"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "password"):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.verify_connectivity()
    
    def verify_connectivity(self):
        """Verify database connectivity"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                logger.info("✅ Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    def setup_schema(self):
        """Set up the complete graph schema"""
        schema_file = Path(__file__).parent.parent.parent / "schemas" / "graph-schema.cypher"
        
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return False
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_cypher = f.read()
        
        try:
            with self.driver.session() as session:
                # Split by comments and execute each statement
                statements = [stmt.strip() for stmt in schema_cypher.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement and not statement.startswith('//'):
                        session.run(statement)
                        logger.info(f"✅ Executed: {statement[:50]}...")
                
                logger.info("🎉 Schema setup completed successfully!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Schema setup failed: {e}")
            return False
    
    def clear_database(self):
        """Clear all data from the database (use with caution!)"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("🗑️ Database cleared")
        except Exception as e:
            logger.error(f"❌ Failed to clear database: {e}")
    
    def import_raga_mappings(self, mappings_file: str):
        """Import raga mappings from JSON file"""
        if not os.path.exists(mappings_file):
            logger.error(f"Mappings file not found: {mappings_file}")
            return False
        
        with open(mappings_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        try:
            with self.driver.session() as session:
                for mapping in mappings:
                    self._create_raga_mapping(session, mapping)
                
                logger.info(f"✅ Imported {len(mappings)} raga mappings")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to import mappings: {e}")
            return False
    
    def _create_raga_mapping(self, session, mapping: Dict[str, Any]):
        """Create a single raga mapping in Neo4j"""
        carnatic = mapping['carnatic_raga']
        hindustani = mapping['hindustani_raga']
        relationship = mapping['relationship']
        
        # Create Carnatic raga node
        carnatic_query = """
        MERGE (c:Raga {id: $carnatic_id})
        SET c.name = $carnatic_name,
            c.tradition = 'carnatic',
            c.melakarta_number = $melakarta_number,
            c.arohana = $carnatic_arohana,
            c.avarohana = $carnatic_avarohana
        """
        
        session.run(carnatic_query, 
                   carnatic_id=carnatic['id'],
                   carnatic_name=carnatic['name'],
                   melakarta_number=carnatic.get('melakarta_number'),
                   carnatic_arohana=carnatic.get('arohana', []),
                   carnatic_avarohana=carnatic.get('avarohana', []))
        
        # Create Hindustani raga node
        hindustani_query = """
        MERGE (h:Raga {id: $hindustani_id})
        SET h.name = $hindustani_name,
            h.tradition = 'hindustani',
            h.thaat = $thaat,
            h.arohana = $hindustani_arohana,
            h.avarohana = $hindustani_avarohana
        """
        
        session.run(hindustani_query,
                   hindustani_id=hindustani['id'],
                   hindustani_name=hindustani['name'],
                   thaat=hindustani.get('thaat'),
                   hindustani_arohana=hindustani.get('arohana', []),
                   hindustani_avarohana=hindustani.get('avarohana', []))
        
        # Create relationship
        rel_type = relationship['type']
        rel_query = f"""
        MATCH (c:Raga {{id: $carnatic_id}}), (h:Raga {{id: $hindustani_id}})
        MERGE (c)-[r:{rel_type}]->(h)
        SET r.confidence = $confidence,
            r.verified_by = $verified_by,
            r.verification_date = $verification_date,
            r.notes = $notes
        """
        
        session.run(rel_query,
                   carnatic_id=carnatic['id'],
                   hindustani_id=hindustani['id'],
                   confidence=relationship['confidence'],
                   verified_by=relationship['verified_by'],
                   verification_date=relationship.get('verification_date'),
                   notes=relationship.get('verification_notes', ''))
    
    def query_raga_relationships(self, raga_name: str, tradition: str = None):
        """Query relationships for a specific raga"""
        query = """
        MATCH (r:Raga {name: $raga_name})
        OPTIONAL MATCH (r)-[rel]-(related:Raga)
        RETURN r, rel, related
        """
        
        params = {"raga_name": raga_name}
        if tradition:
            query = """
            MATCH (r:Raga {name: $raga_name, tradition: $tradition})
            OPTIONAL MATCH (r)-[rel]-(related:Raga)
            RETURN r, rel, related
            """
            params["tradition"] = tradition
        
        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return []
    
    def get_database_stats(self):
        """Get database statistics"""
        queries = {
            "total_ragas": "MATCH (r:Raga) RETURN count(r) as count",
            "carnatic_ragas": "MATCH (r:Raga {tradition: 'carnatic'}) RETURN count(r) as count",
            "hindustani_ragas": "MATCH (r:Raga {tradition: 'hindustani'}) RETURN count(r) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "identical_mappings": "MATCH ()-[r:IDENTICAL]->() RETURN count(r) as count",
            "similar_mappings": "MATCH ()-[r:SIMILAR]->() RETURN count(r) as count"
        }
        
        stats = {}
        try:
            with self.driver.session() as session:
                for stat_name, query in queries.items():
                    result = session.run(query)
                    stats[stat_name] = result.single()['count']
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.driver.close()

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RagaSense Neo4j Database Setup")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--clear", action="store_true", help="Clear database before setup")
    parser.add_argument("--import-mappings", help="Import raga mappings from JSON file")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    
    args = parser.parse_args()
    
    # Initialize database connection
    db = RagaSenseNeo4j(args.uri, args.user, args.password)
    
    try:
        if args.clear:
            db.clear_database()
        
        if args.import_mappings:
            db.import_raga_mappings(args.import_mappings)
        else:
            # Default: setup schema
            db.setup_schema()
        
        if args.stats:
            stats = db.get_database_stats()
            print("\n📊 Database Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()
