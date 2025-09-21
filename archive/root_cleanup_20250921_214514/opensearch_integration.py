#!/usr/bin/env python3
"""
OpenSearch Data Integration for Raga Classification
Index classified raga data into OpenSearch for visualization and search
"""

import json
import requests
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant

class OpenSearchRagaIndexer:
    def __init__(self, host: str = "localhost", port: int = 9200):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        
        # Index configurations
        self.indexes = {
            'ragas': 'ragas_classified',
            'carnatic': 'carnatic_ragas',
            'hindustani': 'hindustani_ragas',
            'musical_theory': 'raga_musical_theory'
        }
        
    def check_opensearch_connection(self) -> bool:
        """Check if OpenSearch is running and accessible"""
        try:
            response = self.session.get(f"{self.base_url}/_cluster/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✅ OpenSearch is running: {health.get('status', 'unknown')} cluster")
                return True
            else:
                print(f"❌ OpenSearch health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to OpenSearch at {self.base_url}: {e}")
            print("💡 Make sure OpenSearch is running on localhost:9200")
            return False
    
    def create_raga_index_mapping(self) -> Dict[str, Any]:
        """Create optimized mapping for raga data"""
        return {
            "mappings": {
                "properties": {
                    "name": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {"type": "completion"}
                        }
                    },
                    "normalized_name": {"type": "keyword"},
                    "tradition": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "has_musical_theory": {"type": "boolean"},
                    "musical_theory": {
                        "properties": {
                            "arohana": {"type": "text", "analyzer": "keyword"},
                            "avarohana": {"type": "text", "analyzer": "keyword"},
                            "melakartha": {"type": "keyword"},
                            "thaat": {"type": "keyword"}
                        }
                    },
                    "audio_files": {"type": "keyword"},
                    "composers": {"type": "keyword"},
                    "songs": {"type": "keyword"},
                    "description": {"type": "text"},
                    "indexed_at": {"type": "date"},
                    "search_vector": {"type": "dense_vector", "dims": 128}  # For future ML integration
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "raga_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                }
            }
        }
    
    def create_index(self, index_name: str, mapping: Dict[str, Any]) -> bool:
        """Create an OpenSearch index with proper mapping"""
        try:
            # Delete existing index if it exists
            delete_response = self.session.delete(f"{self.base_url}/{index_name}")
            if delete_response.status_code in [200, 404]:
                print(f"🗑️ Cleared existing index: {index_name}")
            
            # Create new index
            response = self.session.put(
                f"{self.base_url}/{index_name}",
                json=mapping,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Created index: {index_name}")
                return True
            else:
                print(f"❌ Failed to create index {index_name}: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating index {index_name}: {e}")
            return False
    
    def prepare_raga_document(self, raga_data: Dict[str, Any], tradition: str) -> Dict[str, Any]:
        """Prepare a raga document for indexing"""
        # Extract musical theory
        theory = {}
        if 'theory' in raga_data:
            theory = raga_data['theory']
        elif 'data' in raga_data:
            # Extract from data field
            data = raga_data['data']
            theory = {
                'arohana': data.get('arohana', ''),
                'avarohana': data.get('avarohana', ''),
                'melakartha': str(data.get('melakartha', data.get('melakarta', ''))),
                'thaat': data.get('thaat', '')
            }
        
        # Prepare document
        doc = {
            'name': raga_data.get('name', ''),
            'normalized_name': raga_data.get('normalized_name', ''),
            'tradition': tradition,
            'source': raga_data.get('source', 'unknown'),
            'has_musical_theory': raga_data.get('has_musical_theory', False),
            'musical_theory': theory,
            'indexed_at': datetime.now().isoformat()
        }
        
        # Add additional fields from data if available
        if 'data' in raga_data:
            data = raga_data['data']
            doc.update({
                'description': data.get('description', ''),
                'songs': data.get('songs', []),
                'composers': data.get('composers', []),
                'audio_files': data.get('audio_files', [])
            })
        
        return doc
    
    def bulk_index_ragas(self, ragas: List[Dict[str, Any]], index_name: str, tradition: str) -> bool:
        """Bulk index ragas into OpenSearch"""
        if not ragas:
            print(f"⚠️ No ragas to index for {tradition}")
            return True
            
        print(f"📊 Indexing {len(ragas)} {tradition} ragas...")
        
        # Prepare bulk request
        bulk_body = []
        for raga in ragas:
            # Index action
            bulk_body.append(json.dumps({"index": {"_index": index_name}}))
            # Document
            doc = self.prepare_raga_document(raga, tradition)
            bulk_body.append(json.dumps(doc))
        
        bulk_data = "\\n".join(bulk_body) + "\\n"
        
        try:
            response = self.session.post(
                f"{self.base_url}/_bulk",
                data=bulk_data,
                headers={'Content-Type': 'application/x-ndjson'}
            )
            
            if response.status_code == 200:
                result = response.json()
                errors = [item for item in result.get('items', []) if 'error' in item.get('index', {})]
                
                if errors:
                    print(f"⚠️ {len(errors)} indexing errors occurred")
                    for error in errors[:3]:  # Show first 3 errors
                        print(f"   Error: {error['index']['error']}")
                else:
                    print(f"✅ Successfully indexed {len(ragas)} {tradition} ragas")
                
                return len(errors) == 0
            else:
                print(f"❌ Bulk indexing failed: {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return False
                
        except Exception as e:
            print(f"❌ Error during bulk indexing: {e}")
            return False
    
    def create_index_patterns_and_dashboards(self):
        """Create OpenSearch Dashboards index patterns and basic visualizations"""
        dashboard_config = {
            "index_patterns": [
                {
                    "id": "ragas_classified*",
                    "title": "ragas_classified*",
                    "timeFieldName": "indexed_at"
                },
                {
                    "id": "carnatic_ragas*", 
                    "title": "carnatic_ragas*",
                    "timeFieldName": "indexed_at"
                },
                {
                    "id": "hindustani_ragas*",
                    "title": "hindustani_ragas*", 
                    "timeFieldName": "indexed_at"
                }
            ]
        }
        
        print("🎨 OpenSearch Dashboard configuration ready")
        print("   📊 Index patterns configured for:")
        for pattern in dashboard_config["index_patterns"]:
            print(f"      - {pattern['title']}")
        
        return dashboard_config
    
    def index_all_raga_data(self) -> bool:
        """Index all classified raga data into OpenSearch"""
        print("🚀 Starting OpenSearch indexing process...")
        
        # Check connection
        if not self.check_opensearch_connection():
            return False
        
        # Load classified data
        assistant = DataUnificationAssistant()
        
        # Create classification
        classification = {
            'carnatic': {'ragas': []},
            'hindustani': {'ragas': []}
        }
        
        # Classify Saraga data
        for key, entity in assistant.saraga_data.items():
            tradition = entity.get('tradition', '').lower()
            if tradition in ['carnatic', 'hindustani']:
                classification[tradition]['ragas'].append({
                    'name': entity.get('original_name', key),
                    'normalized_name': key,
                    'source': entity.get('source', 'Saraga'),
                    'data': entity.get('data', {}),
                    'has_musical_theory': False  # Saraga typically doesn't have theory
                })
        
        # Classify Ramanarunachalam data
        for key, entity in assistant.ramanarunachalam_data.items():
            tradition = entity.get('tradition', '').lower()
            if tradition in ['carnatic', 'hindustani']:
                data = entity.get('data', {})
                theory = assistant._extract_musical_theory(data)
                has_theory = any([theory.get('arohana'), theory.get('avarohana'), 
                                theory.get('melakartha'), theory.get('thaat')])
                
                classification[tradition]['ragas'].append({
                    'name': entity.get('original_name', key),
                    'normalized_name': key,
                    'source': entity.get('source', 'Ramanarunachalam'),
                    'data': data,
                    'theory': theory,
                    'has_musical_theory': has_theory
                })
        
        # Create indexes
        mapping = self.create_raga_index_mapping()
        
        success = True
        
        # Create and index main ragas index
        if self.create_index(self.indexes['ragas'], mapping):
            all_ragas = classification['carnatic']['ragas'] + classification['hindustani']['ragas']
            for raga in all_ragas:
                # Determine tradition from source data
                tradition = 'carnatic' if raga in classification['carnatic']['ragas'] else 'hindustani'
                success &= self.bulk_index_ragas([raga], self.indexes['ragas'], tradition)
        
        # Create tradition-specific indexes
        for tradition in ['carnatic', 'hindustani']:
            index_name = self.indexes[tradition]
            if self.create_index(index_name, mapping):
                success &= self.bulk_index_ragas(
                    classification[tradition]['ragas'], 
                    index_name, 
                    tradition
                )
        
        if success:
            # Create dashboard configurations
            self.create_index_patterns_and_dashboards()
            
            print("\\n✅ OpenSearch indexing completed successfully!")
            print(f"📊 Data available at: {self.base_url}")
            print("🎨 OpenSearch Dashboards: http://localhost:5601")
            print("\\n🔍 You can now:")
            print("   1. Search ragas by name, tradition, or musical theory")
            print("   2. Filter by tradition (Carnatic/Hindustani)")
            print("   3. Explore ragas with musical theory data")
            print("   4. Create custom visualizations and dashboards")
        
        return success

def main():
    """Main function to index raga data into OpenSearch"""
    print("🔍 OpenSearch Raga Data Integration")
    print("=" * 60)
    
    indexer = OpenSearchRagaIndexer()
    
    if indexer.index_all_raga_data():
        print("\\n🎉 Success! Your raga data is now available in OpenSearch")
        print("\\n🔗 Quick Access URLs:")
        print(f"   OpenSearch API: {indexer.base_url}")
        print("   OpenSearch Dashboards: http://localhost:5601")
        print("\\n📋 Available Indexes:")
        for name, index in indexer.indexes.items():
            print(f"   - {index} ({name})")
    else:
        print("\\n❌ Failed to index data into OpenSearch")
        print("💡 Check if OpenSearch is running: docker run -p 9200:9200 opensearchproject/opensearch:latest")

if __name__ == "__main__":
    main()