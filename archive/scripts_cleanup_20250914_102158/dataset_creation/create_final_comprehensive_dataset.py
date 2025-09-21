#!/usr/bin/env python3
"""
Create Final Comprehensive Dataset
==================================

This script creates the final comprehensive dataset by integrating:
1. Ramanarunachalam raga definitions (1,341 ragas)
2. Saraga Carnatic metadata (184 recordings, 124 ragas)
3. Cross-referencing and validation
4. Creating the final unified dataset

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_final_comprehensive_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinalDatasetCreator:
    """Create the final comprehensive raga dataset"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "02_raw" / "final_comprehensive_dataset"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Input paths
        self.ramanarunachalam_path = self.base_path / "02_raw" / "complete_unified_ragas" / "complete_unified_raga_dataset.json"
        self.saraga_metadata_path = self.base_path / "02_raw" / "saraga_metadata" / "saraga_combined_metadata.json"
        
    def normalize_raga_name(self, name: str) -> str:
        """Normalize raga name for comparison"""
        if not name:
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = name.lower().strip()
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(raag|raga|raaga)\s+', '', normalized)
        normalized = re.sub(r'\s+(raga|raag|raaga)$', '', normalized)
        
        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def load_ramanarunachalam_data(self) -> Dict[str, Any]:
        """Load Ramanarunachalam raga definitions"""
        logger.info("Loading Ramanarunachalam raga definitions...")
        
        if not self.ramanarunachalam_path.exists():
            raise FileNotFoundError(f"Ramanarunachalam data not found at {self.ramanarunachalam_path}")
        
        with open(self.ramanarunachalam_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} ragas from Ramanarunachalam")
        return data
    
    def load_saraga_metadata(self) -> Dict[str, Any]:
        """Load Saraga metadata"""
        logger.info("Loading Saraga metadata...")
        
        if not self.saraga_metadata_path.exists():
            logger.warning(f"Saraga metadata not found at {self.saraga_metadata_path}")
            return {}
        
        with open(self.saraga_metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} recordings from Saraga")
        return data
    
    def create_raga_index(self, ragas: List[Dict[str, Any]]) -> Dict[str, int]:
        """Create normalized raga name index"""
        index = {}
        for i, raga_data in enumerate(ragas):
            raga_name = raga_data.get('name', '')
            normalized = self.normalize_raga_name(raga_name)
            if normalized:
                index[normalized] = i
        return index
    
    def find_matching_ragas(self, ramanarunachalam_ragas: List[Dict[str, Any]], 
                           saraga_metadata: List[Dict[str, Any]]) -> Dict[int, Any]:
        """Find matching ragas between sources"""
        logger.info("Finding matching ragas between sources...")
        
        # Create normalized index for Ramanarunachalam
        rama_index = self.create_raga_index(ramanarunachalam_ragas)
        
        matches = {}
        saraga_matches = set()
        
        # Process Saraga metadata
        for i, recording_data in enumerate(saraga_metadata):
            raga_name = recording_data.get('raga_name', '')
            if isinstance(raga_name, dict):
                raga_name = raga_name.get('name', '') if 'name' in raga_name else str(raga_name)
            elif isinstance(raga_name, list):
                if raga_name and isinstance(raga_name[0], dict):
                    raga_name = raga_name[0].get('name', '') if 'name' in raga_name[0] else str(raga_name[0])
                else:
                    raga_name = raga_name[0] if raga_name else ''
            
            normalized_saraga = self.normalize_raga_name(raga_name)
            
            if normalized_saraga in rama_index:
                rama_id = rama_index[normalized_saraga]
                
                if rama_id not in matches:
                    matches[rama_id] = {
                        'ramanarunachalam_data': ramanarunachalam_ragas[rama_id],
                        'saraga_recordings': []
                    }
                
                matches[rama_id]['saraga_recordings'].append({
                    'recording_id': i,
                    'metadata': recording_data
                })
                
                saraga_matches.add(normalized_saraga)
        
        logger.info(f"Found {len(matches)} matching ragas with Saraga recordings")
        logger.info(f"Matched {len(saraga_matches)} unique Saraga ragas")
        
        return matches
    
    def create_comprehensive_dataset(self, ramanarunachalam_ragas: List[Dict[str, Any]],
                                   saraga_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create the final comprehensive dataset"""
        logger.info("Creating comprehensive dataset...")
        
        # Find matches
        matches = self.find_matching_ragas(ramanarunachalam_ragas, saraga_metadata)
        
        # Create comprehensive dataset
        comprehensive_dataset = []
        
        for i, raga_data in enumerate(ramanarunachalam_ragas):
            comprehensive_raga = {
                'raga_id': i,
                'raga_name': raga_data.get('name', ''),
                'tradition': raga_data.get('tradition', ''),
                'melakartha': raga_data.get('melakartha', ''),
                'arohana': raga_data.get('arohana', ''),
                'avarohana': raga_data.get('avarohana', ''),
                'source': 'ramanarunachalam',
                'has_audio': False,
                'audio_sources': [],
                'metadata': {
                    'duration': raga_data.get('duration_hours', ''),
                    'form': raga_data.get('form', ''),
                    'taala': raga_data.get('taala', ''),
                    'artist': raga_data.get('artist', ''),
                    'album': raga_data.get('album', ''),
                    'songs_count': raga_data.get('songs_count', 0),
                    'composers_count': raga_data.get('composers_count', 0),
                    'views': raga_data.get('views', 0)
                }
            }
            
            # Add Saraga data if available
            if i in matches:
                saraga_data = matches[i]
                comprehensive_raga['has_audio'] = True
                comprehensive_raga['audio_sources'].append('saraga')
                comprehensive_raga['saraga_recordings'] = saraga_data['saraga_recordings']
                
                # Update metadata with Saraga information
                for recording in saraga_data['saraga_recordings']:
                    rec_metadata = recording['metadata']
                    if rec_metadata.get('form'):
                        comprehensive_raga['metadata']['form'] = rec_metadata['form']
                    if rec_metadata.get('taala'):
                        comprehensive_raga['metadata']['taala'] = rec_metadata['taala']
                    if rec_metadata.get('artist'):
                        comprehensive_raga['metadata']['artist'] = rec_metadata['artist']
            
            comprehensive_dataset.append(comprehensive_raga)
        
        logger.info(f"Created comprehensive dataset with {len(comprehensive_dataset)} ragas")
        return comprehensive_dataset
    
    def generate_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive statistics"""
        logger.info("Generating dataset statistics...")
        
        total_ragas = len(dataset)
        with_audio = sum(1 for raga in dataset if raga['has_audio'])
        without_audio = total_ragas - with_audio
        
        # Tradition breakdown
        tradition_counts = {}
        for raga in dataset:
            tradition = raga.get('tradition', 'Unknown')
            tradition_counts[tradition] = tradition_counts.get(tradition, 0) + 1
        
        # Audio source breakdown
        audio_sources = {}
        for raga in dataset:
            for source in raga.get('audio_sources', []):
                audio_sources[source] = audio_sources.get(source, 0) + 1
        
        # Saraga recording counts
        saraga_recordings = 0
        for raga in dataset:
            if 'saraga_recordings' in raga:
                saraga_recordings += len(raga['saraga_recordings'])
        
        statistics = {
            'creation_date': datetime.now().isoformat(),
            'total_ragas': total_ragas,
            'with_audio': with_audio,
            'without_audio': without_audio,
            'audio_coverage_percentage': round((with_audio / total_ragas) * 100, 2),
            'tradition_breakdown': tradition_counts,
            'audio_sources': audio_sources,
            'saraga_recordings': saraga_recordings,
            'data_quality': {
                'complete_arohana': sum(1 for raga in dataset if raga.get('arohana')),
                'complete_avarohana': sum(1 for raga in dataset if raga.get('avarohana')),
                'with_melakartha': sum(1 for raga in dataset if raga.get('melakartha')),
                'with_tradition': sum(1 for raga in dataset if raga.get('tradition'))
            }
        }
        
        return statistics
    
    def save_dataset(self, dataset: List[Dict[str, Any]], statistics: Dict[str, Any]):
        """Save the comprehensive dataset"""
        logger.info("Saving comprehensive dataset...")
        
        # Save main dataset
        dataset_path = self.output_path / "comprehensive_raga_dataset.json"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_path = self.output_path / "comprehensive_dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary = {
            'dataset_info': {
                'name': 'Comprehensive Raga Dataset',
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'description': 'Unified dataset combining Ramanarunachalam raga definitions with Saraga audio metadata'
            },
            'statistics': statistics,
            'files': {
                'dataset': str(dataset_path.relative_to(self.base_path)),
                'statistics': str(stats_path.relative_to(self.base_path))
            }
        }
        
        summary_path = self.output_path / "dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {self.output_path}")
        logger.info(f"Files created:")
        logger.info(f"  - {dataset_path.name}")
        logger.info(f"  - {stats_path.name}")
        logger.info(f"  - {summary_path.name}")
    
    def run(self):
        """Run the comprehensive dataset creation process"""
        logger.info("🎵 Final Comprehensive Dataset Creation")
        logger.info("=" * 50)
        logger.info("This will create the final comprehensive dataset")
        logger.info("combining all available sources")
        logger.info("=" * 50)
        
        try:
            # Load data
            ramanarunachalam_ragas = self.load_ramanarunachalam_data()
            saraga_metadata = self.load_saraga_metadata()
            
            # Create comprehensive dataset
            comprehensive_dataset = self.create_comprehensive_dataset(
                ramanarunachalam_ragas, saraga_metadata
            )
            
            # Generate statistics
            statistics = self.generate_statistics(comprehensive_dataset)
            
            # Save dataset
            self.save_dataset(comprehensive_dataset, statistics)
            
            logger.info("✅ Comprehensive dataset creation completed successfully!")
            logger.info(f"📊 Total ragas: {statistics['total_ragas']}")
            logger.info(f"🎵 With audio: {statistics['with_audio']}")
            logger.info(f"📈 Audio coverage: {statistics['audio_coverage_percentage']}%")
            logger.info(f"📁 Dataset saved to: {self.output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error creating comprehensive dataset: {e}")
            raise

def main():
    """Main function"""
    creator = FinalDatasetCreator()
    creator.run()

if __name__ == "__main__":
    main()
