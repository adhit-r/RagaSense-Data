#!/usr/bin/env python3
"""
Create Unified Raga Dataset
==========================

This script creates a comprehensive unified dataset by combining:
1. Ramanarunachalam: Raga definitions (Arohana/Avarohana notes)
2. Saraga-Carnatic: Real audio recordings of Carnatic ragas
3. Saraga-Hindustani: Real audio recordings of Hindustani ragas
4. Saraga-Carnatic-Melody-Synth: Synthetic melody data

The goal is to create a clean dataset with:
- Individual ragas only (no combinations)
- Proper raga definitions with musical theory
- Real audio recordings where available
- Clean annotations and metadata

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import json
import zipfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_unified_raga_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedRagaDatasetCreator:
    """Create unified raga dataset from all sources"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "02_raw" / "unified_ragas"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Source paths
        self.ramanarunachalam_path = self.base_path / "01_source" / "ramanarunachalam"
        self.saraga_path = self.base_path / "01_source" / "saraga"
        
        # Results storage
        self.results = {
            'creation_date': datetime.now().isoformat(),
            'sources': {
                'ramanarunachalam_carnatic': {'ragas_processed': 0, 'ragas_skipped': 0},
                'ramanarunachalam_hindustani': {'ragas_processed': 0, 'ragas_skipped': 0},
                'saraga_carnatic': {'audio_files': 0, 'ragas_found': 0},
                'saraga_hindustani': {'audio_files': 0, 'ragas_found': 0}
            },
            'unified_dataset': {
                'total_ragas': 0,
                'carnatic_ragas': 0,
                'hindustani_ragas': 0,
                'cross_tradition_ragas': 0,
                'ragas_with_audio': 0,
                'ragas_without_audio': 0
            }
        }
    
    def is_combination_raga(self, filename: str) -> bool:
        """Check if filename contains combination ragas (e.g., 'Abheri, Mohanam.json')"""
        return ',' in filename or '&' in filename or 'and' in filename.lower()
    
    def extract_raga_info(self, raga_data: dict) -> dict:
        """Extract raga information from Ramanarunachalam data"""
        raga_info = {
            'name': None,
            'melakartha': None,
            'arohana': None,
            'avarohana': None,
            'tradition': None,
            'songs_count': 0,
            'composers_count': 0,
            'duration_hours': 0
        }
        
        # Extract basic info
        if 'info' in raga_data:
            for item in raga_data['info']:
                if item.get('H') == 'Melakartha':
                    raga_info['melakartha'] = item.get('V')
                elif item.get('H') == 'Arohana':
                    arohana_data = item.get('V')
                    if isinstance(arohana_data, list) and len(arohana_data) > 0:
                        raga_info['arohana'] = arohana_data[0]  # Take the first notation
                elif item.get('H') == 'Avarohana':
                    avarohana_data = item.get('V')
                    if isinstance(avarohana_data, list) and len(avarohana_data) > 0:
                        raga_info['avarohana'] = avarohana_data[0]  # Take the first notation
        
        # Extract stats
        if 'stats' in raga_data:
            for stat in raga_data['stats']:
                if stat.get('H') == 'Songs':
                    raga_info['songs_count'] = int(stat.get('C', '0').replace(',', ''))
                elif stat.get('H') == 'Composers':
                    raga_info['composers_count'] = int(stat.get('C', '0').replace(',', ''))
                elif stat.get('H') == 'Duration':
                    duration_str = stat.get('C', '0 H')
                    raga_info['duration_hours'] = float(duration_str.replace(' H', '').replace(',', ''))
        
        return raga_info
    
    def process_ramanarunachalam_tradition(self, tradition: str) -> List[dict]:
        """Process Ramanarunachalam raga definitions for a tradition"""
        logger.info(f"Processing Ramanarunachalam {tradition} tradition...")
        
        raga_dir = self.ramanarunachalam_path / tradition / "raga"
        if not raga_dir.exists():
            logger.error(f"Raga directory not found: {raga_dir}")
            return []
        
        ragas = []
        processed_count = 0
        skipped_count = 0
        
        for raga_file in raga_dir.glob("*.json"):
            # Skip combination ragas
            if self.is_combination_raga(raga_file.name):
                skipped_count += 1
                logger.debug(f"Skipping combination raga: {raga_file.name}")
                continue
            
            try:
                with open(raga_file, 'r', encoding='utf-8') as f:
                    raga_data = json.load(f)
                
                raga_info = self.extract_raga_info(raga_data)
                raga_info['name'] = raga_file.stem
                raga_info['tradition'] = tradition
                raga_info['source'] = 'ramanarunachalam'
                raga_info['source_file'] = str(raga_file)
                
                ragas.append(raga_info)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {raga_file.name}: {e}")
                skipped_count += 1
        
        logger.info(f"Processed {processed_count} {tradition} ragas, skipped {skipped_count} combination ragas")
        self.results['sources'][f'ramanarunachalam_{tradition.lower()}'] = {
            'ragas_processed': processed_count,
            'ragas_skipped': skipped_count
        }
        
        return ragas
    
    def extract_saraga_metadata(self, tradition: str) -> List[dict]:
        """Extract metadata from Saraga zip files"""
        logger.info(f"Extracting Saraga {tradition} metadata...")
        
        zip_path = self.saraga_path / tradition / f"saraga1.5_{tradition}.zip"
        if not zip_path.exists():
            logger.error(f"Saraga zip file not found: {zip_path}")
            return []
        
        metadata = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Look for metadata files
                metadata_files = [f for f in zip_ref.namelist() if f.endswith('.json') and 'metadata' in f.lower()]
                
                for metadata_file in metadata_files:
                    try:
                        with zip_ref.open(metadata_file) as f:
                            data = json.load(f)
                        
                        # Extract raga information from metadata
                        if isinstance(data, dict):
                            raga_name = data.get('raaga', data.get('raag', data.get('raga', 'Unknown')))
                            if raga_name and raga_name != 'Unknown':
                                metadata.append({
                                    'name': raga_name,
                                    'tradition': tradition,
                                    'source': 'saraga',
                                    'metadata_file': metadata_file,
                                    'audio_file': data.get('audio_file', ''),
                                    'duration': data.get('duration', 0),
                                    'artist': data.get('artist', ''),
                                    'album': data.get('album', '')
                                })
                    except Exception as e:
                        logger.warning(f"Error processing metadata file {metadata_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error extracting Saraga {tradition} metadata: {e}")
        
        logger.info(f"Extracted metadata for {len(metadata)} {tradition} recordings")
        self.results['sources'][f'saraga_{tradition.lower()}'] = {
            'audio_files': len(metadata),
            'ragas_found': len(set(item['name'] for item in metadata))
        }
        
        return metadata
    
    def merge_raga_data(self, ramanarunachalam_ragas: List[dict], saraga_metadata: List[dict]) -> List[dict]:
        """Merge raga definitions with audio metadata"""
        logger.info("Merging raga definitions with audio metadata...")
        
        # Create lookup dictionaries
        ramanarunachalam_lookup = {raga['name'].lower(): raga for raga in ramanarunachalam_ragas}
        saraga_lookup = {}
        
        for item in saraga_metadata:
            raga_name = item['name'].lower()
            if raga_name not in saraga_lookup:
                saraga_lookup[raga_name] = []
            saraga_lookup[raga_name].append(item)
        
        # Merge data
        unified_ragas = []
        ragas_with_audio = 0
        ragas_without_audio = 0
        
        # Process Ramanarunachalam ragas
        for raga in ramanarunachalam_ragas:
            raga_name_lower = raga['name'].lower()
            
            # Add audio information if available
            if raga_name_lower in saraga_lookup:
                raga['audio_recordings'] = saraga_lookup[raga_name_lower]
                raga['has_audio'] = True
                ragas_with_audio += 1
            else:
                raga['audio_recordings'] = []
                raga['has_audio'] = False
                ragas_without_audio += 1
            
            unified_ragas.append(raga)
        
        # Add Saraga ragas not in Ramanarunachalam
        for raga_name, recordings in saraga_lookup.items():
            if raga_name not in ramanarunachalam_lookup:
                # Create basic raga info from Saraga data
                raga_info = {
                    'name': recordings[0]['name'],
                    'tradition': recordings[0]['tradition'],
                    'source': 'saraga_only',
                    'melakartha': None,
                    'arohana': None,
                    'avarohana': None,
                    'songs_count': 0,
                    'composers_count': 0,
                    'duration_hours': 0,
                    'audio_recordings': recordings,
                    'has_audio': True
                }
                unified_ragas.append(raga_info)
                ragas_with_audio += 1
        
        logger.info(f"Merged {len(unified_ragas)} ragas: {ragas_with_audio} with audio, {ragas_without_audio} without audio")
        
        return unified_ragas
    
    def classify_traditions(self, unified_ragas: List[dict]) -> List[dict]:
        """Classify ragas by tradition (Carnatic, Hindustani, Both)"""
        logger.info("Classifying ragas by tradition...")
        
        # Count traditions
        tradition_counts = {'Carnatic': 0, 'Hindustani': 0, 'Both': 0}
        
        for raga in unified_ragas:
            tradition = raga.get('tradition', 'Unknown')
            if tradition == 'Carnatic':
                raga['tradition_classification'] = 'Carnatic'
                tradition_counts['Carnatic'] += 1
            elif tradition == 'Hindustani':
                raga['tradition_classification'] = 'Hindustani'
                tradition_counts['Hindustani'] += 1
            else:
                raga['tradition_classification'] = 'Unknown'
        
        logger.info(f"Tradition classification: {tradition_counts}")
        
        return unified_ragas
    
    def save_unified_dataset(self, unified_ragas: List[dict]):
        """Save the unified dataset"""
        logger.info("Saving unified dataset...")
        
        # Save main dataset
        dataset_file = self.output_path / "unified_raga_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(unified_ragas, f, indent=2, ensure_ascii=False)
        
        # Create summary
        summary = {
            'creation_date': datetime.now().isoformat(),
            'total_ragas': len(unified_ragas),
            'tradition_breakdown': {},
            'source_breakdown': {},
            'audio_availability': {
                'ragas_with_audio': sum(1 for r in unified_ragas if r.get('has_audio', False)),
                'ragas_without_audio': sum(1 for r in unified_ragas if not r.get('has_audio', False))
            }
        }
        
        # Count by tradition
        for raga in unified_ragas:
            tradition = raga.get('tradition_classification', 'Unknown')
            summary['tradition_breakdown'][tradition] = summary['tradition_breakdown'].get(tradition, 0) + 1
        
        # Count by source
        for raga in unified_ragas:
            source = raga.get('source', 'Unknown')
            summary['source_breakdown'][source] = summary['source_breakdown'].get(source, 0) + 1
        
        # Save summary
        summary_file = self.output_path / "unified_dataset_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save individual tradition files
        for tradition in ['Carnatic', 'Hindustani']:
            tradition_ragas = [r for r in unified_ragas if r.get('tradition_classification') == tradition]
            if tradition_ragas:
                tradition_file = self.output_path / f"{tradition.lower()}_ragas.json"
                with open(tradition_file, 'w', encoding='utf-8') as f:
                    json.dump(tradition_ragas, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved unified dataset: {dataset_file}")
        logger.info(f"Saved summary: {summary_file}")
        
        return summary
    
    def run_unified_dataset_creation(self):
        """Run the complete unified dataset creation process"""
        logger.info("Starting unified raga dataset creation...")
        
        try:
            # Process Ramanarunachalam traditions
            carnatic_ragas = self.process_ramanarunachalam_tradition('Carnatic')
            hindustani_ragas = self.process_ramanarunachalam_tradition('Hindustani')
            
            # Extract Saraga metadata
            saraga_carnatic_metadata = self.extract_saraga_metadata('carnatic')
            saraga_hindustani_metadata = self.extract_saraga_metadata('hindustani')
            
            # Merge Carnatic data
            unified_carnatic = self.merge_raga_data(carnatic_ragas, saraga_carnatic_metadata)
            
            # Merge Hindustani data
            unified_hindustani = self.merge_raga_data(hindustani_ragas, saraga_hindustani_metadata)
            
            # Combine all ragas
            all_ragas = unified_carnatic + unified_hindustani
            
            # Classify traditions
            classified_ragas = self.classify_traditions(all_ragas)
            
            # Save dataset
            summary = self.save_unified_dataset(classified_ragas)
            
            # Update results
            self.results['unified_dataset'] = summary
            
            logger.info("Unified raga dataset creation completed successfully!")
            return self.results
            
        except Exception as e:
            logger.error(f"Error during dataset creation: {e}")
            raise

def main():
    """Main function"""
    print("🎵 RagaSense Unified Dataset Creation")
    print("=" * 50)
    print("Combining all raga sources:")
    print("• Ramanarunachalam: Raga definitions (Arohana/Avarohana)")
    print("• Saraga-Carnatic: Real audio recordings")
    print("• Saraga-Hindustani: Real audio recordings")
    print("• Clean individual ragas only (no combinations)")
    print("=" * 50)
    
    creator = UnifiedRagaDatasetCreator()
    results = creator.run_unified_dataset_creation()
    
    print(f"\n✅ Unified Dataset Created!")
    print(f"📊 Total ragas: {results['unified_dataset']['total_ragas']}")
    print(f"📊 Carnatic: {results['unified_dataset']['tradition_breakdown'].get('Carnatic', 0)}")
    print(f"📊 Hindustani: {results['unified_dataset']['tradition_breakdown'].get('Hindustani', 0)}")
    print(f"🎵 With audio: {results['unified_dataset']['audio_availability']['ragas_with_audio']}")
    print(f"📁 Dataset saved to: data/02_raw/unified_ragas/")

if __name__ == "__main__":
    main()
