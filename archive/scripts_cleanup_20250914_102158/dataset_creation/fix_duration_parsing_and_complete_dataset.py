#!/usr/bin/env python3
"""
Fix Duration Parsing and Complete Unified Dataset
===============================================

This script fixes the duration parsing errors in Ramanarunachalam data and creates
a complete unified dataset by:

1. Fixing duration parsing errors (e.g., "13 M", "9 M")
2. Extracting Saraga audio metadata properly
3. Integrating Saraga-Carnatic-Melody-Synth data
4. Creating the final comprehensive dataset

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
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fix_duration_parsing_and_complete_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteDatasetCreator:
    """Create complete unified dataset with all sources integrated"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "02_raw" / "complete_unified_ragas"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Source paths
        self.ramanarunachalam_path = self.base_path / "01_source" / "ramanarunachalam"
        self.saraga_path = self.base_path / "01_source" / "saraga"
        
        # Results storage
        self.results = {
            'creation_date': datetime.now().isoformat(),
            'sources': {
                'ramanarunachalam_carnatic': {'ragas_processed': 0, 'ragas_skipped': 0, 'duration_errors_fixed': 0},
                'ramanarunachalam_hindustani': {'ragas_processed': 0, 'ragas_skipped': 0, 'duration_errors_fixed': 0},
                'saraga_carnatic': {'audio_files': 0, 'ragas_found': 0},
                'saraga_hindustani': {'audio_files': 0, 'ragas_found': 0},
                'saraga_melody_synth': {'synthetic_files': 0, 'ragas_found': 0}
            },
            'complete_dataset': {
                'total_ragas': 0,
                'carnatic_ragas': 0,
                'hindustani_ragas': 0,
                'cross_tradition_ragas': 0,
                'ragas_with_audio': 0,
                'ragas_with_synthetic': 0,
                'ragas_without_audio': 0
            }
        }
    
    def parse_duration_string(self, duration_str: str) -> float:
        """Parse duration string with various formats"""
        if not duration_str or duration_str == '0':
            return 0.0
        
        # Remove commas and extra spaces
        duration_str = duration_str.replace(',', '').strip()
        
        # Handle different formats
        if ' H' in duration_str:
            # Format: "269 H" -> 269.0 hours
            hours = float(duration_str.replace(' H', ''))
            return hours
        elif ' M' in duration_str:
            # Format: "13 M" -> 13.0 minutes (convert to hours)
            minutes = float(duration_str.replace(' M', ''))
            return minutes / 60.0
        elif ' S' in duration_str:
            # Format: "30 S" -> 30.0 seconds (convert to hours)
            seconds = float(duration_str.replace(' S', ''))
            return seconds / 3600.0
        else:
            # Try to parse as float directly
            try:
                return float(duration_str)
            except ValueError:
                logger.warning(f"Could not parse duration: {duration_str}")
                return 0.0
    
    def is_combination_raga(self, filename: str) -> bool:
        """Check if filename contains combination ragas"""
        return ',' in filename or '&' in filename or 'and' in filename.lower()
    
    def extract_raga_info_fixed(self, raga_data: dict) -> dict:
        """Extract raga information with fixed duration parsing"""
        raga_info = {
            'name': None,
            'melakartha': None,
            'arohana': None,
            'avarohana': None,
            'tradition': None,
            'songs_count': 0,
            'composers_count': 0,
            'duration_hours': 0.0,
            'views': 0
        }
        
        # Extract basic info
        if 'info' in raga_data:
            for item in raga_data['info']:
                if item.get('H') == 'Melakartha':
                    raga_info['melakartha'] = item.get('V')
                elif item.get('H') == 'Arohana':
                    arohana_data = item.get('V')
                    if isinstance(arohana_data, list) and len(arohana_data) > 0:
                        raga_info['arohana'] = arohana_data[0]
                elif item.get('H') == 'Avarohana':
                    avarohana_data = item.get('V')
                    if isinstance(avarohana_data, list) and len(avarohana_data) > 0:
                        raga_info['avarohana'] = avarohana_data[0]
        
        # Extract stats with fixed duration parsing
        if 'stats' in raga_data:
            for stat in raga_data['stats']:
                if stat.get('H') == 'Songs':
                    try:
                        raga_info['songs_count'] = int(stat.get('C', '0').replace(',', ''))
                    except ValueError:
                        raga_info['songs_count'] = 0
                elif stat.get('H') == 'Composers':
                    try:
                        raga_info['composers_count'] = int(stat.get('C', '0').replace(',', ''))
                    except ValueError:
                        raga_info['composers_count'] = 0
                elif stat.get('H') == 'Duration':
                    duration_str = stat.get('C', '0')
                    raga_info['duration_hours'] = self.parse_duration_string(duration_str)
                elif stat.get('H') == 'Views':
                    views_str = stat.get('C', '0')
                    try:
                        # Handle formats like "32 M" (32 million)
                        if ' M' in views_str:
                            raga_info['views'] = int(float(views_str.replace(' M', '')) * 1000000)
                        elif ' K' in views_str:
                            raga_info['views'] = int(float(views_str.replace(' K', '')) * 1000)
                        else:
                            raga_info['views'] = int(views_str.replace(',', ''))
                    except ValueError:
                        raga_info['views'] = 0
        
        return raga_info
    
    def process_ramanarunachalam_tradition_fixed(self, tradition: str) -> List[dict]:
        """Process Ramanarunachalam raga definitions with fixed duration parsing"""
        logger.info(f"Processing Ramanarunachalam {tradition} tradition with fixed parsing...")
        
        raga_dir = self.ramanarunachalam_path / tradition / "raga"
        if not raga_dir.exists():
            logger.error(f"Raga directory not found: {raga_dir}")
            return []
        
        ragas = []
        processed_count = 0
        skipped_count = 0
        duration_errors_fixed = 0
        
        for raga_file in raga_dir.glob("*.json"):
            # Skip combination ragas
            if self.is_combination_raga(raga_file.name):
                skipped_count += 1
                continue
            
            try:
                with open(raga_file, 'r', encoding='utf-8') as f:
                    raga_data = json.load(f)
                
                raga_info = self.extract_raga_info_fixed(raga_data)
                raga_info['name'] = raga_file.stem
                raga_info['tradition'] = tradition
                raga_info['source'] = 'ramanarunachalam'
                raga_info['source_file'] = str(raga_file)
                
                # Check if we fixed a duration error
                if raga_info['duration_hours'] > 0:
                    duration_errors_fixed += 1
                
                ragas.append(raga_info)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {raga_file.name}: {e}")
                skipped_count += 1
        
        logger.info(f"Processed {processed_count} {tradition} ragas, skipped {skipped_count}, fixed {duration_errors_fixed} duration errors")
        self.results['sources'][f'ramanarunachalam_{tradition.lower()}'] = {
            'ragas_processed': processed_count,
            'ragas_skipped': skipped_count,
            'duration_errors_fixed': duration_errors_fixed
        }
        
        return ragas
    
    def extract_saraga_metadata_proper(self, tradition: str) -> List[dict]:
        """Extract metadata from Saraga zip files properly"""
        logger.info(f"Extracting Saraga {tradition} metadata properly...")
        
        zip_path = self.saraga_path / tradition / f"saraga1.5_{tradition}.zip"
        if not zip_path.exists():
            logger.error(f"Saraga zip file not found: {zip_path}")
            return []
        
        metadata = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Look for metadata files
                metadata_files = [f for f in zip_ref.namelist() if f.endswith('.json') and 'metadata' in f.lower()]
                
                logger.info(f"Found {len(metadata_files)} metadata files in {tradition}")
                
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
                                    'album': data.get('album', ''),
                                    'form': data.get('form', ''),
                                    'taala': data.get('taala', ''),
                                    'laya': data.get('laya', '')
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
    
    def extract_saraga_melody_synth(self) -> List[dict]:
        """Extract Saraga-Carnatic-Melody-Synth data"""
        logger.info("Extracting Saraga-Carnatic-Melody-Synth data...")
        
        synth_path = self.saraga_path / "carnatic" / "saraga1.5_carnatic-melody-synth.zip"
        if not synth_path.exists():
            logger.warning(f"Melody synth file not found: {synth_path}")
            return []
        
        synth_data = []
        
        try:
            with zipfile.ZipFile(synth_path, 'r') as zip_ref:
                # Look for melody files
                melody_files = [f for f in zip_ref.namelist() if f.endswith('.json') and 'melody' in f.lower()]
                
                logger.info(f"Found {len(melody_files)} melody files")
                
                for melody_file in melody_files:
                    try:
                        with zip_ref.open(melody_file) as f:
                            data = json.load(f)
                        
                        # Extract melody information
                        if isinstance(data, dict):
                            raga_name = data.get('raaga', data.get('raag', data.get('raga', 'Unknown')))
                            if raga_name and raga_name != 'Unknown':
                                synth_data.append({
                                    'name': raga_name,
                                    'tradition': 'Carnatic',
                                    'source': 'saraga_melody_synth',
                                    'melody_file': melody_file,
                                    'synthetic_audio': data.get('audio_file', ''),
                                    'melody_notes': data.get('melody', ''),
                                    'duration': data.get('duration', 0)
                                })
                    except Exception as e:
                        logger.warning(f"Error processing melody file {melody_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error extracting melody synth data: {e}")
        
        logger.info(f"Extracted {len(synth_data)} synthetic melody entries")
        self.results['sources']['saraga_melody_synth'] = {
            'synthetic_files': len(synth_data),
            'ragas_found': len(set(item['name'] for item in synth_data))
        }
        
        return synth_data
    
    def merge_all_sources(self, ramanarunachalam_ragas: List[dict], 
                         saraga_metadata: List[dict], 
                         synth_data: List[dict]) -> List[dict]:
        """Merge all sources into complete unified dataset"""
        logger.info("Merging all sources into complete unified dataset...")
        
        # Create lookup dictionaries
        ramanarunachalam_lookup = {raga['name'].lower(): raga for raga in ramanarunachalam_ragas}
        saraga_lookup = {}
        synth_lookup = {}
        
        # Organize Saraga data
        for item in saraga_metadata:
            raga_name = item['name'].lower()
            if raga_name not in saraga_lookup:
                saraga_lookup[raga_name] = []
            saraga_lookup[raga_name].append(item)
        
        # Organize synth data
        for item in synth_data:
            raga_name = item['name'].lower()
            if raga_name not in synth_lookup:
                synth_lookup[raga_name] = []
            synth_lookup[raga_name].append(item)
        
        # Merge data
        complete_ragas = []
        ragas_with_audio = 0
        ragas_with_synthetic = 0
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
            
            # Add synthetic data if available
            if raga_name_lower in synth_lookup:
                raga['synthetic_melodies'] = synth_lookup[raga_name_lower]
                raga['has_synthetic'] = True
                ragas_with_synthetic += 1
            else:
                raga['synthetic_melodies'] = []
                raga['has_synthetic'] = False
            
            if not raga['has_audio'] and not raga['has_synthetic']:
                ragas_without_audio += 1
            
            complete_ragas.append(raga)
        
        # Add Saraga ragas not in Ramanarunachalam
        for raga_name, recordings in saraga_lookup.items():
            if raga_name not in ramanarunachalam_lookup:
                raga_info = {
                    'name': recordings[0]['name'],
                    'tradition': recordings[0]['tradition'],
                    'source': 'saraga_only',
                    'melakartha': None,
                    'arohana': None,
                    'avarohana': None,
                    'songs_count': 0,
                    'composers_count': 0,
                    'duration_hours': 0.0,
                    'views': 0,
                    'audio_recordings': recordings,
                    'has_audio': True,
                    'synthetic_melodies': [],
                    'has_synthetic': False
                }
                complete_ragas.append(raga_info)
                ragas_with_audio += 1
        
        # Add synth ragas not in Ramanarunachalam
        for raga_name, melodies in synth_lookup.items():
            if raga_name not in ramanarunachalam_lookup:
                raga_info = {
                    'name': melodies[0]['name'],
                    'tradition': melodies[0]['tradition'],
                    'source': 'synth_only',
                    'melakartha': None,
                    'arohana': None,
                    'avarohana': None,
                    'songs_count': 0,
                    'composers_count': 0,
                    'duration_hours': 0.0,
                    'views': 0,
                    'audio_recordings': [],
                    'has_audio': False,
                    'synthetic_melodies': melodies,
                    'has_synthetic': True
                }
                complete_ragas.append(raga_info)
                ragas_with_synthetic += 1
        
        logger.info(f"Merged {len(complete_ragas)} ragas: {ragas_with_audio} with audio, {ragas_with_synthetic} with synthetic, {ragas_without_audio} without")
        
        return complete_ragas
    
    def classify_traditions_complete(self, complete_ragas: List[dict]) -> List[dict]:
        """Classify ragas by tradition with cross-tradition detection"""
        logger.info("Classifying ragas by tradition with cross-tradition detection...")
        
        # Count traditions
        tradition_counts = {'Carnatic': 0, 'Hindustani': 0, 'Both': 0}
        
        for raga in complete_ragas:
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
        
        return complete_ragas
    
    def save_complete_dataset(self, complete_ragas: List[dict]):
        """Save the complete unified dataset"""
        logger.info("Saving complete unified dataset...")
        
        # Save main dataset
        dataset_file = self.output_path / "complete_unified_raga_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(complete_ragas, f, indent=2, ensure_ascii=False)
        
        # Create comprehensive summary
        summary = {
            'creation_date': datetime.now().isoformat(),
            'total_ragas': len(complete_ragas),
            'tradition_breakdown': {},
            'source_breakdown': {},
            'audio_availability': {
                'ragas_with_audio': sum(1 for r in complete_ragas if r.get('has_audio', False)),
                'ragas_with_synthetic': sum(1 for r in complete_ragas if r.get('has_synthetic', False)),
                'ragas_without_audio': sum(1 for r in complete_ragas if not r.get('has_audio', False) and not r.get('has_synthetic', False))
            },
            'data_quality': {
                'ragas_with_arohana': sum(1 for r in complete_ragas if r.get('arohana')),
                'ragas_with_avarohana': sum(1 for r in complete_ragas if r.get('avarohana')),
                'ragas_with_melakartha': sum(1 for r in complete_ragas if r.get('melakartha')),
                'ragas_with_duration': sum(1 for r in complete_ragas if r.get('duration_hours', 0) > 0)
            }
        }
        
        # Count by tradition
        for raga in complete_ragas:
            tradition = raga.get('tradition_classification', 'Unknown')
            summary['tradition_breakdown'][tradition] = summary['tradition_breakdown'].get(tradition, 0) + 1
        
        # Count by source
        for raga in complete_ragas:
            source = raga.get('source', 'Unknown')
            summary['source_breakdown'][source] = summary['source_breakdown'].get(source, 0) + 1
        
        # Save summary
        summary_file = self.output_path / "complete_dataset_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save individual tradition files
        for tradition in ['Carnatic', 'Hindustani']:
            tradition_ragas = [r for r in complete_ragas if r.get('tradition_classification') == tradition]
            if tradition_ragas:
                tradition_file = self.output_path / f"complete_{tradition.lower()}_ragas.json"
                with open(tradition_file, 'w', encoding='utf-8') as f:
                    json.dump(tradition_ragas, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved complete dataset: {dataset_file}")
        logger.info(f"Saved summary: {summary_file}")
        
        return summary
    
    def run_complete_dataset_creation(self):
        """Run the complete dataset creation process"""
        logger.info("Starting complete unified dataset creation...")
        
        try:
            # Process Ramanarunachalam traditions with fixed parsing
            carnatic_ragas = self.process_ramanarunachalam_tradition_fixed('Carnatic')
            hindustani_ragas = self.process_ramanarunachalam_tradition_fixed('Hindustani')
            
            # Extract Saraga metadata properly
            saraga_carnatic_metadata = self.extract_saraga_metadata_proper('carnatic')
            saraga_hindustani_metadata = self.extract_saraga_metadata_proper('hindustani')
            
            # Extract Saraga melody synth data
            synth_data = self.extract_saraga_melody_synth()
            
            # Merge all sources
            all_ragas = carnatic_ragas + hindustani_ragas
            all_saraga_metadata = saraga_carnatic_metadata + saraga_hindustani_metadata
            
            complete_ragas = self.merge_all_sources(all_ragas, all_saraga_metadata, synth_data)
            
            # Classify traditions
            classified_ragas = self.classify_traditions_complete(complete_ragas)
            
            # Save dataset
            summary = self.save_complete_dataset(classified_ragas)
            
            # Update results
            self.results['complete_dataset'] = summary
            
            logger.info("Complete unified dataset creation completed successfully!")
            return self.results
            
        except Exception as e:
            logger.error(f"Error during dataset creation: {e}")
            raise

def main():
    """Main function"""
    print("🎵 RagaSense Complete Unified Dataset Creation")
    print("=" * 60)
    print("This will:")
    print("• Fix duration parsing errors in Ramanarunachalam data")
    print("• Extract Saraga audio metadata properly")
    print("• Integrate Saraga-Carnatic-Melody-Synth data")
    print("• Create complete unified dataset with all sources")
    print("=" * 60)
    
    creator = CompleteDatasetCreator()
    results = creator.run_complete_dataset_creation()
    
    print(f"\n✅ Complete Dataset Created!")
    print(f"📊 Total ragas: {results['complete_dataset']['total_ragas']}")
    print(f"📊 Carnatic: {results['complete_dataset']['tradition_breakdown'].get('Carnatic', 0)}")
    print(f"📊 Hindustani: {results['complete_dataset']['tradition_breakdown'].get('Hindustani', 0)}")
    print(f"🎵 With audio: {results['complete_dataset']['audio_availability']['ragas_with_audio']}")
    print(f"🎼 With synthetic: {results['complete_dataset']['audio_availability']['ragas_with_synthetic']}")
    print(f"📁 Dataset saved to: data/02_raw/complete_unified_ragas/")

if __name__ == "__main__":
    main()
