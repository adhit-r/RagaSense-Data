#!/usr/bin/env python3
"""
Extract Saraga Metadata Properly
===============================

This script properly extracts metadata from Saraga zip files and integrates
it with the complete unified dataset.

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_saraga_metadata_proper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SaragaMetadataExtractor:
    """Extract metadata from Saraga zip files properly"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.saraga_path = self.base_path / "01_source" / "saraga"
        self.output_path = self.base_path / "02_raw" / "saraga_metadata"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            'extraction_date': datetime.now().isoformat(),
            'carnatic': {'files_processed': 0, 'ragas_found': 0, 'metadata_extracted': 0},
            'hindustani': {'files_processed': 0, 'ragas_found': 0, 'metadata_extracted': 0}
        }
    
    def extract_zip_contents(self, zip_path: Path) -> List[str]:
        """Extract and list contents of zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                return zip_ref.namelist()
        except Exception as e:
            logger.error(f"Error reading zip file {zip_path}: {e}")
            return []
    
    def extract_metadata_from_zip(self, zip_path: Path, tradition: str) -> List[dict]:
        """Extract metadata from Saraga zip file"""
        logger.info(f"Extracting metadata from {zip_path.name}...")
        
        metadata = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get all files in the zip
                all_files = zip_ref.namelist()
                logger.info(f"Found {len(all_files)} files in {zip_path.name}")
                
                # Look for different types of metadata files
                metadata_patterns = [
                    'metadata.json',
                    'metadata/',
                    '.json',
                    'info.json',
                    'track_info.json'
                ]
                
                metadata_files = []
                for pattern in metadata_patterns:
                    if pattern.endswith('/'):
                        # Directory pattern
                        metadata_files.extend([f for f in all_files if f.startswith(pattern)])
                    else:
                        # File pattern
                        metadata_files.extend([f for f in all_files if pattern in f.lower()])
                
                # Remove duplicates and sort
                metadata_files = sorted(list(set(metadata_files)))
                logger.info(f"Found {len(metadata_files)} potential metadata files")
                
                # Process each metadata file
                for metadata_file in metadata_files:
                    try:
                        # Skip directories
                        if metadata_file.endswith('/'):
                            continue
                        
                        # Extract file content
                        with zip_ref.open(metadata_file) as f:
                            content = f.read()
                        
                        # Try to parse as JSON
                        try:
                            data = json.loads(content.decode('utf-8'))
                        except:
                            # Try different encodings
                            try:
                                data = json.loads(content.decode('latin-1'))
                            except:
                                logger.warning(f"Could not parse {metadata_file} as JSON")
                                continue
                        
                        # Extract raga information
                        raga_info = self.extract_raga_from_metadata(data, metadata_file, tradition)
                        if raga_info:
                            metadata.append(raga_info)
                            
                    except Exception as e:
                        logger.warning(f"Error processing {metadata_file}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error extracting from {zip_path}: {e}")
        
        logger.info(f"Extracted {len(metadata)} metadata entries from {tradition}")
        return metadata
    
    def extract_raga_from_metadata(self, data: dict, file_path: str, tradition: str) -> Optional[dict]:
        """Extract raga information from metadata"""
        raga_info = {
            'tradition': tradition,
            'source': 'saraga',
            'metadata_file': file_path,
            'audio_file': '',
            'raga_name': '',
            'artist': '',
            'album': '',
            'duration': 0,
            'form': '',
            'taala': '',
            'laya': '',
            'tonic': '',
            'pitch_available': False,
            'sections_available': False
        }
        
        # Try different field names for raga
        raga_fields = ['raaga', 'raag', 'raga', 'rāga', 'rāg', 'ragam']
        for field in raga_fields:
            if field in data:
                raga_info['raga_name'] = data[field]
                break
        
        # If no raga found, try nested structures
        if not raga_info['raga_name']:
            for key, value in data.items():
                if isinstance(value, dict):
                    for field in raga_fields:
                        if field in value:
                            raga_info['raga_name'] = value[field]
                            break
                    if raga_info['raga_name']:
                        break
        
        # Skip if no raga name found
        if not raga_info['raga_name'] or raga_info['raga_name'] == 'Unknown':
            return None
        
        # Extract other information
        raga_info['artist'] = data.get('artist', data.get('performer', ''))
        raga_info['album'] = data.get('album', data.get('release', ''))
        raga_info['duration'] = data.get('duration', 0)
        raga_info['form'] = data.get('form', data.get('type', ''))
        raga_info['taala'] = data.get('taala', data.get('tala', ''))
        raga_info['laya'] = data.get('laya', data.get('tempo', ''))
        raga_info['tonic'] = data.get('tonic', data.get('pitch', ''))
        
        # Check for available data
        raga_info['pitch_available'] = 'pitch' in data or 'tonic' in data
        raga_info['sections_available'] = 'sections' in data
        
        # Try to find audio file path
        audio_fields = ['audio_file', 'audio', 'file', 'path', 'filename']
        for field in audio_fields:
            if field in data:
                raga_info['audio_file'] = data[field]
                break
        
        return raga_info
    
    def process_tradition(self, tradition: str) -> List[dict]:
        """Process a tradition's Saraga data"""
        logger.info(f"Processing Saraga {tradition} tradition...")
        
        tradition_path = self.saraga_path / tradition
        if not tradition_path.exists():
            logger.error(f"Tradition path not found: {tradition_path}")
            return []
        
        # Find zip files
        zip_files = list(tradition_path.glob("*.zip"))
        if not zip_files:
            logger.error(f"No zip files found in {tradition_path}")
            return []
        
        all_metadata = []
        
        for zip_file in zip_files:
            logger.info(f"Processing {zip_file.name}...")
            metadata = self.extract_metadata_from_zip(zip_file, tradition)
            all_metadata.extend(metadata)
            self.results[tradition]['files_processed'] += 1
        
        # Remove duplicates and organize
        unique_ragas = {}
        for item in all_metadata:
            raga_name = item['raga_name']
            
            # Handle different data types for raga_name
            if isinstance(raga_name, dict):
                # If it's a dict, try to extract a meaningful name
                if 'name' in raga_name:
                    raga_name = raga_name['name']
                elif 'raga' in raga_name:
                    raga_name = raga_name['raga']
                else:
                    raga_name = str(raga_name)
            elif isinstance(raga_name, list):
                raga_name = raga_name[0] if raga_name else 'Unknown'
            elif not isinstance(raga_name, str):
                raga_name = str(raga_name)
            
            # Clean up the name
            raga_name = str(raga_name).strip().lower()
            if not raga_name or raga_name == 'unknown' or raga_name == 'none':
                continue  # Skip invalid entries
            
            if raga_name not in unique_ragas:
                unique_ragas[raga_name] = []
            unique_ragas[raga_name].append(item)
        
        # Flatten back to list
        final_metadata = []
        for raga_name, recordings in unique_ragas.items():
            for recording in recordings:
                final_metadata.append(recording)
        
        self.results[tradition]['ragas_found'] = len(unique_ragas)
        self.results[tradition]['metadata_extracted'] = len(final_metadata)
        
        logger.info(f"Processed {tradition}: {len(final_metadata)} recordings, {len(unique_ragas)} unique ragas")
        
        return final_metadata
    
    def save_metadata(self, carnatic_metadata: List[dict], hindustani_metadata: List[dict]):
        """Save extracted metadata"""
        logger.info("Saving extracted metadata...")
        
        # Save individual tradition metadata
        carnatic_file = self.output_path / "saraga_carnatic_metadata.json"
        with open(carnatic_file, 'w', encoding='utf-8') as f:
            json.dump(carnatic_metadata, f, indent=2, ensure_ascii=False)
        
        hindustani_file = self.output_path / "saraga_hindustani_metadata.json"
        with open(hindustani_file, 'w', encoding='utf-8') as f:
            json.dump(hindustani_metadata, f, indent=2, ensure_ascii=False)
        
        # Save combined metadata
        combined_metadata = carnatic_metadata + hindustani_metadata
        combined_file = self.output_path / "saraga_combined_metadata.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_metadata, f, indent=2, ensure_ascii=False)
        
        # Create summary
        def get_raga_name(item):
            raga_name = item['raga_name']
            if isinstance(raga_name, dict):
                if 'name' in raga_name:
                    raga_name = raga_name['name']
                elif 'raga' in raga_name:
                    raga_name = raga_name['raga']
                else:
                    raga_name = str(raga_name)
            elif isinstance(raga_name, list):
                raga_name = raga_name[0] if raga_name else 'Unknown'
            return str(raga_name).strip().lower()
        
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'total_recordings': len(combined_metadata),
            'carnatic_recordings': len(carnatic_metadata),
            'hindustani_recordings': len(hindustani_metadata),
            'unique_ragas': len(set(get_raga_name(item) for item in combined_metadata)),
            'carnatic_ragas': len(set(get_raga_name(item) for item in carnatic_metadata)),
            'hindustani_ragas': len(set(get_raga_name(item) for item in hindustani_metadata)),
            'data_availability': {
                'with_artist': sum(1 for item in combined_metadata if item['artist']),
                'with_album': sum(1 for item in combined_metadata if item['album']),
                'with_duration': sum(1 for item in combined_metadata if item['duration'] > 0),
                'with_form': sum(1 for item in combined_metadata if item['form']),
                'with_taala': sum(1 for item in combined_metadata if item['taala']),
                'with_tonic': sum(1 for item in combined_metadata if item['tonic']),
                'with_pitch_data': sum(1 for item in combined_metadata if item['pitch_available']),
                'with_sections': sum(1 for item in combined_metadata if item['sections_available'])
            }
        }
        
        summary_file = self.output_path / "saraga_metadata_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata to {self.output_path}")
        return summary
    
    def run_extraction(self):
        """Run the complete metadata extraction process"""
        logger.info("Starting Saraga metadata extraction...")
        
        try:
            # Process both traditions
            carnatic_metadata = self.process_tradition('carnatic')
            hindustani_metadata = self.process_tradition('hindustani')
            
            # Save metadata
            summary = self.save_metadata(carnatic_metadata, hindustani_metadata)
            
            logger.info("Saraga metadata extraction completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise

def main():
    """Main function"""
    print("🎵 Saraga Metadata Extraction")
    print("=" * 40)
    print("This will extract metadata from Saraga zip files")
    print("and prepare it for integration with the unified dataset")
    print("=" * 40)
    
    extractor = SaragaMetadataExtractor()
    results = extractor.run_extraction()
    
    print(f"\n✅ Metadata Extraction Complete!")
    print(f"📊 Total recordings: {results['total_recordings']}")
    print(f"📊 Carnatic: {results['carnatic_recordings']}")
    print(f"📊 Hindustani: {results['hindustani_recordings']}")
    print(f"🎵 Unique ragas: {results['unique_ragas']}")
    print(f"📁 Metadata saved to: data/02_raw/saraga_metadata/")

if __name__ == "__main__":
    main()
