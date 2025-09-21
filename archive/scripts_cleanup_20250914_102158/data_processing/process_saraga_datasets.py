#!/usr/bin/env python3
"""
RagaSense Saraga Dataset Processor
Processes Saraga 1.5 Carnatic and Hindustani datasets without hardcoded data
"""

import os
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SaragaDatasetProcessor:
    """Processes Saraga datasets without hardcoded data"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.raw_data_path = base_path / "data" / "raw"
        self.processed_data_path = base_path / "data" / "processed"
        self.processed_data_path.mkdir(exist_ok=True)
        
        logger.info(f"🎵 Saraga Dataset Processor initialized")
        logger.info(f"📁 Raw data path: {self.raw_data_path}")
        logger.info(f"📁 Processed data path: {self.processed_data_path}")
    
    def process_saraga_carnatic(self) -> Dict:
        """Process Saraga 1.5 Carnatic dataset"""
        logger.info("🎼 Processing Saraga 1.5 Carnatic dataset...")
        
        carnatic_zip = self.raw_data_path / "saraga_datasets" / "carnatic" / "saraga1.5_carnatic.zip"
        if not carnatic_zip.exists():
            logger.error(f"❌ Saraga 1.5 Carnatic not found at {carnatic_zip}")
            return {}
        
        try:
            # Extract to temporary directory
            temp_dir = self.base_path / "temp_saraga_carnatic"
            temp_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(carnatic_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process extracted data
            carnatic_data = self._process_saraga_directory(temp_dir, "carnatic")
            
            # Save processed data
            output_file = self.processed_data_path / "saraga_carnatic_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(carnatic_data, f, indent=2, ensure_ascii=False)
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
            
            logger.info(f"✅ Saraga Carnatic processed: {carnatic_data.get('total_tracks', 0)} tracks")
            return carnatic_data
            
        except Exception as e:
            logger.error(f"❌ Error processing Saraga Carnatic: {e}")
            return {}
    
    def _process_saraga_directory(self, directory: Path, tradition: str) -> Dict:
        """Process extracted Saraga directory"""
        logger.info(f"📂 Processing {tradition} directory: {directory}")
        
        data = {
            "tradition": tradition,
            "processing_timestamp": datetime.now().isoformat(),
            "total_tracks": 0,
            "total_artists": 0,
            "total_ragas": 0,
            "tracks": [],
            "artists": set(),
            "ragas": set(),
            "metadata_files": [],
            "audio_files": []
        }
        
        # Find all files
        all_files = list(directory.rglob("*"))
        
        for file_path in all_files:
            if file_path.is_file():
                if file_path.suffix.lower() in ['.wav', '.mp3', '.flac']:
                    data["audio_files"].append(str(file_path.relative_to(directory)))
                elif file_path.suffix.lower() == '.json':
                    data["metadata_files"].append(str(file_path.relative_to(directory)))
                    
                    # Process metadata file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # Extract track information
                        track_info = self._extract_track_info(metadata, file_path)
                        if track_info:
                            data["tracks"].append(track_info)
                            
                            # Extract artists and ragas
                            if track_info.get("artist"):
                                data["artists"].add(track_info["artist"])
                            if track_info.get("raga"):
                                data["ragas"].add(track_info["raga"])
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing metadata {file_path}: {e}")
        
        # Convert sets to lists and count
        data["artists"] = list(data["artists"])
        data["ragas"] = list(data["ragas"])
        data["total_tracks"] = len(data["tracks"])
        data["total_artists"] = len(data["artists"])
        data["total_ragas"] = len(data["ragas"])
        
        logger.info(f"📊 {tradition.capitalize()} processing results:")
        logger.info(f"   Tracks: {data['total_tracks']}")
        logger.info(f"   Artists: {data['total_artists']}")
        logger.info(f"   Ragas: {data['total_ragas']}")
        logger.info(f"   Audio files: {len(data['audio_files'])}")
        logger.info(f"   Metadata files: {len(data['metadata_files'])}")
        
        return data
    
    def _extract_track_info(self, metadata: Dict, file_path: Path) -> Optional[Dict]:
        """Extract track information from metadata"""
        try:
            track_info = {
                "file_path": str(file_path.relative_to(file_path.parents[2])),
                "tradition": "carnatic" if "carnatic" in str(file_path).lower() else "hindustani"
            }
            
            # Extract common fields
            if isinstance(metadata, dict):
                # Try different possible field names
                for field in ["title", "name", "track_name"]:
                    if field in metadata:
                        track_info["title"] = metadata[field]
                        break
                
                for field in ["artist", "performer", "singer"]:
                    if field in metadata:
                        track_info["artist"] = metadata[field]
                        break
                
                for field in ["raga", "raag", "raaga"]:
                    if field in metadata:
                        track_info["raga"] = metadata[field]
                        break
                
                for field in ["composer", "lyricist"]:
                    if field in metadata:
                        track_info["composer"] = metadata[field]
                        break
                
                for field in ["duration", "length"]:
                    if field in metadata:
                        track_info["duration"] = metadata[field]
                        break
                
                for field in ["language", "lang"]:
                    if field in metadata:
                        track_info["language"] = metadata[field]
                        break
                
                # Add any additional metadata
                track_info["metadata"] = metadata
            
            elif isinstance(metadata, list):
                # Handle list of tracks
                if len(metadata) > 0 and isinstance(metadata[0], dict):
                    # Take first track as representative
                    first_track = metadata[0]
                    track_info.update(self._extract_track_info(first_track, file_path))
            
            return track_info if track_info.get("title") or track_info.get("artist") else None
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting track info from {file_path}: {e}")
            return None
    
    def run_full_processing(self):
        """Run complete Saraga dataset processing"""
        logger.info("🚀 Starting full Saraga dataset processing...")
        
        try:
            # Process Saraga Carnatic
            carnatic_data = self.process_saraga_carnatic()
            
            # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
                "datasets_processed": ["saraga_carnatic"],
                "total_tracks": carnatic_data.get("total_tracks", 0),
                "total_artists": carnatic_data.get("total_artists", 0),
                "total_ragas": carnatic_data.get("total_ragas", 0)
            }
            
            # Save report
            report_path = self.processed_data_path / "saraga_processing_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
            logger.info("🎉 Saraga dataset processing completed!")
            logger.info(f"📊 Final report: {report}")
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise

def main():
    """Main processing function"""
    base_path = Path(__file__).parent.parent
    
    processor = SaragaDatasetProcessor(base_path)
    processor.run_full_processing()

if __name__ == "__main__":
    main()