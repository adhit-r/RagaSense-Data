#!/usr/bin/env python3
"""
Efficient Saraga Dataset Processing - Metadata Extraction Only
============================================================

This script processes Saraga datasets efficiently by:
1. Extracting metadata from zip files without full unzipping
2. Processing already unzipped datasets
3. Creating comprehensive metadata for integration

Approach:
- Use zipfile to read metadata from compressed files
- Extract only essential information (filenames, structure)
- Avoid full extraction to save disk space
- Focus on metadata that can be used for dataset integration
"""

import json
import os
import zipfile
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('saraga_efficient_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EfficientSaragaProcessor:
    """
    Efficiently processes Saraga datasets by extracting metadata without full unzipping.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.downloads_path = self.project_root / "downloads"
        self.output_path = self.project_root / "data" / "saraga_processed"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.all_processed_data = {
            "artists": {},
            "tracks": {},
            "ragas": {},
            "datasets": {},
            "metadata": {
                "total_artists": 0,
                "total_tracks": 0,
                "total_audio_files": 0,
                "datasets_processed": 0,
                "processing_timestamp": datetime.now().isoformat()
            }
        }

    def extract_zip_metadata(self, zip_path: Path) -> Dict[str, Any]:
        """Extract metadata from a zip file without unzipping."""
        logger.info(f"📦 Extracting metadata from {zip_path.name}...")
        
        metadata = {
            "total_files": 0,
            "audio_files": [],
            "annotation_files": [],
            "metadata_files": [],
            "directory_structure": {},
            "estimated_size_mb": 0
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                metadata["total_files"] = len(file_list)
                
                # Categorize files
                for file_path in file_list:
                    if file_path.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                        metadata["audio_files"].append(file_path)
                    elif file_path.endswith('.json'):
                        if 'annotation' in file_path.lower():
                            metadata["annotation_files"].append(file_path)
                        else:
                            metadata["metadata_files"].append(file_path)
                    
                    # Build directory structure
                    path_parts = file_path.split('/')
                    current_level = metadata["directory_structure"]
                    for part in path_parts[:-1]:  # Exclude filename
                        if part not in current_level:
                            current_level[part] = {}
                        current_level = current_level[part]
                
                # Estimate size
                total_size = sum(info.file_size for info in zip_ref.infolist())
                metadata["estimated_size_mb"] = total_size / (1024 * 1024)
                
                logger.info(f"   Found {len(metadata['audio_files'])} audio files")
                logger.info(f"   Found {len(metadata['annotation_files'])} annotation files")
                logger.info(f"   Estimated size: {metadata['estimated_size_mb']:.1f} MB")
                
        except Exception as e:
            logger.error(f"❌ Failed to extract metadata from {zip_path}: {e}")
            return None
        
        return metadata

    def process_zip_dataset(self, dataset_name: str, zip_path: Path, tradition: str):
        """Process a zip dataset by extracting metadata only."""
        logger.info(f"🎵 Processing {dataset_name} dataset...")
        
        if not zip_path.exists():
            logger.error(f"❌ Zip file not found: {zip_path}")
            return False
        
        # Extract metadata
        zip_metadata = self.extract_zip_metadata(zip_path)
        if not zip_metadata:
            return False
        
        # Process audio files to extract artist/track information
        artists_found = set()
        tracks_found = 0
        
        for audio_file in zip_metadata["audio_files"]:
            # Extract artist and track info from path
            path_parts = audio_file.split('/')
            
            # Look for artist directory (usually first or second level)
            artist_name = "Unknown"
            track_name = Path(audio_file).stem
            
            if len(path_parts) >= 2:
                # Try different patterns
                for part in path_parts:
                    if part and part not in ['audio', 'wav', 'mp3'] and not part.endswith(('.wav', '.mp3')):
                        if len(part) > 3:  # Reasonable artist name length
                            artist_name = part
                            break
            
            artists_found.add(artist_name)
            tracks_found += 1
            
            track_id = f"{dataset_name}_{artist_name}_{track_name}"
            self.all_processed_data["tracks"][track_id] = {
                "track_id": track_id,
                "name": track_name,
                "artist": artist_name,
                "tradition": tradition,
                "source": "saraga1.5",
                "dataset": dataset_name,
                "audio_file": audio_file,
                "file_type": Path(audio_file).suffix
            }
        
        # Process artists
        for artist_name in artists_found:
            artist_tracks = [t for t in self.all_processed_data["tracks"].values() 
                           if t["artist"] == artist_name and t["dataset"] == dataset_name]
            
            self.all_processed_data["artists"][artist_name] = {
                "name": artist_name,
                "tradition": tradition,
                "total_tracks": len(artist_tracks),
                "source": "saraga1.5",
                "dataset": dataset_name
            }
        
        # Store dataset metadata
        self.all_processed_data["datasets"][dataset_name] = {
            "name": dataset_name,
            "tradition": tradition,
            "source": "saraga1.5",
            "zip_file": str(zip_path),
            "total_files": zip_metadata["total_files"],
            "audio_files": len(zip_metadata["audio_files"]),
            "annotation_files": len(zip_metadata["annotation_files"]),
            "estimated_size_mb": zip_metadata["estimated_size_mb"],
            "artists_found": len(artists_found),
            "tracks_found": tracks_found
        }
        
        logger.info(f"✅ {dataset_name} processed: {len(artists_found)} artists, {tracks_found} tracks")
        return True

    def load_melody_synth_data(self):
        """Load the already processed Melody Synth data."""
        logger.info("🎵 Loading Melody Synth processed data...")
        
        melody_synth_file = self.output_path / "melody_synth_processed.json"
        if melody_synth_file.exists():
            with open(melody_synth_file, 'r') as f:
                melody_data = json.load(f)
            
            # Merge into all processed data
            self.all_processed_data["artists"].update(melody_data["artists"])
            self.all_processed_data["tracks"].update(melody_data["tracks"])
            
            # Add dataset info
            self.all_processed_data["datasets"]["saraga_carnatic_melody_synth"] = {
                "name": "saraga_carnatic_melody_synth",
                "tradition": "Carnatic",
                "source": "saraga_carnatic_melody_synth",
                "total_artists": melody_data["metadata"]["total_artists"],
                "total_tracks": melody_data["metadata"]["total_tracks"],
                "total_audio_files": melody_data["metadata"]["total_audio_files"],
                "status": "processed"
            }
            
            logger.info(f"✅ Melody Synth data loaded: {melody_data['metadata']['total_artists']} artists, {melody_data['metadata']['total_tracks']} tracks")
            return True
        else:
            logger.warning("⚠️ Melody Synth processed data not found")
            return False

    def process_all_datasets(self):
        """Process all Saraga datasets efficiently."""
        logger.info("🚀 Starting efficient Saraga datasets processing...")
        start_time = time.time()
        
        # Load already processed Melody Synth data
        self.load_melody_synth_data()
        
        # Process Saraga 1.5 Carnatic
        carnatic_zip = self.downloads_path / "saraga_datasets" / "carnatic" / "saraga1.5_carnatic.zip"
        if carnatic_zip.exists():
            if not self.process_zip_dataset("saraga1.5_carnatic", carnatic_zip, "Carnatic"):
                logger.error("❌ Failed to process Saraga 1.5 Carnatic")
                return False
        else:
            logger.warning("⚠️ Saraga 1.5 Carnatic zip file not found")
        
        # Process Saraga 1.5 Hindustani
        hindustani_zip = self.downloads_path / "saraga_datasets" / "hindustani" / "saraga1.5_hindustani.zip"
        if hindustani_zip.exists():
            if not self.process_zip_dataset("saraga1.5_hindustani", hindustani_zip, "Hindustani"):
                logger.error("❌ Failed to process Saraga 1.5 Hindustani")
                return False
        else:
            logger.warning("⚠️ Saraga 1.5 Hindustani zip file not found")
        
        # Calculate final statistics
        self.all_processed_data["metadata"]["total_artists"] = len(self.all_processed_data["artists"])
        self.all_processed_data["metadata"]["total_tracks"] = len(self.all_processed_data["tracks"])
        self.all_processed_data["metadata"]["total_audio_files"] = sum(
            dataset.get("total_audio_files", 0) for dataset in self.all_processed_data["datasets"].values()
        )
        self.all_processed_data["metadata"]["datasets_processed"] = len(self.all_processed_data["datasets"])
        self.all_processed_data["metadata"]["processing_time_seconds"] = time.time() - start_time
        
        logger.info("🎉 All Saraga datasets processed efficiently!")
        return True

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive processing report."""
        logger.info("📊 Generating comprehensive report...")
        
        # Artist statistics by tradition
        artist_stats = Counter(artist["tradition"] for artist in self.all_processed_data["artists"].values())
        
        # Track statistics by tradition
        track_stats = Counter(track["tradition"] for track in self.all_processed_data["tracks"].values())
        
        # Top artists by track count
        top_artists = sorted(
            self.all_processed_data["artists"].items(),
            key=lambda x: x[1]["total_tracks"],
            reverse=True
        )[:15]
        
        # Dataset summary
        dataset_summary = {}
        for dataset_name, dataset_info in self.all_processed_data["datasets"].items():
            dataset_summary[dataset_name] = {
                "tradition": dataset_info["tradition"],
                "artists": dataset_info.get("artists_found", dataset_info.get("total_artists", 0)),
                "tracks": dataset_info.get("tracks_found", dataset_info.get("total_tracks", 0)),
                "audio_files": dataset_info.get("total_audio_files", 0),
                "size_mb": dataset_info.get("estimated_size_mb", 0)
            }
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "processing_summary": self.all_processed_data["metadata"],
            "tradition_distribution": {
                "artists": dict(artist_stats),
                "tracks": dict(track_stats)
            },
            "dataset_summary": dataset_summary,
            "top_artists": [
                {
                    "name": artist_name,
                    "tradition": artist_data["tradition"],
                    "total_tracks": artist_data["total_tracks"],
                    "source": artist_data["source"]
                }
                for artist_name, artist_data in top_artists
            ],
            "data_quality_metrics": {
                "artists_with_tracks": len([a for a in self.all_processed_data["artists"].values() if a["total_tracks"] > 0]),
                "unique_artists": len(self.all_processed_data["artists"]),
                "total_audio_files": self.all_processed_data["metadata"]["total_audio_files"],
                "datasets_processed": self.all_processed_data["metadata"]["datasets_processed"]
            }
        }
        
        return report

    def save_all_processed_data(self):
        """Save all processed data and reports."""
        logger.info("💾 Saving all processed data...")
        
        # Save comprehensive processed data
        with open(self.output_path / "saraga_all_processed_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.all_processed_data, f, indent=2, ensure_ascii=False)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        with open(self.output_path / "saraga_comprehensive_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save individual components
        for component in ["artists", "tracks", "datasets"]:
            with open(self.output_path / f"saraga_{component}.json", 'w', encoding='utf-8') as f:
                json.dump(self.all_processed_data[component], f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ All processed data saved to {self.output_path}")

    def run_efficient_processing(self):
        """Run the efficient Saraga datasets processing."""
        logger.info("🚀 STARTING EFFICIENT SARAGA DATASETS PROCESSING")
        logger.info("=" * 60)
        
        # Process all datasets
        if not self.process_all_datasets():
            return False
        
        # Save all processed data
        self.save_all_processed_data()
        
        # Print summary
        logger.info("\n🎉 EFFICIENT SARAGA DATASETS PROCESSING COMPLETED!")
        logger.info(f"⏱️ Processing time: {self.all_processed_data['metadata']['processing_time_seconds']:.1f} seconds")
        
        logger.info("\n📊 PROCESSING SUMMARY:")
        logger.info(f"   Datasets processed: {self.all_processed_data['metadata']['datasets_processed']}")
        logger.info(f"   Total artists: {self.all_processed_data['metadata']['total_artists']}")
        logger.info(f"   Total tracks: {self.all_processed_data['metadata']['total_tracks']}")
        logger.info(f"   Total audio files: {self.all_processed_data['metadata']['total_audio_files']}")
        
        return True

def main():
    """Main function to run the efficient Saraga datasets processing."""
    processor = EfficientSaragaProcessor()
    success = processor.run_efficient_processing()
    
    if success:
        logger.info(f"\n🎯 EFFICIENT SARAGA DATASETS PROCESSING COMPLETE!")
        logger.info(f"📋 Processed data saved to: {processor.output_path}")
        logger.info(f"📊 Report saved to: {processor.output_path / 'saraga_comprehensive_report.json'}")
    else:
        logger.error("❌ Efficient Saraga datasets processing failed!")

if __name__ == "__main__":
    main()
