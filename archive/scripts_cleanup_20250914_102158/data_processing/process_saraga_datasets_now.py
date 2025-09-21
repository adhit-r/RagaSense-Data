#!/usr/bin/env python3
"""
Process Saraga Datasets - Immediate Action
==========================================

This script processes the Saraga 1.5 Carnatic and Hindustani datasets
that are already downloaded in the organized_raw directory.
"""

import json
import zipfile
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_saraga_dataset(zip_path: Path, tradition: str):
    """Process a Saraga dataset ZIP file."""
    logger.info(f"🎵 Processing {tradition} dataset: {zip_path.name}")
    
    if not zip_path.exists():
        logger.error(f"❌ Dataset not found: {zip_path}")
        return {}
    
    # Create output directory
    output_dir = Path("data/processed/saraga") / tradition.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        "tracks": 0,
        "artists": 0,
        "ragas": 0,
        "audio_files": 0,
        "metadata_files": 0,
        "processing_time": 0
    }
    
    start_time = datetime.now()
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            
            # Count files
            audio_files = [f for f in file_list if f.endswith(('.wav', '.mp3', '.flac'))]
            metadata_files = [f for f in file_list if f.endswith('.json') and not f.startswith('__MACOSX')]
            
            stats["audio_files"] = len(audio_files)
            stats["metadata_files"] = len(metadata_files)
            
            logger.info(f"📊 Found {len(audio_files)} audio files and {len(metadata_files)} metadata files")
            
            # Process metadata files
            processed_tracks = []
            artists = set()
            ragas = set()
            
            for i, metadata_file in enumerate(metadata_files[:100]):  # Process first 100 for now
                try:
                    with zip_ref.open(metadata_file) as f:
                        data = json.load(f)
                        
                        # Extract track information
                        track_info = {
                            "file_path": metadata_file,
                            "tradition": tradition,
                            "source": "saraga1.5",
                            "processed_at": datetime.now().isoformat()
                        }
                        
                        # Extract common fields
                        if isinstance(data, dict):
                            # Try different possible field names
                            for field in ["title", "name", "track_name"]:
                                if field in data:
                                    track_info["title"] = data[field]
                                    break
                            
                            for field in ["artist", "performer", "singer"]:
                                if field in data:
                                    track_info["artist"] = data[field]
                                    artists.add(data[field])
                                    break
                            
                            for field in ["raga", "raag", "raaga"]:
                                if field in data:
                                    track_info["raga"] = data[field]
                                    ragas.add(data[field])
                                    break
                            
                            for field in ["composer", "lyricist"]:
                                if field in data:
                                    track_info["composer"] = data[field]
                                    break
                            
                            for field in ["duration", "length"]:
                                if field in data:
                                    track_info["duration"] = data[field]
                                    break
                            
                            for field in ["language", "lang"]:
                                if field in data:
                                    track_info["language"] = data[field]
                                    break
                            
                            # Add any additional metadata
                            track_info["metadata"] = data
                        
                        processed_tracks.append(track_info)
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"📝 Processed {i + 1}/{min(100, len(metadata_files))} metadata files")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Error processing {metadata_file}: {e}")
                    continue
            
            stats["tracks"] = len(processed_tracks)
            stats["artists"] = len(artists)
            stats["ragas"] = len(ragas)
            stats["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            # Save processed data
            output_file = output_dir / f"{tradition.lower()}_processed.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "metadata": {
                        "tradition": tradition,
                        "source": "saraga1.5",
                        "processed_at": datetime.now().isoformat(),
                        "total_files": len(file_list),
                        "audio_files": len(audio_files),
                        "metadata_files": len(metadata_files),
                        "processed_tracks": len(processed_tracks)
                    },
                    "statistics": stats,
                    "tracks": processed_tracks,
                    "artists": list(artists),
                    "ragas": list(ragas)
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved processed data to: {output_file}")
            
            return stats
            
    except Exception as e:
        logger.error(f"❌ Failed to process {zip_path}: {e}")
        return {}

def main():
    """Main function to process all Saraga datasets."""
    logger.info("🚀 Starting Saraga dataset processing...")
    
    base_path = Path(__file__).parent
    saraga_path = base_path / "data" / "organized_raw" / "saraga_datasets"
    
    # Process Carnatic dataset
    carnatic_zip = saraga_path / "carnatic" / "saraga1.5_carnatic.zip"
    carnatic_stats = process_saraga_dataset(carnatic_zip, "Carnatic")
    
    # Process Hindustani dataset
    hindustani_zip = saraga_path / "hindustani" / "saraga1.5_hindustani.zip"
    hindustani_stats = process_saraga_dataset(hindustani_zip, "Hindustani")
    
    # Create summary report
    summary = {
        "processing_date": datetime.now().isoformat(),
        "carnatic": carnatic_stats,
        "hindustani": hindustani_stats,
        "total_tracks": carnatic_stats.get("tracks", 0) + hindustani_stats.get("tracks", 0),
        "total_artists": carnatic_stats.get("artists", 0) + hindustani_stats.get("artists", 0),
        "total_ragas": carnatic_stats.get("ragas", 0) + hindustani_stats.get("ragas", 0),
        "total_audio_files": carnatic_stats.get("audio_files", 0) + hindustani_stats.get("audio_files", 0)
    }
    
    # Save summary
    summary_file = base_path / "data" / "processed" / "saraga_processing_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Summary saved to: {summary_file}")
    
    # Log results
    logger.info("🎉 Saraga dataset processing completed!")
    logger.info(f"📊 Total tracks processed: {summary['total_tracks']}")
    logger.info(f"📊 Total artists found: {summary['total_artists']}")
    logger.info(f"📊 Total ragas found: {summary['total_ragas']}")
    logger.info(f"📊 Total audio files: {summary['total_audio_files']}")
    
    return summary

if __name__ == "__main__":
    main()
