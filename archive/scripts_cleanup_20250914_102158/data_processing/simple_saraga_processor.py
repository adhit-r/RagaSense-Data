#!/usr/bin/env python3
"""
Simple Saraga Dataset Processor
"""

import json
import zipfile
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_saraga_carnatic():
    """Process Saraga 1.5 Carnatic dataset"""
    logger.info("🎼 Processing Saraga 1.5 Carnatic dataset...")
    
    base_path = Path(__file__).parent.parent
    carnatic_zip = base_path / "data" / "raw" / "saraga_datasets" / "carnatic" / "saraga1.5_carnatic.zip"
    
    if not carnatic_zip.exists():
        logger.error(f"❌ Saraga 1.5 Carnatic not found at {carnatic_zip}")
        return {}
    
    try:
        # Extract to temporary directory
        temp_dir = base_path / "temp_saraga_carnatic"
        temp_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(carnatic_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Count files
        audio_files = list(temp_dir.rglob("*.wav")) + list(temp_dir.rglob("*.mp3"))
        metadata_files = list(temp_dir.rglob("*.json"))
        
        # Process a few metadata files to get sample data
        sample_tracks = []
        for metadata_file in metadata_files[:10]:  # Process first 10 files
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                sample_tracks.append({
                    "file": str(metadata_file.relative_to(temp_dir)),
                    "metadata": metadata
                })
            except Exception as e:
                logger.warning(f"⚠️ Error processing {metadata_file}: {e}")
        
        # Create result
        result = {
            "tradition": "carnatic",
            "processing_timestamp": datetime.now().isoformat(),
            "total_audio_files": len(audio_files),
            "total_metadata_files": len(metadata_files),
            "sample_tracks": sample_tracks
        }
        
        # Save result
        output_file = base_path / "data" / "processed" / "saraga_carnatic_processed.json"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        logger.info(f"✅ Saraga Carnatic processed: {len(audio_files)} audio files, {len(metadata_files)} metadata files")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error processing Saraga Carnatic: {e}")
        return {}

def main():
    """Main processing function"""
    logger.info("🚀 Starting Saraga dataset processing...")
    
    result = process_saraga_carnatic()
    
    logger.info("🎉 Processing completed!")
    logger.info(f"📊 Result: {result}")

if __name__ == "__main__":
    main()
