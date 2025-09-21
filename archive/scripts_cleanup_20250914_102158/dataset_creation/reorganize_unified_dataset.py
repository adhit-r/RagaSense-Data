#!/usr/bin/env python3
"""
Reorganize RagaSense Data into Unified Dataset Structure
========================================================

This script reorganizes the existing data into a clean, unified structure:
- data/carnatic/ - All Carnatic data (audio, metadata, annotations)
- data/hindustani/ - All Hindustani data (audio, metadata, annotations)
- data/unified/ - Cross-tradition mappings and unified datasets

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reorganize_unified_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedDatasetReorganizer:
    """Reorganize data into unified carnatic/hindustani structure"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.source_path = self.base_path / "organized_raw" / "Ramanarunachalam_Music_Repository"
        self.carnatic_path = self.base_path / "carnatic"
        self.hindustani_path = self.base_path / "hindustani"
        self.unified_path = self.base_path / "unified"
        
        # Create target directories
        self._create_directories()
        
    def _create_directories(self):
        """Create the unified directory structure"""
        directories = [
            self.carnatic_path / "audio",
            self.carnatic_path / "metadata",
            self.carnatic_path / "annotations",
            self.hindustani_path / "audio", 
            self.hindustani_path / "metadata",
            self.hindustani_path / "annotations",
            self.unified_path / "cross_tradition_mappings",
            self.unified_path / "processed_datasets",
            self.unified_path / "ml_ready"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def reorganize_audio_files(self):
        """Copy audio files to tradition-specific directories"""
        logger.info("Reorganizing audio files...")
        
        # Carnatic audio files
        carnatic_audio_source = self.source_path / "Carnatic" / "audio"
        carnatic_audio_target = self.carnatic_path / "audio"
        
        if carnatic_audio_source.exists():
            for audio_file in carnatic_audio_source.glob("*.mp3"):
                target_file = carnatic_audio_target / audio_file.name
                shutil.copy2(audio_file, target_file)
                logger.info(f"Copied Carnatic audio: {audio_file.name}")
        
        # Hindustani audio files
        hindustani_audio_source = self.source_path / "Hindustani" / "audio"
        hindustani_audio_target = self.hindustani_path / "audio"
        
        if hindustani_audio_source.exists():
            for audio_file in hindustani_audio_source.glob("*.mp3"):
                target_file = hindustani_audio_target / audio_file.name
                shutil.copy2(audio_file, target_file)
                logger.info(f"Copied Hindustani audio: {audio_file.name}")
    
    def reorganize_metadata(self):
        """Copy and organize metadata files"""
        logger.info("Reorganizing metadata files...")
        
        # Carnatic metadata
        carnatic_metadata_source = self.source_path / "Carnatic"
        carnatic_metadata_target = self.carnatic_path / "metadata"
        
        carnatic_metadata_files = [
            "raga.json", "artist.json", "composer.json", "song.json",
            "concert.json", "type.json", "about.json"
        ]
        
        for metadata_file in carnatic_metadata_files:
            source_file = carnatic_metadata_source / metadata_file
            if source_file.exists():
                target_file = carnatic_metadata_target / metadata_file
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied Carnatic metadata: {metadata_file}")
        
        # Hindustani metadata
        hindustani_metadata_source = self.source_path / "Hindustani"
        hindustani_metadata_target = self.hindustani_path / "metadata"
        
        hindustani_metadata_files = [
            "raga.json", "artist.json", "composer.json", "song.json",
            "concert.json", "type.json", "about.json"
        ]
        
        for metadata_file in hindustani_metadata_files:
            source_file = hindustani_metadata_source / metadata_file
            if source_file.exists():
                target_file = hindustani_metadata_target / metadata_file
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied Hindustani metadata: {metadata_file}")
    
    def reorganize_annotations(self):
        """Copy raga annotation files"""
        logger.info("Reorganizing annotation files...")
        
        # Carnatic raga annotations
        carnatic_raga_source = self.source_path / "Carnatic" / "raga"
        carnatic_raga_target = self.carnatic_path / "annotations"
        
        if carnatic_raga_source.exists():
            for raga_file in carnatic_raga_source.glob("*.json"):
                target_file = carnatic_raga_target / raga_file.name
                shutil.copy2(raga_file, target_file)
                logger.info(f"Copied Carnatic raga annotation: {raga_file.name}")
        
        # Hindustani raga annotations
        hindustani_raga_source = self.source_path / "Hindustani" / "raga"
        hindustani_raga_target = self.hindustani_path / "annotations"
        
        if hindustani_raga_source.exists():
            for raga_file in hindustani_raga_source.glob("*.json"):
                target_file = hindustani_raga_target / raga_file.name
                shutil.copy2(raga_file, target_file)
                logger.info(f"Copied Hindustani raga annotation: {raga_file.name}")
    
    def copy_unified_datasets(self):
        """Copy existing unified datasets to the new structure"""
        logger.info("Copying unified datasets...")
        
        # Copy processed datasets
        processed_source = self.base_path / "organized_processed"
        processed_target = self.unified_path / "processed_datasets"
        
        unified_files = [
            "unified_ragas_target_achieved.json",
            "unified_cross_tradition_mappings_nat_fixed.json",
            "unified_ragas_nat_fixed.json"
        ]
        
        for unified_file in unified_files:
            source_file = processed_source / unified_file
            if source_file.exists():
                target_file = processed_target / unified_file
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied unified dataset: {unified_file}")
        
        # Copy cross-tradition mappings
        cross_tradition_source = processed_source / "unified_cross_tradition_mappings_nat_fixed.json"
        cross_tradition_target = self.unified_path / "cross_tradition_mappings" / "cross_tradition_mappings.json"
        
        if cross_tradition_source.exists():
            shutil.copy2(cross_tradition_source, cross_tradition_target)
            logger.info("Copied cross-tradition mappings")
    
    def copy_ml_ready_datasets(self):
        """Copy ML-ready datasets to unified structure"""
        logger.info("Copying ML-ready datasets...")
        
        ml_source = self.base_path / "ml_ready"
        ml_target = self.unified_path / "ml_ready"
        
        # Copy key ML datasets
        ml_files = [
            "final_enhanced_ml_ready_dataset.json",
            "final_enhanced_ml_dataset_summary.json",
            "optimized_audio_features.json",
            "optimized_processing_summary.json"
        ]
        
        for ml_file in ml_files:
            source_file = ml_source / ml_file
            if source_file.exists():
                target_file = ml_target / ml_file
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied ML dataset: {ml_file}")
    
    def create_unified_summary(self):
        """Create a summary of the unified dataset structure"""
        logger.info("Creating unified dataset summary...")
        
        summary = {
            "creation_date": datetime.now().isoformat(),
            "dataset_structure": {
                "carnatic": {
                    "audio_files": len(list((self.carnatic_path / "audio").glob("*.mp3"))),
                    "metadata_files": len(list((self.carnatic_path / "metadata").glob("*.json"))),
                    "annotation_files": len(list((self.carnatic_path / "annotations").glob("*.json")))
                },
                "hindustani": {
                    "audio_files": len(list((self.hindustani_path / "audio").glob("*.mp3"))),
                    "metadata_files": len(list((self.hindustani_path / "metadata").glob("*.json"))),
                    "annotation_files": len(list((self.hindustani_path / "annotations").glob("*.json")))
                },
                "unified": {
                    "processed_datasets": len(list((self.unified_path / "processed_datasets").glob("*.json"))),
                    "cross_tradition_mappings": len(list((self.unified_path / "cross_tradition_mappings").glob("*.json"))),
                    "ml_ready_datasets": len(list((self.unified_path / "ml_ready").glob("*.json")))
                }
            },
            "total_audio_files": (
                len(list((self.carnatic_path / "audio").glob("*.mp3"))) +
                len(list((self.hindustani_path / "audio").glob("*.mp3")))
            ),
            "total_metadata_files": (
                len(list((self.carnatic_path / "metadata").glob("*.json"))) +
                len(list((self.hindustani_path / "metadata").glob("*.json")))
            ),
            "total_annotation_files": (
                len(list((self.carnatic_path / "annotations").glob("*.json"))) +
                len(list((self.hindustani_path / "annotations").glob("*.json")))
            )
        }
        
        # Save summary
        summary_file = self.unified_path / "unified_dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Created unified dataset summary: {summary_file}")
        return summary
    
    def reorganize_all(self):
        """Run the complete reorganization process"""
        logger.info("Starting unified dataset reorganization...")
        
        try:
            self.reorganize_audio_files()
            self.reorganize_metadata()
            self.reorganize_annotations()
            self.copy_unified_datasets()
            self.copy_ml_ready_datasets()
            summary = self.create_unified_summary()
            
            logger.info("Unified dataset reorganization completed successfully!")
            logger.info(f"Total audio files: {summary['total_audio_files']}")
            logger.info(f"Total metadata files: {summary['total_metadata_files']}")
            logger.info(f"Total annotation files: {summary['total_annotation_files']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error during reorganization: {e}")
            raise

def main():
    """Main function"""
    print("🎵 RagaSense Unified Dataset Reorganization")
    print("=" * 50)
    
    reorganizer = UnifiedDatasetReorganizer()
    summary = reorganizer.reorganize_all()
    
    print("\n✅ Reorganization Complete!")
    print(f"📁 Carnatic: {summary['dataset_structure']['carnatic']['audio_files']} audio files")
    print(f"📁 Hindustani: {summary['dataset_structure']['hindustani']['audio_files']} audio files")
    print(f"📁 Unified: {summary['dataset_structure']['unified']['processed_datasets']} processed datasets")
    print(f"🎯 Total: {summary['total_audio_files']} audio files organized")

if __name__ == "__main__":
    main()
