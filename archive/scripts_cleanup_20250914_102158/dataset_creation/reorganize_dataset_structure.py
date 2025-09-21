#!/usr/bin/env python3
"""
Reorganize Dataset Structure with Proper Nomenclature
===================================================

This script reorganizes the confusing dataset structure into a clean, logical hierarchy:
- Clear naming conventions
- Proper versioning
- Logical organization by data type and processing stage
- Consistent nomenclature throughout

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
        logging.FileHandler('reorganize_dataset_structure.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetStructureReorganizer:
    """Reorganize dataset structure with proper nomenclature"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.new_structure = {
            # Source data (original, unprocessed)
            '01_source': {
                'description': 'Original source data from repositories',
                'subdirs': ['ramanarunachalam', 'youtube', 'saraga']
            },
            # Raw data (extracted but unprocessed)
            '02_raw': {
                'description': 'Extracted raw data by tradition',
                'subdirs': ['carnatic', 'hindustani', 'cross_tradition']
            },
            # Processed data (cleaned and standardized)
            '03_processed': {
                'description': 'Cleaned and standardized data',
                'subdirs': ['metadata', 'annotations', 'audio_features']
            },
            # ML datasets (ready for machine learning)
            '04_ml_datasets': {
                'description': 'Machine learning ready datasets',
                'subdirs': ['training', 'validation', 'test', 'features']
            },
            # Research datasets (for analysis and research)
            '05_research': {
                'description': 'Research and analysis datasets',
                'subdirs': ['analysis', 'statistics', 'reports']
            },
            # Archive (old versions and backups)
            '99_archive': {
                'description': 'Archived versions and backups',
                'subdirs': ['old_versions', 'backups', 'deprecated']
            }
        }
        
        # Create new structure
        self._create_new_structure()
        
    def _create_new_structure(self):
        """Create the new organized structure"""
        logger.info("Creating new dataset structure...")
        
        for stage, config in self.new_structure.items():
            stage_path = self.base_path / stage
            stage_path.mkdir(exist_ok=True)
            
            # Create README for each stage
            readme_content = f"""# {stage.upper()} - {config['description']}

## Purpose
{config['description']}

## Subdirectories
"""
            for subdir in config['subdirs']:
                readme_content += f"- `{subdir}/` - {self._get_subdir_description(stage, subdir)}\n"
            
            readme_content += f"""
## Usage
This directory contains {config['description'].lower()}.

## Last Updated
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            readme_file = stage_path / "README.md"
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            
            # Create subdirectories
            for subdir in config['subdirs']:
                subdir_path = stage_path / subdir
                subdir_path.mkdir(exist_ok=True)
            
            logger.info(f"Created {stage} structure")
    
    def _get_subdir_description(self, stage: str, subdir: str) -> str:
        """Get description for subdirectory"""
        descriptions = {
            '01_source': {
                'ramanarunachalam': 'Ramanarunachalam Music Repository source data',
                'youtube': 'YouTube dataset source files',
                'saraga': 'Saraga dataset source files'
            },
            '02_raw': {
                'carnatic': 'Raw Carnatic music data',
                'hindustani': 'Raw Hindustani music data',
                'cross_tradition': 'Cross-tradition mappings and relationships'
            },
            '03_processed': {
                'metadata': 'Processed metadata (ragas, artists, composers)',
                'annotations': 'Processed raga annotations and definitions',
                'audio_features': 'Extracted audio features from audio files'
            },
            '04_ml_datasets': {
                'training': 'Training datasets for ML models',
                'validation': 'Validation datasets for ML models',
                'test': 'Test datasets for ML models',
                'features': 'Feature matrices and vectors'
            },
            '05_research': {
                'analysis': 'Research analysis results',
                'statistics': 'Dataset statistics and summaries',
                'reports': 'Research reports and findings'
            },
            '99_archive': {
                'old_versions': 'Previous versions of datasets',
                'backups': 'Backup copies of important data',
                'deprecated': 'Deprecated and unused files'
            }
        }
        
        return descriptions.get(stage, {}).get(subdir, f"{subdir} data")
    
    def migrate_source_data(self):
        """Migrate source data to new structure"""
        logger.info("Migrating source data...")
        
        # Migrate Ramanarunachalam repository
        old_ramanarunachalam = self.base_path / "organized_raw" / "Ramanarunachalam_Music_Repository"
        new_ramanarunachalam = self.base_path / "01_source" / "ramanarunachalam"
        
        if old_ramanarunachalam.exists():
            if new_ramanarunachalam.exists():
                shutil.rmtree(new_ramanarunachalam)
            shutil.copytree(old_ramanarunachalam, new_ramanarunachalam)
            logger.info(f"Migrated Ramanarunachalam repository to {new_ramanarunachalam}")
        
        # Migrate YouTube dataset
        old_youtube = self.base_path / "youtube_dataset"
        new_youtube = self.base_path / "01_source" / "youtube"
        
        if old_youtube.exists():
            if new_youtube.exists():
                shutil.rmtree(new_youtube)
            shutil.copytree(old_youtube, new_youtube)
            logger.info(f"Migrated YouTube dataset to {new_youtube}")
    
    def migrate_raw_data(self):
        """Migrate raw data to new structure"""
        logger.info("Migrating raw data...")
        
        # Migrate Carnatic data
        old_carnatic = self.base_path / "carnatic"
        new_carnatic = self.base_path / "02_raw" / "carnatic"
        
        if old_carnatic.exists():
            if new_carnatic.exists():
                shutil.rmtree(new_carnatic)
            shutil.copytree(old_carnatic, new_carnatic)
            logger.info(f"Migrated Carnatic data to {new_carnatic}")
        
        # Migrate Hindustani data
        old_hindustani = self.base_path / "hindustani"
        new_hindustani = self.base_path / "02_raw" / "hindustani"
        
        if old_hindustani.exists():
            if new_hindustani.exists():
                shutil.rmtree(new_hindustani)
            shutil.copytree(old_hindustani, new_hindustani)
            logger.info(f"Migrated Hindustani data to {new_hindustani}")
        
        # Migrate cross-tradition mappings
        old_cross_tradition = self.base_path / "unified" / "cross_tradition_mappings"
        new_cross_tradition = self.base_path / "02_raw" / "cross_tradition"
        
        if old_cross_tradition.exists():
            if new_cross_tradition.exists():
                shutil.rmtree(new_cross_tradition)
            shutil.copytree(old_cross_tradition, new_cross_tradition)
            logger.info(f"Migrated cross-tradition mappings to {new_cross_tradition}")
    
    def migrate_processed_data(self):
        """Migrate processed data to new structure"""
        logger.info("Migrating processed data...")
        
        # Migrate processed datasets
        old_processed = self.base_path / "organized_processed"
        new_processed = self.base_path / "03_processed"
        
        if old_processed.exists():
            # Copy metadata files
            metadata_files = list(old_processed.glob("*raga*.json"))
            for file in metadata_files:
                shutil.copy2(file, new_processed / "metadata" / file.name)
            
            # Copy annotation files (they're already in raw/carnatic and raw/hindustani)
            logger.info("Metadata files migrated to processed/metadata")
        
        # Migrate audio features
        old_audio_features = self.base_path / "unified" / "comprehensive_audio_features"
        new_audio_features = self.base_path / "03_processed" / "audio_features"
        
        if old_audio_features.exists():
            if new_audio_features.exists():
                shutil.rmtree(new_audio_features)
            shutil.copytree(old_audio_features, new_audio_features)
            logger.info(f"Migrated audio features to {new_audio_features}")
    
    def migrate_ml_datasets(self):
        """Migrate ML datasets to new structure"""
        logger.info("Migrating ML datasets...")
        
        # Migrate ML-ready datasets
        old_ml_ready = self.base_path / "ml_ready"
        new_ml_ready = self.base_path / "04_ml_datasets"
        
        if old_ml_ready.exists():
            # Copy training datasets (files only)
            training_files = [f for f in old_ml_ready.glob("*training*") if f.is_file()] + [f for f in old_ml_ready.glob("*train*") if f.is_file()]
            for file in training_files:
                shutil.copy2(file, new_ml_ready / "training" / file.name)
            
            # Copy validation datasets (files only)
            validation_files = [f for f in old_ml_ready.glob("*validation*") if f.is_file()] + [f for f in old_ml_ready.glob("*val*") if f.is_file()]
            for file in validation_files:
                shutil.copy2(file, new_ml_ready / "validation" / file.name)
            
            # Copy feature files (files only)
            feature_files = [f for f in old_ml_ready.glob("*feature*") if f.is_file()]
            for file in feature_files:
                shutil.copy2(file, new_ml_ready / "features" / file.name)
            
            # Copy remaining ML files (JSON files only)
            remaining_files = [f for f in old_ml_ready.glob("*.json") 
                             if f.is_file() and not any(pattern in f.name for pattern in ['training', 'train', 'validation', 'val', 'feature'])]
            for file in remaining_files:
                shutil.copy2(file, new_ml_ready / "training" / file.name)
            
            # Copy directories (like trained_models)
            for item in old_ml_ready.iterdir():
                if item.is_dir():
                    target_dir = new_ml_ready / "training" / item.name
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.copytree(item, target_dir)
            
            logger.info("ML datasets migrated to new structure")
    
    def migrate_research_data(self):
        """Migrate research data to new structure"""
        logger.info("Migrating research data...")
        
        # Migrate comprehensive dataset
        old_comprehensive = self.base_path / "unified" / "comprehensive_dataset"
        new_research = self.base_path / "05_research"
        
        if old_comprehensive.exists():
            # Copy analysis files
            analysis_files = list(old_comprehensive.glob("*analysis*.json"))
            for file in analysis_files:
                shutil.copy2(file, new_research / "analysis" / file.name)
            
            # Copy statistics files
            stats_files = list(old_comprehensive.glob("*statistics*.json")) + list(old_comprehensive.glob("*summary*.json"))
            for file in stats_files:
                shutil.copy2(file, new_research / "statistics" / file.name)
            
            # Copy reports
            report_files = list(old_comprehensive.glob("*.md"))
            for file in report_files:
                shutil.copy2(file, new_research / "reports" / file.name)
            
            logger.info("Research data migrated to new structure")
    
    def archive_old_structure(self):
        """Archive the old confusing structure"""
        logger.info("Archiving old structure...")
        
        old_dirs = [
            "organized_raw", "organized_processed", "organized_exports",
            "processed", "processing", "raw", "exports",
            "unified", "ml_ready"
        ]
        
        archive_path = self.base_path / "99_archive" / "old_versions"
        archive_path.mkdir(parents=True, exist_ok=True)
        
        for old_dir in old_dirs:
            old_path = self.base_path / old_dir
            if old_path.exists():
                new_path = archive_path / f"{old_dir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(str(old_path), str(new_path))
                logger.info(f"Archived {old_dir} to {new_path}")
    
    def create_dataset_manifest(self):
        """Create a comprehensive dataset manifest"""
        logger.info("Creating dataset manifest...")
        
        manifest = {
            'dataset_name': 'RagaSense Dataset',
            'version': '2.0.0',
            'reorganization_date': datetime.now().isoformat(),
            'structure': {},
            'statistics': {},
            'usage_guide': {
                '01_source': 'Original source data - do not modify',
                '02_raw': 'Extracted raw data - clean and standardize here',
                '03_processed': 'Cleaned data - ready for analysis',
                '04_ml_datasets': 'ML-ready datasets - use for training models',
                '05_research': 'Research outputs - analysis and reports',
                '99_archive': 'Old versions and backups - reference only'
            }
        }
        
        # Count files in each stage
        for stage in self.new_structure.keys():
            stage_path = self.base_path / stage
            if stage_path.exists():
                file_count = sum(1 for _ in stage_path.rglob('*') if _.is_file())
                dir_count = sum(1 for _ in stage_path.rglob('*') if _.is_dir())
                manifest['statistics'][stage] = {
                    'files': file_count,
                    'directories': dir_count
                }
        
        # Save manifest
        manifest_file = self.base_path / "DATASET_MANIFEST.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created dataset manifest: {manifest_file}")
        return manifest
    
    def run_reorganization(self):
        """Run the complete reorganization process"""
        logger.info("Starting dataset structure reorganization...")
        
        try:
            # Migrate data to new structure
            self.migrate_source_data()
            self.migrate_raw_data()
            self.migrate_processed_data()
            self.migrate_ml_datasets()
            self.migrate_research_data()
            
            # Archive old structure
            self.archive_old_structure()
            
            # Create manifest
            manifest = self.create_dataset_manifest()
            
            logger.info("Dataset structure reorganization completed successfully!")
            return manifest
            
        except Exception as e:
            logger.error(f"Error during reorganization: {e}")
            raise

def main():
    """Main function"""
    print("🗂️  RagaSense Dataset Structure Reorganization")
    print("=" * 50)
    
    reorganizer = DatasetStructureReorganizer()
    manifest = reorganizer.run_reorganization()
    
    print(f"\n✅ Dataset Structure Reorganized!")
    print(f"📁 New structure created with proper nomenclature")
    print(f"📁 Old structure archived in 99_archive/")
    print(f"📁 Dataset manifest: DATASET_MANIFEST.json")
    
    print(f"\n📊 New Structure Statistics:")
    for stage, stats in manifest['statistics'].items():
        print(f"  {stage}: {stats['files']} files, {stats['directories']} directories")

if __name__ == "__main__":
    main()
