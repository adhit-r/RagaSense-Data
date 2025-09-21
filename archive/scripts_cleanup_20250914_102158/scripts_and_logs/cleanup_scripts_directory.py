#!/usr/bin/env python3
"""
Clean Scripts Directory - Archive Unused Scripts
================================================

This script identifies and archives unused scripts while keeping only
the essential ones that are currently being used.

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
import json
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_scripts_directory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScriptsDirectoryCleaner:
    """Clean and organize the scripts directory"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.scripts_path = self.base_path / "scripts"
        self.archive_path = self.base_path / "archive"
        self.archive_path.mkdir(exist_ok=True)
        
        # Create timestamped archive subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamped_archive = self.archive_path / f"scripts_cleanup_{timestamp}"
        self.timestamped_archive.mkdir(exist_ok=True)
        
        # Scripts that are currently being used (keep these)
        self.essential_scripts = {
            # Current active scripts
            'scripts/dataset_creation/process_extracted_saraga_data.py',
            'scripts/dataset_creation/separate_combination_ragas_and_count_unique.py',
            'scripts/dataset_creation/organize_ramanarunachalam_and_create_audio.py',
            'scripts/dataset_creation/fix_duplicate_raga_names.py',
            
            # Core utilities that might be needed
            'scripts/utilities/data_migrate_json_to_postgresql_v1.0_stable.py',
            'scripts/utilities/setup_postgresql_migration_v1.0_stable.py',
        }
        
        # Scripts to definitely archive (obsolete/duplicate)
        self.obsolete_scripts = {
            # Duplicate/old versions
            'scripts/data_processing/extract_audio_features_now.py',
            'scripts/data_processing/process_saraga_datasets_now.py',
            'scripts/data_processing/scale_audio_processing_fixed.py',
            'scripts/data_processing/scale_audio_processing_optimized.py',
            'scripts/data_processing/scale_up_audio_processing.py',
            'scripts/data_processing/scale_up_audio_processing_fixed.py',
            'scripts/dataset_creation/create_enhanced_ml_dataset.py',
            'scripts/dataset_creation/create_enhanced_ml_dataset_fixed.py',
            'scripts/dataset_creation/create_final_enhanced_dataset.py',
            'scripts/dataset_creation/create_final_comprehensive_dataset.py',
            'scripts/dataset_creation/reorganize_unified_dataset.py',
            'scripts/dataset_creation/perfect_dataset.py',
            
            # Old analysis scripts
            'scripts/analysis/validate_data_quality.py',
            'scripts/analysis/validate_data_quality_simple.py',
            'scripts/analysis/investigate_nat_raga.py',
            'scripts/analysis/validate_dataset_quality.py',
            
            # Old ML training scripts
            'scripts/ml_training/train_raga_classifier.py',
            
            # Old utility scripts
            'scripts/utilities/fix_nat_mapping_proper.py',
            'scripts/utilities/fix_nat_raga_mapping.py',
            'scripts/utilities/fix_tradition_classification.py',
            'scripts/utilities/fix_tradition_classification_advanced.py',
            'scripts/utilities/update_postgresql_tradition_classification.py',
            
            # Old integration scripts
            'scripts/integration/create_unified_dataset.py',
            'scripts/integration/integrate_all_datasets.py',
            'scripts/integration/unified_dataset_integration.py',
            
            # Old data processing scripts
            'scripts/data_processing/comprehensive_data_processor.py',
            'scripts/data_processing/comprehensive_ramanarunachalam_processor.py',
            'scripts/data_processing/process_saraga_datasets.py',
            'scripts/data_processing/process_saraga_efficient.py',
            'scripts/data_processing/simple_saraga_processor.py',
            'scripts/data_processing/extract_saraga_features_proper.py',
            'scripts/data_processing/ramanarunachalam_proper_decoder.py',
            'scripts/data_processing/decode_ramanarunachalam_data.py',
            'scripts/data_processing/clean_and_deduplicate_ragas.py',
            'scripts/data_processing/comprehensive_data_cleaner.py',
            'scripts/data_processing/correct_cross_tradition_mappings.py',
            'scripts/data_processing/fix_cross_tradition_mappings_accurate.py',
            'scripts/data_processing/fix_ragamalika_classification.py',
            'scripts/data_processing/fix_unknownraga_issue.py',
            'scripts/data_processing/reclassify_combined_ragas.py',
            'scripts/data_processing/update_kalyani_data.py',
            'scripts/data_processing/update_raga_sources.py',
            'scripts/data_processing/youtube_song_analyzer.py',
            
            # Old dataset creation scripts
            'scripts/dataset_creation/create_comprehensive_dataset.py',
            'scripts/dataset_creation/create_corrected_raga_dataset.py',
            'scripts/dataset_creation/create_ml_ready_dataset.py',
            'scripts/dataset_creation/create_proper_raga_detection_system.py',
            'scripts/dataset_creation/create_unified_raga_dataset.py',
            'scripts/dataset_creation/extract_saraga_metadata_proper.py',
            'scripts/dataset_creation/fix_duration_parsing_and_complete_dataset.py',
            'scripts/dataset_creation/reorganize_dataset_structure.py',
            
            # Old analysis scripts
            'scripts/analysis/analyze_real_data.py',
            'scripts/analysis/analyze_saraga_datasets.py',
            'scripts/analysis/comprehensive_data_analysis.py',
            
            # Old exploration scripts
            'scripts/exploration/explore_ragasense_data.py',
            'scripts/exploration/web_explorer.py',
            
            # Old cleanup scripts
            'scripts/cleanup/cleanup_and_organize_workspace.py',
            'scripts/utilities/organize_workspace.py',
            
            # Old utility scripts
            'scripts/utilities/vector_database_schema.py',
            'scripts/create_ragamalika_mapping.py',
            
            # Deprecated scripts
            'scripts/deprecated/achieve_target_tradition_breakdown.py',
            'scripts/deprecated/cleanup_and_organize_repo.py',
            'scripts/deprecated/rename_files_systematically.py',
        }
        
        self.scripts_to_archive = []
        self.scripts_to_keep = []
        
    def analyze_scripts(self):
        """Analyze all scripts and categorize them"""
        logger.info("🔍 Analyzing scripts directory...")
        
        all_scripts = list(self.scripts_path.rglob("*.py"))
        logger.info(f"Found {len(all_scripts)} Python scripts")
        
        for script_path in all_scripts:
            # Convert to relative path for comparison
            rel_path = str(script_path.relative_to(self.base_path))
            
            if rel_path in self.essential_scripts:
                self.scripts_to_keep.append(script_path)
                logger.info(f"  ✅ Keep: {rel_path}")
            elif rel_path in self.obsolete_scripts:
                self.scripts_to_archive.append(script_path)
                logger.info(f"  📦 Archive: {rel_path}")
            else:
                # Check if it's a recent/important script
                if self.is_recent_or_important(script_path):
                    self.scripts_to_keep.append(script_path)
                    logger.info(f"  ✅ Keep (recent/important): {rel_path}")
                else:
                    self.scripts_to_archive.append(script_path)
                    logger.info(f"  📦 Archive (unused): {rel_path}")
        
        logger.info(f"Scripts to keep: {len(self.scripts_to_keep)}")
        logger.info(f"Scripts to archive: {len(self.scripts_to_archive)}")
        
        return {
            'total_scripts': len(all_scripts),
            'keep_count': len(self.scripts_to_keep),
            'archive_count': len(self.scripts_to_archive)
        }
    
    def is_recent_or_important(self, script_path: Path) -> bool:
        """Check if a script is recent or important"""
        # Check modification time (keep scripts modified in last 7 days)
        import time
        mod_time = script_path.stat().st_mtime
        days_old = (time.time() - mod_time) / (24 * 3600)
        
        if days_old < 7:
            return True
        
        # Check for important keywords in filename
        important_keywords = [
            'extract', 'process', 'create', 'organize', 'fix', 'separate',
            'migrate', 'setup', 'postgresql'
        ]
        
        script_name = script_path.name.lower()
        for keyword in important_keywords:
            if keyword in script_name:
                return True
        
        return False
    
    def archive_scripts(self):
        """Archive obsolete scripts"""
        logger.info("📦 Archiving obsolete scripts...")
        
        archived_scripts = []
        
        for script_path in self.scripts_to_archive:
            try:
                # Create subdirectory in archive based on original structure
                rel_path = script_path.relative_to(self.scripts_path)
                archive_subdir = self.timestamped_archive / rel_path.parent
                archive_subdir.mkdir(parents=True, exist_ok=True)
                
                # Move script to archive
                destination = archive_subdir / script_path.name
                shutil.move(str(script_path), str(destination))
                archived_scripts.append({
                    'original': str(script_path),
                    'archived': str(destination),
                    'category': rel_path.parts[0] if len(rel_path.parts) > 1 else 'root'
                })
                logger.info(f"  ✅ Archived: {rel_path}")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to archive {script_path.name}: {e}")
        
        return archived_scripts
    
    def create_clean_structure(self):
        """Create a clean scripts structure"""
        logger.info("🏗️ Creating clean scripts structure...")
        
        # Create a summary of the clean structure
        structure_summary = {
            'cleanup_date': datetime.now().isoformat(),
            'scripts_kept': [str(s.relative_to(self.base_path)) for s in self.scripts_to_keep],
            'scripts_archived_count': len(self.scripts_to_archive),
            'total_scripts_before': len(self.scripts_to_keep) + len(self.scripts_to_archive),
            'total_scripts_after': len(self.scripts_to_keep)
        }
        
        # Save structure summary
        with open(self.base_path / "SCRIPTS_CLEANUP_SUMMARY.json", 'w') as f:
            json.dump(structure_summary, f, indent=2)
        
        logger.info("✅ Clean scripts structure summary saved")
        return structure_summary
    
    def create_archive_readme(self, archived_scripts):
        """Create a README for the archived scripts"""
        logger.info("📝 Creating archive README...")
        
        readme_content = f"""# Scripts Directory Cleanup Archive

**Cleanup Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This archive contains scripts that were moved from the scripts directory during cleanup to maintain a clean, organized workspace.

## Archived Scripts ({len(archived_scripts)} total)

### By Category
"""
        
        # Group scripts by category
        categories = {}
        for script_info in archived_scripts:
            category = script_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(script_info)
        
        for category, scripts in categories.items():
            readme_content += f"\n#### {category.title()} ({len(scripts)} scripts)\n"
            for script_info in scripts:
                script_name = Path(script_info['original']).name
                readme_content += f"- `{script_name}`\n"
        
        readme_content += f"""
## Current Scripts Structure

### Kept Scripts ({len(self.scripts_to_keep)} total)
{chr(10).join(f"- {s.relative_to(self.base_path)}" for s in self.scripts_to_keep)}

## Notes
- All scripts have been safely archived without deletion
- Original functionality is preserved
- Clean scripts structure maintained
- Easy to restore scripts if needed
- Focus on essential, currently-used scripts
"""
        
        # Save README
        readme_path = self.timestamped_archive / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"✅ Archive README created: {readme_path}")
    
    def verify_clean_state(self):
        """Verify the scripts directory is clean"""
        logger.info("🔍 Verifying clean scripts state...")
        
        remaining_scripts = list(self.scripts_path.rglob("*.py"))
        logger.info(f"📊 Scripts directory status:")
        logger.info(f"  Scripts remaining: {len(remaining_scripts)}")
        
        # List remaining scripts
        for script in remaining_scripts:
            rel_path = script.relative_to(self.base_path)
            logger.info(f"  ✅ {rel_path}")
        
        return {
            'remaining_scripts': len(remaining_scripts),
            'scripts_list': [str(s.relative_to(self.base_path)) for s in remaining_scripts]
        }
    
    def run(self):
        """Run the complete cleanup process"""
        logger.info("🧹 Starting Scripts Directory Cleanup")
        logger.info("=" * 50)
        logger.info("This will clean the scripts directory and archive unused scripts")
        logger.info("=" * 50)
        
        try:
            # Step 1: Analyze scripts
            analysis = self.analyze_scripts()
            
            if not self.scripts_to_archive:
                logger.info("✅ Scripts directory is already clean!")
                return
            
            # Step 2: Archive scripts
            archived_scripts = self.archive_scripts()
            
            # Step 3: Create clean structure
            structure_summary = self.create_clean_structure()
            
            # Step 4: Create archive README
            self.create_archive_readme(archived_scripts)
            
            # Step 5: Verify clean state
            verification = self.verify_clean_state()
            
            logger.info("✅ Scripts directory cleanup completed successfully!")
            logger.info(f"📦 Archived {len(archived_scripts)} scripts")
            logger.info(f"📁 Archive location: {self.timestamped_archive}")
            logger.info(f"📊 Scripts reduced from {analysis['total_scripts']} to {verification['remaining_scripts']}")
            
            return {
                'archived_scripts': archived_scripts,
                'structure_summary': structure_summary,
                'verification': verification
            }
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            raise

def main():
    """Main function"""
    cleaner = ScriptsDirectoryCleaner()
    cleaner.run()

if __name__ == "__main__":
    main()
