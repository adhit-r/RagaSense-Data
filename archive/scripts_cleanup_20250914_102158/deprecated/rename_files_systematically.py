#!/usr/bin/env python3
"""
Systematic File Renaming Script
==============================

This script renames all files according to the new naming convention:
[category]_[purpose]_[version]_[status].[extension]

Categories: ml, data, api, util, test, doc, config, script
Purposes: train, predict, extract, process, validate, analyze, integrate, classify
Versions: v1.0, v1.1, v1.1.1, v1.1.1-alpha, v1.1.1-beta, v1.1.1-rc1
Status: stable, dev, experimental, deprecated, archived
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

class FileRenamer:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.rename_log = []
        self.backup_dir = self.base_path / "backup_before_rename"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Define renaming rules
        self.rename_rules = {
            # ML Models
            "raga_classifier_api.py": "api_raga_classifier_v1.0_stable.py",
            "raga_classifier_api_working.py": "api_raga_classifier_v1.1_stable.py",
            "train_simple_raga_classifier.py": "ml_train_classifier_v1.0_stable.py",
            "extract_saraga_audio_features.py": "data_extract_features_v1.0_stable.py",
            "test_model_direct.py": "test_model_direct_v1.0_dev.py",
            "test_model_simple.py": "test_model_simple_v1.0_dev.py",
            "test_model_neural_network.py": "test_model_neural_v1.0_dev.py",
            "real_raga_classifier.py": "ml_train_real_classifier_v1.0_dev.py",
            
            # Processing Scripts
            "phase1_enhancement.py": "script_phase1_enhancement_v1.0_archived.py",
            "phase2_simplified_ml.py": "script_phase2_ml_v1.0_archived.py",
            "phase3_composer_relationship_fix.py": "data_fix_composer_relations_v1.0_stable.py",
            "phase3_youtube_validation.py": "data_validate_youtube_v1.0_stable.py",
            "phase4_youtube_dataset_creation.py": "data_create_youtube_dataset_v1.0_dev.py",
            
            # Analysis Scripts
            "yue_model_analysis_simplified.py": "ml_analyze_yue_model_v1.0_stable.py",
            "yue_model_comprehensive_analysis.py": "ml_analyze_yue_comprehensive_v1.0_stable.py",
            "yue_rhythmic_analysis.py": "ml_analyze_yue_rhythmic_v1.0_stable.py",
            
            # Documentation
            "CORRECTED_RAGA_STATISTICS.md": "doc_raga_statistics_corrected_v1.0_stable.md",
            "REAL_DATA_ANALYSIS_RESULTS.md": "doc_real_data_analysis_v1.0_stable.md",
            "YUE_RHYTHMIC_ANALYSIS_SUMMARY.md": "doc_yue_rhythmic_analysis_v1.0_stable.md",
            "ML_TRAINING_SUCCESS_SUMMARY.md": "doc_ml_training_success_v1.0_stable.md",
            "REAL_DATA_ANALYSIS_AND_FIXES.md": "doc_real_data_analysis_fixes_v1.0_stable.md",
            "COMPREHENSIVE_ORGANIZATION_SUMMARY.md": "doc_organization_summary_v1.0_stable.md",
            "FILE_NAMING_AND_VERSIONING_STANDARDS.md": "doc_naming_versioning_standards_v1.0_stable.md",
            
            # Configuration
            "opensearch_dashboards.yml": "config_opensearch_dashboards_v1.0_stable.yml",
            
            # HTML Files
            "test_raga_classifier.html": "test_raga_classifier_v1.0_dev.html",
            
            # Log Files (archive with timestamp)
            "deduplicate_ragas.log": "log_deduplicate_ragas_v1.0_archived.log",
            "expand_raga_dataset.log": "log_expand_dataset_v1.0_archived.log",
            "opensearch_vector_integration.log": "log_opensearch_vector_v1.0_archived.log",
            "phase1_enhancement.log": "log_phase1_enhancement_v1.0_archived.log",
            "phase2_simplified_ml.log": "log_phase2_ml_v1.0_archived.log",
            "phase3_composer_fix.log": "log_phase3_composer_v1.0_archived.log",
            "phase3_youtube_validation.log": "log_phase3_youtube_v1.0_archived.log",
            "phase4_youtube_dataset.log": "log_phase4_youtube_v1.0_archived.log",
            "unify_data_sources.log": "log_unify_data_sources_v1.0_archived.log",
        }
    
    def backup_file(self, file_path: Path) -> Path:
        """Create backup of file before renaming."""
        backup_path = self.backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def find_files_to_rename(self) -> list:
        """Find all files that need renaming."""
        files_to_rename = []
        
        for root, dirs, files in os.walk(self.base_path):
            # Skip certain directories
            if any(skip_dir in root for skip_dir in ['.git', 'venv', 'node_modules', '__pycache__', '.vercel']):
                continue
                
            for file in files:
                if file in self.rename_rules:
                    file_path = Path(root) / file
                    files_to_rename.append(file_path)
        
        return files_to_rename
    
    def rename_file(self, old_path: Path, new_name: str) -> bool:
        """Rename a single file."""
        try:
            # Create backup
            backup_path = self.backup_file(old_path)
            
            # Create new path
            new_path = old_path.parent / new_name
            
            # Rename file
            old_path.rename(new_path)
            
            # Log the change
            self.rename_log.append({
                'old_path': str(old_path),
                'new_path': str(new_path),
                'backup_path': str(backup_path),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            })
            
            print(f"✅ Renamed: {old_path.name} → {new_name}")
            return True
            
        except Exception as e:
            self.rename_log.append({
                'old_path': str(old_path),
                'new_name': new_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            })
            print(f"❌ Failed to rename {old_path.name}: {e}")
            return False
    
    def rename_all_files(self, dry_run: bool = True) -> dict:
        """Rename all files according to the rules."""
        files_to_rename = self.find_files_to_rename()
        
        print(f"🔍 Found {len(files_to_rename)} files to rename")
        
        if dry_run:
            print("\n📋 DRY RUN - Files that would be renamed:")
            for file_path in files_to_rename:
                new_name = self.rename_rules[file_path.name]
                print(f"  {file_path} → {new_name}")
            return {'dry_run': True, 'files_found': len(files_to_rename)}
        
        # Actually rename files
        success_count = 0
        failed_count = 0
        
        for file_path in files_to_rename:
            new_name = self.rename_rules[file_path.name]
            if self.rename_file(file_path, new_name):
                success_count += 1
            else:
                failed_count += 1
        
        # Save rename log
        log_file = self.base_path / "rename_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.rename_log, f, indent=2)
        
        return {
            'success_count': success_count,
            'failed_count': failed_count,
            'total_files': len(files_to_rename),
            'log_file': str(log_file)
        }
    
    def create_version_directories(self):
        """Create versioned directory structure."""
        version_dirs = [
            "ml_models/core_models/v1.0_stable",
            "ml_models/core_models/v1.1_dev",
            "ml_models/api_models/v1.0_stable",
            "ml_models/api_models/v1.1_stable",
            "ml_models/training_scripts/v1.0_stable",
            "ml_models/training_scripts/v1.1_dev",
            "data/processing/v1.0_stable",
            "data/processing/v1.1_dev",
            "archive/scripts/v1.0_archived",
            "archive/analysis/v1.0_archived",
            "docs/overview/v1.0_stable",
            "docs/processing/v1.0_stable",
            "docs/analysis/v1.0_stable"
        ]
        
        for dir_path in version_dirs:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")

def main():
    """Main function."""
    print("🚀 Starting Systematic File Renaming")
    print("=" * 50)
    
    renamer = FileRenamer()
    
    # First, do a dry run
    print("\n1️⃣ DRY RUN - Checking files to rename...")
    dry_run_result = renamer.rename_all_files(dry_run=True)
    
    if dry_run_result['files_found'] == 0:
        print("✅ No files found that need renaming!")
        return
    
    # Ask for confirmation
    print(f"\n📊 Found {dry_run_result['files_found']} files to rename")
    response = input("\n❓ Proceed with renaming? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Renaming cancelled by user")
        return
    
    # Create version directories
    print("\n2️⃣ Creating version directories...")
    renamer.create_version_directories()
    
    # Actually rename files
    print("\n3️⃣ Renaming files...")
    result = renamer.rename_all_files(dry_run=False)
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 RENAMING RESULTS:")
    print(f"✅ Successfully renamed: {result['success_count']} files")
    print(f"❌ Failed to rename: {result['failed_count']} files")
    print(f"📁 Total files processed: {result['total_files']}")
    print(f"📝 Log saved to: {result['log_file']}")
    
    if result['success_count'] > 0:
        print("\n🎉 File renaming completed successfully!")
        print("📋 Check the rename log for details")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()

