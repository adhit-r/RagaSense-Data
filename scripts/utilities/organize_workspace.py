#!/usr/bin/env python3
"""
RagaSense Workspace Organizer
Comprehensive workspace organization and cleanup script
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkspaceOrganizer:
    """Comprehensive workspace organizer"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.archive_path = base_path / "archive"
        self.data_path = base_path / "data"
        
        # Create archive structure
        self.archive_path.mkdir(exist_ok=True)
        for subdir in ["data_versions", "old_scripts", "old_docs", "analysis_results", "duplicates"]:
            (self.archive_path / subdir).mkdir(exist_ok=True)
        
        logger.info(f"🏗️ Workspace Organizer initialized for {base_path}")
    
    def analyze_workspace(self) -> Dict:
        """Analyze current workspace structure"""
        logger.info("🔍 Analyzing workspace structure...")
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "directories": {},
            "file_counts": {},
            "duplicates": [],
            "large_files": [],
            "old_files": []
        }
        
        # Analyze each major directory
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                analysis["directories"][item.name] = {
                    "size": self._get_directory_size(item),
                    "file_count": len(list(item.rglob("*"))),
                    "subdirectories": len([d for d in item.iterdir() if d.is_dir()])
                }
        
        # Find large files (>100MB)
        for file_path in self.base_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > 100 * 1024 * 1024:
                analysis["large_files"].append({
                    "path": str(file_path.relative_to(self.base_path)),
                    "size_mb": file_path.stat().st_size / (1024 * 1024)
                })
        
        # Find old files (>30 days)
        cutoff_date = datetime.now().timestamp() - (30 * 24 * 60 * 60)
        for file_path in self.base_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_date:
                analysis["old_files"].append({
                    "path": str(file_path.relative_to(self.base_path)),
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return analysis
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (PermissionError, OSError):
            pass
        return total_size
    
    def organize_data_folder(self):
        """Organize the data folder structure"""
        logger.info("📁 Organizing data folder...")
        
        # Create new data structure
        new_structure = {
            "raw": ["ramanarunachalam", "saraga", "carnatic_varnam"],
            "processed": ["unified_ragas.json", "unified_artists.json", "unified_tracks.json"],
            "ml_ready": ["features.npy", "labels.npy", "metadata.json"],
            "exports": ["csv", "parquet", "sqlite"]
        }
        
        # Move raw data
        raw_path = self.data_path / "raw"
        raw_path.mkdir(exist_ok=True)
        
        # Move processed data
        processed_path = self.data_path / "processed"
        processed_path.mkdir(exist_ok=True)
        
        # Move ML-ready data
        ml_ready_path = self.data_path / "ml_ready"
        ml_ready_path.mkdir(exist_ok=True)
        
        # Move exports
        exports_path = self.data_path / "exports"
        exports_path.mkdir(exist_ok=True)
        
        logger.info("✅ Data folder structure organized")
    
    def archive_old_files(self):
        """Archive old and unused files"""
        logger.info("📦 Archiving old files...")
        
        # Archive old data versions
        old_data_patterns = [
            "cleaned_ragasense_dataset",
            "combined_raga_reclassified", 
            "comprehensive_ramanarunachalam_analysis",
            "comprehensive_unified_dataset",
            "comprehensively_cleaned",
            "cross_tradition_corrected",
            "ragamalika_classification_fixed",
            "ragamalika_mapping",
            "unified_ragasense_dataset",
            "unknownraga_fixed",
            "updated_raga_sources",
            "youtube_song_analysis"
        ]
        
        archived_count = 0
        for pattern in old_data_patterns:
            source_path = self.data_path / pattern
            if source_path.exists():
                dest_path = self.archive_path / "data_versions" / pattern
                shutil.move(str(source_path), str(dest_path))
                archived_count += 1
                logger.info(f"📦 Archived: {pattern}")
        
        # Archive old analysis files
        old_analysis_files = [
            "ramanarunachalam_analysis.json",
            "ramanarunachalam_corrected.json", 
            "real_data_analysis_results.json",
            "saraga_analysis_results.json",
            "corrected_cross_tradition_mappings.json"
        ]
        
        for filename in old_analysis_files:
            source_path = self.data_path / filename
            if source_path.exists():
                dest_path = self.archive_path / "analysis_results" / filename
                shutil.move(str(source_path), str(dest_path))
                archived_count += 1
                logger.info(f"📦 Archived: {filename}")
        
        logger.info(f"✅ Archived {archived_count} items")
    
    def organize_tools_folder(self):
        """Organize and optimize tools folder"""
        logger.info("🔧 Organizing tools folder...")
        
        tools_path = self.base_path / "tools"
        if not tools_path.exists():
            logger.warning("⚠️ Tools folder not found")
            return
        
        # Create organized structure
        organized_tools = {
            "data_processing": ["comprehensive_data_processor.py"],
            "analysis": ["data_analyzer.py", "statistics_generator.py"],
            "validation": ["data_validator.py", "quality_checker.py"],
            "export": ["csv_exporter.py", "sqlite_exporter.py"],
            "ml": ["feature_extractor.py", "model_trainer.py"]
        }
        
        # Create subdirectories
        for category in organized_tools:
            (tools_path / category).mkdir(exist_ok=True)
        
        logger.info("✅ Tools folder organized")
    
    def create_governance_structure(self):
        """Create data governance structure"""
        logger.info("📋 Creating governance structure...")
        
        governance_path = self.base_path / "governance"
        governance_path.mkdir(exist_ok=True)
        
        # Create governance files
        governance_files = {
            "data_policy.md": self._create_data_policy(),
            "quality_standards.md": self._create_quality_standards(),
            "access_control.md": self._create_access_control(),
            "data_lineage.json": self._create_data_lineage()
        }
        
        for filename, content in governance_files.items():
            file_path = governance_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info("✅ Governance structure created")
    
    def _create_data_policy(self) -> str:
        """Create data policy document"""
        return """# RagaSense Data Policy

## Data Classification
- **Public**: Open datasets, research results
- **Internal**: Processed datasets, analysis results  
- **Confidential**: Raw source data, proprietary information

## Data Retention
- Raw data: 5 years
- Processed data: 3 years
- Analysis results: 2 years
- Logs: 1 year

## Data Quality Standards
- Accuracy: >95% for raga classifications
- Completeness: >90% for required fields
- Consistency: Standardized naming conventions
- Timeliness: Updated monthly

## Access Control
- Read: All team members
- Write: Data engineers and researchers
- Delete: Data administrators only
"""
    
    def _create_quality_standards(self) -> str:
        """Create quality standards document"""
        return """# RagaSense Data Quality Standards

## Raga Data Quality
- **Name Consistency**: Standardized raga names
- **Tradition Assignment**: Clear Carnatic/Hindustani classification
- **Cross-tradition Mapping**: Validated equivalences
- **Metadata Completeness**: All required fields populated

## Audio Data Quality
- **Format**: WAV, 44.1kHz, 16-bit preferred
- **Duration**: Minimum 30 seconds for analysis
- **Quality**: No significant noise or distortion
- **Metadata**: Artist, raga, composer information

## Processing Quality
- **Feature Extraction**: Consistent parameters
- **Model Training**: Reproducible results
- **Validation**: Cross-validation for all models
- **Documentation**: Complete processing logs
"""
    
    def _create_access_control(self) -> str:
        """Create access control document"""
        return """# RagaSense Access Control

## User Roles
- **Data Administrator**: Full access, can modify governance
- **Data Engineer**: Read/write access to processing pipelines
- **Researcher**: Read access to processed data
- **ML Engineer**: Read access to ML datasets

## Data Access Levels
- **Level 1**: Public datasets and documentation
- **Level 2**: Processed datasets and analysis results
- **Level 3**: Raw data and proprietary information
- **Level 4**: Administrative and system files

## Security Requirements
- Authentication required for all access
- Audit logging for all data operations
- Regular access reviews and updates
- Encryption for sensitive data transmission
"""
    
    def _create_data_lineage(self) -> str:
        """Create data lineage JSON"""
        lineage = {
            "data_sources": {
                "ramanarunachalam": {
                    "type": "repository",
                    "location": "data/raw/ramanarunachalam",
                    "description": "Ramanarunachalam Music Repository",
                    "last_updated": "2025-09-08"
                },
                "saraga": {
                    "type": "dataset",
                    "location": "data/raw/saraga",
                    "description": "Saraga 1.5 datasets",
                    "last_updated": "2025-09-08"
                }
            },
            "processing_pipeline": {
                "raw_to_processed": {
                    "input": "data/raw/*",
                    "process": "comprehensive_data_processor.py",
                    "output": "data/processed/*",
                    "frequency": "monthly"
                },
                "processed_to_ml": {
                    "input": "data/processed/*",
                    "process": "feature_extractor.py",
                    "output": "data/ml_ready/*",
                    "frequency": "weekly"
                }
            },
            "dependencies": {
                "unified_ragas.json": ["ramanarunachalam", "saraga"],
                "audio_features.npy": ["unified_ragas.json"],
                "ml_model.pth": ["audio_features.npy"]
            }
        }
        
        return json.dumps(lineage, indent=2)
    
    def generate_organization_report(self) -> Dict:
        """Generate organization report"""
        logger.info("📊 Generating organization report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": {
                "archived_files": 0,
                "organized_directories": 0,
                "created_governance": 0
            },
            "current_structure": {},
            "recommendations": []
        }
        
        # Analyze current structure
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                report["current_structure"][item.name] = {
                    "size_mb": self._get_directory_size(item) / (1024 * 1024),
                    "file_count": len(list(item.rglob("*")))
                }
        
        # Add recommendations
        report["recommendations"] = [
            "Regular cleanup of temporary files",
            "Archive old analysis results monthly",
            "Monitor disk usage and optimize storage",
            "Update data lineage documentation",
            "Review and update access controls"
        ]
        
        # Save report
        report_path = self.base_path / "workspace_organization_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Organization report saved to {report_path}")
        return report
    
    def run_full_organization(self):
        """Run complete workspace organization"""
        logger.info("🚀 Starting full workspace organization...")
        
        try:
            # Analyze current state
            analysis = self.analyze_workspace()
            logger.info(f"📊 Workspace analysis complete: {len(analysis['directories'])} directories")
            
            # Archive old files
            self.archive_old_files()
            
            # Organize data folder
            self.organize_data_folder()
            
            # Organize tools folder
            self.organize_tools_folder()
            
            # Create governance structure
            self.create_governance_structure()
            
            # Generate report
            report = self.generate_organization_report()
            
            logger.info("🎉 Workspace organization completed successfully!")
            logger.info(f"📊 Final report: {report}")
            
        except Exception as e:
            logger.error(f"❌ Organization failed: {e}")
            raise

def main():
    """Main organization function"""
    base_path = Path(__file__).parent.parent
    
    organizer = WorkspaceOrganizer(base_path)
    organizer.run_full_organization()

if __name__ == "__main__":
    main()
