#!/usr/bin/env python3
"""
Cleanup and Organize Workspace
==============================

This script cleans up and organizes the workspace by:
1. Archiving unused files from scripts/, data/, ml/, tests/ folders
2. Cleaning the root directory
3. Organizing files into proper archive structure
4. Maintaining a clean, organized workspace

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
import json
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_and_organize_workspace.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkspaceCleaner:
    """Clean up and organize the workspace"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.archive_path = self.base_path / "archive"
        self.archive_path.mkdir(exist_ok=True)
        
        # Create archive subdirectories
        self.archive_dirs = {
            'scripts': self.archive_path / "scripts",
            'data': self.archive_path / "data", 
            'ml': self.archive_path / "ml",
            'tests': self.archive_path / "tests",
            'logs': self.archive_path / "logs",
            'temp': self.archive_path / "temp",
            'old_files': self.archive_path / "old_files",
            'root_cleanup': self.archive_path / "root_cleanup"
        }
        
        for archive_dir in self.archive_dirs.values():
            archive_dir.mkdir(exist_ok=True)
        
        # Files to keep in root (essential files)
        self.keep_in_root = {
            'README.md', 'todo.md', 'requirements.txt', 'setup.py', 'pyproject.toml',
            'Dockerfile', 'docker-compose.yml', '.gitignore', '.env.example',
            'LICENSE', 'CHANGELOG.md', 'CONTRIBUTING.md', 'MANIFEST.in'
        }
        
        # Directories to keep in root (essential directories)
        self.keep_dirs_in_root = {
            'data', 'scripts', 'docs', 'config', 'venv', '.git', 'archive'
        }
        
        # File patterns to archive
        self.archive_patterns = {
            'logs': ['*.log', '*.out', '*.err'],
            'temp': ['*.tmp', '*.temp', '*.bak', '*.backup'],
            'old_files': ['*_old.*', '*_backup.*', '*_copy.*', '*.orig'],
            'python_cache': ['__pycache__', '*.pyc', '*.pyo'],
            'jupyter': ['.ipynb_checkpoints'],
            'ide': ['.vscode', '.idea', '*.swp', '*.swo'],
            'os': ['.DS_Store', 'Thumbs.db', 'desktop.ini']
        }
        
        self.cleanup_report = {
            'timestamp': datetime.now().isoformat(),
            'files_archived': 0,
            'directories_archived': 0,
            'root_cleaned': 0,
            'archive_size': 0,
            'details': defaultdict(list)
        }
    
    def identify_unused_files(self, directory: Path, category: str) -> List[Path]:
        """Identify unused files in a directory"""
        unused_files = []
        
        if not directory.exists():
            return unused_files
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                # Check if file is unused based on patterns
                if self.is_unused_file(file_path, category):
                    unused_files.append(file_path)
        
        return unused_files
    
    def is_unused_file(self, file_path: Path, category: str) -> bool:
        """Determine if a file is unused"""
        file_name = file_path.name.lower()
        
        # Archive patterns
        for pattern_category, patterns in self.archive_patterns.items():
            for pattern in patterns:
                if pattern.startswith('*'):
                    suffix = pattern[1:]
                    if file_name.endswith(suffix):
                        return True
                elif pattern.startswith('.'):
                    if file_name == pattern:
                        return True
                else:
                    if pattern in file_name:
                        return True
        
        # Category-specific unused file detection
        if category == 'scripts':
            # Archive old, backup, or test scripts
            if any(keyword in file_name for keyword in ['old', 'backup', 'test_', '_test', 'temp', 'tmp']):
                return True
        
        elif category == 'data':
            # Archive old data files, backups, or temporary files
            if any(keyword in file_name for keyword in ['old', 'backup', 'temp', 'tmp', '_copy', '_bak']):
                return True
        
        elif category == 'ml':
            # Archive old models, experiments, or temporary files
            if any(keyword in file_name for keyword in ['old', 'backup', 'temp', 'tmp', 'experiment_', 'test_']):
                return True
        
        elif category == 'tests':
            # Archive old test files or temporary test data
            if any(keyword in file_name for keyword in ['old', 'backup', 'temp', 'tmp', '_old', '_backup']):
                return True
        
        return False
    
    def archive_files(self, files: List[Path], archive_category: str) -> int:
        """Archive files to the specified category"""
        archived_count = 0
        
        for file_path in files:
            try:
                # Create relative path structure in archive
                relative_path = file_path.relative_to(self.base_path)
                archive_dest = self.archive_dirs[archive_category] / relative_path
                
                # Create parent directories
                archive_dest.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file to archive
                shutil.move(str(file_path), str(archive_dest))
                archived_count += 1
                
                self.cleanup_report['details'][archive_category].append(str(relative_path))
                logger.info(f"Archived: {relative_path} -> {archive_category}")
                
            except Exception as e:
                logger.warning(f"Failed to archive {file_path}: {e}")
        
        return archived_count
    
    def clean_root_directory(self) -> int:
        """Clean up the root directory"""
        cleaned_count = 0
        
        for item in self.base_path.iterdir():
            if item.is_file():
                # Check if file should be kept in root
                if item.name not in self.keep_in_root:
                    # Check if it's a file that should be archived
                    if self.is_unused_file(item, 'root_cleanup'):
                        try:
                            archive_dest = self.archive_dirs['root_cleanup'] / item.name
                            shutil.move(str(item), str(archive_dest))
                            cleaned_count += 1
                            self.cleanup_report['details']['root_cleanup'].append(item.name)
                            logger.info(f"Archived from root: {item.name}")
                        except Exception as e:
                            logger.warning(f"Failed to archive {item}: {e}")
            
            elif item.is_dir():
                # Check if directory should be kept in root
                if item.name not in self.keep_dirs_in_root:
                    # Archive entire directory
                    try:
                        archive_dest = self.archive_dirs['root_cleanup'] / item.name
                        shutil.move(str(item), str(archive_dest))
                        cleaned_count += 1
                        self.cleanup_report['details']['root_cleanup'].append(f"{item.name}/ (directory)")
                        logger.info(f"Archived directory from root: {item.name}")
                    except Exception as e:
                        logger.warning(f"Failed to archive directory {item}: {e}")
        
        return cleaned_count
    
    def clean_directory(self, directory_name: str) -> Dict[str, int]:
        """Clean a specific directory"""
        directory_path = self.base_path / directory_name
        
        if not directory_path.exists():
            logger.warning(f"Directory {directory_name} does not exist")
            return {'files': 0, 'directories': 0}
        
        logger.info(f"Cleaning directory: {directory_name}")
        
        # Identify unused files
        unused_files = self.identify_unused_files(directory_path, directory_name)
        
        # Archive unused files
        archived_files = self.archive_files(unused_files, directory_name)
        
        # Clean up empty directories
        archived_dirs = self.clean_empty_directories(directory_path)
        
        return {'files': archived_files, 'directories': archived_dirs}
    
    def clean_empty_directories(self, directory_path: Path) -> int:
        """Remove empty directories"""
        removed_count = 0
        
        try:
            # Walk through directories bottom-up
            for root, dirs, files in os.walk(directory_path, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        # Try to remove empty directory
                        if not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            removed_count += 1
                            logger.info(f"Removed empty directory: {dir_path.relative_to(self.base_path)}")
                    except OSError:
                        # Directory not empty or permission error
                        pass
        except Exception as e:
            logger.warning(f"Error cleaning empty directories: {e}")
        
        return removed_count
    
    def calculate_archive_size(self) -> int:
        """Calculate total size of archived files"""
        total_size = 0
        
        for archive_dir in self.archive_dirs.values():
            if archive_dir.exists():
                for file_path in archive_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            total_size += file_path.stat().st_size
                        except OSError:
                            pass
        
        return total_size
    
    def create_archive_manifest(self):
        """Create a manifest of archived files"""
        manifest = {
            'archive_date': datetime.now().isoformat(),
            'archive_structure': {},
            'file_counts': {},
            'total_size': self.calculate_archive_size()
        }
        
        for category, archive_dir in self.archive_dirs.items():
            if archive_dir.exists():
                files = list(archive_dir.rglob('*'))
                file_count = len([f for f in files if f.is_file()])
                manifest['file_counts'][category] = file_count
                manifest['archive_structure'][category] = str(archive_dir.relative_to(self.base_path))
        
        manifest_path = self.archive_path / "archive_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Archive manifest created: {manifest_path}")
        return manifest
    
    def run_cleanup(self):
        """Run the complete cleanup process"""
        logger.info("🧹 Starting Workspace Cleanup and Organization")
        logger.info("=" * 60)
        logger.info("This will archive unused files and clean the workspace")
        logger.info("=" * 60)
        
        try:
            # Clean specific directories
            directories_to_clean = ['scripts', 'data', 'ml', 'tests']
            
            for directory in directories_to_clean:
                logger.info(f"Cleaning {directory} directory...")
                results = self.clean_directory(directory)
                self.cleanup_report['files_archived'] += results['files']
                self.cleanup_report['directories_archived'] += results['directories']
                logger.info(f"Archived {results['files']} files and {results['directories']} directories from {directory}")
            
            # Clean root directory
            logger.info("Cleaning root directory...")
            root_cleaned = self.clean_root_directory()
            self.cleanup_report['root_cleaned'] = root_cleaned
            logger.info(f"Cleaned {root_cleaned} items from root directory")
            
            # Calculate archive size
            self.cleanup_report['archive_size'] = self.calculate_archive_size()
            
            # Create archive manifest
            manifest = self.create_archive_manifest()
            
            # Save cleanup report
            report_path = self.archive_path / "cleanup_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.cleanup_report, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Workspace cleanup completed successfully!")
            logger.info(f"📁 Files archived: {self.cleanup_report['files_archived']}")
            logger.info(f"📂 Directories archived: {self.cleanup_report['directories_archived']}")
            logger.info(f"🧹 Root items cleaned: {self.cleanup_report['root_cleaned']}")
            logger.info(f"💾 Archive size: {self.cleanup_report['archive_size'] / (1024*1024):.2f} MB")
            logger.info(f"📋 Reports saved to: {self.archive_path}")
            
            return self.cleanup_report
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            raise

def main():
    """Main function"""
    cleaner = WorkspaceCleaner()
    cleaner.run_cleanup()

if __name__ == "__main__":
    main()
