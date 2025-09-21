#!/usr/bin/env python3
"""
Repository Cleanup and Organization
==================================

This script cleans up and organizes the RagaSense repository by:
1. Analyzing the current structure
2. Archiving unused/old files
3. Creating a proper, clean structure
4. Organizing scripts by purpose
5. Creating comprehensive documentation

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_and_organize_repo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RepositoryOrganizer:
    """Organize and clean up the repository structure"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.archive_path = self.base_path / "archive" / "repository_cleanup"
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # Analysis results
        self.analysis = {
            'cleanup_date': datetime.now().isoformat(),
            'files_analyzed': 0,
            'files_archived': 0,
            'scripts_organized': 0,
            'structure_created': {}
        }
    
    def analyze_current_structure(self):
        """Analyze the current repository structure"""
        logger.info("Analyzing current repository structure...")
        
        analysis = {
            'root_files': [],
            'scripts': [],
            'logs': [],
            'data_directories': [],
            'documentation': [],
            'config_files': []
        }
        
        # Analyze root directory
        for item in self.base_path.iterdir():
            if item.is_file():
                self.analysis['files_analyzed'] += 1
                
                if item.suffix == '.py':
                    analysis['scripts'].append({
                        'name': item.name,
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
                elif item.suffix == '.log':
                    analysis['logs'].append({
                        'name': item.name,
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
                elif item.suffix in ['.md', '.txt']:
                    analysis['documentation'].append({
                        'name': item.name,
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
                elif item.suffix in ['.json', '.yaml', '.yml']:
                    analysis['config_files'].append({
                        'name': item.name,
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
                else:
                    analysis['root_files'].append({
                        'name': item.name,
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
            elif item.is_dir() and item.name.startswith('data'):
                analysis['data_directories'].append({
                    'name': item.name,
                    'files': len(list(item.rglob('*')))
                })
        
        self.analysis['current_structure'] = analysis
        logger.info(f"Analyzed {self.analysis['files_analyzed']} files")
        
        return analysis
    
    def categorize_scripts(self, scripts: List[Dict]) -> Dict[str, List[str]]:
        """Categorize scripts by their purpose"""
        categories = {
            'data_processing': [],
            'dataset_creation': [],
            'ml_training': [],
            'analysis': [],
            'utilities': [],
            'deprecated': []
        }
        
        for script in scripts:
            name = script['name'].lower()
            
            if any(keyword in name for keyword in ['extract', 'process', 'scale', 'audio']):
                categories['data_processing'].append(script['name'])
            elif any(keyword in name for keyword in ['create', 'unified', 'dataset', 'ml_ready']):
                categories['dataset_creation'].append(script['name'])
            elif any(keyword in name for keyword in ['train', 'model', 'classifier', 'yue']):
                categories['ml_training'].append(script['name'])
            elif any(keyword in name for keyword in ['analyze', 'investigate', 'validate', 'perfect']):
                categories['analysis'].append(script['name'])
            elif any(keyword in name for keyword in ['fix', 'update', 'migrate', 'setup']):
                categories['utilities'].append(script['name'])
            else:
                categories['deprecated'].append(script['name'])
        
        return categories
    
    def create_organized_structure(self):
        """Create the new organized structure"""
        logger.info("Creating organized repository structure...")
        
        # Create main directories
        directories = {
            'scripts': {
                'data_processing': 'Scripts for processing raw data',
                'dataset_creation': 'Scripts for creating unified datasets',
                'analysis': 'Scripts for data analysis and validation',
                'utilities': 'Utility scripts for maintenance',
                'deprecated': 'Deprecated scripts (archived)'
            },
            'docs': {
                'api': 'API documentation',
                'datasets': 'Dataset documentation',
                'analysis': 'Analysis reports',
                'architecture': 'System architecture docs'
            },
            'config': 'Configuration files',
            'logs': 'Log files (organized by date)',
            'archive': 'Archived files and old versions'
        }
        
        for main_dir, subdirs in directories.items():
            main_path = self.base_path / main_dir
            main_path.mkdir(exist_ok=True)
            
            if isinstance(subdirs, dict):
                for subdir, description in subdirs.items():
                    sub_path = main_path / subdir
                    sub_path.mkdir(exist_ok=True)
                    
                    # Create README for each directory
                    readme_content = f"""# {subdir.replace('_', ' ').title()}

{description}

## Purpose
This directory contains {description.lower()}.

## Files
- Files will be organized here based on their purpose

## Last Updated
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    readme_file = sub_path / "README.md"
                    with open(readme_file, 'w') as f:
                        f.write(readme_content)
            else:
                # Create README for main directory
                readme_content = f"""# {main_dir.title()}

{subdirs}

## Purpose
This directory contains {subdirs.lower()}.

## Last Updated
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                readme_file = main_path / "README.md"
                with open(readme_file, 'w') as f:
                    f.write(readme_content)
        
        self.analysis['structure_created'] = directories
        logger.info("Created organized directory structure")
    
    def organize_scripts(self, script_categories: Dict[str, List[str]]):
        """Move scripts to their appropriate directories"""
        logger.info("Organizing scripts by category...")
        
        for category, scripts in script_categories.items():
            if not scripts:
                continue
                
            target_dir = self.base_path / "scripts" / category
            target_dir.mkdir(exist_ok=True)
            
            for script_name in scripts:
                source_path = self.base_path / script_name
                if source_path.exists():
                    target_path = target_dir / script_name
                    
                    # Move script
                    shutil.move(str(source_path), str(target_path))
                    self.analysis['scripts_organized'] += 1
                    logger.info(f"Moved {script_name} to scripts/{category}/")
    
    def archive_old_files(self):
        """Archive old and unused files"""
        logger.info("Archiving old and unused files...")
        
        # Archive old logs
        log_files = [f for f in self.base_path.iterdir() 
                    if f.is_file() and f.suffix == '.log']
        
        if log_files:
            logs_dir = self.archive_path / "old_logs"
            logs_dir.mkdir(exist_ok=True)
            
            for log_file in log_files:
                shutil.move(str(log_file), str(logs_dir / log_file.name))
                self.analysis['files_archived'] += 1
                logger.info(f"Archived {log_file.name}")
        
        # Archive old JSON files (except important ones)
        important_json = ['postgresql_migration_report.json', 'rename_log.json']
        json_files = [f for f in self.base_path.iterdir() 
                     if f.is_file() and f.suffix == '.json' and f.name not in important_json]
        
        if json_files:
            json_dir = self.archive_path / "old_json"
            json_dir.mkdir(exist_ok=True)
            
            for json_file in json_files:
                shutil.move(str(json_file), str(json_dir / json_file.name))
                self.analysis['files_archived'] += 1
                logger.info(f"Archived {json_file.name}")
    
    def create_comprehensive_documentation(self):
        """Create comprehensive documentation"""
        logger.info("Creating comprehensive documentation...")
        
        # Main README
        main_readme = """# RagaSense Dataset

A comprehensive dataset for Indian classical music (Carnatic and Hindustani) combining multiple sources.

## Dataset Sources

1. **Ramanarunachalam**: Raga definitions with Arohana/Avarohana notation
2. **Saraga-Carnatic**: Real audio recordings of Carnatic ragas
3. **Saraga-Hindustani**: Real audio recordings of Hindustani ragas
4. **Saraga-Carnatic-Melody-Synth**: Synthetic melody data

## Repository Structure

```
RagaSense-Data/
├── data/                          # Main dataset directory
│   ├── 01_source/                # Original source data
│   ├── 02_raw/                   # Extracted raw data
│   ├── 03_processed/             # Cleaned and processed data
│   ├── 04_ml_datasets/           # ML-ready datasets
│   ├── 05_research/              # Research outputs
│   └── 99_archive/               # Archived versions
├── scripts/                      # Processing scripts
│   ├── data_processing/          # Data processing scripts
│   ├── dataset_creation/         # Dataset creation scripts
│   ├── analysis/                 # Analysis scripts
│   └── utilities/                # Utility scripts
├── docs/                         # Documentation
├── config/                       # Configuration files
└── logs/                         # Log files
```

## Quick Start

1. **View Dataset**: Check `data/02_raw/unified_ragas/` for the main dataset
2. **Run Scripts**: Use scripts in `scripts/` directory
3. **Read Docs**: Check `docs/` for detailed documentation

## Dataset Statistics

- **Total Ragas**: 911 (378 Carnatic + 533 Hindustani)
- **Individual Ragas**: Clean dataset with no combinations
- **Musical Theory**: Complete Arohana/Avarohana notation
- **Source**: Ramanarunachalam Music Repository

## Last Updated

{date}

## License

This dataset is for research and educational purposes.
""".format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        with open(self.base_path / "README.md", 'w') as f:
            f.write(main_readme)
        
        # Dataset documentation
        dataset_docs = """# RagaSense Dataset Documentation

## Overview

The RagaSense dataset is a comprehensive collection of Indian classical music data combining multiple authoritative sources.

## Dataset Components

### 1. Ramanarunachalam Music Repository

**Source**: Ramanarunachalam Music Repository
**Type**: Metadata and musical theory
**Content**: 
- Raga definitions with Arohana/Avarohana notation
- Artist information
- Composer details
- Song metadata
- Performance statistics

**Structure**:
- **Carnatic**: 602 individual ragas, 266 combination ragas
- **Hindustani**: 5,315 raga files
- **Total**: 911 unique individual ragas

### 2. Saraga Dataset

**Source**: MTG Saraga Collections
**Type**: Real audio recordings
**Content**:
- Carnatic: 249 recordings, 96 unique ragas
- Hindustani: 108 recordings, 61 unique ragas
- Total: 357 recordings, 157 unique ragas

### 3. Dataset Organization

The dataset is organized into a clean, logical structure:

```
data/
├── 01_source/           # Original source data (DO NOT MODIFY)
├── 02_raw/             # Extracted raw data by tradition
├── 03_processed/       # Cleaned and standardized data
├── 04_ml_datasets/     # Machine learning ready datasets
├── 05_research/        # Research outputs and analysis
└── 99_archive/         # Old versions and backups
```

## Data Quality

- **Individual Ragas Only**: Filtered out combination ragas
- **Clean Annotations**: Proper Arohana/Avarohana notation
- **Tradition Classification**: Clear Carnatic/Hindustani separation
- **Musical Theory**: Complete raga definitions

## Usage

The dataset is designed for:
- Music research and analysis
- Raga classification studies
- Indian classical music education
- Machine learning applications

## File Formats

- **JSON**: Metadata and annotations
- **MP3**: Audio recordings
- **CSV**: Tabular data exports

## Last Updated

{date}
""".format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        docs_dir = self.base_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        with open(docs_dir / "DATASET.md", 'w') as f:
            f.write(dataset_docs)
        
        logger.info("Created comprehensive documentation")
    
    def create_analysis_report(self):
        """Create analysis report of the cleanup"""
        logger.info("Creating analysis report...")
        
        report = {
            'cleanup_summary': self.analysis,
            'repository_structure': {
                'organized_directories': list(self.analysis['structure_created'].keys()),
                'scripts_organized': self.analysis['scripts_organized'],
                'files_archived': self.analysis['files_archived']
            },
            'recommendations': [
                'Use scripts/ directory for all processing scripts',
                'Keep data/ directory clean and organized',
                'Archive old files in archive/ directory',
                'Update documentation regularly',
                'Follow naming conventions for new files'
            ]
        }
        
        report_file = self.base_path / "docs" / "cleanup_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Created analysis report: {report_file}")
    
    def run_cleanup_and_organization(self):
        """Run the complete cleanup and organization process"""
        logger.info("Starting repository cleanup and organization...")
        
        try:
            # Analyze current structure
            current_structure = self.analyze_current_structure()
            
            # Categorize scripts
            script_categories = self.categorize_scripts(current_structure['scripts'])
            
            # Create organized structure
            self.create_organized_structure()
            
            # Organize scripts
            self.organize_scripts(script_categories)
            
            # Archive old files
            self.archive_old_files()
            
            # Create documentation
            self.create_comprehensive_documentation()
            
            # Create analysis report
            self.create_analysis_report()
            
            logger.info("Repository cleanup and organization completed successfully!")
            return self.analysis
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise

def main():
    """Main function"""
    print("🧹 RagaSense Repository Cleanup and Organization")
    print("=" * 60)
    print("This will:")
    print("• Analyze current repository structure")
    print("• Organize scripts by purpose")
    print("• Archive old and unused files")
    print("• Create proper directory structure")
    print("• Generate comprehensive documentation")
    print("=" * 60)
    
    organizer = RepositoryOrganizer()
    results = organizer.run_cleanup_and_organization()
    
    print(f"\n✅ Repository Cleanup Complete!")
    print(f"📁 Files analyzed: {results['files_analyzed']}")
    print(f"📁 Scripts organized: {results['scripts_organized']}")
    print(f"📁 Files archived: {results['files_archived']}")
    print(f"📁 Structure created: {len(results['structure_created'])} main directories")
    print(f"📁 Check docs/cleanup_report.json for detailed analysis")

if __name__ == "__main__":
    main()
