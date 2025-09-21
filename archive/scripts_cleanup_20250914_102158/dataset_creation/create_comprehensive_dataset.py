#!/usr/bin/env python3
"""
Create Comprehensive RagaSense Dataset
=====================================

This script creates a comprehensive dataset by combining:
- All 148 processed audio files with extracted features
- Raga metadata and annotations (6,183 files)
- Cross-tradition mappings
- ML-ready datasets
- Quality validation and statistics

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays and types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_comprehensive_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDatasetCreator:
    """Create comprehensive RagaSense dataset"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "unified" / "comprehensive_dataset"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Input paths
        self.audio_features_path = self.base_path / "unified" / "comprehensive_audio_features"
        self.processed_datasets_path = self.base_path / "unified" / "processed_datasets"
        self.cross_tradition_path = self.base_path / "unified" / "cross_tradition_mappings"
        self.ml_ready_path = self.base_path / "unified" / "ml_ready"
        self.carnatic_path = self.base_path / "carnatic"
        self.hindustani_path = self.base_path / "hindustani"
        
        # Statistics
        self.dataset_stats = {
            'creation_date': datetime.now().isoformat(),
            'audio_files': 0,
            'metadata_files': 0,
            'annotation_files': 0,
            'traditions': {},
            'ragas': {},
            'features': {}
        }
    
    def load_audio_features(self):
        """Load all processed audio features"""
        logger.info("Loading audio features...")
        
        audio_features = {}
        
        # Load comprehensive audio features
        comprehensive_file = self.audio_features_path / "comprehensive_all_audio_features.json"
        if comprehensive_file.exists():
            with open(comprehensive_file, 'r') as f:
                all_features = json.load(f)
            
            for feature in all_features:
                file_name = feature['file_name']
                tradition = feature['tradition']
                
                audio_features[file_name] = {
                    'tradition': tradition,
                    'features': feature['features'],
                    'duration': feature['duration'],
                    'sample_rate': feature['sample_rate'],
                    'extraction_timestamp': feature['extraction_timestamp']
                }
            
            logger.info(f"Loaded {len(audio_features)} audio features")
            self.dataset_stats['audio_files'] = len(audio_features)
            
            # Update tradition breakdown
            for feature in all_features:
                tradition = feature['tradition']
                if tradition not in self.dataset_stats['traditions']:
                    self.dataset_stats['traditions'][tradition] = 0
                self.dataset_stats['traditions'][tradition] += 1
        
        return audio_features
    
    def load_metadata(self):
        """Load all metadata files"""
        logger.info("Loading metadata...")
        
        metadata = {
            'carnatic': {},
            'hindustani': {}
        }
        
        # Load Carnatic metadata
        carnatic_metadata_path = self.carnatic_path / "metadata"
        if carnatic_metadata_path.exists():
            for metadata_file in carnatic_metadata_path.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        metadata['carnatic'][metadata_file.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {metadata_file}: {e}")
        
        # Load Hindustani metadata
        hindustani_metadata_path = self.hindustani_path / "metadata"
        if hindustani_metadata_path.exists():
            for metadata_file in hindustani_metadata_path.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        metadata['hindustani'][metadata_file.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {metadata_file}: {e}")
        
        # Count metadata files
        total_metadata = sum(len(tradition_data) for tradition_data in metadata.values())
        self.dataset_stats['metadata_files'] = total_metadata
        
        logger.info(f"Loaded {total_metadata} metadata files")
        return metadata
    
    def load_annotations(self):
        """Load all annotation files"""
        logger.info("Loading annotations...")
        
        annotations = {
            'carnatic': {},
            'hindustani': {}
        }
        
        # Load Carnatic annotations
        carnatic_annotations_path = self.carnatic_path / "annotations"
        if carnatic_annotations_path.exists():
            for annotation_file in carnatic_annotations_path.glob("*.json"):
                try:
                    with open(annotation_file, 'r') as f:
                        data = json.load(f)
                        annotations['carnatic'][annotation_file.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {annotation_file}: {e}")
        
        # Load Hindustani annotations
        hindustani_annotations_path = self.hindustani_path / "annotations"
        if hindustani_annotations_path.exists():
            for annotation_file in hindustani_annotations_path.glob("*.json"):
                try:
                    with open(annotation_file, 'r') as f:
                        data = json.load(f)
                        annotations['hindustani'][annotation_file.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {annotation_file}: {e}")
        
        # Count annotation files
        total_annotations = sum(len(tradition_data) for tradition_data in annotations.values())
        self.dataset_stats['annotation_files'] = total_annotations
        
        logger.info(f"Loaded {total_annotations} annotation files")
        return annotations
    
    def load_processed_datasets(self):
        """Load processed datasets"""
        logger.info("Loading processed datasets...")
        
        processed_datasets = {}
        
        if self.processed_datasets_path.exists():
            for dataset_file in self.processed_datasets_path.glob("*.json"):
                try:
                    with open(dataset_file, 'r') as f:
                        data = json.load(f)
                        processed_datasets[dataset_file.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_file}: {e}")
        
        logger.info(f"Loaded {len(processed_datasets)} processed datasets")
        return processed_datasets
    
    def load_cross_tradition_mappings(self):
        """Load cross-tradition mappings"""
        logger.info("Loading cross-tradition mappings...")
        
        cross_tradition_mappings = {}
        
        if self.cross_tradition_path.exists():
            for mapping_file in self.cross_tradition_path.glob("*.json"):
                try:
                    with open(mapping_file, 'r') as f:
                        data = json.load(f)
                        cross_tradition_mappings[mapping_file.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {mapping_file}: {e}")
        
        logger.info(f"Loaded {len(cross_tradition_mappings)} cross-tradition mappings")
        return cross_tradition_mappings
    
    def load_ml_ready_datasets(self):
        """Load ML-ready datasets"""
        logger.info("Loading ML-ready datasets...")
        
        ml_ready_datasets = {}
        
        if self.ml_ready_path.exists():
            for ml_file in self.ml_ready_path.glob("*.json"):
                try:
                    with open(ml_file, 'r') as f:
                        data = json.load(f)
                        ml_ready_datasets[ml_file.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {ml_file}: {e}")
        
        logger.info(f"Loaded {len(ml_ready_datasets)} ML-ready datasets")
        return ml_ready_datasets
    
    def analyze_ragas(self, metadata, annotations):
        """Analyze raga distribution and characteristics"""
        logger.info("Analyzing ragas...")
        
        raga_analysis = {
            'total_unique_ragas': 0,
            'carnatic_ragas': set(),
            'hindustani_ragas': set(),
            'cross_tradition_ragas': set(),
            'raga_frequencies': defaultdict(int),
            'tradition_distribution': defaultdict(int)
        }
        
        # Analyze from metadata
        for tradition, tradition_metadata in metadata.items():
            if 'raga' in tradition_metadata:
                raga_data = tradition_metadata['raga']
                if isinstance(raga_data, list):
                    for raga in raga_data:
                        if isinstance(raga, dict) and 'name' in raga:
                            raga_name = raga['name']
                            raga_analysis['raga_frequencies'][raga_name] += 1
                            raga_analysis['tradition_distribution'][tradition] += 1
                            
                            if tradition == 'carnatic':
                                raga_analysis['carnatic_ragas'].add(raga_name)
                            elif tradition == 'hindustani':
                                raga_analysis['hindustani_ragas'].add(raga_name)
        
        # Find cross-tradition ragas
        raga_analysis['cross_tradition_ragas'] = raga_analysis['carnatic_ragas'].intersection(
            raga_analysis['hindustani_ragas']
        )
        
        raga_analysis['total_unique_ragas'] = len(
            raga_analysis['carnatic_ragas'].union(raga_analysis['hindustani_ragas'])
        )
        
        # Convert sets to lists for JSON serialization
        raga_analysis['carnatic_ragas'] = list(raga_analysis['carnatic_ragas'])
        raga_analysis['hindustani_ragas'] = list(raga_analysis['hindustani_ragas'])
        raga_analysis['cross_tradition_ragas'] = list(raga_analysis['cross_tradition_ragas'])
        raga_analysis['raga_frequencies'] = dict(raga_analysis['raga_frequencies'])
        raga_analysis['tradition_distribution'] = dict(raga_analysis['tradition_distribution'])
        
        self.dataset_stats['ragas'] = raga_analysis
        
        logger.info(f"Found {raga_analysis['total_unique_ragas']} unique ragas")
        logger.info(f"Carnatic: {len(raga_analysis['carnatic_ragas'])} ragas")
        logger.info(f"Hindustani: {len(raga_analysis['hindustani_ragas'])} ragas")
        logger.info(f"Cross-tradition: {len(raga_analysis['cross_tradition_ragas'])} ragas")
        
        return raga_analysis
    
    def analyze_features(self, audio_features):
        """Analyze audio feature characteristics"""
        logger.info("Analyzing audio features...")
        
        feature_analysis = {
            'total_audio_files': len(audio_features),
            'feature_types': set(),
            'tradition_breakdown': defaultdict(int),
            'duration_stats': [],
            'sample_rate_stats': []
        }
        
        for file_name, feature_data in audio_features.items():
            tradition = feature_data['tradition']
            features = feature_data['features']
            duration = feature_data['duration']
            sample_rate = feature_data['sample_rate']
            
            feature_analysis['tradition_breakdown'][tradition] += 1
            feature_analysis['duration_stats'].append(duration)
            feature_analysis['sample_rate_stats'].append(sample_rate)
            
            # Collect feature types
            for feature_type in features.keys():
                feature_analysis['feature_types'].add(feature_type)
        
        # Calculate statistics
        if feature_analysis['duration_stats']:
            feature_analysis['duration_stats'] = {
                'mean': np.mean(feature_analysis['duration_stats']),
                'min': np.min(feature_analysis['duration_stats']),
                'max': np.max(feature_analysis['duration_stats']),
                'std': np.std(feature_analysis['duration_stats'])
            }
        
        if feature_analysis['sample_rate_stats']:
            feature_analysis['sample_rate_stats'] = {
                'mean': np.mean(feature_analysis['sample_rate_stats']),
                'min': np.min(feature_analysis['sample_rate_stats']),
                'max': np.max(feature_analysis['sample_rate_stats']),
                'std': np.std(feature_analysis['sample_rate_stats'])
            }
        
        # Convert sets to lists for JSON serialization
        feature_analysis['feature_types'] = list(feature_analysis['feature_types'])
        feature_analysis['tradition_breakdown'] = dict(feature_analysis['tradition_breakdown'])
        
        self.dataset_stats['features'] = feature_analysis
        
        logger.info(f"Analyzed {len(audio_features)} audio files")
        logger.info(f"Feature types: {len(feature_analysis['feature_types'])}")
        
        return feature_analysis
    
    def create_comprehensive_dataset(self):
        """Create the comprehensive dataset"""
        logger.info("Creating comprehensive dataset...")
        
        # Load all components
        audio_features = self.load_audio_features()
        metadata = self.load_metadata()
        annotations = self.load_annotations()
        processed_datasets = self.load_processed_datasets()
        cross_tradition_mappings = self.load_cross_tradition_mappings()
        ml_ready_datasets = self.load_ml_ready_datasets()
        
        # Analyze components
        raga_analysis = self.analyze_ragas(metadata, annotations)
        feature_analysis = self.analyze_features(audio_features)
        
        # Create comprehensive dataset structure
        comprehensive_dataset = {
            'dataset_info': {
                'name': 'RagaSense Comprehensive Dataset',
                'version': '1.0.0',
                'creation_date': datetime.now().isoformat(),
                'description': 'Comprehensive dataset for Indian classical music raga research and ML',
                'total_size': {
                    'audio_files': len(audio_features),
                    'metadata_files': self.dataset_stats['metadata_files'],
                    'annotation_files': self.dataset_stats['annotation_files'],
                    'unique_ragas': raga_analysis['total_unique_ragas']
                }
            },
            'audio_features': audio_features,
            'metadata': metadata,
            'annotations': annotations,
            'processed_datasets': processed_datasets,
            'cross_tradition_mappings': cross_tradition_mappings,
            'ml_ready_datasets': ml_ready_datasets,
            'statistics': self.dataset_stats,
            'raga_analysis': raga_analysis,
            'feature_analysis': feature_analysis
        }
        
        return comprehensive_dataset
    
    def save_comprehensive_dataset(self, dataset):
        """Save the comprehensive dataset"""
        logger.info("Saving comprehensive dataset...")
        
        # Save main dataset
        main_file = self.output_path / "ragasense_comprehensive_dataset.json"
        with open(main_file, 'w') as f:
            json.dump(dataset, f, cls=NumpyEncoder, indent=2)
        
        logger.info(f"Saved comprehensive dataset to {main_file}")
        
        # Save individual components
        components = {
            'audio_features': dataset['audio_features'],
            'metadata': dataset['metadata'],
            'annotations': dataset['annotations'],
            'processed_datasets': dataset['processed_datasets'],
            'cross_tradition_mappings': dataset['cross_tradition_mappings'],
            'ml_ready_datasets': dataset['ml_ready_datasets'],
            'statistics': dataset['statistics'],
            'raga_analysis': dataset['raga_analysis'],
            'feature_analysis': dataset['feature_analysis']
        }
        
        for component_name, component_data in components.items():
            component_file = self.output_path / f"{component_name}.json"
            with open(component_file, 'w') as f:
                json.dump(component_data, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved {component_name} to {component_file}")
        
        # Create summary report
        self.create_summary_report(dataset)
    
    def create_summary_report(self, dataset):
        """Create a human-readable summary report"""
        logger.info("Creating summary report...")
        
        report = f"""# RagaSense Comprehensive Dataset Summary

## Dataset Overview
- **Name**: {dataset['dataset_info']['name']}
- **Version**: {dataset['dataset_info']['version']}
- **Creation Date**: {dataset['dataset_info']['creation_date']}
- **Description**: {dataset['dataset_info']['description']}

## Dataset Statistics
- **Total Audio Files**: {dataset['dataset_info']['total_size']['audio_files']}
- **Total Metadata Files**: {dataset['dataset_info']['total_size']['metadata_files']}
- **Total Annotation Files**: {dataset['dataset_info']['total_size']['annotation_files']}
- **Unique Ragas**: {dataset['dataset_info']['total_size']['unique_ragas']}

## Tradition Breakdown
"""
        
        for tradition, count in dataset['statistics']['traditions'].items():
            report += f"- **{tradition}**: {count} files\n"
        
        report += f"""
## Raga Analysis
- **Total Unique Ragas**: {dataset['raga_analysis']['total_unique_ragas']}
- **Carnatic Ragas**: {len(dataset['raga_analysis']['carnatic_ragas'])}
- **Hindustani Ragas**: {len(dataset['raga_analysis']['hindustani_ragas'])}
- **Cross-Tradition Ragas**: {len(dataset['raga_analysis']['cross_tradition_ragas'])}

## Audio Feature Analysis
- **Total Audio Files**: {dataset['feature_analysis']['total_audio_files']}
- **Feature Types**: {len(dataset['feature_analysis']['feature_types'])}
- **Feature Types Available**: {', '.join(dataset['feature_analysis']['feature_types'])}

## Duration Statistics
- **Mean Duration**: {dataset['feature_analysis']['duration_stats']['mean']:.3f} seconds
- **Min Duration**: {dataset['feature_analysis']['duration_stats']['min']:.3f} seconds
- **Max Duration**: {dataset['feature_analysis']['duration_stats']['max']:.3f} seconds
- **Standard Deviation**: {dataset['feature_analysis']['duration_stats']['std']:.3f} seconds

## Dataset Components
1. **Audio Features**: Comprehensive audio feature extraction from all available files
2. **Metadata**: Raga, artist, composer, and song information
3. **Annotations**: Detailed raga definitions and characteristics
4. **Processed Datasets**: Cleaned and standardized raga data
5. **Cross-Tradition Mappings**: Relationships between Carnatic and Hindustani ragas
6. **ML-Ready Datasets**: Preprocessed data ready for machine learning

## Usage
This comprehensive dataset is designed for:
- **Research**: Academic research on Indian classical music
- **Machine Learning**: Training raga classification models
- **Music Analysis**: Understanding raga characteristics and relationships
- **Cross-Tradition Studies**: Comparing Carnatic and Hindustani traditions

## File Structure
```
data/unified/comprehensive_dataset/
├── ragasense_comprehensive_dataset.json    # Complete dataset
├── audio_features.json                     # Audio feature data
├── metadata.json                          # Metadata information
├── annotations.json                       # Raga annotations
├── processed_datasets.json                # Processed raga data
├── cross_tradition_mappings.json          # Cross-tradition mappings
├── ml_ready_datasets.json                 # ML-ready datasets
├── statistics.json                        # Dataset statistics
├── raga_analysis.json                     # Raga analysis results
├── feature_analysis.json                  # Feature analysis results
└── comprehensive_dataset_summary.md       # This summary report
```

## Next Steps
1. **Data Exploration**: Use the comprehensive dataset for research and analysis
2. **ML Model Training**: Utilize ML-ready datasets for model development
3. **Feature Engineering**: Build upon the extracted audio features
4. **Cross-Tradition Analysis**: Study relationships between traditions
5. **Dataset Expansion**: Add more audio files and metadata as available

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        report_file = self.output_path / "comprehensive_dataset_summary.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Created summary report: {report_file}")
    
    def run_comprehensive_dataset_creation(self):
        """Run the complete comprehensive dataset creation process"""
        logger.info("Starting comprehensive dataset creation...")
        
        try:
            # Create comprehensive dataset
            dataset = self.create_comprehensive_dataset()
            
            # Save dataset
            self.save_comprehensive_dataset(dataset)
            
            logger.info("Comprehensive dataset creation completed successfully!")
            return dataset
            
        except Exception as e:
            logger.error(f"Error in comprehensive dataset creation: {e}")
            raise

def main():
    """Main function"""
    print("🎵 RagaSense Comprehensive Dataset Creation")
    print("=" * 50)
    
    creator = ComprehensiveDatasetCreator()
    dataset = creator.run_comprehensive_dataset_creation()
    
    print(f"\n✅ Comprehensive Dataset Created!")
    print(f"📁 Total audio files: {dataset['dataset_info']['total_size']['audio_files']}")
    print(f"📁 Total metadata files: {dataset['dataset_info']['total_size']['metadata_files']}")
    print(f"📁 Total annotation files: {dataset['dataset_info']['total_size']['annotation_files']}")
    print(f"📁 Unique ragas: {dataset['dataset_info']['total_size']['unique_ragas']}")
    print(f"📁 Output directory: {creator.output_path}")

if __name__ == "__main__":
    main()
