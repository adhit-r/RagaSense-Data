#!/usr/bin/env python3
"""
Perfect Dataset for Raga Detection
=================================

This script addresses the key issues in our current dataset:
1. Severe class imbalance (Nat: 100 samples, others: 1-5 samples)
2. Limited dataset size (149 samples total)
3. High-dimensional features (209,305 dimensions)
4. Need to process more audio files (413 available, 150 processed)
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import librosa
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPerfection:
    """Comprehensive dataset perfection system."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.raw_data_path = base_path / "data" / "organized_raw"
        self.processed_data_path = base_path / "data" / "processed"
        self.ml_ready_path = base_path / "data" / "ml_ready"
        
        # Create directories
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.ml_ready_path.mkdir(parents=True, exist_ok=True)
        
        # Audio processing parameters
        self.sample_rate = 22050  # Reduced from 44100 for efficiency
        self.max_duration = 30  # 30 seconds max
        self.n_mfcc = 13
        self.n_mels = 64  # Reduced from 128
        self.n_chroma = 12
        
        logger.info("🎯 Dataset Perfection System initialized")
    
    def analyze_current_dataset(self) -> Dict:
        """Analyze the current dataset to identify issues."""
        logger.info("📊 Analyzing current dataset...")
        
        # Load current dataset summary
        summary_path = self.ml_ready_path / "ml_dataset_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                current_summary = json.load(f)
        else:
            logger.error("❌ Current dataset summary not found!")
            return {}
        
        # Load current ML dataset
        ml_dataset_path = self.ml_ready_path / "ml_ready_dataset.json"
        if ml_dataset_path.exists():
            with open(ml_dataset_path, 'r') as f:
                ml_data = json.load(f)
        else:
            logger.error("❌ Current ML dataset not found!")
            return {}
        
        # Analyze issues
        raga_distribution = current_summary['raga_distribution']
        total_samples = current_summary['total_samples']
        
        # Calculate imbalance metrics
        max_samples = max(raga_distribution.values())
        min_samples = min(raga_distribution.values())
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        # Identify severely underrepresented ragas
        underrepresented_ragas = [raga for raga, count in raga_distribution.items() if count < 5]
        overrepresented_ragas = [raga for raga, count in raga_distribution.items() if count > 20]
        
        analysis = {
            'current_status': {
                'total_samples': total_samples,
                'unique_ragas': len(raga_distribution),
                'feature_dimensions': current_summary['feature_dimensions'],
                'imbalance_ratio': imbalance_ratio,
                'max_samples': max_samples,
                'min_samples': min_samples
            },
            'issues': {
                'severe_imbalance': imbalance_ratio > 10,
                'small_dataset': total_samples < 1000,
                'high_dimensionality': current_summary['feature_dimensions'] > 100000,
                'underrepresented_ragas': len(underrepresented_ragas),
                'overrepresented_ragas': len(overrepresented_ragas)
            },
            'raga_analysis': {
                'underrepresented': underrepresented_ragas,
                'overrepresented': overrepresented_ragas,
                'distribution': raga_distribution
            }
        }
        
        logger.info(f"📈 Current dataset analysis:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Unique ragas: {len(raga_distribution)}")
        logger.info(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
        logger.info(f"  Feature dimensions: {current_summary['feature_dimensions']:,}")
        logger.info(f"  Underrepresented ragas: {len(underrepresented_ragas)}")
        logger.info(f"  Overrepresented ragas: {len(overrepresented_ragas)}")
        
        return analysis
    
    def discover_available_audio(self) -> Dict:
        """Discover all available audio files and their potential raga mappings."""
        logger.info("🔍 Discovering available audio files...")
        
        # Find all audio files
        audio_files = list(self.raw_data_path.rglob("*.mp3")) + \
                     list(self.raw_data_path.rglob("*.wav")) + \
                     list(self.raw_data_path.rglob("*.m4a"))
        
        logger.info(f"📁 Found {len(audio_files)} audio files")
        
        # Analyze file structure
        file_analysis = {
            'total_files': len(audio_files),
            'by_extension': {},
            'by_directory': {},
            'potential_ragas': set()
        }
        
        for audio_file in audio_files:
            # Count by extension
            ext = audio_file.suffix.lower()
            file_analysis['by_extension'][ext] = file_analysis['by_extension'].get(ext, 0) + 1
            
            # Count by directory
            rel_path = audio_file.relative_to(self.raw_data_path)
            dir_key = str(rel_path.parent)
            file_analysis['by_directory'][dir_key] = file_analysis['by_directory'].get(dir_key, 0) + 1
            
            # Extract potential raga names from file paths
            path_parts = str(rel_path).lower().split('/')
            for part in path_parts:
                if 'raga' in part or any(raga_word in part for raga_word in ['kalyani', 'bhairavi', 'todi', 'yaman']):
                    file_analysis['potential_ragas'].add(part)
        
        file_analysis['potential_ragas'] = list(file_analysis['potential_ragas'])
        
        logger.info(f"📊 Audio file analysis:")
        logger.info(f"  By extension: {file_analysis['by_extension']}")
        logger.info(f"  Potential ragas found: {len(file_analysis['potential_ragas'])}")
        
        return file_analysis
    
    def create_balanced_sampling_strategy(self, current_analysis: Dict) -> Dict:
        """Create a strategy for balanced sampling."""
        logger.info("⚖️ Creating balanced sampling strategy...")
        
        raga_distribution = current_analysis['raga_analysis']['distribution']
        
        # Define target samples per raga
        target_samples_per_raga = {
            'high_priority': 20,  # For underrepresented ragas
            'medium_priority': 15,  # For moderately represented ragas
            'low_priority': 10,   # For overrepresented ragas
            'max_samples': 30     # Maximum samples per raga
        }
        
        # Categorize ragas by current representation
        raga_categories = {
            'underrepresented': [],  # < 5 samples
            'moderate': [],          # 5-15 samples
            'overrepresented': []    # > 15 samples
        }
        
        for raga, count in raga_distribution.items():
            if count < 5:
                raga_categories['underrepresented'].append(raga)
            elif count <= 15:
                raga_categories['moderate'].append(raga)
            else:
                raga_categories['overrepresented'].append(raga)
        
        # Create sampling strategy
        sampling_strategy = {
            'target_samples': target_samples_per_raga,
            'raga_categories': raga_categories,
            'priority_order': [
                'underrepresented',  # Focus on these first
                'moderate',
                'overrepresented'
            ],
            'estimated_total_samples': (
                len(raga_categories['underrepresented']) * target_samples_per_raga['high_priority'] +
                len(raga_categories['moderate']) * target_samples_per_raga['medium_priority'] +
                len(raga_categories['overrepresented']) * target_samples_per_raga['low_priority']
            )
        }
        
        logger.info(f"📋 Balanced sampling strategy:")
        logger.info(f"  Underrepresented ragas: {len(raga_categories['underrepresented'])} (target: {target_samples_per_raga['high_priority']} each)")
        logger.info(f"  Moderate ragas: {len(raga_categories['moderate'])} (target: {target_samples_per_raga['medium_priority']} each)")
        logger.info(f"  Overrepresented ragas: {len(raga_categories['overrepresented'])} (target: {target_samples_per_raga['low_priority']} each)")
        logger.info(f"  Estimated total samples: {sampling_strategy['estimated_total_samples']}")
        
        return sampling_strategy
    
    def optimize_feature_extraction(self) -> Dict:
        """Optimize feature extraction parameters for better performance."""
        logger.info("🔧 Optimizing feature extraction parameters...")
        
        # Current parameters are too high-dimensional
        current_params = {
            'sample_rate': 44100,
            'n_mels': 128,
            'n_mfcc': 13,
            'n_chroma': 12,
            'max_duration': 30
        }
        
        # Optimized parameters for better performance
        optimized_params = {
            'sample_rate': 22050,  # Reduced for efficiency
            'n_mels': 64,          # Reduced from 128
            'n_mfcc': 13,          # Keep same
            'n_chroma': 12,        # Keep same
            'max_duration': 30,    # Keep same
            'hop_length': 512,     # Standard
            'n_fft': 2048          # Standard
        }
        
        # Calculate estimated feature dimensions
        time_frames = int(optimized_params['max_duration'] * optimized_params['sample_rate'] / optimized_params['hop_length'])
        estimated_dimensions = (
            optimized_params['n_mels'] * time_frames +  # Mel-spectrogram
            optimized_params['n_mfcc'] * time_frames +  # MFCC
            optimized_params['n_chroma'] * time_frames +  # Chroma
            4 * time_frames  # Spectral features (centroid, rolloff, zcr, tonnetz)
        )
        
        optimization = {
            'current_params': current_params,
            'optimized_params': optimized_params,
            'estimated_dimensions': estimated_dimensions,
            'reduction_factor': current_params.get('feature_dimensions', 209305) / estimated_dimensions,
            'benefits': [
                'Reduced computational complexity',
                'Faster training and inference',
                'Better generalization',
                'Lower memory requirements'
            ]
        }
        
        logger.info(f"🎯 Feature optimization:")
        logger.info(f"  Current dimensions: ~209,305")
        logger.info(f"  Optimized dimensions: ~{estimated_dimensions:,}")
        logger.info(f"  Reduction factor: {optimization['reduction_factor']:.1f}x")
        
        return optimization
    
    def create_perfection_plan(self) -> Dict:
        """Create a comprehensive plan to perfect the dataset."""
        logger.info("📋 Creating dataset perfection plan...")
        
        # Analyze current state
        current_analysis = self.analyze_current_dataset()
        if not current_analysis:
            return {}
        
        # Discover available resources
        audio_analysis = self.discover_available_audio()
        
        # Create sampling strategy
        sampling_strategy = self.create_balanced_sampling_strategy(current_analysis)
        
        # Optimize features
        feature_optimization = self.optimize_feature_extraction()
        
        # Create comprehensive plan
        perfection_plan = {
            'current_issues': current_analysis['issues'],
            'available_resources': audio_analysis,
            'sampling_strategy': sampling_strategy,
            'feature_optimization': feature_optimization,
            'implementation_phases': {
                'phase_1': {
                    'name': 'Feature Optimization',
                    'description': 'Implement optimized feature extraction parameters',
                    'estimated_time': '2 hours',
                    'priority': 'high'
                },
                'phase_2': {
                    'name': 'Audio Processing Scale-up',
                    'description': 'Process remaining audio files with balanced sampling',
                    'estimated_time': '8 hours',
                    'priority': 'high'
                },
                'phase_3': {
                    'name': 'Dataset Balancing',
                    'description': 'Apply data augmentation and balancing techniques',
                    'estimated_time': '4 hours',
                    'priority': 'medium'
                },
                'phase_4': {
                    'name': 'Quality Validation',
                    'description': 'Validate improved dataset quality',
                    'estimated_time': '2 hours',
                    'priority': 'medium'
                }
            },
            'expected_improvements': {
                'dataset_size': f"{sampling_strategy['estimated_total_samples']} samples (vs current {current_analysis['current_status']['total_samples']})",
                'class_balance': 'Significantly improved (target: 10-30 samples per raga)',
                'feature_dimensions': f"~{feature_optimization['estimated_dimensions']:,} (vs current 209,305)",
                'processing_efficiency': f"{feature_optimization['reduction_factor']:.1f}x faster"
            }
        }
        
        # Save plan
        plan_path = self.ml_ready_path / "dataset_perfection_plan.json"
        with open(plan_path, 'w') as f:
            json.dump(perfection_plan, f, indent=2)
        
        logger.info("✅ Dataset perfection plan created!")
        logger.info(f"📁 Plan saved to: {plan_path}")
        
        return perfection_plan
    
    def generate_perfection_report(self) -> str:
        """Generate a comprehensive report on dataset perfection."""
        logger.info("📊 Generating dataset perfection report...")
        
        plan = self.create_perfection_plan()
        if not plan:
            return "❌ Failed to create perfection plan"
        
        report = f"""
# Dataset Perfection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Current Issues Identified

### 1. Severe Class Imbalance
- **Imbalance Ratio**: {plan['current_issues'].get('imbalance_ratio', 'N/A')}:1
- **Overrepresented Ragas**: {plan['current_issues'].get('overrepresented_ragas', 0)}
- **Underrepresented Ragas**: {plan['current_issues'].get('underrepresented_ragas', 0)}

### 2. Limited Dataset Size
- **Current Samples**: {plan['current_issues'].get('total_samples', 0)}
- **Target Samples**: {plan['sampling_strategy']['estimated_total_samples']}
- **Improvement**: {plan['sampling_strategy']['estimated_total_samples'] / plan['current_issues'].get('total_samples', 1):.1f}x

### 3. High Dimensionality
- **Current Dimensions**: 209,305
- **Optimized Dimensions**: {plan['feature_optimization']['estimated_dimensions']:,}
- **Reduction Factor**: {plan['feature_optimization']['reduction_factor']:.1f}x

## Available Resources

### Audio Files
- **Total Available**: {plan['available_resources']['total_files']}
- **Currently Processed**: 150
- **Remaining to Process**: {plan['available_resources']['total_files'] - 150}

### File Distribution
{json.dumps(plan['available_resources']['by_extension'], indent=2)}

## Implementation Plan

### Phase 1: Feature Optimization (High Priority)
- Implement optimized feature extraction parameters
- Reduce feature dimensions by {plan['feature_optimization']['reduction_factor']:.1f}x
- Estimated time: 2 hours

### Phase 2: Audio Processing Scale-up (High Priority)
- Process remaining {plan['available_resources']['total_files'] - 150} audio files
- Apply balanced sampling strategy
- Estimated time: 8 hours

### Phase 3: Dataset Balancing (Medium Priority)
- Apply data augmentation techniques
- Implement class balancing
- Estimated time: 4 hours

### Phase 4: Quality Validation (Medium Priority)
- Validate improved dataset quality
- Generate quality metrics
- Estimated time: 2 hours

## Expected Improvements

- **Dataset Size**: {plan['expected_improvements']['dataset_size']}
- **Class Balance**: {plan['expected_improvements']['class_balance']}
- **Feature Dimensions**: {plan['expected_improvements']['feature_dimensions']}
- **Processing Efficiency**: {plan['expected_improvements']['processing_efficiency']}

## Next Steps

1. **Immediate**: Implement Phase 1 (Feature Optimization)
2. **Short-term**: Execute Phase 2 (Audio Processing Scale-up)
3. **Medium-term**: Complete Phase 3 (Dataset Balancing)
4. **Final**: Validate with Phase 4 (Quality Validation)

---
*This report provides a comprehensive roadmap for perfecting the raga detection dataset.*
"""
        
        # Save report
        report_path = self.ml_ready_path / "dataset_perfection_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to: {report_path}")
        return report

def main():
    """Main function to perfect the dataset."""
    logger.info("🎯 Dataset Perfection System")
    logger.info("=" * 60)
    
    # Initialize system
    base_path = Path("/Users/adhi/axonome/RagaSense-Data")
    perfection_system = DatasetPerfection(base_path)
    
    # Generate comprehensive report
    report = perfection_system.generate_perfection_report()
    
    logger.info("🎉 Dataset perfection analysis complete!")
    logger.info("📋 Check the generated report for detailed recommendations")

if __name__ == "__main__":
    main()
