#!/usr/bin/env python3
"""
Create Enhanced ML Dataset
==========================

This script creates an enhanced ML-ready dataset by:
1. Combining optimized audio features (413 files processed)
2. Applying corrected Nat raga mapping
3. Creating balanced training/validation splits
4. Implementing proper class balancing strategies
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMLDatasetCreator:
    """Create enhanced ML-ready dataset with optimized features and balanced sampling."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.data_path = base_path / "data"
        self.ml_ready_path = self.data_path / "ml_ready"
        self.processed_path = self.data_path / "organized_processed"
        
        logger.info("🎯 Enhanced ML Dataset Creator initialized")
    
    def load_optimized_audio_features(self) -> List[Dict]:
        """Load the optimized audio features."""
        logger.info("📁 Loading optimized audio features...")
        
        features_path = self.ml_ready_path / "optimized_audio_features.json"
        if not features_path.exists():
            logger.error("❌ Optimized audio features not found!")
            return []
        
        with open(features_path, 'r') as f:
            audio_features = json.load(f)
        
        logger.info(f"✅ Loaded {len(audio_features)} audio features")
        return audio_features
    
    def load_corrected_raga_metadata(self) -> Dict:
        """Load the corrected raga metadata."""
        logger.info("📚 Loading corrected raga metadata...")
        
        ragas_path = self.processed_path / "unified_ragas_nat_fixed.json"
        if not ragas_path.exists():
            logger.error("❌ Corrected raga metadata not found!")
            return {}
        
        with open(ragas_path, 'r') as f:
            ragas_data = json.load(f)
        
        logger.info(f"✅ Loaded {len(ragas_data)} raga entries")
        return ragas_data
    
    def extract_raga_from_file_path(self, file_path: str) -> Optional[str]:
        """Extract raga name from file path."""
        try:
            path_parts = Path(file_path).parts
            
            # Look for raga in the path
            for part in path_parts:
                if 'raga' in part.lower():
                    # Find the raga directory
                    raga_dir_index = path_parts.index(part)
                    if raga_dir_index + 1 < len(path_parts):
                        raga_name = path_parts[raga_dir_index + 1]
                        # Remove file extension if present
                        return Path(raga_name).stem
                elif part.endswith('.json') and 'raga' in part.lower():
                    # Handle raga JSON files
                    return Path(part).stem
            
            # Fallback: look for common raga patterns
            for part in path_parts:
                if any(raga_word in part.lower() for raga_word in 
                      ['kalyani', 'bhairavi', 'todi', 'yaman', 'nat', 'shree', 'des']):
                    return part
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Could not extract raga from {file_path}: {e}")
            return None
    
    def determine_tradition_from_path(self, file_path: str) -> str:
        """Determine tradition from file path."""
        path_str = file_path.lower()
        
        if 'hindustani' in path_str:
            return 'Hindustani'
        elif 'carnatic' in path_str:
            return 'Carnatic'
        else:
            # Default based on directory structure
            if 'hindustani' in path_str:
                return 'Hindustani'
            else:
                return 'Carnatic'  # Default to Carnatic for Ramanarunachalam repository
    
    def create_enhanced_samples(self, audio_features: List[Dict], ragas_data: Dict) -> List[Dict]:
        """Create enhanced samples with proper raga classification."""
        logger.info("🔧 Creating enhanced samples...")
        
        enhanced_samples = []
        raga_counts = {}
        tradition_counts = {'Carnatic': 0, 'Hindustani': 0, 'Both': 0}
        
        for audio_data in audio_features:
            file_path = audio_data['file_path']
            features = audio_data['features']
            
            # Extract raga from file path
            raga_name = self.extract_raga_from_file_path(file_path)
            if not raga_name:
                logger.warning(f"⚠️ Could not determine raga for {file_path}")
                continue
            
            # Determine tradition
            tradition = self.determine_tradition_from_path(file_path)
            
            # Special handling for Nat raga (corrected mapping)
            if raga_name.lower() == 'nat':
                tradition = 'Hindustani'  # Nat is Hindustani, not Carnatic
                raga_name = 'Nat'
            
            # Create enhanced sample
            sample = {
                'file_path': file_path,
                'raga_name': raga_name,
                'tradition': tradition,
                'features': features,
                'duration': audio_data['duration'],
                'sample_rate': audio_data['sample_rate'],
                'extraction_time': audio_data['extraction_time'],
                'enhanced': True,
                'enhancement_date': datetime.now().isoformat()
            }
            
            enhanced_samples.append(sample)
            
            # Update counts
            raga_counts[raga_name] = raga_counts.get(raga_name, 0) + 1
            tradition_counts[tradition] = tradition_counts.get(tradition, 0) + 1
        
        logger.info(f"✅ Created {len(enhanced_samples)} enhanced samples")
        logger.info(f"📊 Raga distribution: {dict(sorted(raga_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
        logger.info(f"📊 Tradition distribution: {tradition_counts}")
        
        return enhanced_samples, raga_counts, tradition_counts
    
    def apply_balanced_sampling(self, samples: List[Dict], raga_counts: Dict) -> List[Dict]:
        """Apply balanced sampling to reduce class imbalance."""
        logger.info("⚖️ Applying balanced sampling...")
        
        # Define sampling strategy
        max_samples_per_raga = 30  # Maximum samples per raga
        min_samples_per_raga = 5   # Minimum samples per raga
        
        balanced_samples = []
        sampling_stats = {}
        
        for raga_name, count in raga_counts.items():
            raga_samples = [s for s in samples if s['raga_name'] == raga_name]
            
            if count > max_samples_per_raga:
                # Downsample overrepresented ragas
                selected_samples = np.random.choice(
                    raga_samples, size=max_samples_per_raga, replace=False
                ).tolist()
                sampling_stats[raga_name] = {
                    'original_count': count,
                    'sampled_count': max_samples_per_raga,
                    'action': 'downsampled'
                }
            elif count < min_samples_per_raga:
                # Keep underrepresented ragas as-is (we'll handle with data augmentation later)
                selected_samples = raga_samples
                sampling_stats[raga_name] = {
                    'original_count': count,
                    'sampled_count': count,
                    'action': 'kept_as_is'
                }
            else:
                # Keep moderate ragas as-is
                selected_samples = raga_samples
                sampling_stats[raga_name] = {
                    'original_count': count,
                    'sampled_count': count,
                    'action': 'kept_as_is'
                }
            
            balanced_samples.extend(selected_samples)
        
        logger.info(f"✅ Balanced sampling applied:")
        logger.info(f"  Original samples: {len(samples)}")
        logger.info(f"  Balanced samples: {len(balanced_samples)}")
        logger.info(f"  Reduction: {len(samples) - len(balanced_samples)} samples")
        
        return balanced_samples, sampling_stats
    
    def create_ml_ready_features(self, samples: List[Dict]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create ML-ready feature vectors and labels."""
        logger.info("🤖 Creating ML-ready features...")
        
        features_list = []
        raga_labels = []
        tradition_labels = []
        
        for sample in samples:
            features = sample['features']
            
            # Flatten all features into a single vector
            feature_vector = []
            
            # Add mel-spectrogram
            if 'mel_spectrogram' in features:
                feature_vector.extend(np.array(features['mel_spectrogram']).flatten())
            
            # Add MFCC
            if 'mfcc' in features:
                feature_vector.extend(np.array(features['mfcc']).flatten())
            
            # Add chroma
            if 'chroma' in features:
                feature_vector.extend(np.array(features['chroma']).flatten())
            
            # Add spectral features
            for spec_feat in ['spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate', 'tonnetz']:
                if spec_feat in features:
                    feature_vector.extend(np.array(features[spec_feat]).flatten())
            
            # Add tempo
            if 'tempo' in features:
                feature_vector.append(features['tempo'])
            
            features_list.append(feature_vector)
            raga_labels.append(sample['raga_name'])
            tradition_labels.append(sample['tradition'])
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y_raga = np.array(raga_labels)
        y_tradition = np.array(tradition_labels)
        
        logger.info(f"✅ ML-ready features created:")
        logger.info(f"  Feature matrix shape: {X.shape}")
        logger.info(f"  Unique ragas: {len(np.unique(y_raga))}")
        logger.info(f"  Unique traditions: {len(np.unique(y_tradition))}")
        
        return X, y_raga, y_tradition
    
    def create_train_val_splits(self, X: np.ndarray, y_raga: np.ndarray, y_tradition: np.ndarray) -> Dict:
        """Create training and validation splits."""
        logger.info("📊 Creating train/validation splits...")
        
        # Create stratified split based on raga labels
        X_train, X_val, y_raga_train, y_raga_val, y_tradition_train, y_tradition_val = train_test_split(
            X, y_raga, y_tradition, test_size=0.2, random_state=42, stratify=y_raga
        )
        
        # Encode labels
        raga_encoder = LabelEncoder()
        tradition_encoder = LabelEncoder()
        
        y_raga_train_encoded = raga_encoder.fit_transform(y_raga_train)
        y_raga_val_encoded = raga_encoder.transform(y_raga_val)
        y_tradition_train_encoded = tradition_encoder.fit_transform(y_tradition_train)
        y_tradition_val_encoded = tradition_encoder.transform(y_tradition_val)
        
        # Create splits dictionary
        splits = {
            'X_train': X_train.tolist(),
            'X_val': X_val.tolist(),
            'y_raga_train': y_raga_train_encoded.tolist(),
            'y_raga_val': y_raga_val_encoded.tolist(),
            'y_tradition_train': y_tradition_train_encoded.tolist(),
            'y_tradition_val': y_tradition_val_encoded.tolist(),
            'raga_encoder_classes': raga_encoder.classes_.tolist(),
            'tradition_encoder_classes': tradition_encoder.classes_.tolist(),
            'feature_dimensions': X.shape[1],
            'num_ragas': len(raga_encoder.classes_),
            'num_traditions': len(tradition_encoder.classes_)
        }
        
        logger.info(f"✅ Train/validation splits created:")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val)}")
        logger.info(f"  Feature dimensions: {X.shape[1]}")
        logger.info(f"  Number of ragas: {len(raga_encoder.classes_)}")
        
        return splits
    
    def save_enhanced_dataset(self, samples: List[Dict], splits: Dict, sampling_stats: Dict, 
                            raga_counts: Dict, tradition_counts: Dict):
        """Save the enhanced ML dataset."""
        logger.info("💾 Saving enhanced ML dataset...")
        
        # Create enhanced dataset
        enhanced_dataset = {
            'creation_date': datetime.now().isoformat(),
            'dataset_type': 'enhanced_ml_ready',
            'total_samples': len(samples),
            'feature_dimensions': splits['feature_dimensions'],
            'num_ragas': splits['num_ragas'],
            'num_traditions': splits['num_traditions'],
            'training_samples': len(splits['X_train']),
            'validation_samples': len(splits['X_val']),
            'raga_distribution': raga_counts,
            'tradition_distribution': tradition_counts,
            'sampling_stats': sampling_stats,
            'splits': splits,
            'samples': samples
        }
        
        # Save enhanced dataset
        dataset_path = self.ml_ready_path / "enhanced_ml_ready_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(enhanced_dataset, f, indent=2)
        
        # Create summary
        summary = {
            'creation_date': datetime.now().isoformat(),
            'dataset_type': 'enhanced_ml_ready',
            'total_samples': len(samples),
            'feature_dimensions': splits['feature_dimensions'],
            'num_ragas': splits['num_ragas'],
            'num_traditions': splits['num_traditions'],
            'training_samples': len(splits['X_train']),
            'validation_samples': len(splits['X_val']),
            'raga_distribution': dict(sorted(raga_counts.items(), key=lambda x: x[1], reverse=True)),
            'tradition_distribution': tradition_counts,
            'improvements': {
                'nat_mapping_fixed': True,
                'optimized_features': True,
                'balanced_sampling': True,
                'enhanced_classification': True
            }
        }
        
        summary_path = self.ml_ready_path / "enhanced_ml_dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Enhanced dataset saved:")
        logger.info(f"  Dataset: {dataset_path}")
        logger.info(f"  Summary: {summary_path}")
        
        return enhanced_dataset, summary

def main():
    """Main function to create enhanced ML dataset."""
    logger.info("🎯 Create Enhanced ML Dataset")
    logger.info("=" * 60)
    
    # Initialize creator
    base_path = Path("/Users/adhi/axonome/RagaSense-Data")
    creator = EnhancedMLDatasetCreator(base_path)
    
    try:
        # Step 1: Load optimized audio features
        audio_features = creator.load_optimized_audio_features()
        if not audio_features:
            logger.error("❌ No audio features loaded!")
            return
        
        # Step 2: Load corrected raga metadata
        ragas_data = creator.load_corrected_raga_metadata()
        if not ragas_data:
            logger.error("❌ No raga metadata loaded!")
            return
        
        # Step 3: Create enhanced samples
        samples, raga_counts, tradition_counts = creator.create_enhanced_samples(audio_features, ragas_data)
        
        # Step 4: Apply balanced sampling
        balanced_samples, sampling_stats = creator.apply_balanced_sampling(samples, raga_counts)
        
        # Step 5: Create ML-ready features
        X, y_raga, y_tradition = creator.create_ml_ready_features(balanced_samples)
        
        # Step 6: Create train/validation splits
        splits = creator.create_train_val_splits(X, y_raga, y_tradition)
        
        # Step 7: Save enhanced dataset
        enhanced_dataset, summary = creator.save_enhanced_dataset(
            balanced_samples, splits, sampling_stats, raga_counts, tradition_counts
        )
        
        logger.info("🎉 Enhanced ML dataset created successfully!")
        logger.info(f"📊 Total samples: {len(balanced_samples)}")
        logger.info(f"🎯 Feature dimensions: {splits['feature_dimensions']}")
        logger.info(f"📈 Number of ragas: {splits['num_ragas']}")
        logger.info(f"⚖️ Tradition distribution: {tradition_counts}")
        
    except Exception as e:
        logger.error(f"❌ Error creating enhanced dataset: {e}")
        raise

if __name__ == "__main__":
    main()
