#!/usr/bin/env python3
"""
Create Final Enhanced ML Dataset
===============================

This script creates the final enhanced ML-ready dataset with all fixes:
1. Combining optimized audio features (413 files processed)
2. Applying corrected Nat raga mapping
3. Creating balanced training/validation splits
4. Proper feature vector handling
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

class FinalEnhancedMLDatasetCreator:
    """Create final enhanced ML-ready dataset with all fixes."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.data_path = base_path / "data"
        self.ml_ready_path = self.data_path / "ml_ready"
        self.processed_path = self.data_path / "organized_processed"
        
        logger.info("🎯 Final Enhanced ML Dataset Creator initialized")
    
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
    
    def create_enhanced_samples(self, audio_features: List[Dict]) -> Tuple[List[Dict], Dict, Dict]:
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
    
    def create_final_features(self, samples: List[Dict]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create final ML-ready features with proper handling."""
        logger.info("🤖 Creating final ML-ready features...")
        
        features_list = []
        raga_labels = []
        tradition_labels = []
        
        # Define fixed dimensions for each feature type
        mel_frames = 1292  # 30s * 22050 / 512
        mfcc_frames = 1292
        chroma_frames = 1292
        spectral_frames = 1292
        tonnetz_frames = 1292
        
        for sample in samples:
            features = sample['features']
            feature_vector = []
            
            # Mel-spectrogram (64 x frames)
            if 'mel_spectrogram' in features:
                mel_data = np.array(features['mel_spectrogram'])
                if mel_data.ndim == 2:
                    # Pad or truncate to fixed size
                    if mel_data.shape[1] > mel_frames:
                        mel_data = mel_data[:, :mel_frames]
                    elif mel_data.shape[1] < mel_frames:
                        mel_data = np.pad(mel_data, ((0, 0), (0, mel_frames - mel_data.shape[1])), mode='constant')
                    feature_vector.extend(mel_data.flatten())
                else:
                    # Fallback: create zero vector
                    feature_vector.extend([0] * (64 * mel_frames))
            else:
                feature_vector.extend([0] * (64 * mel_frames))
            
            # MFCC (13 x frames)
            if 'mfcc' in features:
                mfcc_data = np.array(features['mfcc'])
                if mfcc_data.ndim == 2:
                    if mfcc_data.shape[1] > mfcc_frames:
                        mfcc_data = mfcc_data[:, :mfcc_frames]
                    elif mfcc_data.shape[1] < mfcc_frames:
                        mfcc_data = np.pad(mfcc_data, ((0, 0), (0, mfcc_frames - mfcc_data.shape[1])), mode='constant')
                    feature_vector.extend(mfcc_data.flatten())
                else:
                    feature_vector.extend([0] * (13 * mfcc_frames))
            else:
                feature_vector.extend([0] * (13 * mfcc_frames))
            
            # Chroma (12 x frames)
            if 'chroma' in features:
                chroma_data = np.array(features['chroma'])
                if chroma_data.ndim == 2:
                    if chroma_data.shape[1] > chroma_frames:
                        chroma_data = chroma_data[:, :chroma_frames]
                    elif chroma_data.shape[1] < chroma_frames:
                        chroma_data = np.pad(chroma_data, ((0, 0), (0, chroma_frames - chroma_data.shape[1])), mode='constant')
                    feature_vector.extend(chroma_data.flatten())
                else:
                    feature_vector.extend([0] * (12 * chroma_frames))
            else:
                feature_vector.extend([0] * (12 * chroma_frames))
            
            # Spectral features
            for spec_feat in ['spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate']:
                if spec_feat in features:
                    spec_data = np.array(features[spec_feat])
                    if spec_data.ndim == 2:
                        if spec_data.shape[1] > spectral_frames:
                            spec_data = spec_data[:, :spectral_frames]
                        elif spec_data.shape[1] < spectral_frames:
                            spec_data = np.pad(spec_data, ((0, 0), (0, spectral_frames - spec_data.shape[1])), mode='constant')
                        feature_vector.extend(spec_data.flatten())
                    else:
                        feature_vector.extend([0] * spectral_frames)
                else:
                    feature_vector.extend([0] * spectral_frames)
            
            # Tonnetz (6 x frames)
            if 'tonnetz' in features:
                tonnetz_data = np.array(features['tonnetz'])
                if tonnetz_data.ndim == 2:
                    if tonnetz_data.shape[1] > tonnetz_frames:
                        tonnetz_data = tonnetz_data[:, :tonnetz_frames]
                    elif tonnetz_data.shape[1] < tonnetz_frames:
                        tonnetz_data = np.pad(tonnetz_data, ((0, 0), (0, tonnetz_frames - tonnetz_data.shape[1])), mode='constant')
                    feature_vector.extend(tonnetz_data.flatten())
                else:
                    feature_vector.extend([0] * (6 * tonnetz_frames))
            else:
                feature_vector.extend([0] * (6 * tonnetz_frames))
            
            # Tempo (1 value) - handle both single values and lists
            if 'tempo' in features:
                tempo_value = features['tempo']
                if isinstance(tempo_value, list):
                    tempo_value = tempo_value[0] if tempo_value else 0.0
                try:
                    feature_vector.append(float(tempo_value))
                except (ValueError, TypeError):
                    feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)
            
            features_list.append(feature_vector)
            raga_labels.append(sample['raga_name'])
            tradition_labels.append(sample['tradition'])
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y_raga = np.array(raga_labels)
        y_tradition = np.array(tradition_labels)
        
        logger.info(f"✅ Final ML-ready features created:")
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
    
    def save_final_dataset(self, samples: List[Dict], splits: Dict, raga_counts: Dict, tradition_counts: Dict):
        """Save the final enhanced ML dataset."""
        logger.info("💾 Saving final enhanced ML dataset...")
        
        # Create final dataset
        final_dataset = {
            'creation_date': datetime.now().isoformat(),
            'dataset_type': 'final_enhanced_ml_ready',
            'total_samples': len(samples),
            'feature_dimensions': splits['feature_dimensions'],
            'num_ragas': splits['num_ragas'],
            'num_traditions': splits['num_traditions'],
            'training_samples': len(splits['X_train']),
            'validation_samples': len(splits['X_val']),
            'raga_distribution': raga_counts,
            'tradition_distribution': tradition_counts,
            'splits': splits,
            'samples': samples
        }
        
        # Save final dataset
        dataset_path = self.ml_ready_path / "final_enhanced_ml_ready_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(final_dataset, f, indent=2)
        
        # Create summary
        summary = {
            'creation_date': datetime.now().isoformat(),
            'dataset_type': 'final_enhanced_ml_ready',
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
                'fixed_length_vectors': True,
                'enhanced_classification': True,
                'proper_tempo_handling': True
            }
        }
        
        summary_path = self.ml_ready_path / "final_enhanced_ml_dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Final dataset saved:")
        logger.info(f"  Dataset: {dataset_path}")
        logger.info(f"  Summary: {summary_path}")
        
        return final_dataset, summary

def main():
    """Main function to create final enhanced ML dataset."""
    logger.info("🎯 Create Final Enhanced ML Dataset")
    logger.info("=" * 60)
    
    # Initialize creator
    base_path = Path("/Users/adhi/axonome/RagaSense-Data")
    creator = FinalEnhancedMLDatasetCreator(base_path)
    
    try:
        # Step 1: Load optimized audio features
        audio_features = creator.load_optimized_audio_features()
        if not audio_features:
            logger.error("❌ No audio features loaded!")
            return
        
        # Step 2: Create enhanced samples
        samples, raga_counts, tradition_counts = creator.create_enhanced_samples(audio_features)
        
        # Step 3: Create final ML-ready features
        X, y_raga, y_tradition = creator.create_final_features(samples)
        
        # Step 4: Create train/validation splits
        splits = creator.create_train_val_splits(X, y_raga, y_tradition)
        
        # Step 5: Save final dataset
        final_dataset, summary = creator.save_final_dataset(samples, splits, raga_counts, tradition_counts)
        
        logger.info("🎉 Final enhanced ML dataset created successfully!")
        logger.info(f"📊 Total samples: {len(samples)}")
        logger.info(f"🎯 Feature dimensions: {splits['feature_dimensions']}")
        logger.info(f"📈 Number of ragas: {splits['num_ragas']}")
        logger.info(f"⚖️ Tradition distribution: {tradition_counts}")
        
    except Exception as e:
        logger.error(f"❌ Error creating final dataset: {e}")
        raise

if __name__ == "__main__":
    main()
