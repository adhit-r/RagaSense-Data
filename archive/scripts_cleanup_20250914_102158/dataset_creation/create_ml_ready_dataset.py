#!/usr/bin/env python3
"""
Create ML-Ready Dataset
======================

This script creates a comprehensive ML-ready dataset by:
1. Connecting audio features to raga labels from metadata
2. Creating training/validation splits for ML
3. Integrating with corrected tradition classification
4. Preparing the dataset for model training
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLDatasetCreator:
    """Create ML-ready dataset from audio features and raga metadata."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.ml_ready_path = self.base_path / "data" / "ml_ready"
        self.ml_ready_path.mkdir(parents=True, exist_ok=True)
        
        # Load corrected raga dataset
        self.raga_dataset_path = self.base_path / "data" / "organized_processed" / "unified_ragas_target_achieved.json"
        self.ragas_data = self._load_raga_dataset()
        
        logger.info(f"🎯 ML Dataset Creator initialized")
        logger.info(f"📁 ML ready path: {self.ml_ready_path}")
        logger.info(f"📊 Loaded {len(self.ragas_data)} ragas from corrected dataset")
    
    def _load_raga_dataset(self):
        """Load the corrected raga dataset with tradition classification."""
        try:
            with open(self.raga_dataset_path, 'r') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded raga dataset with {len(data)} entries")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to load raga dataset: {e}")
            return []
    
    def _extract_raga_from_filename(self, filename: str) -> str:
        """Extract raga name from audio filename."""
        # Common patterns in Saraga filenames
        filename = filename.lower()
        
        # Remove common prefixes/suffixes
        filename = filename.replace('.wav', '').replace('.mp3', '').replace('.flac', '')
        filename = filename.replace('_', ' ').replace('-', ' ')
        
        # Special handling for Hindustani files with "Raag" prefix
        if 'raag ' in filename:
            # Extract raga name after "raag "
            parts = filename.split('raag ')
            if len(parts) > 1:
                raga_part = parts[1].split()[0]  # Get first word after "raag"
                # Try to match with known ragas
                for raga_key, raga_data in self.ragas_data.items():
                    if isinstance(raga_data, dict) and 'name' in raga_data:
                        raga_name = raga_data['name'].lower()
                        if raga_part in raga_name or raga_name in raga_part:
                            return raga_data['name']
        
        # Try to match with known ragas (direct match)
        for raga_key, raga_data in self.ragas_data.items():
            if isinstance(raga_data, dict) and 'name' in raga_data:
                raga_name = raga_data['name'].lower()
                if raga_name in filename:
                    return raga_data['name']
        
        # If no direct match, try partial matches
        for raga_key, raga_data in self.ragas_data.items():
            if isinstance(raga_data, dict) and 'name' in raga_data:
                raga_name = raga_data['name'].lower()
                raga_words = raga_name.split()
                if any(word in filename for word in raga_words if len(word) > 3):
                    return raga_data['name']
        
        return "Unknown"
    
    def _load_audio_features(self, features_file: Path) -> list:
        """Load audio features from JSON file."""
        try:
            logger.info(f"📂 Loading audio features from {features_file.name}")
            with open(features_file, 'r') as f:
                data = json.load(f)
            
            features = data.get('features', [])
            logger.info(f"✅ Loaded {len(features)} audio feature entries")
            
            # Debug: Check the structure of the first feature
            if features:
                logger.info(f"🔍 First feature keys: {list(features[0].keys()) if isinstance(features[0], dict) else 'Not a dict'}")
                if isinstance(features[0], dict) and 'metadata' in features[0]:
                    logger.info(f"🔍 Metadata keys: {list(features[0]['metadata'].keys())}")
            
            return features
        except Exception as e:
            logger.error(f"❌ Failed to load audio features from {features_file}: {e}")
            return []
    
    def _create_feature_vector(self, audio_features: dict) -> np.ndarray:
        """Create a flattened feature vector from audio features."""
        try:
            feature_vector = []
            
            # MFCC features (flatten)
            if 'mfcc' in audio_features:
                mfcc = np.array(audio_features['mfcc'])
                feature_vector.extend(mfcc.flatten())
            
            # Mel-spectrogram (flatten)
            if 'mel_spectrogram' in audio_features:
                mel_spec = np.array(audio_features['mel_spectrogram'])
                feature_vector.extend(mel_spec.flatten())
            
            # Chroma features (flatten)
            if 'chroma' in audio_features:
                chroma = np.array(audio_features['chroma'])
                feature_vector.extend(chroma.flatten())
            
            # Spectral features (flatten)
            if 'spectral_centroid' in audio_features:
                spec_cent = np.array(audio_features['spectral_centroid'])
                feature_vector.extend(spec_cent.flatten())
            
            if 'spectral_rolloff' in audio_features:
                spec_roll = np.array(audio_features['spectral_rolloff'])
                feature_vector.extend(spec_roll.flatten())
            
            if 'zero_crossing_rate' in audio_features:
                zcr = np.array(audio_features['zero_crossing_rate'])
                feature_vector.extend(zcr.flatten())
            
            # Tempo (single value)
            if 'tempo' in audio_features:
                feature_vector.append(audio_features['tempo'])
            
            # Tonnetz features (flatten)
            if 'tonnetz' in audio_features:
                tonnetz = np.array(audio_features['tonnetz'])
                feature_vector.extend(tonnetz.flatten())
            
            return np.array(feature_vector)
            
        except Exception as e:
            logger.error(f"❌ Error creating feature vector: {e}")
            return np.array([])
    
    def _get_raga_metadata(self, raga_name: str) -> dict:
        """Get raga metadata from the corrected dataset."""
        for raga_key, raga_data in self.ragas_data.items():
            if isinstance(raga_data, dict) and raga_data.get('name') == raga_name:
                return raga_data
        return {
            'name': raga_name,
            'tradition': 'Unknown',
            'melakarta': None,
            'arohana': None,
            'avarohana': None
        }
    
    def create_ml_dataset(self):
        """Create the complete ML-ready dataset."""
        logger.info("🚀 Creating ML-ready dataset...")
        
        # Load all audio features
        audio_features_dir = self.ml_ready_path / "audio_features"
        
        all_features = []
        all_labels = []
        all_metadata = []
        
        # Process each audio features file
        for features_file in audio_features_dir.glob("*_audio_features.json"):
            logger.info(f"📊 Processing {features_file.name}")
            
            features_data = self._load_audio_features(features_file)
            
            for audio_feature in features_data:
                try:
                    # Extract raga from filename
                    file_path = audio_feature.get('metadata', {}).get('file_path', '')
                    raga_name = self._extract_raga_from_filename(file_path)
                    
                    # Debug: Log first few filenames and extracted ragas
                    if len(all_features) < 5:
                        logger.info(f"🔍 File: {file_path} -> Raga: {raga_name}")
                    
                    # Skip unknown ragas
                    if raga_name == "Unknown":
                        continue
                    
                    # Create feature vector
                    feature_vector = self._create_feature_vector(audio_feature)
                    
                    if len(feature_vector) == 0:
                        continue
                    
                    # Get raga metadata
                    raga_metadata = self._get_raga_metadata(raga_name)
                    
                    # Store data
                    all_features.append(feature_vector)
                    all_labels.append(raga_name)
                    all_metadata.append({
                        'raga_name': raga_name,
                        'tradition': raga_metadata.get('tradition', 'Unknown'),
                        'melakarta': raga_metadata.get('melakarta'),
                        'arohana': raga_metadata.get('arohana'),
                        'avarohana': raga_metadata.get('avarohana'),
                        'file_path': file_path,
                        'source': audio_feature.get('source', 'unknown'),
                        'duration': audio_feature.get('metadata', {}).get('duration', 0)
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing audio feature: {e}")
                    continue
        
        logger.info(f"📊 Total samples collected: {len(all_features)}")
        logger.info(f"📊 Unique ragas: {len(set(all_labels))}")
        
        if len(all_features) == 0:
            logger.error("❌ No valid features collected!")
            return {}
        
        # Convert to numpy arrays - ensure consistent dimensions
        if all_features:
            # Find the maximum length to pad all vectors
            max_length = max(len(fv) for fv in all_features)
            
            # Pad all feature vectors to the same length
            padded_features = []
            for fv in all_features:
                if len(fv) < max_length:
                    # Pad with zeros
                    padded_fv = np.pad(fv, (0, max_length - len(fv)), 'constant')
                else:
                    padded_fv = fv
                padded_features.append(padded_fv)
            
            X = np.array(padded_features)
            y = np.array(all_labels)
        else:
            X = np.array([])
            y = np.array([])
        
        # Create label encoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Create training/validation splits (non-stratified due to some classes having only 1 sample)
        X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
            X, y_encoded, all_metadata, test_size=0.2, random_state=42
        )
        
        # Create dataset summary
        dataset_summary = {
            'creation_date': datetime.now().isoformat(),
            'total_samples': int(len(all_features)),
            'training_samples': int(len(X_train)),
            'validation_samples': int(len(X_val)),
            'unique_ragas': int(len(set(all_labels))),
            'feature_dimensions': int(X.shape[1]),
            'raga_distribution': {str(k): int(v) for k, v in dict(pd.Series(all_labels).value_counts()).items()},
            'tradition_distribution': {str(k): int(v) for k, v in dict(pd.Series([m['tradition'] for m in all_metadata]).value_counts()).items()}
        }
        
        # Save the ML-ready dataset
        ml_dataset = {
            'metadata': dataset_summary,
            'label_encoder': {
                'classes': label_encoder.classes_.tolist(),
                'n_classes': int(len(label_encoder.classes_))
            },
            'training_data': {
                'X_train': X_train.tolist(),
                'y_train': [int(x) for x in y_train.tolist()],
                'metadata_train': metadata_train
            },
            'validation_data': {
                'X_val': X_val.tolist(),
                'y_val': [int(x) for x in y_val.tolist()],
                'metadata_val': metadata_val
            }
        }
        
        # Save to file
        output_file = self.ml_ready_path / "ml_ready_dataset.json"
        with open(output_file, 'w') as f:
            json.dump(ml_dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 ML-ready dataset saved to: {output_file}")
        
        # Save summary separately
        summary_file = self.ml_ready_path / "ml_dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(dataset_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Dataset summary saved to: {summary_file}")
        
        # Log results
        logger.info("🎉 ML-ready dataset creation completed!")
        logger.info(f"📊 Training samples: {len(X_train)}")
        logger.info(f"📊 Validation samples: {len(X_val)}")
        logger.info(f"📊 Feature dimensions: {X.shape[1]}")
        logger.info(f"📊 Unique ragas: {len(set(all_labels))}")
        
        return dataset_summary

def main():
    """Main function to create ML-ready dataset."""
    creator = MLDatasetCreator()
    summary = creator.create_ml_dataset()
    return summary

if __name__ == "__main__":
    main()
