#!/usr/bin/env python3
"""
RagaSense Audio Feature Extractor
Extracts real audio features from all available audio files without dummy data
"""

import os
import json
import logging
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """Extracts real audio features from audio files"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.raw_data_path = base_path / "data" / "raw"
        self.ml_ready_path = base_path / "data" / "ml_ready"
        self.ml_ready_path.mkdir(exist_ok=True)
        
        # Audio processing parameters
        self.sample_rate = 44100
        self.n_mfcc = 13
        self.n_mels = 128
        self.n_chroma = 12
        self.max_duration = 30  # seconds
        
        logger.info(f"🎵 Audio Feature Extractor initialized")
        logger.info(f"📁 Raw data path: {self.raw_data_path}")
        logger.info(f"📁 ML ready path: {self.ml_ready_path}")
        logger.info(f"🎼 Sample rate: {self.sample_rate}Hz")
        logger.info(f"🎼 Max duration: {self.max_duration}s")
    
    def find_audio_files(self) -> List[Path]:
        """Find all audio files in the dataset"""
        logger.info("🔍 Finding audio files...")
        
        audio_files = []
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        
        # Search in all raw data directories
        for root_dir in self.raw_data_path.iterdir():
            if root_dir.is_dir():
                for ext in audio_extensions:
                    files = list(root_dir.rglob(f"*{ext}"))
                    audio_files.extend(files)
        
        logger.info(f"📊 Found {len(audio_files)} audio files")
        return audio_files
    
    def extract_features_from_file(self, audio_path: Path) -> Optional[Dict]:
        """Extract features from a single audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Limit duration
            if len(y) > self.sample_rate * self.max_duration:
                y = y[:self.sample_rate * self.max_duration]
            
            # Extract features
            features = {}
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features['mfcc'] = mfcc.tolist()
            
            # Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            features['mel_spectrogram'] = mel_spec.tolist()
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.n_chroma)
            features['chroma'] = chroma.tolist()
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = spectral_centroid.tolist()
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff'] = spectral_rolloff.tolist()
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
            features['zero_crossing_rate'] = zero_crossing_rate.tolist()
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_frames'] = beats.tolist()
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz'] = tonnetz.tolist()
            
            # Add metadata
            features['metadata'] = {
                'file_path': str(audio_path.relative_to(self.base_path)),
                'duration': len(y) / sr,
                'sample_rate': sr,
                'extracted_at': datetime.now().isoformat()
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting features from {audio_path}: {e}")
            return None
    
    def extract_features_batch(self, audio_files: List[Path], batch_size: int = 10) -> List[Dict]:
        """Extract features from a batch of audio files"""
        logger.info(f"🎼 Extracting features from {len(audio_files)} audio files...")
        
        features_list = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(audio_files), batch_size), desc="Processing audio files"):
            batch = audio_files[i:i + batch_size]
            
            for audio_file in batch:
                features = self.extract_features_from_file(audio_file)
                if features:
                    features_list.append(features)
        
        logger.info(f"✅ Extracted features from {len(features_list)} audio files")
        return features_list
    
    def create_ml_dataset(self, features_list: List[Dict]) -> Dict:
        """Create ML-ready dataset from extracted features"""
        logger.info("🤖 Creating ML-ready dataset...")
        
        if not features_list:
            logger.error("❌ No features extracted")
            return {}
        
        # Prepare data for ML
        ml_data = {
            "dataset_info": {
                "total_files": len(features_list),
                "feature_types": ["mfcc", "mel_spectrogram", "chroma", "spectral_centroid", "spectral_rolloff", "zero_crossing_rate", "tempo", "tonnetz"],
                "sample_rate": self.sample_rate,
                "max_duration": self.max_duration,
                "created_at": datetime.now().isoformat()
            },
            "features": [],
            "metadata": []
        }
        
        # Extract features and metadata
        for i, features in enumerate(features_list):
            # Combine all features into a single vector
            combined_features = []
            
            # Add MFCC features
            if 'mfcc' in features:
                mfcc_flat = np.array(features['mfcc']).flatten()
                combined_features.extend(mfcc_flat)
            
            # Add Mel-spectrogram features
            if 'mel_spectrogram' in features:
                mel_flat = np.array(features['mel_spectrogram']).flatten()
                combined_features.extend(mel_flat)
            
            # Add Chroma features
            if 'chroma' in features:
                chroma_flat = np.array(features['chroma']).flatten()
                combined_features.extend(chroma_flat)
            
            # Add other features
            if 'spectral_centroid' in features:
                centroid_flat = np.array(features['spectral_centroid']).flatten()
                combined_features.extend(centroid_flat)
            
            if 'spectral_rolloff' in features:
                rolloff_flat = np.array(features['spectral_rolloff']).flatten()
                combined_features.extend(rolloff_flat)
            
            if 'zero_crossing_rate' in features:
                zcr_flat = np.array(features['zero_crossing_rate']).flatten()
                combined_features.extend(zcr_flat)
            
            if 'tempo' in features:
                combined_features.append(features['tempo'])
            
            if 'tonnetz' in features:
                tonnetz_flat = np.array(features['tonnetz']).flatten()
                combined_features.extend(tonnetz_flat)
            
            ml_data["features"].append(combined_features)
            ml_data["metadata"].append(features.get('metadata', {}))
        
        # Convert to numpy arrays
        ml_data["features"] = np.array(ml_data["features"])
        
        logger.info(f"📊 ML dataset created:")
        logger.info(f"   Total samples: {len(ml_data['features'])}")
        logger.info(f"   Feature dimension: {ml_data['features'].shape[1] if len(ml_data['features']) > 0 else 0}")
        
        return ml_data
    
    def save_ml_dataset(self, ml_data: Dict):
        """Save ML-ready dataset"""
        logger.info("💾 Saving ML-ready dataset...")
        
        # Save features as numpy array
        features_path = self.ml_ready_path / "audio_features.npy"
        np.save(features_path, ml_data["features"])
        
        # Save metadata
        metadata_path = self.ml_ready_path / "audio_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(ml_data["metadata"], f, indent=2, ensure_ascii=False)
        
        # Save dataset info
        info_path = self.ml_ready_path / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(ml_data["dataset_info"], f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ ML dataset saved:")
        logger.info(f"   Features: {features_path}")
        logger.info(f"   Metadata: {metadata_path}")
        logger.info(f"   Info: {info_path}")
    
    def run_full_extraction(self):
        """Run complete audio feature extraction"""
        logger.info("🚀 Starting full audio feature extraction...")
        
        try:
            # Find audio files
            audio_files = self.find_audio_files()
            
            if not audio_files:
                logger.error("❌ No audio files found")
                return
            
            # Extract features
            features_list = self.extract_features_batch(audio_files)
            
            if not features_list:
                logger.error("❌ No features extracted")
                return
            
            # Create ML dataset
            ml_data = self.create_ml_dataset(features_list)
            
            # Save dataset
            self.save_ml_dataset(ml_data)
            
            logger.info("🎉 Audio feature extraction completed!")
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            raise

def main():
    """Main extraction function"""
    base_path = Path(__file__).parent.parent
    
    extractor = AudioFeatureExtractor(base_path)
    extractor.run_full_extraction()

if __name__ == "__main__":
    main()
