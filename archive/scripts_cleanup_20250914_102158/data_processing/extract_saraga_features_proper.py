#!/usr/bin/env python3
"""
Extract Features from Real Saraga Dataset
========================================

This script extracts audio features from the complete Saraga dataset:
- Carnatic: 249 recordings, 96 unique ragas, 52.7 hours
- Hindustani: 108 recordings, 61 unique ragas, 43.6 hours
- Total: 357 recordings, 157 unique ragas, 96.3 hours

This is the REAL dataset we should be using for raga detection!

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import json
import zipfile
import librosa
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_saraga_features_proper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

class SaragaFeatureExtractor:
    """Extract features from the complete Saraga dataset"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.saraga_path = self.base_path / "01_source" / "saraga"
        self.output_path = self.base_path / "03_processed" / "audio_features"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Audio processing parameters
        self.sample_rate = 22050
        self.max_duration = 60  # 1 minute max per file
        self.n_mfcc = 13
        self.n_mels = 128
        self.n_chroma = 12
        
        # Results storage
        self.results = {
            'extraction_date': datetime.now().isoformat(),
            'carnatic': {'processed': 0, 'errors': 0, 'features': []},
            'hindustani': {'processed': 0, 'errors': 0, 'features': []},
            'total_processed': 0,
            'total_errors': 0
        }
    
    def extract_zip_contents(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract zip file contents"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return False
    
    def find_audio_files(self, directory: Path) -> list:
        """Find all audio files in directory"""
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(directory.rglob(f'*{ext}'))
        
        return audio_files
    
    def extract_features_from_audio(self, audio_path: Path) -> dict:
        """Extract audio features from a single file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_duration)
            
            if len(y) == 0:
                logger.warning(f"Skipping {audio_path.name}: Audio file is empty or too short")
                return None
            
            features = {}
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features['mfcc'] = mfccs.mean(axis=1)
            
            # Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            features['mel_spectrogram'] = librosa.power_to_db(mel_spectrogram, ref=np.max).mean(axis=1)
            
            # Chroma features
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.n_chroma)
                features['chroma'] = chroma.mean(axis=1)
            except Exception as e:
                logger.warning(f"Chroma extraction failed for {audio_path.name}: {e}")
                features['chroma'] = np.zeros(self.n_chroma)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = spectral_centroids.mean()
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff'] = spectral_rolloff.mean()
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth'] = spectral_bandwidth.mean()
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate'] = zcr.mean()
            
            # RMS energy
            rms = librosa.feature.rms(y=y)
            features['rms_energy'] = rms.mean()
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz'] = tonnetz.mean(axis=1)
            
            # Poly features
            poly_features = librosa.feature.poly_features(y=y, sr=sr)
            features['poly_features'] = poly_features.mean(axis=1)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast'] = spectral_contrast.mean(axis=1)
            
            # Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            features['spectral_flatness'] = spectral_flatness.mean()
            
            return {
                'file_path': str(audio_path),
                'file_name': audio_path.name,
                'features': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in features.items()},
                'duration_seconds': librosa.get_duration(y=y, sr=sr),
                'sample_rate': sr
            }
            
        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {e}")
            return None
    
    def process_tradition(self, tradition: str) -> dict:
        """Process a tradition (Carnatic or Hindustani)"""
        logger.info(f"Processing {tradition} tradition...")
        
        zip_path = self.saraga_path / tradition / f"saraga1.5_{tradition}.zip"
        if not zip_path.exists():
            logger.error(f"Zip file not found: {zip_path}")
            return {'processed': 0, 'errors': 1, 'features': []}
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract zip file
            if not self.extract_zip_contents(zip_path, temp_path):
                return {'processed': 0, 'errors': 1, 'features': []}
            
            # Find audio files
            audio_files = self.find_audio_files(temp_path)
            logger.info(f"Found {len(audio_files)} audio files in {tradition}")
            
            if not audio_files:
                logger.warning(f"No audio files found in {tradition}")
                return {'processed': 0, 'errors': 0, 'features': []}
            
            # Process audio files in parallel
            features_list = []
            processed_count = 0
            error_count = 0
            
            with ProcessPoolExecutor(max_workers=4) as executor:
                future_to_file = {executor.submit(self.extract_features_from_audio, audio_file): audio_file 
                                for audio_file in audio_files}
                
                for future in as_completed(future_to_file):
                    audio_file = future_to_file[future]
                    try:
                        result = future.result()
                        if result is not None:
                            features_list.append(result)
                            processed_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        logger.error(f"Error processing {audio_file}: {e}")
                        error_count += 1
            
            logger.info(f"{tradition} processing complete: {processed_count} processed, {error_count} errors")
            
            return {
                'processed': processed_count,
                'errors': error_count,
                'features': features_list
            }
    
    def save_features(self, tradition: str, features: list):
        """Save extracted features to JSON file"""
        output_file = self.output_path / f"saraga_{tradition}_features.json"
        
        with open(output_file, 'w') as f:
            json.dump(features, f, cls=NumpyEncoder, indent=2)
        
        logger.info(f"Saved {len(features)} {tradition} features to {output_file}")
    
    def run_extraction(self):
        """Run the complete feature extraction process"""
        logger.info("Starting Saraga feature extraction...")
        
        try:
            # Process Carnatic tradition
            carnatic_results = self.process_tradition('carnatic')
            self.results['carnatic'] = carnatic_results
            
            if carnatic_results['features']:
                self.save_features('carnatic', carnatic_results['features'])
            
            # Process Hindustani tradition
            hindustani_results = self.process_tradition('hindustani')
            self.results['hindustani'] = hindustani_results
            
            if hindustani_results['features']:
                self.save_features('hindustani', hindustani_results['features'])
            
            # Calculate totals
            self.results['total_processed'] = carnatic_results['processed'] + hindustani_results['processed']
            self.results['total_errors'] = carnatic_results['errors'] + hindustani_results['errors']
            
            # Save summary
            summary_file = self.output_path / "saraga_extraction_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.results, f, cls=NumpyEncoder, indent=2)
            
            logger.info("Saraga feature extraction completed successfully!")
            logger.info(f"Total processed: {self.results['total_processed']}")
            logger.info(f"Total errors: {self.results['total_errors']}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise

def main():
    """Main function"""
    print("🎵 Saraga Feature Extraction - REAL Dataset")
    print("=" * 50)
    print("Processing the complete Saraga dataset:")
    print("• Carnatic: 249 recordings, 96 ragas, 52.7 hours")
    print("• Hindustani: 108 recordings, 61 ragas, 43.6 hours")
    print("• Total: 357 recordings, 157 ragas, 96.3 hours")
    print("=" * 50)
    
    extractor = SaragaFeatureExtractor()
    results = extractor.run_extraction()
    
    print(f"\n✅ Saraga Feature Extraction Complete!")
    print(f"📊 Carnatic: {results['carnatic']['processed']} files processed")
    print(f"📊 Hindustani: {results['hindustani']['processed']} files processed")
    print(f"📊 Total: {results['total_processed']} files processed")
    print(f"❌ Errors: {results['total_errors']}")
    print(f"📁 Features saved to: data/03_processed/audio_features/")

if __name__ == "__main__":
    main()
