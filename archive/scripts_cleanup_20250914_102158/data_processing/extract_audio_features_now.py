#!/usr/bin/env python3
"""
Extract Audio Features - Immediate Action
========================================

This script extracts real audio features from the Saraga datasets
that were just processed, creating ML-ready feature vectors.
"""

import json
import zipfile
import logging
import librosa
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """Extract audio features from Saraga datasets."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.sample_rate = 22050  # Standard for ML
        self.max_duration = 30  # seconds
        self.n_mfcc = 13
        self.n_mels = 128
        self.n_chroma = 12
        
        # Create output directory
        self.output_dir = self.base_path / "data" / "ml_ready" / "audio_features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎵 Audio Feature Extractor initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎼 Sample rate: {self.sample_rate}Hz")
        logger.info(f"🎼 Max duration: {self.max_duration}s")
    
    def extract_features_from_audio(self, audio_data: bytes, file_path: str) -> dict:
        """Extract features from audio data."""
        try:
            # Load audio from bytes
            y, sr = librosa.load(io.BytesIO(audio_data), sr=self.sample_rate)
            
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
                'file_path': file_path,
                'duration': len(y) / sr,
                'sample_rate': sr,
                'extracted_at': datetime.now().isoformat()
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features from {file_path}: {e}")
            return {}
    
    def process_saraga_dataset(self, zip_path: Path, tradition: str) -> dict:
        """Process a Saraga dataset and extract audio features."""
        logger.info(f"🎵 Processing {tradition} dataset: {zip_path.name}")
        
        if not zip_path.exists():
            logger.error(f"❌ Dataset not found: {zip_path}")
            return {}
        
        # Statistics
        stats = {
            "audio_files_processed": 0,
            "features_extracted": 0,
            "errors": 0,
            "processing_time": 0
        }
        
        start_time = datetime.now()
        extracted_features = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of audio files
                audio_files = [f for f in zip_ref.namelist() 
                             if f.endswith(('.wav', '.mp3', '.flac')) and not f.startswith('__MACOSX')]
                
                logger.info(f"📊 Found {len(audio_files)} audio files")
                
                # Process audio files (limit to first 50 for now)
                for i, audio_file in enumerate(audio_files[:50]):
                    try:
                        # Read audio data
                        audio_data = zip_ref.read(audio_file)
                        
                        # Extract features
                        features = self.extract_features_from_audio(audio_data, audio_file)
                        
                        if features:
                            features['tradition'] = tradition
                            features['source'] = 'saraga1.5'
                            extracted_features.append(features)
                            stats["features_extracted"] += 1
                        
                        stats["audio_files_processed"] += 1
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"🎵 Processed {i + 1}/{min(50, len(audio_files))} audio files")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing {audio_file}: {e}")
                        stats["errors"] += 1
                        continue
            
            stats["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            # Save extracted features
            output_file = self.output_dir / f"{tradition.lower()}_audio_features.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "metadata": {
                        "tradition": tradition,
                        "source": "saraga1.5",
                        "extracted_at": datetime.now().isoformat(),
                        "total_audio_files": len(audio_files),
                        "features_extracted": len(extracted_features)
                    },
                    "statistics": stats,
                    "features": extracted_features
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved audio features to: {output_file}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to process {zip_path}: {e}")
            return {}
    
    def process_melody_synth_dataset(self) -> dict:
        """Process the Melody Synth dataset (already extracted)."""
        logger.info("🎵 Processing Melody Synth dataset...")
        
        melody_synth_path = self.base_path / "data" / "organized_raw" / "saraga_carnatic_melody_synth" / "Saraga-Carnatic-Melody-Synth"
        
        if not melody_synth_path.exists():
            logger.error(f"❌ Melody Synth dataset not found: {melody_synth_path}")
            return {}
        
        # Find audio files
        audio_files = list(melody_synth_path.rglob("*.wav")) + list(melody_synth_path.rglob("*.mp3"))
        
        logger.info(f"📊 Found {len(audio_files)} audio files in Melody Synth")
        
        # Statistics
        stats = {
            "audio_files_processed": 0,
            "features_extracted": 0,
            "errors": 0,
            "processing_time": 0
        }
        
        start_time = datetime.now()
        extracted_features = []
        
        # Process audio files (limit to first 50 for now)
        for i, audio_file in enumerate(audio_files[:50]):
            try:
                # Load audio file
                y, sr = librosa.load(audio_file, sr=self.sample_rate)
                
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
                    'file_path': str(audio_file.relative_to(self.base_path)),
                    'duration': len(y) / sr,
                    'sample_rate': sr,
                    'extracted_at': datetime.now().isoformat()
                }
                
                features['tradition'] = 'Carnatic'
                features['source'] = 'melody_synth'
                extracted_features.append(features)
                
                stats["audio_files_processed"] += 1
                stats["features_extracted"] += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"🎵 Processed {i + 1}/{min(50, len(audio_files))} audio files")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing {audio_file}: {e}")
                stats["errors"] += 1
                continue
        
        stats["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        # Save extracted features
        output_file = self.output_dir / "melody_synth_audio_features.json"
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "tradition": "Carnatic",
                    "source": "melody_synth",
                    "extracted_at": datetime.now().isoformat(),
                    "total_audio_files": len(audio_files),
                    "features_extracted": len(extracted_features)
                },
                "statistics": stats,
                "features": extracted_features
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved Melody Synth features to: {output_file}")
        
        return stats

def main():
    """Main function to extract audio features from all datasets."""
    logger.info("🚀 Starting audio feature extraction...")
    
    extractor = AudioFeatureExtractor()
    
    base_path = Path(__file__).parent
    saraga_path = base_path / "data" / "organized_raw" / "saraga_datasets"
    
    # Process Melody Synth dataset (already extracted)
    melody_synth_stats = extractor.process_melody_synth_dataset()
    
    # Process Saraga Carnatic dataset
    carnatic_zip = saraga_path / "carnatic" / "saraga1.5_carnatic.zip"
    carnatic_stats = extractor.process_saraga_dataset(carnatic_zip, "Carnatic")
    
    # Process Saraga Hindustani dataset
    hindustani_zip = saraga_path / "hindustani" / "saraga1.5_hindustani.zip"
    hindustani_stats = extractor.process_saraga_dataset(hindustani_zip, "Hindustani")
    
    # Create summary report
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "melody_synth": melody_synth_stats,
        "carnatic": carnatic_stats,
        "hindustani": hindustani_stats,
        "total_features_extracted": (
            melody_synth_stats.get("features_extracted", 0) +
            carnatic_stats.get("features_extracted", 0) +
            hindustani_stats.get("features_extracted", 0)
        ),
        "total_audio_files_processed": (
            melody_synth_stats.get("audio_files_processed", 0) +
            carnatic_stats.get("audio_files_processed", 0) +
            hindustani_stats.get("audio_files_processed", 0)
        )
    }
    
    # Save summary
    summary_file = base_path / "data" / "ml_ready" / "audio_feature_extraction_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Summary saved to: {summary_file}")
    
    # Log results
    logger.info("🎉 Audio feature extraction completed!")
    logger.info(f"📊 Total features extracted: {summary['total_features_extracted']}")
    logger.info(f"📊 Total audio files processed: {summary['total_audio_files_processed']}")
    
    return summary

if __name__ == "__main__":
    import io  # Import io for BytesIO
    main()
