#!/usr/bin/env python3
"""
Scale Up Audio Processing for Comprehensive Dataset Creation
===========================================================

This script processes ALL available audio files to create a comprehensive dataset:
- Processes all 148+ audio files from both traditions
- Extracts optimized audio features for ML
- Creates comprehensive metadata and annotations
- Generates unified dataset for research and ML

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import json
import librosa
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scale_up_audio_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

class ComprehensiveAudioProcessor:
    """Process all available audio files for comprehensive dataset creation"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "unified" / "comprehensive_audio_features"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Optimized parameters for comprehensive processing
        self.sample_rate = 22050  # Reduced from 44100 for efficiency
        self.max_duration = 30    # 30 seconds max per file
        self.n_mfcc = 13         # Standard MFCC count
        self.n_mels = 128        # Reduced from 256 for efficiency
        self.n_chroma = 12       # Standard chroma features
        
        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.total_files = 0
        
    def find_all_audio_files(self):
        """Find all audio files in the dataset"""
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
        audio_files = []
        
        # Search in all relevant directories
        search_dirs = [
            self.base_path / "carnatic" / "audio",
            self.base_path / "hindustani" / "audio",
            self.base_path / "organized_raw" / "Ramanarunachalam_Music_Repository" / "Carnatic" / "audio",
            self.base_path / "organized_raw" / "Ramanarunachalam_Music_Repository" / "Hindustani" / "audio",
            self.base_path / "youtube_dataset"
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in audio_extensions:
                    audio_files.extend(search_dir.glob(f"*{ext}"))
        
        # Remove duplicates and sort
        audio_files = list(set(audio_files))
        audio_files.sort()
        
        self.total_files = len(audio_files)
        logger.info(f"Found {self.total_files} audio files to process")
        
        return audio_files
    
    def extract_audio_features(self, audio_file: Path):
        """Extract comprehensive audio features from a single file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=self.sample_rate, duration=self.max_duration)
            
            # Extract comprehensive features
            features = {}
            
            # 1. MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features['mfcc'] = mfcc
            
            # 2. Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            features['mel_spectrogram'] = mel_spec
            
            # 3. Chroma features
            chroma = librosa.feature.chroma(y=y, sr=sr, n_chroma=self.n_chroma)
            features['chroma'] = chroma
            
            # 4. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            features['spectral_centroid'] = spectral_centroids
            features['spectral_rolloff'] = spectral_rolloff
            features['spectral_bandwidth'] = spectral_bandwidth
            features['zero_crossing_rate'] = zero_crossing_rate
            
            # 5. Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_frames'] = beats
            
            # 6. Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz'] = tonnetz
            
            # 7. RMS energy
            rms = librosa.feature.rms(y=y)
            features['rms_energy'] = rms
            
            # 8. Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast'] = spectral_contrast
            
            # 9. Poly features
            poly_features = librosa.feature.poly_features(y=y, sr=sr)
            features['poly_features'] = poly_features
            
            # 10. Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            features['harmonic_energy'] = np.mean(y_harmonic**2)
            features['percussive_energy'] = np.mean(y_percussive**2)
            
            # Create feature summary
            feature_summary = {
                'file_path': str(audio_file),
                'file_name': audio_file.name,
                'tradition': self._determine_tradition(audio_file),
                'duration': len(y) / sr,
                'sample_rate': sr,
                'features': features,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            return feature_summary
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            return None
    
    def _determine_tradition(self, audio_file: Path):
        """Determine tradition based on file path"""
        path_str = str(audio_file).lower()
        if 'carnatic' in path_str:
            return 'Carnatic'
        elif 'hindustani' in path_str:
            return 'Hindustani'
        elif 'youtube' in path_str:
            return 'YouTube'  # Could be either tradition
        else:
            return 'Unknown'
    
    def process_audio_file(self, audio_file: Path):
        """Process a single audio file and return results"""
        try:
            features = self.extract_audio_features(audio_file)
            if features:
                self.processed_count += 1
                logger.info(f"Processed {audio_file.name} ({self.processed_count}/{self.total_files})")
                return features
            else:
                self.error_count += 1
                return None
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to process {audio_file}: {e}")
            return None
    
    def process_all_audio_files(self, max_workers: int = None):
        """Process all audio files using parallel processing"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers max
        
        audio_files = self.find_all_audio_files()
        
        if not audio_files:
            logger.warning("No audio files found to process")
            return []
        
        logger.info(f"Starting parallel processing with {max_workers} workers")
        
        all_features = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_audio_file, audio_file): audio_file 
                for audio_file in audio_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        all_features.append(result)
                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")
                    self.error_count += 1
        
        logger.info(f"Processing complete: {self.processed_count} successful, {self.error_count} errors")
        return all_features
    
    def save_comprehensive_features(self, all_features: list):
        """Save comprehensive audio features to JSON"""
        if not all_features:
            logger.warning("No features to save")
            return
        
        # Save individual tradition files
        carnatic_features = [f for f in all_features if f['tradition'] == 'Carnatic']
        hindustani_features = [f for f in all_features if f['tradition'] == 'Hindustani']
        youtube_features = [f for f in all_features if f['tradition'] == 'YouTube']
        
        # Save tradition-specific files
        if carnatic_features:
            carnatic_file = self.output_path / "comprehensive_carnatic_features.json"
            with open(carnatic_file, 'w') as f:
                json.dump(carnatic_features, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved {len(carnatic_features)} Carnatic features to {carnatic_file}")
        
        if hindustani_features:
            hindustani_file = self.output_path / "comprehensive_hindustani_features.json"
            with open(hindustani_file, 'w') as f:
                json.dump(hindustani_features, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved {len(hindustani_features)} Hindustani features to {hindustani_file}")
        
        if youtube_features:
            youtube_file = self.output_path / "comprehensive_youtube_features.json"
            with open(youtube_file, 'w') as f:
                json.dump(youtube_features, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved {len(youtube_features)} YouTube features to {youtube_file}")
        
        # Save comprehensive combined file
        comprehensive_file = self.output_path / "comprehensive_all_audio_features.json"
        with open(comprehensive_file, 'w') as f:
            json.dump(all_features, f, cls=NumpyEncoder, indent=2)
        logger.info(f"Saved {len(all_features)} total features to {comprehensive_file}")
        
        # Create summary
        self.create_processing_summary(all_features)
    
    def create_processing_summary(self, all_features: list):
        """Create comprehensive processing summary"""
        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_files_processed': len(all_features),
            'total_files_found': self.total_files,
            'success_rate': len(all_features) / self.total_files if self.total_files > 0 else 0,
            'error_count': self.error_count,
            'tradition_breakdown': {},
            'feature_statistics': {},
            'processing_parameters': {
                'sample_rate': self.sample_rate,
                'max_duration': self.max_duration,
                'n_mfcc': self.n_mfcc,
                'n_mels': self.n_mels,
                'n_chroma': self.n_chroma
            }
        }
        
        # Tradition breakdown
        for feature in all_features:
            tradition = feature['tradition']
            if tradition not in summary['tradition_breakdown']:
                summary['tradition_breakdown'][tradition] = 0
            summary['tradition_breakdown'][tradition] += 1
        
        # Feature statistics
        if all_features:
            durations = [f['duration'] for f in all_features]
            summary['feature_statistics'] = {
                'total_duration_minutes': sum(durations) / 60,
                'average_duration_seconds': np.mean(durations),
                'min_duration_seconds': np.min(durations),
                'max_duration_seconds': np.max(durations),
                'total_audio_files': len(all_features)
            }
        
        # Save summary
        summary_file = self.output_path / "comprehensive_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Created processing summary: {summary_file}")
        return summary
    
    def run_comprehensive_processing(self):
        """Run the complete comprehensive audio processing pipeline"""
        logger.info("Starting comprehensive audio processing...")
        
        try:
            # Process all audio files
            all_features = self.process_all_audio_files()
            
            # Save results
            self.save_comprehensive_features(all_features)
            
            logger.info("Comprehensive audio processing completed successfully!")
            return all_features
            
        except Exception as e:
            logger.error(f"Error in comprehensive processing: {e}")
            raise

def main():
    """Main function"""
    print("🎵 RagaSense Comprehensive Audio Processing")
    print("=" * 50)
    
    processor = ComprehensiveAudioProcessor()
    all_features = processor.run_comprehensive_processing()
    
    print(f"\n✅ Comprehensive Processing Complete!")
    print(f"📁 Total files processed: {len(all_features)}")
    print(f"📁 Success rate: {processor.processed_count}/{processor.total_files}")
    print(f"📁 Output directory: {processor.output_path}")

if __name__ == "__main__":
    main()
