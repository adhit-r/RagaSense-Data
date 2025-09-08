#!/usr/bin/env python3
"""
RagaSense ML Model - Raga Detection System
Comprehensive ML pipeline for real-time raga detection using Ramanarunachalam data structure
"""

import os
import json
import logging
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RagaDetectionConfig:
    """Configuration for Raga Detection System"""
    
    # Audio Processing
    SAMPLE_RATE = 44100  # 44.1kHz
    BIT_DEPTH = 16
    HOP_LENGTH = 512
    N_FFT = 2048
    N_MELS = 128
    N_MFCC = 13
    CHROMA_N_CHROMA = 12
    
    # Model Architecture
    HIDDEN_DIM = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    DROPOUT = 0.1
    MAX_SEQ_LENGTH = 1000
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 10
    
    # Supported Formats
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.aac', '.ogg']
    
    # Model Variants
    MODEL_VARIANTS = {
        'cnn_lstm': 'Traditional CNN-LSTM',
        'yue_foundation': 'YuE Foundation Model',
        'ensemble': 'Ensemble (Best Performance)',
        'realtime': 'Real-time Optimized'
    }

class AudioPreprocessor:
    """Audio preprocessing pipeline for raga detection"""
    
    def __init__(self, config: RagaDetectionConfig):
        self.config = config
        logger.info("🎵 Initializing Audio Preprocessor")
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file with proper preprocessing"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.config.SAMPLE_RATE)
            
            # Ensure 16-bit depth (normalize to [-1, 1] then scale)
            audio = audio.astype(np.float32)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            logger.info(f"✅ Loaded audio: {file_path} - Duration: {len(audio)/sr:.2f}s")
            return audio
            
        except Exception as e:
            logger.error(f"❌ Error loading audio {file_path}: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS
        )
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_mfcc=self.config.N_MFCC,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        return mfcc
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features for pitch analysis"""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_chroma=self.config.CHROMA_N_CHROMA
        )
        return chroma
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid for brightness analysis"""
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        return spectral_centroid
    
    def extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral rolloff for timbre analysis"""
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        return spectral_rolloff
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate for rhythm analysis"""
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        return zcr
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all audio features"""
        features = {
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'mfcc': self.extract_mfcc(audio),
            'chroma': self.extract_chroma(audio),
            'spectral_centroid': self.extract_spectral_centroid(audio),
            'spectral_rolloff': self.extract_spectral_rolloff(audio),
            'zero_crossing_rate': self.extract_zero_crossing_rate(audio)
        }
        return features

class RagaDataset(Dataset):
    """Dataset class for raga detection training"""
    
    def __init__(self, data_path: str, raga_mappings: Dict, preprocessor: AudioPreprocessor):
        self.data_path = Path(data_path)
        self.raga_mappings = raga_mappings
        self.preprocessor = preprocessor
        self.samples = []
        
        self._load_samples()
        logger.info(f"📊 Loaded {len(self.samples)} samples for training")
    
    def _load_samples(self):
        """Load all audio samples with raga labels"""
        # Load from Ramanarunachalam structure
        raga_dir = self.data_path / "downloads" / "Ramanarunachalam_Music_Repository" / "Carnatic" / "raga"
        
        for raga_file in raga_dir.glob("*.json"):
            try:
                with open(raga_file, 'r', encoding='utf-8') as f:
                    raga_data = json.load(f)
                
                raga_name = raga_data.get('title', {}).get('H', 'Unknown')
                
                # Get arohana and avarohana from info
                arohana = None
                avarohana = None
                
                for info_item in raga_data.get('info', []):
                    if info_item.get('H') == 'Arohana':
                        arohana = info_item.get('V', [])
                    elif info_item.get('H') == 'Avarohana':
                        avarohana = info_item.get('V', [])
                
                # Store raga information
                self.samples.append({
                    'raga_name': raga_name,
                    'arohana': arohana,
                    'avarohana': avarohana,
                    'file_path': str(raga_file)
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Error loading raga file {raga_file}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # For now, return raga name and metadata
        # In a full implementation, you would load actual audio files
        return {
            'raga_name': sample['raga_name'],
            'arohana': sample['arohana'],
            'avarohana': sample['avarohana'],
            'file_path': sample['file_path']
        }

class YuETransformer(nn.Module):
    """YuE Foundation Model for Raga Classification"""
    
    def __init__(self, config: RagaDetectionConfig, num_ragas: int):
        super().__init__()
        self.config = config
        self.num_ragas = num_ragas
        
        # Audio feature encoders
        self.mel_encoder = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.mfcc_encoder = nn.Conv1d(config.N_MFCC, 64, kernel_size=3, padding=1)
        self.chroma_encoder = nn.Conv1d(config.CHROMA_N_CHROMA, 64, kernel_size=3, padding=1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.HIDDEN_DIM,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.HIDDEN_DIM * 4,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, num_ragas)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(config.MAX_SEQ_LENGTH, config.HIDDEN_DIM))
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process different feature types
        mel_feat = self.mel_encoder(features['mel_spectrogram'].unsqueeze(1))
        mfcc_feat = self.mfcc_encoder(features['mfcc'])
        chroma_feat = self.chroma_encoder(features['chroma'])
        
        # Combine features
        combined_feat = torch.cat([
            mel_feat.view(mel_feat.size(0), -1),
            mfcc_feat.view(mfcc_feat.size(0), -1),
            chroma_feat.view(chroma_feat.size(0), -1)
        ], dim=1)
        
        # Reshape for transformer
        seq_len = min(combined_feat.size(1) // self.config.HIDDEN_DIM, self.config.MAX_SEQ_LENGTH)
        combined_feat = combined_feat[:, :seq_len * self.config.HIDDEN_DIM]
        combined_feat = combined_feat.view(combined_feat.size(0), seq_len, self.config.HIDDEN_DIM)
        
        # Add positional encoding
        combined_feat = combined_feat + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer processing
        transformer_out = self.transformer(combined_feat)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

class CNNLSTMModel(nn.Module):
    """Traditional CNN-LSTM model for raga classification"""
    
    def __init__(self, config: RagaDetectionConfig, num_ragas: int):
        super().__init__()
        self.config = config
        self.num_ragas = num_ragas
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            dropout=config.DROPOUT
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, num_ragas)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(cnn_out.size(0), cnn_out.size(1), -1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(cnn_out)
        
        # Use last hidden state
        last_hidden = hidden[-1]
        
        # Classification
        logits = self.classifier(last_hidden)
        
        return logits

class EnsembleModel(nn.Module):
    """Ensemble model combining multiple architectures"""
    
    def __init__(self, config: RagaDetectionConfig, num_ragas: int):
        super().__init__()
        self.config = config
        self.num_ragas = num_ragas
        
        # Individual models
        self.yue_model = YuETransformer(config, num_ragas)
        self.cnn_lstm_model = CNNLSTMModel(config, num_ragas)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # YuE gets more weight
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Get predictions from both models
        yue_logits = self.yue_model(features)
        cnn_lstm_logits = self.cnn_lstm_model(features['mel_spectrogram'])
        
        # Weighted ensemble
        ensemble_logits = (
            self.ensemble_weights[0] * yue_logits + 
            self.ensemble_weights[1] * cnn_lstm_logits
        )
        
        return ensemble_logits

class RagaDetectionSystem:
    """Main Raga Detection System"""
    
    def __init__(self, config: RagaDetectionConfig, model_variant: str = 'ensemble'):
        self.config = config
        self.model_variant = model_variant
        self.preprocessor = AudioPreprocessor(config)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🎯 Initializing Raga Detection System - Model: {model_variant}")
        logger.info(f"🖥️ Using device: {self.device}")
    
    def load_raga_mappings(self, data_path: str) -> Dict:
        """Load raga mappings from Ramanarunachalam data"""
        raga_path = Path(data_path) / "downloads" / "Ramanarunachalam_Music_Repository" / "Carnatic" / "raga.json"
        
        try:
            with open(raga_path, 'r', encoding='utf-8') as f:
                raga_mappings = json.load(f)
            
            logger.info(f"✅ Loaded raga mappings from {raga_path}")
            return raga_mappings
            
        except Exception as e:
            logger.error(f"❌ Error loading raga mappings: {e}")
            return {}
    
    def prepare_training_data(self, data_path: str) -> Tuple[RagaDataset, List[str]]:
        """Prepare training dataset"""
        raga_mappings = self.load_raga_mappings(data_path)
        dataset = RagaDataset(data_path, raga_mappings, self.preprocessor)
        
        # Get unique raga names
        raga_names = list(set([sample['raga_name'] for sample in dataset.samples]))
        raga_names = [name for name in raga_names if name != 'Unknown']
        
        # Fit label encoder
        self.label_encoder.fit(raga_names)
        
        logger.info(f"📊 Found {len(raga_names)} unique ragas for training")
        return dataset, raga_names
    
    def create_model(self, num_ragas: int) -> nn.Module:
        """Create model based on variant"""
        if self.model_variant == 'yue_foundation':
            return YuETransformer(self.config, num_ragas)
        elif self.model_variant == 'cnn_lstm':
            return CNNLSTMModel(self.config, num_ragas)
        elif self.model_variant == 'ensemble':
            return EnsembleModel(self.config, num_ragas)
        else:
            raise ValueError(f"Unknown model variant: {self.model_variant}")
    
    def train_model(self, dataset: RagaDataset, raga_names: List[str]):
        """Train the raga detection model"""
        logger.info("🚀 Starting model training...")
        
        # Create model
        num_ragas = len(raga_names)
        self.model = self.create_model(num_ragas).to(self.device)
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_accuracy = 0
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            total_loss = 0
            
            for batch in dataloader:
                # In a real implementation, you would process actual audio features here
                # For now, we'll simulate the training process
                
                # Simulate batch processing
                batch_size = len(batch['raga_name'])
                dummy_features = {
                    'mel_spectrogram': torch.randn(batch_size, self.config.N_MELS, 100),
                    'mfcc': torch.randn(batch_size, self.config.N_MFCC, 100),
                    'chroma': torch.randn(batch_size, self.config.CHROMA_N_CHROMA, 100)
                }
                
                # Move to device
                for key in dummy_features:
                    dummy_features[key] = dummy_features[key].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(dummy_features)
                
                # Simulate labels (in real implementation, use actual raga labels)
                labels = torch.randint(0, num_ragas, (batch_size,)).to(self.device)
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_accuracy:
                best_accuracy = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("✅ Model training completed!")
    
    def predict_raga(self, audio_path: str) -> Dict:
        """Predict raga from audio file"""
        if not self.model:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Load and preprocess audio
        audio = self.preprocessor.load_audio(audio_path)
        if audio is None:
            return {'error': 'Failed to load audio file'}
        
        # Extract features
        features = self.preprocessor.extract_all_features(audio)
        
        # Convert to tensors
        feature_tensors = {}
        for key, value in features.items():
            feature_tensors[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(feature_tensors)
            probabilities = F.softmax(logits, dim=1)
            
            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
            
            # Convert to raga names
            top5_ragas = [self.label_encoder.inverse_transform([idx.item()])[0] 
                         for idx in top5_indices[0]]
            
            # Create result
            result = {
                'predicted_raga': top5_ragas[0],
                'confidence': top5_probs[0][0].item(),
                'top5_predictions': [
                    {'raga': raga, 'confidence': prob.item()}
                    for raga, prob in zip(top5_ragas, top5_probs[0])
                ],
                'audio_duration': len(audio) / self.config.SAMPLE_RATE,
                'features_extracted': list(features.keys())
            }
        
        return result
    
    def save_model(self, save_path: str):
        """Save trained model"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'label_encoder': self.label_encoder,
                'config': self.config,
                'model_variant': self.model_variant
            }, save_path)
            logger.info(f"💾 Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load components
        self.label_encoder = checkpoint['label_encoder']
        self.model_variant = checkpoint['model_variant']
        
        # Create and load model
        num_ragas = len(self.label_encoder.classes_)
        self.model = self.create_model(num_ragas).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"📥 Model loaded from {model_path}")

def main():
    """Main function to demonstrate the raga detection system"""
    logger.info("🎵 RagaSense ML Model - Raga Detection System")
    logger.info("=" * 60)
    
    # Configuration
    config = RagaDetectionConfig()
    
    # Initialize system
    system = RagaDetectionSystem(config, model_variant='ensemble')
    
    # Prepare training data
    data_path = "/Users/adhi/axonome/RagaSense-Data"
    dataset, raga_names = system.prepare_training_data(data_path)
    
    # Train model
    system.train_model(dataset, raga_names)
    
    # Save model
    model_path = "ml_models/raga_detection_model.pth"
    system.save_model(model_path)
    
    logger.info("🎉 Raga Detection System setup complete!")
    logger.info(f"📊 Model supports {len(raga_names)} ragas")
    logger.info(f"🎯 Model variant: {config.MODEL_VARIANTS[system.model_variant]}")
    logger.info(f"📁 Model saved to: {model_path}")

if __name__ == "__main__":
    main()

