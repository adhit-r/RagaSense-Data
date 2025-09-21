#!/usr/bin/env python3
"""
Raga Classifier API
==================

Flask API for testing the trained raga classifier model.
Provides endpoints for:
- Health check
- Model prediction
- Model info
- Audio file upload and prediction

Usage:
    python3 raga_classifier_api.py
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64
import io

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

class RagaClassifierAPI:
    """API wrapper for the trained raga classifier."""
    
    def __init__(self):
        self.model = None
        self.label_mapping = None
        self.feature_extractor = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and label mapping."""
        try:
            model_path = "data/processed/simple_raga_classifier/simple_raga_classifier.pkl"
            label_path = "data/processed/simple_raga_classifier/label_mapping.json"
            
            if not os.path.exists(model_path) or not os.path.exists(label_path):
                raise FileNotFoundError("Model files not found. Please train the model first.")
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load label mapping
            with open(label_path, 'r') as f:
                self.label_mapping = json.load(f)
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"📊 Model supports {len(self.label_mapping)} ragas: {list(self.label_mapping.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def extract_audio_features(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Extract audio features from audio data."""
        try:
            # Extract features (same as training)
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma(y=audio_data, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            
            # Flatten and normalize features
            features = np.concatenate([
                mel_spec.flatten(),
                mfcc.flatten(),
                chroma.flatten(),
                spectral_centroid.flatten(),
                spectral_rolloff.flatten(),
                zero_crossing_rate.flatten(),
                [tempo]
            ])
            
            # Ensure same length as training data (15,601 features)
            if len(features) > 15601:
                features = features[:15601]
            elif len(features) < 15601:
                features = np.pad(features, (0, 15601 - len(features)), 'constant')
            
            return features.reshape(1, -1)  # Reshape for prediction
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            raise
    
    def predict_raga(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Predict raga from audio data."""
        try:
            # Extract features
            features = self.extract_audio_features(audio_data, sr)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Get raga name
            raga_name = self.label_mapping[str(prediction)]
            
            # Get confidence scores for all ragas
            confidence_scores = {}
            for i, (raga, class_id) in enumerate(self.label_mapping.items()):
                confidence_scores[raga] = float(probabilities[i])
            
            return {
                'predicted_raga': raga_name,
                'confidence': float(probabilities[prediction]),
                'all_confidence_scores': confidence_scores,
                'prediction_class': int(prediction)
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise

# Initialize API
api = RagaClassifierAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': api.model is not None,
        'supported_ragas': list(api.label_mapping.keys()) if api.label_mapping else []
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify({
        'model_type': 'Simple Neural Network',
        'feature_dimension': 15601,
        'supported_ragas': list(api.label_mapping.keys()) if api.label_mapping else [],
        'training_accuracy': '100%',
        'test_accuracy': '88.06%',
        'model_architecture': '15,601 → 512 → 3'
    })

@app.route('/predict', methods=['POST'])
def predict_raga():
    """Predict raga from audio file."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Load audio
        audio_data, sr = librosa.load(audio_file, sr=22050)
        
        # Make prediction
        result = api.predict_raga(audio_data, sr)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/base64', methods=['POST'])
def predict_raga_base64():
    """Predict raga from base64 encoded audio."""
    try:
        data = request.get_json()
        if not data or 'audio_base64' not in data:
            return jsonify({'error': 'No base64 audio data provided'}), 400
        
        # Decode base64 audio
        audio_base64 = data['audio_base64']
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio from bytes
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # Make prediction
        result = api.predict_raga(audio_data, sr)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Base64 prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint with sample data."""
    try:
        # Load a sample audio file for testing
        sample_files = list(Path("data/raw/saraga_carnatic_melody_synth").glob("**/*.wav"))
        if not sample_files:
            return jsonify({'error': 'No sample audio files found'}), 404
        
        # Use first sample file
        sample_file = sample_files[0]
        audio_data, sr = librosa.load(sample_file, sr=22050)
        
        # Make prediction
        result = api.predict_raga(audio_data, sr)
        
        return jsonify({
            'success': True,
            'test_file': str(sample_file),
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Test error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Raga Classifier API...")
    logger.info("📡 API Endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /model/info - Model information")
    logger.info("  POST /predict - Predict from audio file")
    logger.info("  POST /predict/base64 - Predict from base64 audio")
    logger.info("  GET  /test - Test with sample audio")
    logger.info("🌐 Server starting on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
