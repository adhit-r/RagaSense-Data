#!/usr/bin/env python3
"""
Simple Model Test
================

Test the trained model with mock features (same as training).
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path

def test_model():
    """Test the trained model with mock features."""
    print("🧪 Testing Raga Classifier Model (Mock Features)")
    print("=" * 50)
    
    # Load model
    model_path = "data/processed/simple_raga_classifier/simple_raga_classifier.pkl"
    label_path = "data/processed/simple_raga_classifier/label_mapping.json"
    
    if not os.path.exists(model_path) or not os.path.exists(label_path):
        print("❌ Model files not found!")
        return
    
    print("✅ Loading model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(label_path, 'r') as f:
        label_mapping = json.load(f)
    
    print(f"📊 Model supports {len(label_mapping)} ragas: {list(label_mapping.keys())}")
    
    # Create mock features (same as training)
    print("🔧 Creating mock features...")
    
    # Fixed dimensions (same as training)
    n_mels = 128
    n_mfcc = 13
    n_chroma = 12
    fixed_mel_frames = 100  # Same as training
    fixed_spectral_frames = 100  # Same as training
    
    # Generate mock features
    mel_spectrogram = np.random.rand(n_mels, fixed_mel_frames).astype(np.float32)
    mfcc = np.random.rand(n_mfcc, fixed_mel_frames).astype(np.float32)
    chroma = np.random.rand(n_chroma, fixed_mel_frames).astype(np.float32)
    spectral_centroid = np.random.rand(fixed_spectral_frames).astype(np.float32)
    spectral_rolloff = np.random.rand(fixed_spectral_frames).astype(np.float32)
    zero_crossing_rate = np.random.rand(fixed_spectral_frames).astype(np.float32)
    tempo = np.random.uniform(60, 180)
    
    # Create feature vector (same as training)
    feature_vector = np.concatenate([
        mel_spectrogram.flatten(),
        mfcc.flatten(),
        chroma.flatten(),
        spectral_centroid,
        spectral_rolloff,
        zero_crossing_rate,
        [tempo]
    ])
    
    # Ensure correct shape
    features = feature_vector.reshape(1, -1)
    print(f"📊 Feature vector shape: {features.shape}")
    print(f"📊 Expected shape: (1, 15601)")
    
    # Make prediction
    print("🎯 Making prediction...")
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get results
    raga_name = label_mapping[str(prediction)]
    confidence = probabilities[prediction]
    
    print("\n🎉 PREDICTION RESULTS:")
    print(f"   Predicted Raga: {raga_name}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Prediction Class: {prediction}")
    
    print("\n📊 All Confidence Scores:")
    for i, (raga, class_id) in enumerate(label_mapping.items()):
        print(f"   {raga}: {probabilities[i]:.2%}")
    
    print("\n✅ Model test successful!")
    
    # Test with different random seeds
    print("\n🔄 Testing with different random seeds...")
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        
        # Generate new mock features
        mel_spectrogram = np.random.rand(n_mels, fixed_mel_frames).astype(np.float32)
        mfcc = np.random.rand(n_mfcc, fixed_mel_frames).astype(np.float32)
        chroma = np.random.rand(n_chroma, fixed_mel_frames).astype(np.float32)
        spectral_centroid = np.random.rand(fixed_spectral_frames).astype(np.float32)
        spectral_rolloff = np.random.rand(fixed_spectral_frames).astype(np.float32)
        zero_crossing_rate = np.random.rand(fixed_spectral_frames).astype(np.float32)
        tempo = np.random.uniform(60, 180)
        
        feature_vector = np.concatenate([
            mel_spectrogram.flatten(),
            mfcc.flatten(),
            chroma.flatten(),
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate,
            [tempo]
        ])
        
        features = feature_vector.reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        raga_name = label_mapping[str(prediction)]
        confidence = probabilities[prediction]
        
        print(f"   Seed {seed}: {raga_name} ({confidence:.2%})")

if __name__ == "__main__":
    test_model()
