#!/usr/bin/env python3
"""
Direct Model Test
================

Test the trained model directly without Flask API.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
import librosa

def test_model():
    """Test the trained model directly."""
    print("🧪 Testing Raga Classifier Model Directly")
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
    
    # Find a sample audio file
    sample_files = list(Path("data/raw/saraga_carnatic_melody_synth").glob("**/*.wav"))
    if not sample_files:
        print("❌ No sample audio files found!")
        return
    
    # Test with first sample
    sample_file = sample_files[0]
    print(f"🎵 Testing with: {sample_file}")
    
    try:
        # Load audio
        audio_data, sr = librosa.load(sample_file, sr=22050)
        print(f"📊 Audio loaded: {len(audio_data)} samples, {sr} Hz")
        
        # Extract features
        print("🔧 Extracting features...")
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # Flatten features
        features = np.concatenate([
            mel_spec.flatten(),
            mfcc.flatten(),
            chroma.flatten(),
            spectral_centroid.flatten(),
            spectral_rolloff.flatten(),
            zero_crossing_rate.flatten(),
            [tempo]
        ])
        
        # Ensure same length as training data
        if len(features) > 15601:
            features = features[:15601]
        elif len(features) < 15601:
            features = np.pad(features, (0, 15601 - len(features)), 'constant')
        
        features = features.reshape(1, -1)
        print(f"📊 Feature vector shape: {features.shape}")
        
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
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
