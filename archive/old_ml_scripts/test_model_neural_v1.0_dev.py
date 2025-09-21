#!/usr/bin/env python3
"""
Neural Network Model Test
========================

Test the trained neural network model.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path

class SimpleRagaClassifier:
    """Simple neural network classifier for raga recognition."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.weights = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        # Hidden layer
        z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
        a1 = np.maximum(0, z1)  # ReLU activation
        
        # Output layer
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        
        # Softmax activation
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return a2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.forward(X)

def test_model():
    """Test the trained neural network model."""
    print("🧪 Testing Neural Network Raga Classifier")
    print("=" * 50)
    
    # Load model
    model_path = "data/processed/simple_raga_classifier/simple_raga_classifier.pkl"
    label_path = "data/processed/simple_raga_classifier/label_mapping.json"
    
    if not os.path.exists(model_path) or not os.path.exists(label_path):
        print("❌ Model files not found!")
        return
    
    print("✅ Loading model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(label_path, 'r') as f:
        label_mapping = json.load(f)
    
    # Create model instance
    model = SimpleRagaClassifier(
        input_dim=model_data['input_dim'],
        hidden_dim=model_data['hidden_dim'],
        num_classes=model_data['num_classes']
    )
    model.weights = model_data['weights']
    
    print(f"📊 Model architecture: {model.input_dim} → {model.hidden_dim} → {model.num_classes}")
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
    
    # Get results (reverse mapping)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    raga_name = reverse_mapping[prediction]
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
        raga_name = reverse_mapping[prediction]
        confidence = probabilities[prediction]
        
        print(f"   Seed {seed}: {raga_name} ({confidence:.2%})")
    
    # Show training history
    if 'training_history' in model_data:
        history = model_data['training_history']
        print(f"\n📈 Training History:")
        final_train = history.get('final_train_accuracy', 'N/A')
        final_val = history.get('final_val_accuracy', 'N/A')
        final_test = history.get('final_test_accuracy', 'N/A')
        
        if isinstance(final_train, (int, float)):
            print(f"   Final Training Accuracy: {final_train:.2%}")
        else:
            print(f"   Final Training Accuracy: {final_train}")
            
        if isinstance(final_val, (int, float)):
            print(f"   Final Validation Accuracy: {final_val:.2%}")
        else:
            print(f"   Final Validation Accuracy: {final_val}")
            
        if isinstance(final_test, (int, float)):
            print(f"   Final Test Accuracy: {final_test:.2%}")
        else:
            print(f"   Final Test Accuracy: {final_test}")

if __name__ == "__main__":
    test_model()
