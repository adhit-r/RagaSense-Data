#!/usr/bin/env python3
"""
Simple Raga Classifier Training Script
=====================================

This script trains a simple raga classifier using the extracted audio features
from the Saraga dataset. It demonstrates how to use our current data for ML training.

Features:
- Uses extracted audio features (15,601 dimensions)
- Simple neural network classifier
- Cross-validation evaluation
- Model persistence
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleRagaClassifier:
    """Simple neural network classifier for raga recognition."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Simple neural network weights (mock implementation)
        self.weights = {
            'W1': np.random.randn(input_dim, hidden_dim) * 0.01,
            'b1': np.zeros((1, hidden_dim)),
            'W2': np.random.randn(hidden_dim, num_classes) * 0.01,
            'b2': np.zeros((1, num_classes))
        }
        
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass through the network."""
        # First layer
        z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
        a1 = self.relu(z1)
        
        # Second layer
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self.softmax(z2)
        
        # Cache for backpropagation
        cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }
        
        return a2, cache
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        m = y_pred.shape[0]
        # Avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def backward(self, cache: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass (gradient computation)."""
        m = cache['X'].shape[0]
        
        # Output layer gradients
        dz2 = cache['a2'] - y_true
        dW2 = np.dot(cache['a1'].T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.weights['W2'].T)
        dz1 = da1 * (cache['a1'] > 0)  # ReLU derivative
        dW1 = np.dot(cache['X'].T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
    
    def update_weights(self, gradients: Dict[str, np.ndarray], learning_rate: float = 0.01):
        """Update weights using gradient descent."""
        self.weights['W1'] -= learning_rate * gradients['dW1']
        self.weights['b1'] -= learning_rate * gradients['db1']
        self.weights['W2'] -= learning_rate * gradients['dW2']
        self.weights['b2'] -= learning_rate * gradients['db2']
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, learning_rate: float = 0.01,
              batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the classifier."""
        logger.info(f"🚀 Starting training for {epochs} epochs")
        logger.info(f"📊 Training samples: {X_train.shape[0]}")
        logger.info(f"📊 Validation samples: {X_val.shape[0]}")
        logger.info(f"📊 Feature dimension: {X_train.shape[1]}")
        logger.info(f"📊 Number of classes: {y_train.shape[1]}")
        
        for epoch in range(epochs):
            # Training
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Mini-batch training
            for i in range(0, X_train.shape[0], batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward pass
                y_pred, cache = self.forward(batch_X)
                
                # Compute loss
                loss = self.compute_loss(y_pred, batch_y)
                train_loss += loss
                
                # Compute accuracy
                predictions = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                train_correct += np.sum(predictions == true_labels)
                train_total += batch_y.shape[0]
                
                # Backward pass
                gradients = self.backward(cache, batch_y)
                self.update_weights(gradients, learning_rate)
            
            # Validation
            val_pred, _ = self.forward(X_val)
            val_loss = self.compute_loss(val_pred, y_val)
            val_predictions = np.argmax(val_pred, axis=1)
            val_true = np.argmax(y_val, axis=1)
            val_accuracy = np.mean(val_predictions == val_true)
            
            # Store metrics
            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / (X_train.shape[0] // batch_size)
            
            self.training_history['loss'].append(avg_train_loss)
            self.training_history['accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch:3d}: "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {train_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_accuracy:.4f}")
        
        logger.info("✅ Training completed!")
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model."""
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)
        
        accuracy = np.mean(y_pred == y_true)
        
        # Per-class metrics
        class_metrics = {}
        for i in range(self.num_classes):
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
                class_metrics[f'class_{i}_accuracy'] = class_accuracy
        
        return {
            'accuracy': accuracy,
            'class_metrics': class_metrics
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'weights': self.weights,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_dim': self.hidden_dim,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            input_dim=model_data['input_dim'],
            num_classes=model_data['num_classes'],
            hidden_dim=model_data['hidden_dim']
        )
        
        model.weights = model_data['weights']
        model.training_history = model_data['training_history']
        
        logger.info(f"📂 Model loaded from: {filepath}")
        return model

def load_saraga_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the Saraga ML dataset."""
    dataset_path = Path("data/processed/saraga_audio_features/saraga_ml_dataset.pkl")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    metadata = data['metadata']
    
    logger.info(f"📊 Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"📊 Unique ragas: {len(set(y))}")
    
    return X, y, metadata

def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Encode string labels to one-hot vectors."""
    unique_labels = sorted(list(set(y)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert to one-hot encoding
    y_encoded = np.zeros((len(y), len(unique_labels)))
    for i, label in enumerate(y):
        y_encoded[i, label_to_idx[label]] = 1
    
    logger.info(f"📊 Encoded {len(unique_labels)} unique ragas")
    logger.info(f"📊 Raga labels: {unique_labels}")
    
    return y_encoded, label_to_idx

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
    """Split data into train/validation/test sets."""
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    # Calculate split sizes
    test_end = int(n_samples * test_size)
    val_end = test_end + int(n_samples * val_size)
    
    # Split indices
    test_indices = indices[:test_end]
    val_indices = indices[test_end:val_end]
    train_indices = indices[val_end:]
    
    # Split data
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    logger.info(f"📊 Data split:")
    logger.info(f"   Train: {X_train.shape[0]} samples")
    logger.info(f"   Validation: {X_val.shape[0]} samples")
    logger.info(f"   Test: {X_test.shape[0]} samples")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test)

def main():
    """Main training function."""
    print("🎵 Simple Raga Classifier Training")
    print("=" * 50)
    
    try:
        # Load dataset
        logger.info("📂 Loading Saraga dataset...")
        X, y_strings, metadata = load_saraga_dataset()
        
        # Encode labels
        logger.info("🔤 Encoding labels...")
        y, label_to_idx = encode_labels(y_strings)
        
        # Split data
        logger.info("✂️ Splitting data...")
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
        
        # Normalize features
        logger.info("📏 Normalizing features...")
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8  # Avoid division by zero
        
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
        
        # Create and train model
        logger.info("🤖 Creating classifier...")
        classifier = SimpleRagaClassifier(
            input_dim=X.shape[1],
            num_classes=len(label_to_idx),
            hidden_dim=512
        )
        
        # Train model
        training_history = classifier.train(
            X_train, y_train, X_val, y_val,
            epochs=50, learning_rate=0.01, batch_size=16
        )
        
        # Evaluate on test set
        logger.info("📊 Evaluating on test set...")
        test_metrics = classifier.evaluate(X_test, y_test)
        
        logger.info(f"🎯 Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        # Save model and results
        output_dir = Path("data/processed/simple_raga_classifier")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / "simple_raga_classifier.pkl"
        classifier.save_model(str(model_path))
        
        # Save label mapping
        label_mapping_path = output_dir / "label_mapping.json"
        with open(label_mapping_path, 'w') as f:
            json.dump(label_to_idx, f, indent=2)
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save test results
        results = {
            'test_accuracy': test_metrics['accuracy'],
            'test_metrics': test_metrics,
            'num_classes': len(label_to_idx),
            'feature_dimension': X.shape[1],
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'test_samples': X_test.shape[0],
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎯 Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"📊 Number of ragas: {len(label_to_idx)}")
        print(f"📊 Feature dimension: {X.shape[1]}")
        
        print("\n📊 Files created:")
        print("   • simple_raga_classifier.pkl - Trained model")
        print("   • label_mapping.json - Raga label mapping")
        print("   • training_history.json - Training metrics")
        print("   • training_results.json - Test results")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
