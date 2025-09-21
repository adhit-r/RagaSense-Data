#!/usr/bin/env python3
"""
Create Proper Raga Detection System
==================================

This script creates a production-ready raga detection system using:
- Clean dataset structure (04_ml_datasets)
- Real audio features (148 processed files)
- Proper ML models (RandomForest, SVM, Neural Networks)
- Comprehensive evaluation and deployment pipeline

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pickle

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_proper_raga_detection_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RagaDataset(Dataset):
    """PyTorch Dataset for Raga Classification"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class RagaNeuralNetwork(nn.Module):
    """Neural Network for Raga Classification"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [512, 256, 128]):
        super(RagaNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ProperRagaDetectionSystem:
    """Production-ready Raga Detection System"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.ml_datasets_path = self.data_path / "04_ml_datasets"
        self.models_path = self.ml_datasets_path / "training" / "trained_models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_metrics = {}
        
        # Results storage
        self.results = {
            'creation_date': datetime.now().isoformat(),
            'models_trained': [],
            'best_model': None,
            'evaluation_metrics': {},
            'deployment_info': {}
        }
    
    def load_ml_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load the ML-ready dataset"""
        logger.info("Loading ML-ready dataset...")
        
        # Use the basic dataset which has proper raga labels
        basic_dataset_path = self.ml_datasets_path / "training" / "ml_ready_dataset.json"
        if basic_dataset_path.exists():
            with open(basic_dataset_path, 'r') as f:
                dataset = json.load(f)
            
            # Extract features and labels
            X_train = np.array(dataset['training_data']['X_train'])
            y_train = np.array(dataset['training_data']['y_train'])
            X_val = np.array(dataset['validation_data']['X_val'])
            y_val = np.array(dataset['validation_data']['y_val'])
            
            # Combine training and validation for full dataset
            X = np.vstack([X_train, X_val])
            y = np.hstack([y_train, y_val])
            
            # Get raga names from label encoder
            raga_names = dataset['label_encoder']['classes']
            
            logger.info(f"Loaded basic dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(raga_names)} ragas")
            logger.info(f"Raga names: {raga_names[:10]}...")  # Show first 10 ragas
            
        else:
            raise FileNotFoundError("No ML-ready dataset found")
        
        return X, y, raga_names
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        logger.info("Preparing data for training...")
        
        # Check class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # Filter out classes with less than 2 samples
        valid_classes = unique_classes[class_counts >= 2]
        valid_indices = np.isin(y, valid_classes)
        
        X_filtered = X[valid_indices]
        y_filtered = y[valid_indices]
        
        logger.info(f"Filtered dataset: {X_filtered.shape[0]} samples, {len(valid_classes)} classes")
        
        # Split data (without stratification to avoid issues with small classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['main'] = scaler
        
        logger.info(f"Data prepared: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
        
        metrics = {
            'model_name': 'RandomForest',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.models['random_forest'] = best_rf
        self.model_metrics['random_forest'] = metrics
        
        logger.info(f"Random Forest trained - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return metrics
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train SVM model"""
        logger.info("Training SVM model...")
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        svm = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)  # Reduced CV for speed
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        y_pred = best_svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_svm, X_train, y_train, cv=3)
        
        metrics = {
            'model_name': 'SVM',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.models['svm'] = best_svm
        self.model_metrics['svm'] = metrics
        
        logger.info(f"SVM trained - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return metrics
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, num_classes: int) -> Dict[str, Any]:
        """Train Neural Network model"""
        logger.info("Training Neural Network model...")
        
        # Create datasets
        train_dataset = RagaDataset(X_train, y_train)
        test_dataset = RagaDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = RagaNeuralNetwork(X_train.shape[1], num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_accuracy = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            accuracy = correct / total
            scheduler.step(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.models_path / 'neural_network_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Accuracy = {accuracy:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(self.models_path / 'neural_network_best.pth'))
        
        # Final evaluation
        model.eval()
        y_pred = []
        with torch.no_grad():
            for batch_features, _ in test_loader:
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
        
        y_pred = np.array(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'model_name': 'NeuralNetwork',
            'accuracy': accuracy,
            'best_accuracy': best_accuracy,
            'epochs_trained': epoch + 1,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.models['neural_network'] = model
        self.model_metrics['neural_network'] = metrics
        
        logger.info(f"Neural Network trained - Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def train_mlp_classifier(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train MLP Classifier (sklearn)"""
        logger.info("Training MLP Classifier...")
        
        # Hyperparameter tuning
        param_grid = {
            'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        mlp = MLPClassifier(random_state=42, max_iter=500)
        grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_mlp = grid_search.best_estimator_
        y_pred = best_mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_mlp, X_train, y_train, cv=3)
        
        metrics = {
            'model_name': 'MLPClassifier',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.models['mlp_classifier'] = best_mlp
        self.model_metrics['mlp_classifier'] = metrics
        
        logger.info(f"MLP Classifier trained - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return metrics
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate all trained models and select the best one"""
        logger.info("Evaluating all models...")
        
        best_model_name = None
        best_accuracy = 0
        
        for model_name, metrics in self.model_metrics.items():
            accuracy = metrics['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
        
        self.results['best_model'] = {
            'name': best_model_name,
            'accuracy': best_accuracy,
            'metrics': self.model_metrics[best_model_name]
        }
        
        self.results['evaluation_metrics'] = self.model_metrics
        
        logger.info(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
        
        return self.results
    
    def save_models(self):
        """Save all trained models and components"""
        logger.info("Saving models and components...")
        
        # Save sklearn models
        for model_name, model in self.models.items():
            if model_name != 'neural_network':  # PyTorch model saved separately
                model_path = self.models_path / f"{model_name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = self.models_path / f"{scaler_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved {scaler_name} scaler to {scaler_path}")
        
        # Save results
        results_path = self.models_path / "raga_detection_system_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved results to {results_path}")
    
    def create_deployment_pipeline(self):
        """Create deployment pipeline for production use"""
        logger.info("Creating deployment pipeline...")
        
        best_model_name = self.results['best_model']['name']
        best_model = self.models[best_model_name]
        scaler = self.scalers['main']
        
        # Create prediction pipeline
        class RagaDetectionPipeline:
            def __init__(self, model, scaler, raga_names):
                self.model = model
                self.scaler = scaler
                self.raga_names = raga_names
            
            def predict(self, features):
                # Scale features
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Predict
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features_scaled)[0]
                    prediction = self.model.predict(features_scaled)[0]
                else:
                    # For neural network
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features_scaled)
                        outputs = self.model(features_tensor)
                        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                        prediction = torch.argmax(outputs, dim=1).numpy()[0]
                
                return {
                    'predicted_raga': self.raga_names[prediction],
                    'confidence': float(probabilities[prediction]),
                    'all_probabilities': {
                        self.raga_names[i]: float(prob) 
                        for i, prob in enumerate(probabilities)
                    }
                }
        
        # Create pipeline
        pipeline = RagaDetectionPipeline(best_model, scaler, self.raga_names)
        
        # Save pipeline
        pipeline_path = self.models_path / "raga_detection_pipeline.pkl"
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        self.results['deployment_info'] = {
            'pipeline_path': str(pipeline_path),
            'best_model': best_model_name,
            'model_accuracy': self.results['best_model']['accuracy'],
            'deployment_date': datetime.now().isoformat()
        }
        
        logger.info(f"Deployment pipeline created: {pipeline_path}")
    
    def run_complete_training(self):
        """Run the complete raga detection system training"""
        logger.info("Starting complete raga detection system training...")
        
        try:
            # Load data
            X, y, raga_names = self.load_ml_dataset()
            self.raga_names = raga_names
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)
            
            # Train models
            self.train_random_forest(X_train, y_train, X_test, y_test)
            self.train_svm(X_train, y_train, X_test, y_test)
            self.train_mlp_classifier(X_train, y_train, X_test, y_test)
            self.train_neural_network(X_train, y_train, X_test, y_test, len(raga_names))
            
            # Evaluate models
            self.evaluate_models()
            
            # Save models
            self.save_models()
            
            # Create deployment pipeline
            self.create_deployment_pipeline()
            
            logger.info("Raga detection system training completed successfully!")
            return self.results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

def main():
    """Main function"""
    print("🎵 RagaSense Proper Raga Detection System")
    print("=" * 50)
    
    system = ProperRagaDetectionSystem()
    results = system.run_complete_training()
    
    print(f"\n✅ Raga Detection System Created!")
    print(f"🏆 Best Model: {results['best_model']['name']}")
    print(f"📊 Accuracy: {results['best_model']['accuracy']:.4f}")
    print(f"📁 Models saved to: data/04_ml_datasets/training/trained_models/")
    print(f"🚀 Deployment pipeline ready!")

if __name__ == "__main__":
    main()
