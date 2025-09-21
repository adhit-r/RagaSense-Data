#!/usr/bin/env python3
"""
Train Raga Classifier
====================

Train a machine learning model to classify ragas from audio features.
Uses the real audio features dataset created earlier.
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RagaClassifierTrainer:
    """Train and evaluate raga classification models."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.ml_ready_path = self.base_path / "data" / "ml_ready"
        self.models_path = self.ml_ready_path / "trained_models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Load ML dataset
        self.ml_dataset = self._load_ml_dataset()
        
        logger.info(f"🎯 Raga Classifier Trainer initialized")
        logger.info(f"📁 Models will be saved to: {self.models_path}")
    
    def _load_ml_dataset(self):
        """Load the ML-ready dataset."""
        try:
            ml_file = self.ml_ready_path / "ml_ready_dataset.json"
            with open(ml_file, 'r') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded ML dataset with {data['metadata']['total_samples']} samples")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to load ML dataset: {e}")
            return None
    
    def prepare_data(self):
        """Prepare training and validation data."""
        logger.info("🔧 Preparing training data...")
        
        if not self.ml_dataset:
            return None, None, None, None
        
        # Extract features and labels
        X_train = np.array(self.ml_dataset['training_data']['X_train'])
        y_train = np.array(self.ml_dataset['training_data']['y_train'])
        X_val = np.array(self.ml_dataset['validation_data']['X_val'])
        y_val = np.array(self.ml_dataset['validation_data']['y_val'])
        
        # Get raga names
        raga_names = self.ml_dataset['label_encoder']['classes']
        
        logger.info(f"📊 Training data shape: {X_train.shape}")
        logger.info(f"📊 Validation data shape: {X_val.shape}")
        logger.info(f"📊 Number of classes: {len(raga_names)}")
        
        return X_train, y_train, X_val, y_val, raga_names
    
    def train_models(self, X_train, y_train, X_val, y_val, raga_names):
        """Train multiple classification models."""
        logger.info("🚀 Training multiple models...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Define models to train
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                C=1.0,
                gamma='scale'
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"🎯 Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train_scaled)
                y_val_pred = model.predict(X_val_scaled)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                
                # Generate classification report - only for classes present in validation set
                unique_classes = sorted(list(set(y_val)))
                present_raga_names = [raga_names[i] for i in unique_classes]
                
                val_report = classification_report(y_val, y_val_pred, 
                                                 labels=unique_classes,
                                                 target_names=present_raga_names, 
                                                 output_dict=True, zero_division=0)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'train_accuracy': float(train_accuracy),
                    'val_accuracy': float(val_accuracy),
                    'classification_report': val_report,
                    'y_val_true': y_val.tolist(),
                    'y_val_pred': y_val_pred.tolist()
                }
                
                logger.info(f"✅ {model_name} - Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}")
                
                # Save model
                model_file = self.models_path / f"{model_name.lower()}_model.joblib"
                joblib.dump(model, model_file)
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Save scaler
        scaler_file = self.models_path / "scaler.joblib"
        joblib.dump(scaler, scaler_file)
        
        return results, scaler
    
    def evaluate_models(self, results, raga_names):
        """Evaluate and compare model performance."""
        logger.info("📊 Evaluating model performance...")
        
        evaluation_report = {
            'evaluation_date': datetime.now().isoformat(),
            'dataset_info': self.ml_dataset['metadata'],
            'model_comparison': {},
            'best_model': None,
            'best_accuracy': 0
        }
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
            
            model_info = {
                'train_accuracy': result['train_accuracy'],
                'val_accuracy': result['val_accuracy'],
                'overfitting': result['train_accuracy'] - result['val_accuracy'],
                'classification_report': result['classification_report']
            }
            
            evaluation_report['model_comparison'][model_name] = model_info
            
            # Track best model
            if result['val_accuracy'] > evaluation_report['best_accuracy']:
                evaluation_report['best_accuracy'] = result['val_accuracy']
                evaluation_report['best_model'] = model_name
        
        # Save evaluation report
        report_file = self.models_path / "model_evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Evaluation report saved to: {report_file}")
        
        return evaluation_report
    
    def print_evaluation_summary(self, evaluation_report):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("🎯 RAGA CLASSIFIER TRAINING RESULTS")
        print("="*60)
        
        if 'dataset_info' in evaluation_report:
            info = evaluation_report['dataset_info']
            print(f"📈 Dataset: {info.get('total_samples', 'N/A')} samples, {info.get('unique_ragas', 'N/A')} ragas")
            print(f"📈 Training/Validation: {info.get('training_samples', 'N/A')}/{info.get('validation_samples', 'N/A')}")
        
        print(f"\n🏆 Best Model: {evaluation_report.get('best_model', 'N/A')}")
        print(f"🏆 Best Accuracy: {evaluation_report.get('best_accuracy', 0):.3f}")
        
        print(f"\n📊 Model Comparison:")
        for model_name, model_info in evaluation_report.get('model_comparison', {}).items():
            print(f"   • {model_name}:")
            print(f"     - Train Accuracy: {model_info['train_accuracy']:.3f}")
            print(f"     - Val Accuracy: {model_info['val_accuracy']:.3f}")
            print(f"     - Overfitting: {model_info['overfitting']:.3f}")
        
        print("="*60)
    
    def train_and_evaluate(self):
        """Main training and evaluation pipeline."""
        logger.info("🚀 Starting raga classifier training pipeline...")
        
        # Prepare data
        data = self.prepare_data()
        if data is None:
            logger.error("❌ Failed to prepare data")
            return None
        
        X_train, y_train, X_val, y_val, raga_names = data
        
        # Train models
        results, scaler = self.train_models(X_train, y_train, X_val, y_val, raga_names)
        
        # Evaluate models
        evaluation_report = self.evaluate_models(results, raga_names)
        
        # Print summary
        self.print_evaluation_summary(evaluation_report)
        
        return evaluation_report

def main():
    """Main function to train raga classifier."""
    trainer = RagaClassifierTrainer()
    report = trainer.train_and_evaluate()
    return report

if __name__ == "__main__":
    main()
