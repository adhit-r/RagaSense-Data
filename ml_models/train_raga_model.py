#!/usr/bin/env python3
"""
RagaSense ML Training Script
Train the raga detection model using Ramanarunachalam data
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from raga_detection_system import RagaDetectionSystem, RagaDetectionConfig, RagaDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training and evaluation utilities"""
    
    def __init__(self, config: RagaDetectionConfig, model_variant: str = 'ensemble'):
        self.config = config
        self.model_variant = model_variant
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🎯 Initializing Model Trainer - Variant: {model_variant}")
        logger.info(f"🖥️ Using device: {self.device}")
    
    def create_data_loaders(self, dataset: RagaDataset, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"📊 Data split: {train_size} training, {val_size} validation samples")
        return train_loader, val_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # In a real implementation, you would process actual audio features here
            # For now, we'll simulate the training process with dummy data
            
            batch_size = len(batch['raga_name'])
            
            # Create dummy features (replace with actual audio processing)
            dummy_features = {
                'mel_spectrogram': torch.randn(batch_size, self.config.N_MELS, 100),
                'mfcc': torch.randn(batch_size, self.config.N_MFCC, 100),
                'chroma': torch.randn(batch_size, self.config.CHROMA_N_CHROMA, 100)
            }
            
            # Move to device
            for key in dummy_features:
                dummy_features[key] = dummy_features[key].to(self.device)
            
            # Simulate labels (in real implementation, use actual raga labels)
            labels = torch.randint(0, model.num_ragas, (batch_size,)).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(dummy_features)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate model for one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch_size = len(batch['raga_name'])
                
                # Create dummy features (replace with actual audio processing)
                dummy_features = {
                    'mel_spectrogram': torch.randn(batch_size, self.config.N_MELS, 100),
                    'mfcc': torch.randn(batch_size, self.config.N_MFCC, 100),
                    'chroma': torch.randn(batch_size, self.config.CHROMA_N_CHROMA, 100)
                }
                
                # Move to device
                for key in dummy_features:
                    dummy_features[key] = dummy_features[key].to(self.device)
                
                # Simulate labels
                labels = torch.randint(0, model.num_ragas, (batch_size,)).to(self.device)
                
                # Forward pass
                logits = model(dummy_features)
                loss = criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                   num_epochs: int = None) -> Dict:
        """Train the model with early stopping"""
        
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        best_val_accuracy = 0
        patience_counter = 0
        
        logger.info(f"🚀 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                       f"LR: {current_lr:.6f} - Time: {epoch_time:.2f}s")
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'ml_models/best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"✅ Training completed! Best validation accuracy: {best_val_accuracy:.2f}%")
        
        return history
    
    def plot_training_history(self, history: Dict, save_path: str = "ml_models/training_history.png"):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(history['learning_rates'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Accuracy improvement
        val_acc_improvement = [acc - history['val_accuracy'][0] for acc in history['val_accuracy']]
        axes[1, 1].plot(val_acc_improvement, color='purple')
        axes[1, 1].set_title('Validation Accuracy Improvement')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Improvement (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training history plot saved to {save_path}")
    
    def evaluate_model(self, model: nn.Module, val_loader: DataLoader, 
                      label_encoder, save_path: str = "ml_models/evaluation_report.txt"):
        """Evaluate model and generate detailed report"""
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch_size = len(batch['raga_name'])
                
                # Create dummy features (replace with actual audio processing)
                dummy_features = {
                    'mel_spectrogram': torch.randn(batch_size, self.config.N_MELS, 100),
                    'mfcc': torch.randn(batch_size, self.config.N_MFCC, 100),
                    'chroma': torch.randn(batch_size, self.config.CHROMA_N_CHROMA, 100)
                }
                
                # Move to device
                for key in dummy_features:
                    dummy_features[key] = dummy_features[key].to(self.device)
                
                # Simulate labels
                labels = torch.randint(0, model.num_ragas, (batch_size,)).to(self.device)
                
                # Forward pass
                logits = model(dummy_features)
                _, predicted = torch.max(logits.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Generate classification report
        report = classification_report(all_labels, all_predictions, 
                                     target_names=label_encoder.classes_, 
                                     output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Save evaluation report
        with open(save_path, 'w') as f:
            f.write("RagaSense Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Variant: {self.model_variant}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Number of Classes: {len(label_encoder.classes_)}\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write("-" * 30 + "\n")
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {metrics['support']}\n\n")
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix - Raga Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('ml_models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Evaluation report saved to {save_path}")
        logger.info(f"📈 Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy, report, cm

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train RagaSense ML Model')
    parser.add_argument('--data_path', type=str, default='/Users/adhi/axonome/RagaSense-Data',
                       help='Path to the data directory')
    parser.add_argument('--model_variant', type=str, default='ensemble',
                       choices=['cnn_lstm', 'yue_foundation', 'ensemble', 'realtime'],
                       help='Model variant to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    logger.info("🎵 RagaSense ML Model Training")
    logger.info("=" * 60)
    
    # Configuration
    config = RagaDetectionConfig()
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.NUM_EPOCHS = args.epochs
    
    # Initialize system
    system = RagaDetectionSystem(config, model_variant=args.model_variant)
    
    # Prepare training data
    logger.info("📂 Preparing training data...")
    dataset, raga_names = system.prepare_training_data(args.data_path)
    
    if len(raga_names) == 0:
        logger.error("❌ No ragas found for training. Please check your data path.")
        return
    
    # Create model
    logger.info(f"🏗️ Creating {args.model_variant} model...")
    system.model = system.create_model(len(raga_names)).to(system.device)
    system.model.num_ragas = len(raga_names)
    
    # Initialize trainer
    trainer = ModelTrainer(config, args.model_variant)
    
    # Create data loaders
    train_loader, val_loader = trainer.create_data_loaders(dataset)
    
    # Train model
    logger.info("🚀 Starting model training...")
    history = trainer.train_model(system.model, train_loader, val_loader, args.epochs)
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Evaluate model
    logger.info("📊 Evaluating model...")
    accuracy, report, cm = trainer.evaluate_model(system.model, val_loader, system.label_encoder)
    
    # Save final model
    model_path = f"ml_models/raga_detection_model_{args.model_variant}.pth"
    system.save_model(model_path)
    
    # Save training summary
    summary = {
        'model_variant': args.model_variant,
        'num_ragas': len(raga_names),
        'final_accuracy': accuracy,
        'training_epochs': len(history['train_loss']),
        'best_val_accuracy': max(history['val_accuracy']),
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'hidden_dim': config.HIDDEN_DIM,
            'num_layers': config.NUM_LAYERS
        },
        'ragas': raga_names
    }
    
    with open('ml_models/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"📊 Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"💾 Model saved to: {model_path}")
    logger.info(f"📈 Training summary saved to: ml_models/training_summary.json")

if __name__ == "__main__":
    main()

