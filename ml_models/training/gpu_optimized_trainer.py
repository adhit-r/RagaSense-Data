#!/usr/bin/env python3
"""
RagaSense GPU-Optimized ML Trainer
High-performance training with Mac GPU support and W&B integration
"""

import os
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RagaDataset(Dataset):
    """GPU-optimized dataset for raga classification"""
    
    def __init__(self, features: np.ndarray, labels: List[str], label_encoder):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = label_encoder.transform(labels)
        self.label_encoder = label_encoder
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class GPUOptimizedRagaClassifier(nn.Module):
    """GPU-optimized raga classification model"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Ensure input has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

class GPUOptimizedTrainer:
    """GPU-optimized trainer with W&B integration"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Initialize W&B
        wandb.init(
            project=config.get('wandb_project', 'ragasense-ml-training'),
            config=config,
            name=f"ragasense-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        logger.info(f"🚀 GPU Optimized Trainer initialized")
        logger.info(f"🖥️ Using device: {self.device}")
        logger.info(f"📊 W&B project: {config.get('wandb_project', 'ragasense-ml-training')}")
        
        # Log device info
        if self.device.type == 'cuda':
            logger.info(f"🔥 CUDA available: {torch.cuda.get_device_name()}")
            logger.info(f"🔥 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif self.device.type == 'mps':
            logger.info(f"🍎 MPS (Mac GPU) available")
        
    def load_data(self) -> Tuple[DataLoader, DataLoader, LabelEncoder]:
        """Load and prepare data for training"""
        logger.info("📂 Loading training data...")
        
        # Load ML-ready data
        ml_ready_path = Path(__file__).parent.parent.parent / "data" / "ml_ready"
        
        if not ml_ready_path.exists():
            raise FileNotFoundError("ML-ready data not found. Please run data processing first.")
        
        # Load features and labels
        features = np.load(ml_ready_path / "features.npy")
        labels = np.load(ml_ready_path / "labels.npy")
        
        # Create label encoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Create dataset
        dataset = RagaDataset(features, labels, label_encoder)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True if self.device.type in ['cuda', 'mps'] else False,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True if self.device.type in ['cuda', 'mps'] else False,
            persistent_workers=True
        )
        
        logger.info(f"📊 Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        logger.info(f"🎯 Number of classes: {len(label_encoder.classes_)}")
        
        return train_loader, val_loader, label_encoder
    
    def create_model(self, num_classes: int) -> nn.Module:
        """Create GPU-optimized model"""
        logger.info(f"🏗️ Creating model for {num_classes} classes...")
        
        model = GPUOptimizedRagaClassifier(
            input_dim=128,  # Mel-spectrogram features
            num_classes=num_classes,
            hidden_dim=self.config.get('hidden_dim', 512)
        ).to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log to W&B
        wandb.log({
            "model_total_params": total_params,
            "model_trainable_params": trainable_params,
            "model_num_classes": num_classes
        })
        
        return model
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                wandb.log({
                    "train_batch_loss": loss.item(),
                    "train_batch_accuracy": 100 * correct / total
                })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate model for one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, label_encoder: LabelEncoder):
        """Train the model with GPU optimization"""
        logger.info("🚀 Starting GPU-optimized training...")
        
        # Optimizer and loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # Training history
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - start_time
            
            # Log to W&B
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
                "epoch_time": epoch_time
            })
            
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                       f"LR: {current_lr:.6f} - Time: {epoch_time:.2f}s")
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                
                # Save best model
                model_path = Path(__file__).parent / "models" / "best_raga_model.pth"
                model_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'label_encoder': label_encoder,
                    'config': self.config,
                    'epoch': epoch,
                    'val_accuracy': val_acc
                }, model_path)
                
                # Log to W&B
                wandb.log({"best_val_accuracy": val_acc})
                
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.get('patience', 10):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"✅ Training completed! Best validation accuracy: {best_val_accuracy:.2f}%")
        
        # Log final metrics
        wandb.log({
            "final_best_accuracy": best_val_accuracy,
            "total_epochs": epoch + 1
        })
        
        return best_val_accuracy

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='GPU-Optimized RagaSense Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--wandb_project', type=str, default='ragasense-ml-training', help='W&B project name')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_dim': args.hidden_dim,
        'wandb_project': args.wandb_project,
        'patience': 15,
        'weight_decay': 1e-4
    }
    
    logger.info("🎵 RagaSense GPU-Optimized Training")
    logger.info("=" * 60)
    
    try:
        # Initialize trainer
        trainer = GPUOptimizedTrainer(config)
        
        # Load data
        train_loader, val_loader, label_encoder = trainer.load_data()
        
        # Create model
        model = trainer.create_model(len(label_encoder.classes_))
        
        # Train model
        best_accuracy = trainer.train(model, train_loader, val_loader, label_encoder)
        
        logger.info(f"🎉 Training completed! Best accuracy: {best_accuracy:.2f}%")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()

