#!/usr/bin/env python3
"""
RagaSense-Data: MacBook GPU Acceleration Utility
Optimizes all dataset operations using MacBook's Metal Performance Shaders (MPS)
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import warnings

# MacBook GPU libraries
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import tensorflow as tf
    from numba import jit, cuda
    MACBOOK_GPU_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MacBook GPU libraries not available: {e}")
    MACBOOK_GPU_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacBookGPUAccelerator:
    """MacBook GPU acceleration for dataset operations"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.gpu_available = MACBOOK_GPU_AVAILABLE and self._check_mps_availability()
        
        if self.gpu_available:
            logger.info(f"🍎 MacBook GPU acceleration enabled on device: {self.device}")
            self._setup_gpu_optimizations()
        else:
            logger.warning("⚠️ MacBook GPU not available, using CPU")
            self.device = "cpu"
    
    def _setup_device(self, device: str) -> str:
        """Setup MacBook GPU device"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _check_mps_availability(self) -> bool:
        """Check if Metal Performance Shaders are available"""
        try:
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except:
            return False
    
    def _setup_gpu_optimizations(self):
        """Setup GPU optimizations for MacBook"""
        if self.gpu_available:
            try:
                # Clear GPU cache
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Set memory fraction
                if self.device == "mps":
                    # MPS doesn't have memory fraction setting
                    pass
                elif self.device == "cuda":
                    torch.cuda.set_per_process_memory_fraction(0.8)
                
                # Enable optimizations
                torch.backends.mps.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                logger.info("✅ MacBook GPU optimizations enabled")
            except Exception as e:
                logger.warning(f"GPU optimization setup failed: {e}")
                self.gpu_available = False
    
    def accelerate_data_ingestion(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """GPU-accelerated data ingestion processing"""
        if not self.gpu_available or not data_batch:
            return data_batch
        
        try:
            # Convert to tensors for GPU processing
            processed_batch = self._process_ingestion_batch_gpu(data_batch)
            return processed_batch
        except Exception as e:
            logger.error(f"GPU ingestion acceleration failed: {e}")
            return data_batch
    
    def _process_ingestion_batch_gpu(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process ingestion batch using GPU"""
        processed_batch = []
        
        # Batch process metadata validation
        metadata_tensors = self._create_metadata_tensors(data_batch)
        
        for i, data in enumerate(data_batch):
            try:
                # GPU-accelerated metadata processing
                if i < len(metadata_tensors):
                    processed_metadata = self._process_metadata_gpu(metadata_tensors[i], data)
                    data.update(processed_metadata)
                
                # GPU-accelerated quality scoring
                quality_score = self._calculate_quality_score_gpu(data)
                data['quality_score'] = quality_score
                
                # GPU-accelerated validation
                validation_result = self._validate_metadata_gpu(data)
                data['validation_result'] = validation_result
                
                processed_batch.append(data)
                
            except Exception as e:
                logger.error(f"GPU processing error for item {i}: {e}")
                processed_batch.append(data)
        
        return processed_batch
    
    def _create_metadata_tensors(self, data_batch: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Create tensors from metadata for GPU processing"""
        tensors = []
        
        for data in data_batch:
            try:
                # Extract numerical features
                features = []
                
                # Raga confidence
                features.append(data.get('raga', {}).get('confidence', 0.5))
                
                # Audio duration (normalized)
                duration = data.get('audio', {}).get('duration_seconds', 0)
                features.append(min(duration / 3600, 1.0))  # Normalize to 1 hour
                
                # Sample rate (normalized)
                sample_rate = data.get('audio', {}).get('sample_rate', 44100)
                features.append(sample_rate / 48000)  # Normalize to 48kHz
                
                # Quality score
                features.append(data.get('audio', {}).get('quality_score', 0.5))
                
                # Convert to tensor
                tensor = torch.tensor(features, dtype=torch.float32)
                if self.gpu_available:
                    tensor = tensor.to(self.device)
                
                tensors.append(tensor)
                
            except Exception as e:
                logger.error(f"Tensor creation error: {e}")
                # Create default tensor
                default_tensor = torch.zeros(4, dtype=torch.float32)
                if self.gpu_available:
                    default_tensor = default_tensor.to(self.device)
                tensors.append(default_tensor)
        
        return tensors
    
    def _process_metadata_gpu(self, tensor: torch.Tensor, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata using GPU operations"""
        try:
            # GPU-accelerated feature extraction
            features = tensor.cpu().numpy()
            
            # Calculate derived features
            processed_metadata = {
                'gpu_processed': True,
                'feature_vector': features.tolist(),
                'feature_magnitude': float(torch.norm(tensor).cpu().item()),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return processed_metadata
            
        except Exception as e:
            logger.error(f"GPU metadata processing error: {e}")
            return {'gpu_processed': False, 'error': str(e)}
    
    def _calculate_quality_score_gpu(self, data: Dict[str, Any]) -> float:
        """GPU-accelerated quality score calculation"""
        try:
            # Extract quality indicators
            indicators = []
            
            # Schema completeness
            required_fields = ['id', 'tradition', 'raga', 'audio', 'performance']
            completeness = sum(1 for field in required_fields if field in data) / len(required_fields)
            indicators.append(completeness)
            
            # Raga confidence
            raga_confidence = data.get('raga', {}).get('confidence', 0.5)
            indicators.append(raga_confidence)
            
            # Audio quality
            audio_quality = data.get('audio', {}).get('quality_score', 0.5)
            indicators.append(audio_quality)
            
            # Metadata completeness
            metadata_completeness = len(data.get('metadata', {})) / 5  # Expected 5 metadata fields
            indicators.append(min(metadata_completeness, 1.0))
            
            # Convert to tensor and process on GPU
            indicators_tensor = torch.tensor(indicators, dtype=torch.float32)
            if self.gpu_available:
                indicators_tensor = indicators_tensor.to(self.device)
            
            # GPU-accelerated weighted average
            weights = torch.tensor([0.3, 0.3, 0.2, 0.2], dtype=torch.float32)
            if self.gpu_available:
                weights = weights.to(self.device)
            
            quality_score = torch.sum(indicators_tensor * weights).cpu().item()
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"GPU quality score calculation error: {e}")
            return 0.5
    
    def _validate_metadata_gpu(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """GPU-accelerated metadata validation"""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'gpu_validated': True
            }
            
            # GPU-accelerated field validation
            required_fields = ['id', 'tradition', 'raga', 'audio']
            field_tensor = torch.tensor([1 if field in data else 0 for field in required_fields], dtype=torch.float32)
            if self.gpu_available:
                field_tensor = field_tensor.to(self.device)
            
            missing_fields = torch.sum(1 - field_tensor).cpu().item()
            if missing_fields > 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing {int(missing_fields)} required fields")
            
            # GPU-accelerated data type validation
            if 'raga' in data and 'confidence' in data['raga']:
                confidence = data['raga']['confidence']
                if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                    validation_result['warnings'].append("Invalid confidence score")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"GPU validation error: {e}")
            return {
                'is_valid': False,
                'errors': [f"GPU validation failed: {e}"],
                'warnings': [],
                'gpu_validated': False
            }
    
    def accelerate_cross_tradition_mapping(self, mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """GPU-accelerated cross-tradition mapping analysis"""
        if not self.gpu_available or not mappings:
            return mappings
        
        try:
            # Create similarity matrix using GPU
            similarity_matrix = self._calculate_similarity_matrix_gpu(mappings)
            
            # Process mappings with GPU acceleration
            processed_mappings = []
            for i, mapping in enumerate(mappings):
                try:
                    # GPU-accelerated similarity analysis
                    similarity_scores = similarity_matrix[i] if i < len(similarity_matrix) else []
                    mapping['gpu_similarity_scores'] = similarity_scores.tolist() if hasattr(similarity_scores, 'tolist') else similarity_scores
                    
                    # GPU-accelerated confidence calculation
                    confidence = self._calculate_mapping_confidence_gpu(mapping)
                    mapping['gpu_confidence'] = confidence
                    
                    processed_mappings.append(mapping)
                    
                except Exception as e:
                    logger.error(f"GPU mapping processing error: {e}")
                    processed_mappings.append(mapping)
            
            return processed_mappings
            
        except Exception as e:
            logger.error(f"GPU cross-tradition mapping acceleration failed: {e}")
            return mappings
    
    def _calculate_similarity_matrix_gpu(self, mappings: List[Dict[str, Any]]) -> torch.Tensor:
        """Calculate similarity matrix using GPU"""
        try:
            # Extract features for similarity calculation
            features = []
            for mapping in mappings:
                feature_vector = []
                
                # Relationship type encoding
                rel_type = mapping.get('relationship', {}).get('type', 'UNKNOWN')
                type_encoding = {'SAME': 1.0, 'SIMILAR': 0.8, 'RELATED': 0.6, 'DERIVED': 0.4, 'UNIQUE': 0.0}.get(rel_type, 0.5)
                feature_vector.append(type_encoding)
                
                # Confidence score
                confidence = mapping.get('relationship', {}).get('confidence', 0.5)
                feature_vector.append(confidence)
                
                # Tradition similarity (simplified)
                carnatic = mapping.get('carnatic_raga', {})
                hindustani = mapping.get('hindustani_raga', {})
                tradition_similarity = 1.0 if carnatic and hindustani else 0.0
                feature_vector.append(tradition_similarity)
                
                features.append(feature_vector)
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32)
            if self.gpu_available:
                features_tensor = features_tensor.to(self.device)
            
            # Calculate cosine similarity matrix on GPU
            normalized_features = F.normalize(features_tensor, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"GPU similarity matrix calculation error: {e}")
            # Return identity matrix as fallback
            n = len(mappings)
            return torch.eye(n, dtype=torch.float32)
    
    def _calculate_mapping_confidence_gpu(self, mapping: Dict[str, Any]) -> float:
        """GPU-accelerated mapping confidence calculation"""
        try:
            # Extract confidence indicators
            indicators = []
            
            # Base confidence
            base_confidence = mapping.get('relationship', {}).get('confidence', 0.5)
            indicators.append(base_confidence)
            
            # Expert verification
            verified = mapping.get('relationship', {}).get('verified_by') is not None
            indicators.append(1.0 if verified else 0.0)
            
            # Supporting data
            supporting_data = mapping.get('supporting_data', {})
            has_examples = len(supporting_data.get('example_recordings', [])) > 0
            indicators.append(1.0 if has_examples else 0.0)
            
            # Convert to tensor and process
            indicators_tensor = torch.tensor(indicators, dtype=torch.float32)
            if self.gpu_available:
                indicators_tensor = indicators_tensor.to(self.device)
            
            # Weighted average
            weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
            if self.gpu_available:
                weights = weights.to(self.device)
            
            confidence = torch.sum(indicators_tensor * weights).cpu().item()
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"GPU confidence calculation error: {e}")
            return 0.5
    
    def accelerate_metadata_management(self, metadata_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """GPU-accelerated metadata management operations"""
        if not self.gpu_available or not metadata_batch:
            return metadata_batch
        
        try:
            # GPU-accelerated metadata processing
            processed_batch = []
            
            for metadata in metadata_batch:
                try:
                    # GPU-accelerated metadata enhancement
                    enhanced_metadata = self._enhance_metadata_gpu(metadata)
                    processed_batch.append(enhanced_metadata)
                    
                except Exception as e:
                    logger.error(f"GPU metadata enhancement error: {e}")
                    processed_batch.append(metadata)
            
            return processed_batch
            
        except Exception as e:
            logger.error(f"GPU metadata management acceleration failed: {e}")
            return metadata_batch
    
    def _enhance_metadata_gpu(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """GPU-accelerated metadata enhancement"""
        try:
            enhanced = metadata.copy()
            
            # GPU-accelerated feature extraction
            features = self._extract_metadata_features_gpu(metadata)
            enhanced['gpu_features'] = features
            
            # GPU-accelerated quality assessment
            quality_metrics = self._assess_metadata_quality_gpu(metadata)
            enhanced['gpu_quality_metrics'] = quality_metrics
            
            # GPU-accelerated categorization
            category = self._categorize_metadata_gpu(metadata)
            enhanced['gpu_category'] = category
            
            enhanced['gpu_enhanced'] = True
            enhanced['gpu_processing_time'] = datetime.now().isoformat()
            
            return enhanced
            
        except Exception as e:
            logger.error(f"GPU metadata enhancement error: {e}")
            metadata['gpu_enhanced'] = False
            metadata['gpu_error'] = str(e)
            return metadata
    
    def _extract_metadata_features_gpu(self, metadata: Dict[str, Any]) -> List[float]:
        """Extract metadata features using GPU operations"""
        try:
            features = []
            
            # Tradition encoding
            tradition = metadata.get('tradition', 'unknown')
            tradition_encoding = 1.0 if tradition == 'carnatic' else 0.0
            features.append(tradition_encoding)
            
            # Raga complexity (based on arohana length)
            arohana = metadata.get('raga', {}).get('arohana', [])
            complexity = len(arohana) / 8.0  # Normalize to 8 swaras
            features.append(min(complexity, 1.0))
            
            # Audio quality
            audio_quality = metadata.get('audio', {}).get('quality_score', 0.5)
            features.append(audio_quality)
            
            # Performance type encoding
            perf_type = metadata.get('performance', {}).get('performance_type', 'unknown')
            type_encoding = {'concert': 1.0, 'studio': 0.8, 'live': 0.9, 'practice': 0.6, 'teaching': 0.7}.get(perf_type, 0.5)
            features.append(type_encoding)
            
            # Convert to tensor for GPU processing
            features_tensor = torch.tensor(features, dtype=torch.float32)
            if self.gpu_available:
                features_tensor = features_tensor.to(self.device)
            
            # GPU-accelerated feature normalization
            normalized_features = F.normalize(features_tensor.unsqueeze(0), p=2, dim=1).squeeze()
            
            return normalized_features.cpu().tolist()
            
        except Exception as e:
            logger.error(f"GPU feature extraction error: {e}")
            return [0.0] * 4
    
    def _assess_metadata_quality_gpu(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """GPU-accelerated metadata quality assessment"""
        try:
            quality_metrics = {}
            
            # Completeness score
            required_sections = ['raga', 'audio', 'performance', 'metadata']
            completeness = sum(1 for section in required_sections if section in metadata) / len(required_sections)
            quality_metrics['completeness'] = completeness
            
            # Consistency score
            consistency_indicators = []
            
            # Check tradition consistency
            tradition = metadata.get('tradition')
            raga_tradition = metadata.get('raga', {}).get('tradition', tradition)
            consistency_indicators.append(1.0 if tradition == raga_tradition else 0.0)
            
            # Check ID consistency
            id_consistency = 1.0 if metadata.get('id') else 0.0
            consistency_indicators.append(id_consistency)
            
            # Convert to tensor and calculate
            consistency_tensor = torch.tensor(consistency_indicators, dtype=torch.float32)
            if self.gpu_available:
                consistency_tensor = consistency_tensor.to(self.device)
            
            consistency = torch.mean(consistency_tensor).cpu().item()
            quality_metrics['consistency'] = consistency
            
            # Overall quality score
            overall_score = (completeness + consistency) / 2.0
            quality_metrics['overall_score'] = overall_score
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"GPU quality assessment error: {e}")
            return {'completeness': 0.5, 'consistency': 0.5, 'overall_score': 0.5}
    
    def _categorize_metadata_gpu(self, metadata: Dict[str, Any]) -> str:
        """GPU-accelerated metadata categorization"""
        try:
            # Extract features for categorization
            features = []
            
            # Tradition
            tradition = metadata.get('tradition', 'unknown')
            features.append(1.0 if tradition == 'carnatic' else 0.0)
            
            # Raga type
            melakarta = metadata.get('raga', {}).get('melakarta_number')
            features.append(1.0 if melakarta else 0.0)
            
            # Performance type
            perf_type = metadata.get('performance', {}).get('performance_type', 'unknown')
            features.append(1.0 if perf_type == 'concert' else 0.0)
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32)
            if self.gpu_available:
                features_tensor = features_tensor.to(self.device)
            
            # Simple categorization logic
            if features_tensor[0] > 0.5:  # Carnatic
                if features_tensor[1] > 0.5:  # Melakarta
                    category = "carnatic_melakarta"
                else:
                    category = "carnatic_janya"
            else:  # Hindustani
                category = "hindustani"
            
            return category
            
        except Exception as e:
            logger.error(f"GPU categorization error: {e}")
            return "unknown"
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get MacBook GPU information"""
        gpu_info = {
            'gpu_available': self.gpu_available,
            'device': self.device,
            'mps_available': torch.backends.mps.is_available() if 'torch' in globals() else False,
            'cuda_available': torch.cuda.is_available() if 'torch' in globals() else False
        }
        
        if self.gpu_available:
            if self.device == "mps":
                gpu_info.update({
                    'gpu_type': 'Metal Performance Shaders',
                    'gpu_name': 'Apple Silicon GPU',
                    'mps_built': torch.backends.mps.is_built()
                })
            elif self.device == "cuda":
                gpu_info.update({
                    'gpu_type': 'CUDA',
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                    'gpu_memory_allocated': torch.cuda.memory_allocated(0)
                })
        
        return gpu_info

def main():
    """Test MacBook GPU accelerator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MacBook GPU Accelerator Test")
    parser.add_argument("--device", default="auto", help="Device to use (auto, mps, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = MacBookGPUAccelerator(device=args.device)
    
    # Print GPU info
    gpu_info = accelerator.get_gpu_info()
    print("🍎 MacBook GPU Information:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    # Test with sample data
    sample_data = [
        {
            'id': 'test_001',
            'tradition': 'carnatic',
            'raga': {'confidence': 0.9, 'arohana': ['S', 'R2', 'G3', 'M2', 'P', 'D2', 'N3', 'S\'']},
            'audio': {'duration_seconds': 180, 'sample_rate': 44100, 'quality_score': 0.8},
            'performance': {'performance_type': 'concert'}
        }
    ]
    
    print(f"\n🚀 Testing GPU acceleration with {len(sample_data)} sample records...")
    
    # Test ingestion acceleration
    accelerated_data = accelerator.accelerate_data_ingestion(sample_data)
    print(f"✅ Ingestion acceleration: {accelerated_data[0].get('gpu_processed', False)}")
    
    # Test mapping acceleration
    sample_mappings = [
        {
            'relationship': {'type': 'SAME', 'confidence': 0.95},
            'carnatic_raga': {'name': 'Kalyani'},
            'hindustani_raga': {'name': 'Yaman'}
        }
    ]
    
    accelerated_mappings = accelerator.accelerate_cross_tradition_mapping(sample_mappings)
    print(f"✅ Mapping acceleration: {accelerated_mappings[0].get('gpu_confidence', 0.0):.3f}")
    
    print("🎉 MacBook GPU acceleration test completed!")

if __name__ == "__main__":
    main()
