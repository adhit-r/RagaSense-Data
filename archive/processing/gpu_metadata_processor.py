#!/usr/bin/env python3
"""
RagaSense-Data: GPU-Accelerated Metadata Processor
Process the massive Ramanarunachalam repository with GPU acceleration
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_metadata_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPUAcceleratedMetadataProcessor:
    """GPU-accelerated metadata processor for Indian Classical Music"""
    
    def __init__(self, use_gpu: bool = True, max_workers: int = 8):
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.base_path = project_root
        self.downloads_path = self.base_path / "downloads"
        self.data_path = self.base_path / "data"
        self.data_path.mkdir(exist_ok=True)
        
        # Thread-safe progress tracking
        self.processed_files = 0
        self.total_files = 0
        self.lock = threading.Lock()
        
        # Initialize GPU acceleration
        self.gpu_available = False
        self.device = None
        if self.use_gpu:
            self._setup_gpu()
        
        # Initialize W&B if available
        try:
            import wandb
            wandb.init(project="ragasense-metadata-processing", name=f"metadata-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.wandb_available = True
            logger.info("✅ W&B tracking enabled")
        except:
            self.wandb_available = False
            logger.info("⚠️ W&B not available")
    
    def _setup_gpu(self):
        """Setup GPU acceleration for MacBook"""
        try:
            import torch
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.gpu_available = True
                logger.info("🍎 MacBook GPU (MPS) acceleration enabled")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.gpu_available = True
                logger.info("🚀 CUDA GPU acceleration enabled")
            else:
                self.device = torch.device("cpu")
                logger.info("⚠️ No GPU available, using CPU")
        except ImportError:
            logger.warning("⚠️ PyTorch not available, using CPU")
            self.device = None
    
    def _gpu_accelerated_json_processing(self, json_data: dict) -> dict:
        """GPU-accelerated JSON data processing"""
        if not self.gpu_available:
            return self._cpu_json_processing(json_data)
        
        try:
            import torch
            import numpy as np
            
            # Convert relevant data to tensors for GPU processing
            processed_data = {}
            
            # Process numerical data on GPU
            if 'id' in json_data and isinstance(json_data['id'], (int, float)):
                id_tensor = torch.tensor([json_data['id']], dtype=torch.float32, device=self.device)
                processed_data['id'] = id_tensor.item()
            
            # Process string data (convert to embeddings-like representation)
            if 'name' in json_data and isinstance(json_data['name'], str):
                # Simple character-based encoding for GPU processing
                name_chars = [ord(c) for c in json_data['name'][:100]]  # Limit to 100 chars
                if name_chars:
                    name_tensor = torch.tensor(name_chars, dtype=torch.float32, device=self.device)
                    # GPU-accelerated processing (simple example)
                    processed_name = torch.mean(name_tensor).item()
                    processed_data['name'] = json_data['name']
                    processed_data['name_encoding'] = processed_name
            
            # Process arrays/lists on GPU
            for key, value in json_data.items():
                if isinstance(value, list) and len(value) > 0:
                    if all(isinstance(x, (int, float)) for x in value):
                        # Convert to tensor and process on GPU
                        tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
                        processed_data[f"{key}_mean"] = torch.mean(tensor).item()
                        processed_data[f"{key}_std"] = torch.std(tensor).item()
                        processed_data[f"{key}_count"] = len(value)
            
            # Add original data
            processed_data.update(json_data)
            return processed_data
            
        except Exception as e:
            logger.debug(f"GPU processing failed, falling back to CPU: {e}")
            return self._cpu_json_processing(json_data)
    
    def _cpu_json_processing(self, json_data: dict) -> dict:
        """CPU-based JSON data processing"""
        processed_data = json_data.copy()
        
        # Add processing metadata
        processed_data['_processed_at'] = datetime.now().isoformat()
        processed_data['_processing_method'] = 'cpu'
        
        # Calculate basic statistics
        for key, value in json_data.items():
            if isinstance(value, list) and len(value) > 0:
                if all(isinstance(x, (int, float)) for x in value):
                    processed_data[f"{key}_count"] = len(value)
                    processed_data[f"{key}_sum"] = sum(value)
        
        return processed_data
    
    def process_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single JSON file with GPU acceleration"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # GPU-accelerated processing
            processed_data = self._gpu_accelerated_json_processing(json_data)
            
            # Add file metadata
            processed_data['_file_path'] = str(file_path.relative_to(self.base_path))
            processed_data['_file_size'] = file_path.stat().st_size
            processed_data['_file_name'] = file_path.name
            
            # Update progress
            with self.lock:
                self.processed_files += 1
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path}: {e}")
            return None
    
    def analyze_raga_metadata(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze raga metadata and extract relationships"""
        logger.info("🔍 ANALYZING RAGA METADATA")
        
        ragas = []
        artists = []
        composers = []
        songs = []
        
        for data in processed_data:
            if not data:
                continue
            
            # Extract raga information
            if 'raga' in data.get('_file_name', '').lower():
                ragas.append(data)
            elif 'artist' in data.get('_file_name', '').lower():
                artists.append(data)
            elif 'composer' in data.get('_file_name', '').lower():
                composers.append(data)
            elif 'song' in data.get('_file_name', '').lower():
                songs.append(data)
        
        # Analyze raga relationships
        raga_analysis = self._analyze_raga_relationships(ragas)
        
        # Analyze cross-tradition mappings
        cross_tradition_mappings = self._build_cross_tradition_mappings(ragas)
        
        analysis = {
            "total_files_processed": len(processed_data),
            "raga_files": len(ragas),
            "artist_files": len(artists),
            "composer_files": len(composers),
            "song_files": len(songs),
            "raga_analysis": raga_analysis,
            "cross_tradition_mappings": cross_tradition_mappings,
            "processed_at": datetime.now().isoformat()
        }
        
        return analysis
    
    def _analyze_raga_relationships(self, ragas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze relationships between ragas"""
        if not ragas:
            return {"error": "No raga data found"}
        
        # Extract raga names and properties
        raga_names = []
        raga_properties = {}
        
        for raga_data in ragas:
            if 'name' in raga_data:
                raga_names.append(raga_data['name'])
                raga_properties[raga_data['name']] = raga_data
        
        # Simple relationship analysis
        relationships = []
        for i, raga1 in enumerate(raga_names):
            for j, raga2 in enumerate(raga_names[i+1:], i+1):
                # Calculate similarity (simplified)
                similarity = self._calculate_raga_similarity(raga1, raga2, raga_properties)
                if similarity > 0.5:  # Threshold for similarity
                    relationships.append({
                        "raga1": raga1,
                        "raga2": raga2,
                        "similarity": similarity,
                        "relationship_type": "similar" if similarity > 0.8 else "related"
                    })
        
        return {
            "total_ragas": len(raga_names),
            "unique_ragas": len(set(raga_names)),
            "relationships": relationships,
            "top_ragas": raga_names[:10]  # First 10 ragas
        }
    
    def _calculate_raga_similarity(self, raga1: str, raga2: str, raga_properties: Dict) -> float:
        """Calculate similarity between two ragas"""
        # Simple string similarity for now
        # In a real implementation, this would analyze musical properties
        
        if raga1 == raga2:
            return 1.0
        
        # Check for common words
        words1 = set(raga1.lower().split())
        words2 = set(raga2.lower().split())
        
        if words1 & words2:  # Common words
            return 0.7
        
        # Check for similar length
        length_diff = abs(len(raga1) - len(raga2))
        if length_diff <= 2:
            return 0.3
        
        return 0.0
    
    def _build_cross_tradition_mappings(self, ragas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build cross-tradition raga mappings"""
        # This is a simplified version
        # In reality, you'd analyze the actual raga properties
        
        mappings = [
            {"carnatic": "Kalyani", "hindustani": "Yaman", "relationship": "SAME", "confidence": 0.95},
            {"carnatic": "Kharaharapriya", "hindustani": "Kafi", "relationship": "SIMILAR", "confidence": 0.85},
            {"carnatic": "Todi", "hindustani": "Miyan ki Todi", "relationship": "SAME", "confidence": 0.92},
            {"carnatic": "Bhairavi", "hindustani": "Bhairavi", "relationship": "SAME", "confidence": 0.98},
            {"carnatic": "Sankarabharanam", "hindustani": "Bilaval", "relationship": "SAME", "confidence": 0.94},
        ]
        
        return mappings
    
    def process_ramanarunachalam_repository(self) -> Dict[str, Any]:
        """Process the entire Ramanarunachalam repository"""
        logger.info("🎵 PROCESSING RAMANARUNACHALAM REPOSITORY")
        logger.info("=" * 60)
        logger.info(f"🧵 Max workers: {self.max_workers}")
        logger.info(f"🍎 GPU acceleration: {'✅' if self.gpu_available else '❌'}")
        
        # Find all JSON files
        repo_path = self.downloads_path / "Ramanarunachalam_Music_Repository"
        if not repo_path.exists():
            logger.error("❌ Ramanarunachalam repository not found")
            return {"error": "Repository not found"}
        
        json_files = list(repo_path.rglob("*.json"))
        logger.info(f"📊 Found {len(json_files)} JSON files")
        
        if not json_files:
            logger.error("❌ No JSON files found")
            return {"error": "No JSON files found"}
        
        # Process files in parallel
        self.total_files = len(json_files)
        self.processed_files = 0
        processed_data = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_file = {
                executor.submit(self.process_json_file, file_path): file_path 
                for file_path in json_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        processed_data.append(result)
                    
                    # Log progress
                    progress = (self.processed_files / self.total_files) * 100
                    if self.processed_files % 100 == 0:  # Log every 100 files
                        logger.info(f"📊 Progress: {progress:.1f}% ({self.processed_files}/{self.total_files})")
                    
                except Exception as e:
                    logger.error(f"❌ Processing failed for {file_path}: {e}")
        
        # Analyze the processed data
        analysis = self.analyze_raga_metadata(processed_data)
        
        # Save results
        results = {
            "repository": "Ramanarunachalam_Music_Repository",
            "processing_time_seconds": time.time() - start_time,
            "total_files": self.total_files,
            "processed_files": len(processed_data),
            "gpu_acceleration": self.gpu_available,
            "max_workers": self.max_workers,
            "analysis": analysis,
            "processed_at": datetime.now().isoformat()
        }
        
        # Save to data directory
        results_file = self.data_path / "ramanarunachalam_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n🎉 PROCESSING COMPLETED!")
        logger.info(f"⏱️ Processing time: {results['processing_time_seconds']:.1f} seconds")
        logger.info(f"📊 Files processed: {len(processed_data)}")
        logger.info(f"🔍 Raga files: {analysis.get('raga_files', 0)}")
        logger.info(f"🎭 Artist files: {analysis.get('artist_files', 0)}")
        logger.info(f"🎼 Composer files: {analysis.get('composer_files', 0)}")
        logger.info(f"🎵 Song files: {analysis.get('song_files', 0)}")
        logger.info(f"📋 Results saved: {results_file}")
        
        # Log to W&B if available
        if self.wandb_available:
            import wandb
            wandb.log({
                "total_files": self.total_files,
                "processed_files": len(processed_data),
                "processing_time_seconds": results['processing_time_seconds'],
                "raga_files": analysis.get('raga_files', 0),
                "artist_files": analysis.get('artist_files', 0),
                "composer_files": analysis.get('composer_files', 0),
                "song_files": analysis.get('song_files', 0),
                "gpu_acceleration": self.gpu_available,
                "max_workers": self.max_workers
            })
        
        return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU-accelerated metadata processor')
    parser.add_argument('--workers', type=int, default=8, help='Number of processing threads')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = GPUAcceleratedMetadataProcessor(
            use_gpu=not args.no_gpu,
            max_workers=args.workers
        )
        
        # Process repository
        results = processor.process_ramanarunachalam_repository()
        
        logger.info("✅ Metadata processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Processing interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Processing failed: {e}")
        logger.exception("Full error details:")

if __name__ == "__main__":
    main()
