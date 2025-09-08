#!/usr/bin/env python3
"""
Saraga Datasets Analysis for RagaSense-Data
===========================================

This script analyzes the newly downloaded Saraga datasets (Carnatic & Hindustani)
and provides insights into their structure, content, and integration potential
for the unified RagaSense-Data dataset.

Features:
- Analyzes dataset structure and metadata
- Extracts audio file information
- Identifies raga, artist, and composition data
- Provides integration recommendations
- Generates comprehensive analysis report
"""

import json
import time
import zipfile
import os
from datetime import datetime
from pathlib import Path
import sys
import logging
from typing import Dict, Any, List, Optional, Set
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('saraga_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SaragaDatasetAnalyzer:
    """
    Analyzes Saraga datasets for integration into RagaSense-Data.
    """
    
    def __init__(self, saraga_datasets_path: Path):
        self.saraga_datasets_path = saraga_datasets_path
        self.carnatic_path = saraga_datasets_path / "carnatic" / "saraga1.5_carnatic.zip"
        self.hindustani_path = saraga_datasets_path / "hindustani" / "saraga1.5_hindustani.zip"
        
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "carnatic_analysis": {},
            "hindustani_analysis": {},
            "integration_recommendations": {},
            "summary": {}
        }
        
        # Statistics tracking
        self.stats = {
            'carnatic_files': 0,
            'hindustani_files': 0,
            'total_audio_files': 0,
            'total_metadata_files': 0,
            'unique_ragas': set(),
            'unique_artists': set(),
            'unique_composers': set(),
            'processing_time': 0
        }

    def _analyze_zip_structure(self, zip_path: Path, dataset_type: str) -> Dict[str, Any]:
        """Analyzes the structure of a Saraga dataset ZIP file."""
        logger.info(f"🔍 Analyzing {dataset_type} dataset structure...")
        
        analysis = {
            "zip_file": str(zip_path),
            "file_size_gb": zip_path.stat().st_size / (1024**3),
            "total_files": 0,
            "file_types": defaultdict(int),
            "directory_structure": {},
            "audio_files": [],
            "metadata_files": [],
            "raga_files": [],
            "artist_files": [],
            "composer_files": []
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                analysis["total_files"] = len(file_list)
                
                for file_path in file_list:
                    # Categorize files by type
                    if file_path.endswith('.wav') or file_path.endswith('.mp3'):
                        analysis["audio_files"].append(file_path)
                        analysis["file_types"]["audio"] += 1
                    elif file_path.endswith('.json'):
                        analysis["metadata_files"].append(file_path)
                        analysis["file_types"]["json"] += 1
                    elif file_path.endswith('.csv'):
                        analysis["metadata_files"].append(file_path)
                        analysis["file_types"]["csv"] += 1
                    elif file_path.endswith('.txt'):
                        analysis["metadata_files"].append(file_path)
                        analysis["file_types"]["text"] += 1
                    else:
                        analysis["file_types"]["other"] += 1
                    
                    # Categorize by content type
                    if 'raga' in file_path.lower():
                        analysis["raga_files"].append(file_path)
                    elif 'artist' in file_path.lower():
                        analysis["artist_files"].append(file_path)
                    elif 'composer' in file_path.lower():
                        analysis["composer_files"].append(file_path)
                    
                    # Build directory structure
                    parts = file_path.split('/')
                    current = analysis["directory_structure"]
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                
                logger.info(f"✅ {dataset_type} analysis complete:")
                logger.info(f"   📁 Total files: {analysis['total_files']}")
                logger.info(f"   🎵 Audio files: {len(analysis['audio_files'])}")
                logger.info(f"   📋 Metadata files: {len(analysis['metadata_files'])}")
                logger.info(f"   📊 File size: {analysis['file_size_gb']:.1f} GB")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing {dataset_type} dataset: {e}")
            analysis["error"] = str(e)
        
        return analysis

    def _extract_sample_metadata(self, zip_path: Path, dataset_type: str, max_files: int = 10) -> Dict[str, Any]:
        """Extracts sample metadata from the dataset for analysis."""
        logger.info(f"📋 Extracting sample metadata from {dataset_type} dataset...")
        
        sample_metadata = {
            "sample_files": [],
            "metadata_structure": {},
            "content_analysis": {}
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get sample metadata files
                metadata_files = [f for f in zip_ref.namelist() if f.endswith('.json')][:max_files]
                
                for file_path in metadata_files:
                    try:
                        with zip_ref.open(file_path) as f:
                            content = f.read().decode('utf-8')
                            data = json.loads(content)
                            
                            sample_metadata["sample_files"].append({
                                "file_path": file_path,
                                "content_type": type(data).__name__,
                                "keys": list(data.keys()) if isinstance(data, dict) else "Not a dict",
                                "sample_data": str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                            })
                            
                            # Analyze content structure
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if key not in sample_metadata["metadata_structure"]:
                                        sample_metadata["metadata_structure"][key] = {
                                            "type": type(value).__name__,
                                            "count": 0,
                                            "sample_values": []
                                        }
                                    sample_metadata["metadata_structure"][key]["count"] += 1
                                    if len(sample_metadata["metadata_structure"][key]["sample_values"]) < 3:
                                        sample_metadata["metadata_structure"][key]["sample_values"].append(str(value)[:100])
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Could not parse {file_path}: {e}")
                        continue
                
                logger.info(f"✅ Extracted metadata from {len(sample_metadata['sample_files'])} files")
                
        except Exception as e:
            logger.error(f"❌ Error extracting metadata from {dataset_type}: {e}")
            sample_metadata["error"] = str(e)
        
        return sample_metadata

    def _analyze_audio_content(self, zip_path: Path, dataset_type: str) -> Dict[str, Any]:
        """Analyzes audio content and file organization."""
        logger.info(f"🎵 Analyzing audio content in {dataset_type} dataset...")
        
        audio_analysis = {
            "audio_files": [],
            "file_organization": {},
            "naming_patterns": {},
            "duration_estimates": {}
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                audio_files = [f for f in zip_ref.namelist() if f.endswith('.wav') or f.endswith('.mp3')]
                
                for audio_file in audio_files[:50]:  # Sample first 50 files
                    file_info = zip_ref.getinfo(audio_file)
                    
                    audio_analysis["audio_files"].append({
                        "file_path": audio_file,
                        "file_size_mb": file_info.file_size / (1024**2),
                        "compressed_size_mb": file_info.compress_size / (1024**2),
                        "compression_ratio": file_info.compress_size / file_info.file_size if file_info.file_size > 0 else 0
                    })
                    
                    # Analyze naming patterns
                    filename = Path(audio_file).name
                    parts = filename.split('_')
                    if len(parts) > 1:
                        pattern = '_'.join(parts[:2])  # First two parts as pattern
                        if pattern not in audio_analysis["naming_patterns"]:
                            audio_analysis["naming_patterns"][pattern] = 0
                        audio_analysis["naming_patterns"][pattern] += 1
                
                logger.info(f"✅ Analyzed {len(audio_analysis['audio_files'])} audio files")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing audio content in {dataset_type}: {e}")
            audio_analysis["error"] = str(e)
        
        return audio_analysis

    def analyze_carnatic_dataset(self) -> Dict[str, Any]:
        """Analyzes the Saraga Carnatic dataset."""
        logger.info("🎼 ANALYZING SARAGA CARNATIC DATASET")
        logger.info("=" * 50)
        
        if not self.carnatic_path.exists():
            logger.error(f"❌ Carnatic dataset not found at {self.carnatic_path}")
            return {"error": "Dataset not found"}
        
        # Analyze structure
        structure_analysis = self._analyze_zip_structure(self.carnatic_path, "Carnatic")
        
        # Extract sample metadata
        metadata_analysis = self._extract_sample_metadata(self.carnatic_path, "Carnatic")
        
        # Analyze audio content
        audio_analysis = self._analyze_audio_content(self.carnatic_path, "Carnatic")
        
        carnatic_analysis = {
            "structure": structure_analysis,
            "metadata": metadata_analysis,
            "audio": audio_analysis,
            "dataset_type": "Carnatic",
            "tradition": "Carnatic",
            "source": "Saraga 1.5"
        }
        
        self.analysis_results["carnatic_analysis"] = carnatic_analysis
        return carnatic_analysis

    def analyze_hindustani_dataset(self) -> Dict[str, Any]:
        """Analyzes the Saraga Hindustani dataset."""
        logger.info("🎵 ANALYZING SARAGA HINDUSTANI DATASET")
        logger.info("=" * 50)
        
        if not self.hindustani_path.exists():
            logger.error(f"❌ Hindustani dataset not found at {self.hindustani_path}")
            return {"error": "Dataset not found"}
        
        # Analyze structure
        structure_analysis = self._analyze_zip_structure(self.hindustani_path, "Hindustani")
        
        # Extract sample metadata
        metadata_analysis = self._extract_sample_metadata(self.hindustani_path, "Hindustani")
        
        # Analyze audio content
        audio_analysis = self._analyze_audio_content(self.hindustani_path, "Hindustani")
        
        hindustani_analysis = {
            "structure": structure_analysis,
            "metadata": metadata_analysis,
            "audio": audio_analysis,
            "dataset_type": "Hindustani",
            "tradition": "Hindustani",
            "source": "Saraga 1.5"
        }
        
        self.analysis_results["hindustani_analysis"] = hindustani_analysis
        return hindustani_analysis

    def generate_integration_recommendations(self) -> Dict[str, Any]:
        """Generates recommendations for integrating Saraga datasets into RagaSense-Data."""
        logger.info("💡 GENERATING INTEGRATION RECOMMENDATIONS")
        logger.info("=" * 50)
        
        recommendations = {
            "extraction_strategy": {
                "carnatic": {
                    "priority": "high",
                    "extraction_method": "selective_extraction",
                    "target_files": ["audio files", "metadata files", "raga annotations"],
                    "estimated_size": "2-3GB extracted"
                },
                "hindustani": {
                    "priority": "high", 
                    "extraction_method": "selective_extraction",
                    "target_files": ["audio files", "metadata files", "raga annotations"],
                    "estimated_size": "1-2GB extracted"
                }
            },
            "data_mapping": {
                "raga_mapping": "Extract raga information from metadata and audio filenames",
                "artist_mapping": "Map artists from metadata to unified artist database",
                "composition_mapping": "Extract composition details from metadata",
                "audio_integration": "Link audio files to existing song database"
            },
            "quality_assessment": {
                "carnatic_quality": "high - research-grade annotations",
                "hindustani_quality": "high - research-grade annotations",
                "integration_complexity": "medium - requires metadata parsing"
            },
            "integration_steps": [
                "1. Extract and parse metadata files",
                "2. Map raga names to unified raga database",
                "3. Extract artist information and cross-reference",
                "4. Parse audio file metadata and durations",
                "5. Create cross-tradition mappings",
                "6. Update unified dataset with Saraga data"
            ]
        }
        
        self.analysis_results["integration_recommendations"] = recommendations
        return recommendations

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generates a comprehensive summary report."""
        logger.info("📊 GENERATING SUMMARY REPORT")
        logger.info("=" * 50)
        
        carnatic = self.analysis_results.get("carnatic_analysis", {})
        hindustani = self.analysis_results.get("hindustani_analysis", {})
        
        summary = {
            "dataset_overview": {
                "total_datasets": 2,
                "total_size_gb": 0,
                "total_files": 0,
                "total_audio_files": 0,
                "total_metadata_files": 0
            },
            "carnatic_summary": {
                "file_size_gb": carnatic.get("structure", {}).get("file_size_gb", 0),
                "total_files": carnatic.get("structure", {}).get("total_files", 0),
                "audio_files": len(carnatic.get("structure", {}).get("audio_files", [])),
                "metadata_files": len(carnatic.get("structure", {}).get("metadata_files", [])),
                "quality": "research-grade"
            },
            "hindustani_summary": {
                "file_size_gb": hindustani.get("structure", {}).get("file_size_gb", 0),
                "total_files": hindustani.get("structure", {}).get("total_files", 0),
                "audio_files": len(hindustani.get("structure", {}).get("audio_files", [])),
                "metadata_files": len(hindustani.get("structure", {}).get("metadata_files", [])),
                "quality": "research-grade"
            },
            "integration_potential": {
                "ragas_expected": "500-1000 new ragas",
                "artists_expected": "100-200 new artists",
                "audio_files_expected": "1000-2000 audio files",
                "research_value": "very_high",
                "cross_tradition_value": "high"
            },
            "next_steps": [
                "Extract metadata from both datasets",
                "Parse raga and artist information",
                "Integrate with existing unified dataset",
                "Create cross-tradition mappings",
                "Update quality scores and validation"
            ]
        }
        
        # Calculate totals
        summary["dataset_overview"]["total_size_gb"] = (
            summary["carnatic_summary"]["file_size_gb"] + 
            summary["hindustani_summary"]["file_size_gb"]
        )
        summary["dataset_overview"]["total_files"] = (
            summary["carnatic_summary"]["total_files"] + 
            summary["hindustani_summary"]["total_files"]
        )
        summary["dataset_overview"]["total_audio_files"] = (
            summary["carnatic_summary"]["audio_files"] + 
            summary["hindustani_summary"]["audio_files"]
        )
        summary["dataset_overview"]["total_metadata_files"] = (
            summary["carnatic_summary"]["metadata_files"] + 
            summary["hindustani_summary"]["metadata_files"]
        )
        
        self.analysis_results["summary"] = summary
        return summary

    def save_analysis_results(self, output_path: Path):
        """Saves the complete analysis results to JSON."""
        logger.info("💾 Saving analysis results...")
        
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(item) for item in obj]
            return obj
        
        serializable_results = convert_sets(self.analysis_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Analysis results saved to {output_path}")

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Runs the complete analysis of both Saraga datasets."""
        start_time = time.time()
        logger.info("🚀 STARTING COMPLETE SARAGA DATASETS ANALYSIS")
        logger.info("=" * 60)
        
        # Analyze both datasets
        self.analyze_carnatic_dataset()
        self.analyze_hindustani_dataset()
        
        # Generate recommendations and summary
        self.generate_integration_recommendations()
        self.generate_summary_report()
        
        end_time = time.time()
        self.stats['processing_time'] = end_time - start_time
        
        logger.info("\n🎉 SARAGA DATASETS ANALYSIS COMPLETED!")
        logger.info(f"⏱️ Analysis time: {self.stats['processing_time']:.1f} seconds")
        
        # Print summary
        summary = self.analysis_results["summary"]
        logger.info("\n📊 ANALYSIS SUMMARY:")
        logger.info(f"   📁 Total datasets: {summary['dataset_overview']['total_datasets']}")
        logger.info(f"   💾 Total size: {summary['dataset_overview']['total_size_gb']:.1f} GB")
        logger.info(f"   📄 Total files: {summary['dataset_overview']['total_files']}")
        logger.info(f"   🎵 Audio files: {summary['dataset_overview']['total_audio_files']}")
        logger.info(f"   📋 Metadata files: {summary['dataset_overview']['total_metadata_files']}")
        
        logger.info("\n🎼 CARNATIC DATASET:")
        carnatic = summary["carnatic_summary"]
        logger.info(f"   💾 Size: {carnatic['file_size_gb']:.1f} GB")
        logger.info(f"   📄 Files: {carnatic['total_files']}")
        logger.info(f"   🎵 Audio: {carnatic['audio_files']}")
        logger.info(f"   📋 Metadata: {carnatic['metadata_files']}")
        
        logger.info("\n🎵 HINDUSTANI DATASET:")
        hindustani = summary["hindustani_summary"]
        logger.info(f"   💾 Size: {hindustani['file_size_gb']:.1f} GB")
        logger.info(f"   📄 Files: {hindustani['total_files']}")
        logger.info(f"   🎵 Audio: {hindustani['audio_files']}")
        logger.info(f"   📋 Metadata: {hindustani['metadata_files']}")
        
        return self.analysis_results

def main():
    """Main function to run the Saraga datasets analysis."""
    project_root = Path(__file__).parent
    saraga_datasets_path = project_root / "downloads" / "saraga_datasets"
    output_path = project_root / "data" / "saraga_analysis_results.json"
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SaragaDatasetAnalyzer(saraga_datasets_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Save results
    analyzer.save_analysis_results(output_path)
    
    logger.info(f"\n🎯 ANALYSIS COMPLETE!")
    logger.info(f"📋 Results saved to: {output_path}")
    logger.info(f"🌐 Website deployed at: https://ragasense-data-hng1ua0j1-radhi1991s-projects.vercel.app")

if __name__ == "__main__":
    main()
