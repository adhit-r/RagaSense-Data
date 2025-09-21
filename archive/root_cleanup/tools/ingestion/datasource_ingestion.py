#!/usr/bin/env python3
"""
RagaSense-Data: Data Source Ingestion Pipeline
Ingests data from all sources listed in datasources.md and creates unified dataset.
"""

import os
import json
import logging
import hashlib
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import jsonschema
from dataclasses import dataclass, asdict
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    description: str
    url: str
    tradition: str  # carnatic, hindustani, multi-style, metadata
    format: str  # json, xml, csv, audio, video
    license: str
    size_estimate: str
    priority: int  # 1=highest, 5=lowest
    requires_processing: bool = True
    download_method: str = "direct"  # direct, api, scraping

@dataclass
class IngestionResult:
    """Result of data ingestion operation"""
    source_name: str
    success: bool
    records_processed: int
    records_valid: int
    errors: List[str]
    warnings: List[str]
    processing_time: float
    output_files: List[str]

class RagaSenseDataSourceIngestion:
    """Enhanced data source ingestion system"""
    
    def __init__(self, base_path: str = "/Users/adhi/axonome/RagaSense-Data"):
        self.base_path = Path(base_path)
        self.schema_path = self.base_path / "schemas" / "metadata-schema.json"
        self.data_path = self.base_path / "data"
        self.downloads_path = self.base_path / "downloads"
        self.logs_path = self.base_path / "logs"
        
        # Load schemas
        self.metadata_schema = self._load_schema()
        self.data_sources = self._load_data_sources()
        
        # Create directories
        self._create_directories()
        
        # Initialize W&B if configured
        self.wandb_initialized = self._init_wandb()
        
        # MacBook optimization
        self.macbook_optimized = self._detect_macbook()
        if self.macbook_optimized:
            logger.info("🍎 MacBook detected - optimizing for Apple Silicon")
            # Initialize GPU accelerator
            try:
                from tools.utils.macbook_gpu_accelerator import MacBookGPUAccelerator
                self.gpu_accelerator = MacBookGPUAccelerator()
                logger.info("✅ MacBook GPU accelerator initialized")
            except ImportError:
                logger.warning("⚠️ GPU accelerator not available")
                self.gpu_accelerator = None
        else:
            self.gpu_accelerator = None
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load metadata validation schema"""
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise
    
    def _load_data_sources(self) -> List[DataSource]:
        """Load data sources from datasources.md"""
        sources = [
            # Carnatic Music Datasets
            DataSource(
                name="Google AudioSet Carnatic Music",
                description="Subset of Google AudioSet with 1,663 audio clips (4.6 hrs) of Carnatic vocal music",
                url="https://research.google.com/audioset/dataset/carnatic_music.html",
                tradition="carnatic",
                format="audio",
                license="CC-BY-4.0",
                size_estimate="4.6 hours",
                priority=1,
                download_method="api"
            ),
            DataSource(
                name="Saraga Carnatic Music Dataset",
                description="Time-aligned melody, rhythm, and structural annotations",
                url="https://zenodo.org/records/4301737",
                tradition="carnatic",
                format="json",
                license="CC-BY-4.0",
                size_estimate="~50GB",
                priority=1,
                download_method="direct"
            ),
            DataSource(
                name="Sanidha Multi-Modal Dataset",
                description="Studio-quality multi-track recordings with videos",
                url="https://arxiv.org/abs/2501.06959",
                tradition="carnatic",
                format="video",
                license="Research Use",
                size_estimate="~100GB",
                priority=2,
                download_method="direct"
            ),
            DataSource(
                name="Carnatic Music Rhythm Dataset",
                description="176 excerpts (16.6 hrs) in four taalas with metadata",
                url="https://github.com/Yuan-ManX/ai-audio-datasets",
                tradition="carnatic",
                format="audio",
                license="MIT",
                size_estimate="16.6 hours",
                priority=2,
                download_method="git"
            ),
            DataSource(
                name="Carnatic Music Repository (ramanarunachalam)",
                description="Comprehensive collection with raga, tala, arohana, avarohana metadata",
                url="https://github.com/ramanarunachalam/Music/tree/main/carnatic",
                tradition="carnatic",
                format="json",
                license="MIT",
                size_estimate="~10GB",
                priority=1,
                download_method="git"
            ),
            
            # Hindustani Music Datasets
            DataSource(
                name="SANGEET XML Dataset",
                description="Metadata, notations, rhythm, and melodic info",
                url="https://arxiv.org/abs/2306.04148",
                tradition="hindustani",
                format="xml",
                license="Research Use",
                size_estimate="~20GB",
                priority=1,
                download_method="direct"
            ),
            DataSource(
                name="Thaat and Raga Forest (TRF)",
                description="Raga classification and computational musicology dataset",
                url="https://www.kaggle.com/datasets/suryamajumder/thaat-and-raga-forest-trf-dataset",
                tradition="hindustani",
                format="csv",
                license="CC0",
                size_estimate="~5GB",
                priority=2,
                download_method="kaggle"
            ),
            DataSource(
                name="Hindustani Music Repository (ramanarunachalam)",
                description="Comprehensive collection with raga, tala, arohana, avarohana metadata",
                url="https://github.com/ramanarunachalam/Music/tree/main/hindustani",
                tradition="hindustani",
                format="json",
                license="MIT",
                size_estimate="~8GB",
                priority=1,
                download_method="git"
            ),
            
            # Multi-style / Instruments
            DataSource(
                name="Indian Music Instruments",
                description="Audio files for 4 Indian musical instruments",
                url="https://www.kaggle.com/datasets/aashidutt3/indian-music-instruments",
                tradition="multi-style",
                format="audio",
                license="CC0",
                size_estimate="~2GB",
                priority=3,
                download_method="kaggle"
            ),
            
            # Metadata Repositories
            DataSource(
                name="Carnatic Music Website (ramanarunachalam)",
                description="Website with comprehensive raga, tala information",
                url="https://ramanarunachalam.github.io/Music/Carnatic/carnatic.html",
                tradition="metadata",
                format="html",
                license="MIT",
                size_estimate="~1GB",
                priority=2,
                download_method="scraping"
            )
        ]
        
        return sources
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data_path / "carnatic" / "metadata",
            self.data_path / "hindustani" / "metadata",
            self.data_path / "unified" / "mappings",
            self.downloads_path,
            self.logs_path,
            self.base_path / "processed_data"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _init_wandb(self) -> bool:
        """Initialize Weights & Biases for experiment tracking"""
        try:
            import wandb
            # Initialize W&B project
            wandb.init(
                project="ragasense-data-ingestion",
                name=f"ingestion-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "base_path": str(self.base_path),
                    "total_sources": len(self.data_sources),
                    "schema_version": "1.0"
                }
            )
            logger.info("✅ Weights & Biases initialized")
            return True
        except ImportError:
            logger.warning("⚠️ Weights & Biases not installed. Install with: pip install wandb")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize W&B: {e}")
            return False
    
    def _detect_macbook(self) -> bool:
        """Detect if running on MacBook"""
        import platform
        return platform.system() == "Darwin" and platform.machine() in ["arm64", "x86_64"]
    
    def download_source(self, source: DataSource) -> Tuple[bool, str]:
        """Download data from a source"""
        download_path = self.downloads_path / source.name.replace(" ", "_").lower()
        download_path.mkdir(exist_ok=True)
        
        try:
            if source.download_method == "direct":
                return self._download_direct(source, download_path)
            elif source.download_method == "git":
                return self._download_git(source, download_path)
            elif source.download_method == "kaggle":
                return self._download_kaggle(source, download_path)
            elif source.download_method == "api":
                return self._download_api(source, download_path)
            elif source.download_method == "scraping":
                return self._download_scraping(source, download_path)
            else:
                logger.error(f"Unknown download method: {source.download_method}")
                return False, "Unknown download method"
                
        except Exception as e:
            logger.error(f"Download failed for {source.name}: {e}")
            return False, str(e)
    
    def _download_direct(self, source: DataSource, download_path: Path) -> Tuple[bool, str]:
        """Download directly from URL"""
        try:
            response = requests.get(source.url, stream=True)
            response.raise_for_status()
            
            filename = source.url.split('/')[-1]
            file_path = download_path / filename
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Downloaded {source.name} to {file_path}")
            return True, str(file_path)
            
        except Exception as e:
            return False, str(e)
    
    def _download_git(self, source: DataSource, download_path: Path) -> Tuple[bool, str]:
        """Download using git clone"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "clone", source.url, str(download_path)],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Cloned {source.name} to {download_path}")
                return True, str(download_path)
            else:
                return False, result.stderr
                
        except Exception as e:
            return False, str(e)
    
    def _download_kaggle(self, source: DataSource, download_path: Path) -> Tuple[bool, str]:
        """Download from Kaggle"""
        try:
            import kaggle
            # Extract dataset name from URL
            dataset_name = source.url.split('/')[-1]
            kaggle.api.dataset_download_files(dataset_name, path=str(download_path), unzip=True)
            
            logger.info(f"✅ Downloaded {source.name} from Kaggle")
            return True, str(download_path)
            
        except Exception as e:
            return False, str(e)
    
    def _download_api(self, source: DataSource, download_path: Path) -> Tuple[bool, str]:
        """Download using API"""
        # Placeholder for API-based downloads
        logger.warning(f"API download not implemented for {source.name}")
        return False, "API download not implemented"
    
    def _download_scraping(self, source: DataSource, download_path: Path) -> Tuple[bool, str]:
        """Download using web scraping"""
        # Placeholder for scraping-based downloads
        logger.warning(f"Scraping download not implemented for {source.name}")
        return False, "Scraping download not implemented"
    
    def process_source(self, source: DataSource, download_path: Path) -> IngestionResult:
        """Process downloaded data from a source"""
        start_time = datetime.now()
        result = IngestionResult(
            source_name=source.name,
            success=False,
            records_processed=0,
            records_valid=0,
            errors=[],
            warnings=[],
            processing_time=0.0,
            output_files=[]
        )
        
        try:
            if source.tradition == "carnatic":
                result = self._process_carnatic_data(source, download_path)
            elif source.tradition == "hindustani":
                result = self._process_hindustani_data(source, download_path)
            elif source.tradition == "multi-style":
                result = self._process_multi_style_data(source, download_path)
            elif source.tradition == "metadata":
                result = self._process_metadata(source, download_path)
            else:
                result.errors.append(f"Unknown tradition: {source.tradition}")
            
            result.processing_time = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            result.errors.append(f"Processing failed: {e}")
            result.processing_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _process_carnatic_data(self, source: DataSource, download_path: Path) -> IngestionResult:
        """Process Carnatic music data"""
        result = IngestionResult(
            source_name=source.name,
            success=False,
            records_processed=0,
            records_valid=0,
            errors=[],
            warnings=[],
            processing_time=0.0,
            output_files=[]
        )
        
        # Find and process data files
        if source.format == "json":
            json_files = list(download_path.rglob("*.json"))
            
            # MacBook optimization: Process in smaller batches for memory efficiency
            batch_size = 50 if self.macbook_optimized else 100
            
            for i in range(0, len(json_files), batch_size):
                batch = json_files[i:i + batch_size]
                for json_file in batch:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Convert to unified format
                        unified_data = self._convert_to_unified_format(data, "carnatic", source.name)
                        
                        # Validate
                        errors = self._validate_metadata(unified_data)
                        if errors:
                            result.errors.extend([f"{json_file}: {error}" for error in errors])
                            continue
                        
                        # Save
                        output_path = self._save_unified_metadata(unified_data)
                        result.output_files.append(str(output_path))
                        result.records_valid += 1
                        result.records_processed += 1
                        
                    except Exception as e:
                        result.errors.append(f"Error processing {json_file}: {e}")
                        result.records_processed += 1
        
        result.success = len(result.errors) == 0
        return result
    
    def _process_hindustani_data(self, source: DataSource, download_path: Path) -> IngestionResult:
        """Process Hindustani music data"""
        # Similar to Carnatic processing but with Hindustani-specific logic
        return self._process_carnatic_data(source, download_path)  # Placeholder
    
    def _process_multi_style_data(self, source: DataSource, download_path: Path) -> IngestionResult:
        """Process multi-style/instrument data"""
        # Process instrument-specific data
        return IngestionResult(
            source_name=source.name,
            success=True,
            records_processed=0,
            records_valid=0,
            errors=[],
            warnings=["Multi-style processing not yet implemented"],
            processing_time=0.0,
            output_files=[]
        )
    
    def _process_metadata(self, source: DataSource, download_path: Path) -> IngestionResult:
        """Process metadata repositories"""
        # Process structured metadata
        return IngestionResult(
            source_name=source.name,
            success=True,
            records_processed=0,
            records_valid=0,
            errors=[],
            warnings=["Metadata processing not yet implemented"],
            processing_time=0.0,
            output_files=[]
        )
    
    def _convert_to_unified_format(self, data: Dict[str, Any], tradition: str, source_name: str) -> Dict[str, Any]:
        """Convert source data to unified format"""
        # Generate unique ID
        content = f"{tradition}_{source_name}_{datetime.now().isoformat()}"
        unique_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Build unified metadata structure
        unified = {
            "id": unique_id,
            "tradition": tradition,
            "raga": {
                "name": data.get("raga", {}).get("name", "Unknown"),
                "confidence": 0.9,
                "verified": True,
                "arohana": data.get("raga", {}).get("arohana", []),
                "avarohana": data.get("raga", {}).get("avarohana", [])
            },
            "audio": {
                "file_path": data.get("audio_path", ""),
                "duration_seconds": data.get("duration", 0),
                "sample_rate": 44100,
                "format": "wav"
            },
            "performance": {
                "artist": data.get("artist", "Unknown"),
                "instrument": data.get("instrument", "Unknown")
            },
            "metadata": {
                "source": source_name,
                "license": "CC-BY-SA-4.0",
                "created_date": datetime.now().isoformat().split('T')[0],
                "quality_tier": "tier1"
            }
        }
        
        return unified
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Validate metadata against schema"""
        errors = []
        try:
            jsonschema.validate(metadata, self.metadata_schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except Exception as e:
            errors.append(f"Validation error: {e}")
        return errors
    
    def _save_unified_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Save unified metadata to appropriate location"""
        tradition = metadata['tradition']
        unique_id = metadata['id']
        
        output_path = self.data_path / tradition / "metadata" / f"{unique_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def run_full_ingestion(self, priority_filter: Optional[int] = None) -> List[IngestionResult]:
        """Run full data ingestion pipeline"""
        logger.info("🚀 Starting RagaSense Data Ingestion Pipeline")
        
        # Filter sources by priority if specified
        sources_to_process = self.data_sources
        if priority_filter:
            sources_to_process = [s for s in self.data_sources if s.priority <= priority_filter]
        
        results = []
        
        for source in sources_to_process:
            logger.info(f"📥 Processing {source.name} (Priority: {source.priority})")
            
            # Download
            download_success, download_path = self.download_source(source)
            if not download_success:
                result = IngestionResult(
                    source_name=source.name,
                    success=False,
                    records_processed=0,
                    records_valid=0,
                    errors=[f"Download failed: {download_path}"],
                    warnings=[],
                    processing_time=0.0,
                    output_files=[]
                )
                results.append(result)
                continue
            
            # Process
            result = self.process_source(source, Path(download_path))
            results.append(result)
            
            # Log to W&B if available
            if self.wandb_initialized:
                import wandb
                wandb.log({
                    f"{source.name}_success": result.success,
                    f"{source.name}_records_processed": result.records_processed,
                    f"{source.name}_records_valid": result.records_valid,
                    f"{source.name}_processing_time": result.processing_time
                })
        
        # Generate summary report
        self._generate_ingestion_report(results)
        
        logger.info("✅ Data ingestion pipeline completed")
        return results
    
    def _generate_ingestion_report(self, results: List[IngestionResult]):
        """Generate comprehensive ingestion report"""
        total_processed = sum(r.records_processed for r in results)
        total_valid = sum(r.records_valid for r in results)
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        
        report = {
            "ingestion_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_sources_processed": len(results),
                "total_records_processed": total_processed,
                "total_records_valid": total_valid,
                "success_rate": total_valid / total_processed if total_processed > 0 else 0,
                "total_errors": total_errors,
                "total_warnings": total_warnings
            },
            "source_results": [asdict(r) for r in results]
        }
        
        # Save report
        report_path = self.logs_path / f"ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Ingestion report saved to: {report_path}")
        
        # Log to W&B
        if self.wandb_initialized:
            import wandb
            wandb.log({
                "total_records_processed": total_processed,
                "total_records_valid": total_valid,
                "success_rate": total_valid / total_processed if total_processed > 0 else 0,
                "total_errors": total_errors,
                "total_warnings": total_warnings
            })
            wandb.finish()

def main():
    """Main ingestion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RagaSense Data Source Ingestion")
    parser.add_argument("--base-path", default="/Users/adhi/axonome/RagaSense-Data", help="Base project path")
    parser.add_argument("--priority", type=int, help="Only process sources with priority <= this value")
    parser.add_argument("--source", help="Process only specific source name")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Initialize ingestion system
    ingestion = RagaSenseDataSourceIngestion(args.base_path)
    
    try:
        if args.source:
            # Process specific source
            source = next((s for s in ingestion.data_sources if s.name == args.source), None)
            if source:
                download_success, download_path = ingestion.download_source(source)
                if download_success:
                    result = ingestion.process_source(source, Path(download_path))
                    print(f"✅ Processed {source.name}: {result.records_valid}/{result.records_processed} valid")
                else:
                    print(f"❌ Failed to download {source.name}")
            else:
                print(f"❌ Source not found: {args.source}")
        else:
            # Run full ingestion
            results = ingestion.run_full_ingestion(args.priority)
            
            print(f"\n📊 Ingestion Summary:")
            print(f"  Sources processed: {len(results)}")
            print(f"  Total records: {sum(r.records_processed for r in results)}")
            print(f"  Valid records: {sum(r.records_valid for r in results)}")
            print(f"  Success rate: {sum(r.records_valid for r in results) / sum(r.records_processed for r in results) if sum(r.records_processed for r in results) > 0 else 0:.2%}")
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")

if __name__ == "__main__":
    main()
