#!/usr/bin/env python3
"""
RagaSense-Data: Data Validation and Quality Assurance
Validates data quality, schema compliance, and cross-tradition consistency.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import jsonschema
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# GPU-accelerated libraries
try:
    import torch
    import cudf
    import cuml
    from cuml.preprocessing import StandardScaler
    from cuml.cluster import KMeans
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    file_path: str
    is_valid: bool
    schema_errors: List[str]
    quality_issues: List[str]
    warnings: List[str]
    quality_score: float
    validation_time: float

@dataclass
class QualityMetrics:
    """Data quality metrics"""
    total_files: int
    valid_files: int
    schema_compliance_rate: float
    average_quality_score: float
    common_issues: Dict[str, int]
    tradition_distribution: Dict[str, int]
    raga_distribution: Dict[str, int]

class RagaSenseDataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self, base_path: Optional[str] = None, use_gpu: bool = False, wandb_enabled: bool = False):
        # Resolve base path: CLI arg > env > repo root
        env_base = os.environ.get("RAGASENSE_BASE_PATH")
        if base_path:
            self.base_path = Path(base_path)
        elif env_base:
            self.base_path = Path(env_base)
        else:
            self.base_path = Path(__file__).resolve().parents[2]
        self.schema_path = self.base_path / "schemas" / "metadata-schema.json"
        self.mapping_schema_path = self.base_path / "schemas" / "mapping-schema.json"
        self.data_path = self.base_path / "data"
        self.logs_path = self.base_path / "logs"
        
        # GPU configuration
        env_gpu = os.environ.get("RAGASENSE_GPU", "").lower() in ("1", "true", "yes")
        self.use_gpu = (use_gpu or env_gpu) and GPU_AVAILABLE
        self.gpu_available = GPU_AVAILABLE
        
        if self.use_gpu:
            logger.info("✅ GPU acceleration enabled for validation")
            self._setup_gpu()
            # Initialize MacBook GPU accelerator
            try:
                from tools.utils.macbook_gpu_accelerator import MacBookGPUAccelerator
                self.gpu_accelerator = MacBookGPUAccelerator()
                logger.info("🍎 MacBook GPU accelerator initialized for validation")
            except ImportError:
                logger.warning("⚠️ GPU accelerator not available")
                self.gpu_accelerator = None
        else:
            logger.info("⚠️ Using CPU for validation")
            self.gpu_accelerator = None
        
        # Load schemas
        self.metadata_schema = self._load_schema(self.schema_path)
        self.mapping_schema = self._load_schema(self.mapping_schema_path)
        
        # Create logs directory
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B if available
        self.wandb_enabled = wandb_enabled or (os.environ.get("RAGASENSE_WANDB", "").lower() in ("1", "true", "yes"))
        self.wandb_initialized = self._init_wandb()
    
    def _setup_gpu(self):
        """Setup GPU for validation tasks"""
        if self.gpu_available:
            try:
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(0.7)
                
                logger.info(f"GPU setup complete: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"GPU setup failed: {e}")
                self.use_gpu = False
    
    def _load_schema(self, schema_path: Path) -> Dict[str, Any]:
        """Load JSON schema"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema {schema_path}: {e}")
            raise
    
    def _init_wandb(self) -> bool:
        """Initialize Weights & Biases for validation tracking"""
        if not getattr(self, "wandb_enabled", False):
            return False
        try:
            import wandb
            wandb.init(
                project="ragasense-data-validation",
                name=f"validation-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "base_path": str(self.base_path),
                    "schema_version": "1.0"
                }
            )
            logger.info("✅ Weights & Biases initialized for validation")
            return True
        except ImportError:
            logger.warning("⚠️ Weights & Biases not installed")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize W&B: {e}")
            return False
    
    def validate_metadata_file(self, file_path: Path) -> ValidationResult:
        """Validate a single metadata file"""
        start_time = datetime.now()
        
        result = ValidationResult(
            file_path=str(file_path),
            is_valid=True,
            schema_errors=[],
            quality_issues=[],
            warnings=[],
            quality_score=1.0,
            validation_time=0.0
        )
        
        try:
            # Load file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Schema validation
            schema_errors = self._validate_schema(data)
            result.schema_errors = schema_errors
            
            if schema_errors:
                result.is_valid = False
            
            # Quality validation
            quality_issues, warnings = self._validate_quality(data)
            result.quality_issues = quality_issues
            result.warnings = warnings
            
            # Calculate quality score
            result.quality_score = self._calculate_quality_score(data, quality_issues)
            
            # Update validity based on quality
            if result.quality_score < 0.7:
                result.is_valid = False
            
        except json.JSONDecodeError as e:
            result.is_valid = False
            result.schema_errors.append(f"Invalid JSON: {e}")
        except Exception as e:
            result.is_valid = False
            result.schema_errors.append(f"Validation error: {e}")
        
        result.validation_time = (datetime.now() - start_time).total_seconds()
        return result
    
    def _validate_schema(self, data: Dict[str, Any]) -> List[str]:
        """Validate data against JSON schema"""
        errors = []
        
        try:
            jsonschema.validate(data, self.metadata_schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except Exception as e:
            errors.append(f"Schema validation failed: {e}")
        
        return errors
    
    def _validate_quality(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate data quality beyond schema compliance"""
        issues = []
        warnings = []
        
        # Check required fields
        required_fields = ["id", "tradition", "raga", "audio", "performance", "metadata"]
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")
        
        # Validate raga information
        if "raga" in data:
            raga_data = data["raga"]
            
            # Check raga name
            if not raga_data.get("name") or raga_data["name"] == "Unknown":
                warnings.append("Raga name is missing or generic")
            
            # Check confidence score
            confidence = raga_data.get("confidence", 0)
            if confidence < 0.5:
                issues.append(f"Low confidence score: {confidence}")
            elif confidence < 0.8:
                warnings.append(f"Moderate confidence score: {confidence}")
            
            # Check arohana/avarohana
            arohana = raga_data.get("arohana", [])
            avarohana = raga_data.get("avarohana", [])
            
            if not arohana or not avarohana:
                issues.append("Missing arohana or avarohana")
            elif len(arohana) < 5 or len(avarohana) < 5:
                warnings.append("Arohana/avarohana seems too short")
            
            # Check for standard swara notation
            standard_swaras = {"S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"}
            all_swaras = set(arohana + avarohana)
            non_standard = all_swaras - standard_swaras
            if non_standard:
                warnings.append(f"Non-standard swara notation: {non_standard}")
        
        # Validate audio information
        if "audio" in data:
            audio_data = data["audio"]
            
            # Check file path
            if not audio_data.get("file_path"):
                issues.append("Missing audio file path")
            
            # Check duration
            duration = audio_data.get("duration_seconds", 0)
            if duration <= 0:
                issues.append("Invalid or missing duration")
            elif duration < 10:
                warnings.append("Very short audio duration")
            elif duration > 3600:
                warnings.append("Very long audio duration")
            
            # Check sample rate
            sample_rate = audio_data.get("sample_rate", 0)
            if sample_rate not in [22050, 44100, 48000]:
                warnings.append(f"Non-standard sample rate: {sample_rate}")
        
        # Validate performance information
        if "performance" in data:
            perf_data = data["performance"]
            
            if not perf_data.get("artist") or perf_data["artist"] == "Unknown":
                warnings.append("Missing or generic artist name")
            
            if not perf_data.get("instrument"):
                warnings.append("Missing instrument information")
        
        # Validate metadata
        if "metadata" in data:
            meta_data = data["metadata"]
            
            if not meta_data.get("source"):
                issues.append("Missing source information")
            
            if not meta_data.get("license"):
                warnings.append("Missing license information")
            
            if not meta_data.get("created_date"):
                warnings.append("Missing creation date")
        
        return issues, warnings
    
    def _calculate_quality_score(self, data: Dict[str, Any], issues: List[str]) -> float:
        """Calculate overall quality score (0.0 to 1.0)"""
        base_score = 1.0
        
        # Deduct for issues
        for issue in issues:
            if "Missing required field" in issue:
                base_score -= 0.3
            elif "Low confidence" in issue:
                base_score -= 0.2
            elif "Missing arohana" in issue or "Missing avarohana" in issue:
                base_score -= 0.2
            elif "Invalid" in issue:
                base_score -= 0.15
            else:
                base_score -= 0.1
        
        # Bonus for complete information
        if data.get("raga", {}).get("vadi") and data.get("raga", {}).get("samvadi"):
            base_score += 0.05
        
        if data.get("raga", {}).get("time_of_day"):
            base_score += 0.05
        
        if data.get("segments") and len(data["segments"]) > 0:
            base_score += 0.05
        
        return max(0.0, min(1.0, base_score))
    
    def validate_tradition_consistency(self, tradition: str) -> Dict[str, Any]:
        """Validate consistency within a tradition"""
        tradition_path = self.data_path / tradition / "metadata"
        
        if not tradition_path.exists():
            return {"error": f"Tradition path not found: {tradition_path}"}
        
        results = []
        raga_names = []
        swara_patterns = defaultdict(list)
        
        # Validate all files in tradition
        for file_path in tradition_path.glob("*.json"):
            result = self.validate_metadata_file(file_path)
            results.append(result)
            
            # Extract raga information for consistency checks
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                raga_name = data.get("raga", {}).get("name")
                if raga_name:
                    raga_names.append(raga_name)
                
                arohana = data.get("raga", {}).get("arohana", [])
                avarohana = data.get("raga", {}).get("avarohana", [])
                
                if arohana and avarohana:
                    swara_patterns[raga_name].append({
                        "arohana": arohana,
                        "avarohana": avarohana
                    })
            
            except Exception as e:
                logger.warning(f"Error processing {file_path} for consistency check: {e}")
        
        # Analyze consistency
        consistency_issues = []
        
        # Check for duplicate raga names with different patterns
        raga_counter = Counter(raga_names)
        for raga_name, count in raga_counter.items():
            if count > 1 and raga_name in swara_patterns:
                patterns = swara_patterns[raga_name]
                if len(set(str(p) for p in patterns)) > 1:
                    consistency_issues.append(f"Raga '{raga_name}' has inconsistent swara patterns")
        
        return {
            "tradition": tradition,
            "total_files": len(results),
            "valid_files": sum(1 for r in results if r.is_valid),
            "average_quality_score": np.mean([r.quality_score for r in results]) if results else 0,
            "consistency_issues": consistency_issues,
            "unique_ragas": len(set(raga_names)),
            "most_common_ragas": raga_counter.most_common(10)
        }
    
    def validate_cross_tradition_mappings(self) -> Dict[str, Any]:
        """Validate cross-tradition raga mappings"""
        mappings_path = self.data_path / "unified" / "mappings"
        
        if not mappings_path.exists():
            return {"error": "Mappings directory not found"}
        
        mapping_files = list(mappings_path.glob("*.json"))
        mapping_results = []
        
        for mapping_file in mapping_files:
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                
                # Validate mapping schema
                try:
                    jsonschema.validate(mapping_data, self.mapping_schema)
                    mapping_results.append({
                        "file": str(mapping_file),
                        "valid": True,
                        "errors": []
                    })
                except jsonschema.ValidationError as e:
                    mapping_results.append({
                        "file": str(mapping_file),
                        "valid": False,
                        "errors": [str(e)]
                    })
            
            except Exception as e:
                mapping_results.append({
                    "file": str(mapping_file),
                    "valid": False,
                    "errors": [str(e)]
                })
        
        return {
            "total_mappings": len(mapping_results),
            "valid_mappings": sum(1 for r in mapping_results if r["valid"]),
            "mapping_results": mapping_results
        }
    
    def run_full_validation(self) -> QualityMetrics:
        """Run comprehensive validation across all data"""
        logger.info("🔍 Starting comprehensive data validation")
        
        all_results = []
        tradition_stats = {}
        common_issues = defaultdict(int)
        raga_distribution = defaultdict(int)
        tradition_distribution = defaultdict(int)
        
        # Validate each tradition
        for tradition in ["carnatic", "hindustani"]:
            logger.info(f"Validating {tradition} tradition...")
            
            tradition_path = self.data_path / tradition / "metadata"
            if tradition_path.exists():
                tradition_results = []
                
                for file_path in tradition_path.glob("*.json"):
                    result = self.validate_metadata_file(file_path)
                    tradition_results.append(result)
                    all_results.append(result)
                    
                    # Collect statistics
                    common_issues.update(Counter(result.quality_issues))
                    
                    # Extract raga and tradition info
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        raga_name = data.get("raga", {}).get("name", "Unknown")
                        raga_distribution[raga_name] += 1
                        tradition_distribution[tradition] += 1
                    
                    except Exception:
                        pass
                
                tradition_stats[tradition] = {
                    "total_files": len(tradition_results),
                    "valid_files": sum(1 for r in tradition_results if r.is_valid),
                    "average_quality_score": np.mean([r.quality_score for r in tradition_results]) if tradition_results else 0
                }
        
        # Validate cross-tradition mappings
        mapping_validation = self.validate_cross_tradition_mappings()
        
        # Calculate overall metrics
        total_files = len(all_results)
        valid_files = sum(1 for r in all_results if r.is_valid)
        schema_compliance_rate = valid_files / total_files if total_files > 0 else 0
        average_quality_score = np.mean([r.quality_score for r in all_results]) if all_results else 0
        
        metrics = QualityMetrics(
            total_files=total_files,
            valid_files=valid_files,
            schema_compliance_rate=schema_compliance_rate,
            average_quality_score=average_quality_score,
            common_issues=dict(common_issues),
            tradition_distribution=dict(tradition_distribution),
            raga_distribution=dict(raga_distribution)
        )
        
        # Generate validation report
        self._generate_validation_report(metrics, tradition_stats, mapping_validation)
        
        # Log to W&B
        if self.wandb_initialized:
            import wandb
            wandb.log({
                "total_files": metrics.total_files,
                "valid_files": metrics.valid_files,
                "schema_compliance_rate": metrics.schema_compliance_rate,
                "average_quality_score": metrics.average_quality_score,
                "carnatic_files": tradition_distribution.get("carnatic", 0),
                "hindustani_files": tradition_distribution.get("hindustani", 0),
                "unique_ragas": len(raga_distribution)
            })
            wandb.finish()
        
        logger.info("✅ Data validation completed")
        return metrics
    
    def _generate_validation_report(self, metrics: QualityMetrics, 
                                  tradition_stats: Dict[str, Any], 
                                  mapping_validation: Dict[str, Any]):
        """Generate comprehensive validation report"""
        report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_files_validated": metrics.total_files,
                "valid_files": metrics.valid_files,
                "schema_compliance_rate": metrics.schema_compliance_rate,
                "average_quality_score": metrics.average_quality_score
            },
            "tradition_statistics": tradition_stats,
            "mapping_validation": mapping_validation,
            "quality_analysis": {
                "common_issues": metrics.common_issues,
                "tradition_distribution": metrics.tradition_distribution,
                "top_ragas": dict(Counter(metrics.raga_distribution).most_common(20))
            }
        }
        
        # Save report
        report_path = self.logs_path / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Validation report saved to: {report_path}")
        
        # Print summary
        print(f"\n📊 Validation Summary:")
        print(f"  Total files: {metrics.total_files}")
        print(f"  Valid files: {metrics.valid_files}")
        print(f"  Schema compliance: {metrics.schema_compliance_rate:.2%}")
        print(f"  Average quality score: {metrics.average_quality_score:.2f}")
        print(f"  Unique ragas: {len(metrics.raga_distribution)}")
        
        if metrics.common_issues:
            print(f"\n🔍 Most common issues:")
            for issue, count in Counter(metrics.common_issues).most_common(5):
                print(f"    {issue}: {count}")

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RagaSense Data Validation")
    parser.add_argument("--base-path", default=os.environ.get("RAGASENSE_BASE_PATH"), help="Base project path (defaults to repo root or $RAGASENSE_BASE_PATH)")
    parser.add_argument("--tradition", choices=["carnatic", "hindustani"], help="Validate specific tradition")
    parser.add_argument("--file", help="Validate specific file")
    parser.add_argument("--mappings", action="store_true", help="Validate cross-tradition mappings only")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (or set RAGASENSE_WANDB=1)")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU if available (or set RAGASENSE_GPU=1)")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = RagaSenseDataValidator(base_path=args.base_path, use_gpu=args.gpu, wandb_enabled=args.wandb)
    
    try:
        if args.file:
            # Validate specific file
            result = validator.validate_metadata_file(Path(args.file))
            print(f"File: {result.file_path}")
            print(f"Valid: {result.is_valid}")
            print(f"Quality Score: {result.quality_score:.2f}")
            if result.schema_errors:
                print(f"Schema Errors: {result.schema_errors}")
            if result.quality_issues:
                print(f"Quality Issues: {result.quality_issues}")
        
        elif args.tradition:
            # Validate specific tradition
            stats = validator.validate_tradition_consistency(args.tradition)
            print(f"\n📊 {args.tradition.title()} Tradition Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        elif args.mappings:
            # Validate mappings only
            mapping_validation = validator.validate_cross_tradition_mappings()
            print(f"\n📊 Mapping Validation Results:")
            for key, value in mapping_validation.items():
                print(f"  {key}: {value}")
        
        else:
            # Run full validation
            metrics = validator.run_full_validation()
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")

if __name__ == "__main__":
    main()
