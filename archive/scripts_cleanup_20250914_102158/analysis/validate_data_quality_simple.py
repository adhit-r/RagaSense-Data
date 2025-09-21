#!/usr/bin/env python3
"""
Simple Data Quality Validation
==============================

Memory-efficient data quality validation focusing on key metrics.
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDataQualityValidator:
    """Memory-efficient data quality validation."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.ml_ready_path = self.base_path / "data" / "ml_ready"
        self.results_path = self.ml_ready_path / "quality_validation"
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.ml_dataset = self._load_ml_dataset()
        self.raga_dataset = self._load_raga_dataset()
        
        logger.info(f"🔍 Simple Data Quality Validator initialized")
    
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
    
    def _load_raga_dataset(self):
        """Load the corrected raga dataset."""
        try:
            raga_file = self.base_path / "data" / "organized_processed" / "unified_ragas_target_achieved.json"
            with open(raga_file, 'r') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded raga dataset with {len(data)} entries")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to load raga dataset: {e}")
            return None
    
    def validate_basic_metrics(self):
        """Validate basic dataset metrics."""
        logger.info("🔍 Validating basic metrics...")
        
        if not self.ml_dataset:
            return {}
        
        metadata = self.ml_dataset['metadata']
        
        # Basic statistics
        results = {
            'total_samples': metadata.get('total_samples', 0),
            'training_samples': metadata.get('training_samples', 0),
            'validation_samples': metadata.get('validation_samples', 0),
            'unique_ragas': metadata.get('unique_ragas', 0),
            'feature_dimensions': metadata.get('feature_dimensions', 0),
            'train_val_ratio': metadata.get('training_samples', 0) / max(metadata.get('validation_samples', 1), 1)
        }
        
        logger.info(f"✅ Basic metrics validation completed")
        return results
    
    def validate_raga_distribution(self):
        """Validate raga distribution."""
        logger.info("🔍 Validating raga distribution...")
        
        if not self.ml_dataset:
            return {}
        
        raga_dist = self.ml_dataset['metadata'].get('raga_distribution', {})
        
        # Calculate distribution metrics
        raga_counts = list(raga_dist.values())
        min_samples = min(raga_counts) if raga_counts else 0
        max_samples = max(raga_counts) if raga_counts else 0
        avg_samples = np.mean(raga_counts) if raga_counts else 0
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        results = {
            'raga_distribution': raga_dist,
            'min_samples_per_raga': int(min_samples),
            'max_samples_per_raga': int(max_samples),
            'avg_samples_per_raga': float(avg_samples),
            'imbalance_ratio': float(imbalance_ratio),
            'total_unique_ragas': len(raga_dist)
        }
        
        logger.info(f"✅ Raga distribution validation completed")
        logger.info(f"📊 Imbalance ratio: {imbalance_ratio:.2f}")
        
        return results
    
    def validate_tradition_distribution(self):
        """Validate tradition distribution."""
        logger.info("🔍 Validating tradition distribution...")
        
        if not self.ml_dataset:
            return {}
        
        tradition_dist = self.ml_dataset['metadata'].get('tradition_distribution', {})
        
        # Calculate distribution metrics
        tradition_counts = list(tradition_dist.values())
        total_samples = sum(tradition_counts)
        
        results = {
            'tradition_distribution': tradition_dist,
            'total_samples': int(total_samples),
            'carnatic_percentage': float(tradition_dist.get('Carnatic', 0) / max(total_samples, 1) * 100),
            'hindustani_percentage': float(tradition_dist.get('Hindustani', 0) / max(total_samples, 1) * 100),
            'both_percentage': float(tradition_dist.get('Both', 0) / max(total_samples, 1) * 100)
        }
        
        logger.info(f"✅ Tradition distribution validation completed")
        logger.info(f"📊 Carnatic: {results['carnatic_percentage']:.1f}%")
        logger.info(f"📊 Hindustani: {results['hindustani_percentage']:.1f}%")
        
        return results
    
    def validate_label_consistency(self):
        """Validate label consistency."""
        logger.info("🔍 Validating label consistency...")
        
        if not self.ml_dataset or not self.raga_dataset:
            return {}
        
        # Get raga names from label encoder
        label_encoder = self.ml_dataset['label_encoder']
        raga_names = label_encoder['classes']
        
        # Check against raga dataset
        valid_ragas = set()
        for raga_key, raga_data in self.raga_dataset.items():
            if isinstance(raga_data, dict) and 'name' in raga_data:
                valid_ragas.add(raga_data['name'])
        
        invalid_ragas = [name for name in raga_names if name not in valid_ragas]
        
        results = {
            'total_ragas_in_dataset': len(raga_names),
            'valid_ragas': len(valid_ragas),
            'invalid_ragas': invalid_ragas,
            'consistency_rate': (len(raga_names) - len(invalid_ragas)) / max(len(raga_names), 1)
        }
        
        logger.info(f"✅ Label consistency validation completed")
        logger.info(f"📊 Consistency rate: {results['consistency_rate']:.2%}")
        
        return results
    
    def validate_feature_dimensions(self):
        """Validate feature dimensions without loading full matrices."""
        logger.info("🔍 Validating feature dimensions...")
        
        if not self.ml_dataset:
            return {}
        
        feature_dims = self.ml_dataset['metadata'].get('feature_dimensions', 0)
        
        # Sample a few features to check for issues
        X_train = self.ml_dataset['training_data']['X_train']
        X_val = self.ml_dataset['validation_data']['X_val']
        
        # Check first few samples
        sample_issues = []
        for i, sample in enumerate(X_train[:5]):  # Check first 5 samples
            if len(sample) != feature_dims:
                sample_issues.append(f"Sample {i}: {len(sample)} != {feature_dims}")
        
        results = {
            'feature_dimensions': int(feature_dims),
            'training_samples_count': len(X_train),
            'validation_samples_count': len(X_val),
            'dimension_consistency_issues': sample_issues,
            'all_samples_consistent': len(sample_issues) == 0
        }
        
        logger.info(f"✅ Feature dimensions validation completed")
        logger.info(f"📊 Feature dimensions: {feature_dims}")
        logger.info(f"📊 All samples consistent: {results['all_samples_consistent']}")
        
        return results
    
    def generate_quality_report(self):
        """Generate comprehensive quality validation report."""
        logger.info("📊 Generating quality report...")
        
        # Run all validations
        basic_metrics = self.validate_basic_metrics()
        raga_distribution = self.validate_raga_distribution()
        tradition_distribution = self.validate_tradition_distribution()
        label_consistency = self.validate_label_consistency()
        feature_dimensions = self.validate_feature_dimensions()
        
        # Compile report
        quality_report = {
            'validation_date': datetime.now().isoformat(),
            'basic_metrics': basic_metrics,
            'raga_distribution': raga_distribution,
            'tradition_distribution': tradition_distribution,
            'label_consistency': label_consistency,
            'feature_dimensions': feature_dimensions,
            'overall_quality_score': self._calculate_quality_score(
                basic_metrics, raga_distribution, tradition_distribution, 
                label_consistency, feature_dimensions
            )
        }
        
        # Save report
        report_file = self.results_path / "simple_quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Quality report saved to: {report_file}")
        
        # Print summary
        self._print_quality_summary(quality_report)
        
        return quality_report
    
    def _calculate_quality_score(self, basic_metrics, raga_distribution, tradition_distribution, 
                                label_consistency, feature_dimensions):
        """Calculate overall quality score."""
        score = 0
        max_score = 100
        
        # Basic metrics (20 points)
        if basic_metrics.get('total_samples', 0) > 100:
            score += 10
        if basic_metrics.get('train_val_ratio', 0) > 3:  # Good train/val ratio
            score += 10
        
        # Raga distribution (25 points)
        if raga_distribution.get('imbalance_ratio', float('inf')) < 20:
            score += 15
        if raga_distribution.get('total_unique_ragas', 0) > 10:
            score += 10
        
        # Tradition distribution (25 points)
        carnatic_pct = tradition_distribution.get('carnatic_percentage', 0)
        hindustani_pct = tradition_distribution.get('hindustani_percentage', 0)
        if 20 < carnatic_pct < 80 and 20 < hindustani_pct < 80:  # Balanced
            score += 25
        
        # Label consistency (20 points)
        if label_consistency.get('consistency_rate', 0) > 0.9:
            score += 20
        
        # Feature dimensions (10 points)
        if feature_dimensions.get('all_samples_consistent', False):
            score += 10
        
        return min(score, max_score)
    
    def _print_quality_summary(self, report):
        """Print quality validation summary."""
        print("\n" + "="*60)
        print("📊 DATA QUALITY VALIDATION SUMMARY")
        print("="*60)
        
        if 'basic_metrics' in report:
            bm = report['basic_metrics']
            print(f"📈 Total Samples: {bm.get('total_samples', 'N/A')}")
            print(f"📈 Training/Validation: {bm.get('training_samples', 'N/A')}/{bm.get('validation_samples', 'N/A')}")
            print(f"📈 Unique Ragas: {bm.get('unique_ragas', 'N/A')}")
            print(f"📈 Feature Dimensions: {bm.get('feature_dimensions', 'N/A')}")
        
        print(f"\n🎯 Overall Quality Score: {report.get('overall_quality_score', 0):.1f}/100")
        
        if 'raga_distribution' in report:
            rd = report['raga_distribution']
            print(f"\n🏷️  Raga Distribution:")
            print(f"   • Unique ragas: {rd.get('total_unique_ragas', 'N/A')}")
            print(f"   • Imbalance ratio: {rd.get('imbalance_ratio', 'N/A'):.2f}")
            print(f"   • Min/Max samples: {rd.get('min_samples_per_raga', 'N/A')}/{rd.get('max_samples_per_raga', 'N/A')}")
        
        if 'tradition_distribution' in report:
            td = report['tradition_distribution']
            print(f"\n🎭 Tradition Distribution:")
            print(f"   • Carnatic: {td.get('carnatic_percentage', 'N/A'):.1f}%")
            print(f"   • Hindustani: {td.get('hindustani_percentage', 'N/A'):.1f}%")
            print(f"   • Both: {td.get('both_percentage', 'N/A'):.1f}%")
        
        if 'label_consistency' in report:
            lc = report['label_consistency']
            print(f"\n✅ Label Consistency:")
            print(f"   • Consistency rate: {lc.get('consistency_rate', 0):.2%}")
            print(f"   • Invalid ragas: {len(lc.get('invalid_ragas', []))}")
        
        if 'feature_dimensions' in report:
            fd = report['feature_dimensions']
            print(f"\n🔍 Feature Dimensions:")
            print(f"   • Dimensions: {fd.get('feature_dimensions', 'N/A')}")
            print(f"   • All consistent: {fd.get('all_samples_consistent', 'N/A')}")
        
        print("="*60)

def main():
    """Main function to run data quality validation."""
    validator = SimpleDataQualityValidator()
    report = validator.generate_quality_report()
    return report

if __name__ == "__main__":
    main()
