#!/usr/bin/env python3
"""
Data Quality Validation
======================

This script performs comprehensive data quality validation on the ML-ready dataset:
1. Feature distribution analysis
2. Raga label validation
3. Tradition classification verification
4. Audio feature quality assessment
5. Dataset balance analysis
6. Cross-validation with original sources
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Comprehensive data quality validation for ML-ready dataset."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.ml_ready_path = self.base_path / "data" / "ml_ready"
        self.results_path = self.ml_ready_path / "quality_validation"
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.ml_dataset = self._load_ml_dataset()
        self.raga_dataset = self._load_raga_dataset()
        
        logger.info(f"🔍 Data Quality Validator initialized")
        logger.info(f"📁 Results will be saved to: {self.results_path}")
    
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
    
    def validate_feature_quality(self):
        """Validate the quality of extracted audio features."""
        logger.info("🔍 Validating feature quality...")
        
        if not self.ml_dataset:
            return {}
        
        # Extract features
        X_train = np.array(self.ml_dataset['training_data']['X_train'])
        X_val = np.array(self.ml_dataset['validation_data']['X_val'])
        X_all = np.vstack([X_train, X_val])
        
        # Basic statistics
        feature_stats = {
            'total_features': int(X_all.shape[1]),
            'total_samples': int(X_all.shape[0]),
            'feature_mean': float(np.mean(X_all)),
            'feature_std': float(np.std(X_all)),
            'feature_min': float(np.min(X_all)),
            'feature_max': float(np.max(X_all)),
            'zero_features': int(np.sum(np.all(X_all == 0, axis=0))),
            'constant_features': int(np.sum(np.std(X_all, axis=0) == 0)),
            'nan_features': int(np.sum(np.isnan(X_all))),
            'inf_features': int(np.sum(np.isinf(X_all)))
        }
        
        # Feature distribution analysis
        feature_means = np.mean(X_all, axis=0)
        feature_stds = np.std(X_all, axis=0)
        
        distribution_stats = {
            'mean_of_means': float(np.mean(feature_means)),
            'std_of_means': float(np.std(feature_means)),
            'mean_of_stds': float(np.mean(feature_stds)),
            'std_of_stds': float(np.std(feature_stds))
        }
        
        # Correlation analysis
        correlation_matrix = np.corrcoef(X_all.T)
        high_correlation_pairs = np.sum(np.abs(correlation_matrix) > 0.95) - X_all.shape[1]  # Subtract diagonal
        
        correlation_stats = {
            'high_correlation_pairs': int(high_correlation_pairs),
            'max_correlation': float(np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        }
        
        results = {
            'feature_quality': feature_stats,
            'distribution_analysis': distribution_stats,
            'correlation_analysis': correlation_stats
        }
        
        logger.info(f"✅ Feature validation completed")
        logger.info(f"📊 Total features: {feature_stats['total_features']}")
        logger.info(f"📊 Zero features: {feature_stats['zero_features']}")
        logger.info(f"📊 Constant features: {feature_stats['constant_features']}")
        logger.info(f"📊 High correlation pairs: {correlation_stats['high_correlation_pairs']}")
        
        return results
    
    def validate_raga_labels(self):
        """Validate raga label consistency and accuracy."""
        logger.info("🔍 Validating raga labels...")
        
        if not self.ml_dataset or not self.raga_dataset:
            return {}
        
        # Extract labels and metadata
        y_train = self.ml_dataset['training_data']['y_train']
        y_val = self.ml_dataset['validation_data']['y_val']
        metadata_train = self.ml_dataset['training_data']['metadata_train']
        metadata_val = self.ml_dataset['validation_data']['metadata_val']
        
        all_labels = y_train + y_val
        all_metadata = metadata_train + metadata_val
        
        # Get raga names from label encoder
        label_encoder = self.ml_dataset['label_encoder']
        raga_names = [label_encoder['classes'][i] for i in all_labels]
        
        # Validate against raga dataset
        valid_ragas = set()
        invalid_ragas = set()
        
        for raga_key, raga_data in self.raga_dataset.items():
            if isinstance(raga_data, dict) and 'name' in raga_data:
                valid_ragas.add(raga_data['name'])
        
        for raga_name in raga_names:
            if raga_name not in valid_ragas:
                invalid_ragas.add(raga_name)
        
        # Label distribution analysis
        label_counts = Counter(all_labels)
        raga_counts = Counter(raga_names)
        
        # Check for class imbalance
        min_samples = min(label_counts.values())
        max_samples = max(label_counts.values())
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        results = {
            'total_unique_ragas': len(set(raga_names)),
            'valid_ragas': len(valid_ragas),
            'invalid_ragas': list(invalid_ragas),
            'label_distribution': dict(raga_counts),
            'class_imbalance_ratio': float(imbalance_ratio),
            'min_samples_per_class': int(min_samples),
            'max_samples_per_class': int(max_samples)
        }
        
        logger.info(f"✅ Raga label validation completed")
        logger.info(f"📊 Unique ragas: {results['total_unique_ragas']}")
        logger.info(f"📊 Invalid ragas: {len(results['invalid_ragas'])}")
        logger.info(f"📊 Class imbalance ratio: {results['class_imbalance_ratio']:.2f}")
        
        return results
    
    def validate_tradition_classification(self):
        """Validate tradition classification consistency."""
        logger.info("🔍 Validating tradition classification...")
        
        if not self.ml_dataset or not self.raga_dataset:
            return {}
        
        # Extract metadata
        metadata_train = self.ml_dataset['training_data']['metadata_train']
        metadata_val = self.ml_dataset['validation_data']['metadata_val']
        all_metadata = metadata_train + metadata_val
        
        # Analyze tradition distribution
        tradition_counts = Counter([m['tradition'] for m in all_metadata])
        
        # Validate against raga dataset
        tradition_consistency = {}
        for metadata in all_metadata:
            raga_name = metadata['tradition']
            dataset_tradition = metadata['tradition']
            
            # Find raga in dataset
            raga_tradition = None
            for raga_key, raga_data in self.raga_dataset.items():
                if isinstance(raga_data, dict) and raga_data.get('name') == raga_name:
                    raga_tradition = raga_data.get('tradition')
                    break
            
            if raga_tradition:
                if raga_name not in tradition_consistency:
                    tradition_consistency[raga_name] = {
                        'dataset_tradition': dataset_tradition,
                        'raga_tradition': raga_tradition,
                        'consistent': dataset_tradition == raga_tradition
                    }
        
        # Calculate consistency metrics
        consistent_count = sum(1 for v in tradition_consistency.values() if v['consistent'])
        total_count = len(tradition_consistency)
        consistency_rate = consistent_count / total_count if total_count > 0 else 0
        
        results = {
            'tradition_distribution': dict(tradition_counts),
            'consistency_rate': float(consistency_rate),
            'consistent_classifications': int(consistent_count),
            'total_classifications': int(total_count),
            'tradition_consistency_details': tradition_consistency
        }
        
        logger.info(f"✅ Tradition classification validation completed")
        logger.info(f"📊 Tradition distribution: {dict(tradition_counts)}")
        logger.info(f"📊 Consistency rate: {consistency_rate:.2%}")
        
        return results
    
    def analyze_dataset_balance(self):
        """Analyze dataset balance and diversity."""
        logger.info("🔍 Analyzing dataset balance...")
        
        if not self.ml_dataset:
            return {}
        
        # Extract data
        metadata_train = self.ml_dataset['training_data']['metadata_train']
        metadata_val = self.ml_dataset['validation_data']['metadata_val']
        all_metadata = metadata_train + metadata_val
        
        # Analyze by tradition
        tradition_balance = Counter([m['tradition'] for m in all_metadata])
        
        # Analyze by source
        source_balance = Counter([m['source'] for m in all_metadata])
        
        # Analyze by raga
        raga_balance = Counter([m['raga_name'] for m in all_metadata])
        
        # Calculate balance metrics
        tradition_entropy = -sum((count/len(all_metadata)) * np.log2(count/len(all_metadata)) 
                                for count in tradition_balance.values())
        raga_entropy = -sum((count/len(all_metadata)) * np.log2(count/len(all_metadata)) 
                           for count in raga_balance.values())
        
        results = {
            'tradition_balance': dict(tradition_balance),
            'source_balance': dict(source_balance),
            'raga_balance': dict(raga_balance),
            'tradition_entropy': float(tradition_entropy),
            'raga_entropy': float(raga_entropy),
            'max_entropy_tradition': float(np.log2(len(tradition_balance))),
            'max_entropy_raga': float(np.log2(len(raga_balance)))
        }
        
        logger.info(f"✅ Dataset balance analysis completed")
        logger.info(f"📊 Tradition entropy: {tradition_entropy:.3f}")
        logger.info(f"📊 Raga entropy: {raga_entropy:.3f}")
        
        return results
    
    def generate_quality_report(self):
        """Generate comprehensive quality validation report."""
        logger.info("📊 Generating comprehensive quality report...")
        
        # Run all validations
        feature_quality = self.validate_feature_quality()
        raga_labels = self.validate_raga_labels()
        tradition_classification = self.validate_tradition_classification()
        dataset_balance = self.analyze_dataset_balance()
        
        # Compile comprehensive report
        quality_report = {
            'validation_date': datetime.now().isoformat(),
            'dataset_info': self.ml_dataset['metadata'] if self.ml_dataset else {},
            'feature_quality': feature_quality,
            'raga_label_validation': raga_labels,
            'tradition_classification_validation': tradition_classification,
            'dataset_balance_analysis': dataset_balance,
            'overall_quality_score': self._calculate_quality_score(
                feature_quality, raga_labels, tradition_classification, dataset_balance
            )
        }
        
        # Save report
        report_file = self.results_path / "data_quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Quality report saved to: {report_file}")
        
        # Print summary
        self._print_quality_summary(quality_report)
        
        return quality_report
    
    def _calculate_quality_score(self, feature_quality, raga_labels, tradition_classification, dataset_balance):
        """Calculate overall quality score."""
        score = 0
        max_score = 100
        
        # Feature quality (30 points)
        if feature_quality:
            fq = feature_quality.get('feature_quality', {})
            if fq.get('nan_features', 0) == 0:
                score += 10
            if fq.get('inf_features', 0) == 0:
                score += 10
            if fq.get('zero_features', 0) < fq.get('total_features', 1) * 0.1:
                score += 10
        
        # Raga label validation (25 points)
        if raga_labels:
            if len(raga_labels.get('invalid_ragas', [])) == 0:
                score += 15
            if raga_labels.get('class_imbalance_ratio', float('inf')) < 10:
                score += 10
        
        # Tradition classification (25 points)
        if tradition_classification:
            consistency_rate = tradition_classification.get('consistency_rate', 0)
            score += int(consistency_rate * 25)
        
        # Dataset balance (20 points)
        if dataset_balance:
            tradition_entropy = dataset_balance.get('tradition_entropy', 0)
            max_entropy = dataset_balance.get('max_entropy_tradition', 1)
            if max_entropy > 0:
                balance_score = (tradition_entropy / max_entropy) * 20
                score += balance_score
        
        return min(score, max_score)
    
    def _print_quality_summary(self, report):
        """Print quality validation summary."""
        print("\n" + "="*60)
        print("📊 DATA QUALITY VALIDATION SUMMARY")
        print("="*60)
        
        if 'dataset_info' in report:
            info = report['dataset_info']
            print(f"📈 Total Samples: {info.get('total_samples', 'N/A')}")
            print(f"📈 Training Samples: {info.get('training_samples', 'N/A')}")
            print(f"📈 Validation Samples: {info.get('validation_samples', 'N/A')}")
            print(f"📈 Unique Ragas: {info.get('unique_ragas', 'N/A')}")
            print(f"📈 Feature Dimensions: {info.get('feature_dimensions', 'N/A')}")
        
        print(f"\n🎯 Overall Quality Score: {report.get('overall_quality_score', 0):.1f}/100")
        
        if 'feature_quality' in report:
            fq = report['feature_quality'].get('feature_quality', {})
            print(f"\n🔍 Feature Quality:")
            print(f"   • Zero features: {fq.get('zero_features', 'N/A')}")
            print(f"   • Constant features: {fq.get('constant_features', 'N/A')}")
            print(f"   • NaN features: {fq.get('nan_features', 'N/A')}")
            print(f"   • Inf features: {fq.get('inf_features', 'N/A')}")
        
        if 'raga_label_validation' in report:
            rlv = report['raga_label_validation']
            print(f"\n🏷️  Raga Label Validation:")
            print(f"   • Unique ragas: {rlv.get('total_unique_ragas', 'N/A')}")
            print(f"   • Invalid ragas: {len(rlv.get('invalid_ragas', []))}")
            print(f"   • Class imbalance ratio: {rlv.get('class_imbalance_ratio', 'N/A'):.2f}")
        
        if 'tradition_classification_validation' in report:
            tcv = report['tradition_classification_validation']
            print(f"\n🎭 Tradition Classification:")
            print(f"   • Consistency rate: {tcv.get('consistency_rate', 0):.2%}")
            print(f"   • Distribution: {tcv.get('tradition_distribution', {})}")
        
        print("="*60)

def main():
    """Main function to run data quality validation."""
    validator = DataQualityValidator()
    report = validator.generate_quality_report()
    return report

if __name__ == "__main__":
    main()
