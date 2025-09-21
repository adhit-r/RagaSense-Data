#!/usr/bin/env python3
"""
Validate Dataset Quality
========================

This script performs comprehensive validation of the final comprehensive dataset
including data quality checks, consistency validation, and cross-referencing.

Author: RagaSense Data Team
Date: 2025-01-13
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import re
from collections import Counter, defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validate_dataset_quality.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Comprehensive dataset validation and quality analysis"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / "02_raw" / "final_comprehensive_dataset" / "comprehensive_raga_dataset.json"
        self.output_path = self.base_path / "05_research" / "validation_reports"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the comprehensive dataset"""
        logger.info("Loading comprehensive dataset...")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} ragas for validation")
        return dataset
    
    def validate_data_completeness(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data completeness and missing fields"""
        logger.info("Validating data completeness...")
        
        completeness_report = {
            'total_ragas': len(dataset),
            'missing_fields': defaultdict(int),
            'empty_fields': defaultdict(int),
            'field_completeness': {},
            'issues': []
        }
        
        # Define required fields
        required_fields = ['raga_name', 'tradition', 'arohana', 'avarohana']
        optional_fields = ['melakartha', 'has_audio', 'audio_sources', 'metadata']
        
        for raga in dataset:
            raga_id = raga.get('raga_id', 'unknown')
            raga_name = raga.get('raga_name', 'unknown')
            
            # Check required fields
            for field in required_fields:
                if field not in raga:
                    completeness_report['missing_fields'][field] += 1
                    completeness_report['issues'].append(f"Raga {raga_id} ({raga_name}): Missing required field '{field}'")
                elif not raga[field] or raga[field] == '':
                    completeness_report['empty_fields'][field] += 1
                    completeness_report['issues'].append(f"Raga {raga_id} ({raga_name}): Empty required field '{field}'")
            
            # Check optional fields
            for field in optional_fields:
                if field not in raga:
                    completeness_report['missing_fields'][field] += 1
        
        # Calculate field completeness percentages
        for field in required_fields + optional_fields:
            total = len(dataset)
            missing = completeness_report['missing_fields'][field]
            empty = completeness_report['empty_fields'][field]
            complete = total - missing - empty
            completeness_report['field_completeness'][field] = {
                'complete': complete,
                'missing': missing,
                'empty': empty,
                'completeness_percentage': round((complete / total) * 100, 2)
            }
        
        logger.info(f"Data completeness validation completed. Found {len(completeness_report['issues'])} issues")
        return completeness_report
    
    def validate_tradition_consistency(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate tradition classification consistency"""
        logger.info("Validating tradition consistency...")
        
        tradition_report = {
            'tradition_counts': Counter(),
            'tradition_consistency': {},
            'cross_tradition_ragas': [],
            'issues': []
        }
        
        # Count traditions
        for raga in dataset:
            tradition = raga.get('tradition', 'Unknown')
            tradition_report['tradition_counts'][tradition] += 1
        
        # Check for cross-tradition ragas (ragas that appear in both traditions)
        raga_names_by_tradition = defaultdict(set)
        for raga in dataset:
            raga_name = raga.get('raga_name', '').lower().strip()
            tradition = raga.get('tradition', 'Unknown')
            if raga_name:
                raga_names_by_tradition[tradition].add(raga_name)
        
        # Find cross-tradition ragas
        carnatic_ragas = raga_names_by_tradition.get('Carnatic', set())
        hindustani_ragas = raga_names_by_tradition.get('Hindustani', set())
        cross_tradition = carnatic_ragas.intersection(hindustani_ragas)
        
        tradition_report['cross_tradition_ragas'] = list(cross_tradition)
        
        # Check tradition consistency
        for tradition, count in tradition_report['tradition_counts'].items():
            tradition_report['tradition_consistency'][tradition] = {
                'count': count,
                'percentage': round((count / len(dataset)) * 100, 2)
            }
        
        if cross_tradition:
            tradition_report['issues'].append(f"Found {len(cross_tradition)} ragas that appear in both traditions")
        
        logger.info(f"Tradition validation completed. Found {len(tradition_report['issues'])} issues")
        return tradition_report
    
    def validate_musical_consistency(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate musical theory consistency (Arohana/Avarohana)"""
        logger.info("Validating musical consistency...")
        
        musical_report = {
            'arohana_patterns': Counter(),
            'avarohana_patterns': Counter(),
            'melakartha_consistency': {},
            'invalid_patterns': [],
            'issues': []
        }
        
        # Define valid swaras
        valid_swaras = {'sa', 'ri', 'ga', 'ma', 'pa', 'da', 'ni'}
        valid_variations = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        
        for raga in dataset:
            raga_id = raga.get('raga_id', 'unknown')
            raga_name = raga.get('raga_name', 'unknown')
            
            # Validate Arohana
            arohana = raga.get('arohana', '')
            if arohana:
                if not self.validate_swara_pattern(arohana, valid_swaras, valid_variations):
                    musical_report['invalid_patterns'].append(f"Raga {raga_id} ({raga_name}): Invalid Arohana pattern '{arohana}'")
                else:
                    musical_report['arohana_patterns'][arohana] += 1
            
            # Validate Avarohana
            avarohana = raga.get('avarohana', '')
            if avarohana:
                if not self.validate_swara_pattern(avarohana, valid_swaras, valid_variations):
                    musical_report['invalid_patterns'].append(f"Raga {raga_id} ({raga_name}): Invalid Avarohana pattern '{avarohana}'")
                else:
                    musical_report['avarohana_patterns'][avarohana] += 1
            
            # Check Melakartha consistency
            melakartha = raga.get('melakartha', '')
            if melakartha:
                if not self.validate_melakartha_format(melakartha):
                    musical_report['issues'].append(f"Raga {raga_id} ({raga_name}): Invalid Melakartha format '{melakartha}'")
        
        musical_report['issues'].extend(musical_report['invalid_patterns'])
        
        logger.info(f"Musical consistency validation completed. Found {len(musical_report['issues'])} issues")
        return musical_report
    
    def validate_swara_pattern(self, pattern: str, valid_swaras: Set[str], valid_variations: Set[str]) -> bool:
        """Validate swara pattern format"""
        if not pattern:
            return False
        
        # Split pattern into individual swaras
        swaras = pattern.lower().split()
        
        for swara in swaras:
            # Check if swara contains valid base swara
            has_valid_swara = any(base_swara in swara for base_swara in valid_swaras)
            if not has_valid_swara:
                return False
            
            # Check if variation is valid (if present)
            if len(swara) > 2:  # More than base swara + variation
                variation = swara[-1]
                if variation not in valid_variations:
                    return False
        
        return True
    
    def validate_melakartha_format(self, melakartha: str) -> bool:
        """Validate Melakartha format"""
        if not melakartha:
            return False
        
        # Expected format: "number name" (e.g., "28 harikAmbhOji")
        parts = melakartha.split()
        if len(parts) < 2:
            return False
        
        # Check if first part is a number
        try:
            int(parts[0])
            return True
        except ValueError:
            return False
    
    def validate_audio_metadata(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate audio metadata and connections"""
        logger.info("Validating audio metadata...")
        
        audio_report = {
            'total_with_audio': 0,
            'audio_sources': Counter(),
            'saraga_recordings': 0,
            'metadata_quality': {},
            'issues': []
        }
        
        for raga in dataset:
            has_audio = raga.get('has_audio', False)
            if has_audio:
                audio_report['total_with_audio'] += 1
                
                # Count audio sources
                audio_sources = raga.get('audio_sources', [])
                for source in audio_sources:
                    audio_report['audio_sources'][source] += 1
                
                # Count Saraga recordings
                if 'saraga_recordings' in raga:
                    saraga_count = len(raga['saraga_recordings'])
                    audio_report['saraga_recordings'] += saraga_count
                    
                    # Validate Saraga metadata
                    for recording in raga['saraga_recordings']:
                        metadata = recording.get('metadata', {})
                        if not metadata:
                            audio_report['issues'].append(f"Raga {raga.get('raga_name', 'unknown')}: Empty Saraga metadata")
            
            # Check metadata quality
            metadata = raga.get('metadata', {})
            if metadata:
                for key, value in metadata.items():
                    if key not in audio_report['metadata_quality']:
                        audio_report['metadata_quality'][key] = {'present': 0, 'empty': 0}
                    
                    if value and value != '':
                        audio_report['metadata_quality'][key]['present'] += 1
                    else:
                        audio_report['metadata_quality'][key]['empty'] += 1
        
        logger.info(f"Audio metadata validation completed. Found {len(audio_report['issues'])} issues")
        return audio_report
    
    def generate_quality_score(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall quality score"""
        logger.info("Generating quality score...")
        
        total_ragas = reports['completeness']['total_ragas']
        total_issues = 0
        
        # Count issues from all reports
        for report_name, report_data in reports.items():
            if 'issues' in report_data:
                total_issues += len(report_data['issues'])
        
        # Calculate quality score (0-100)
        if total_ragas > 0:
            issue_rate = total_issues / total_ragas
            quality_score = max(0, 100 - (issue_rate * 10))  # Penalize 10 points per issue per raga
        else:
            quality_score = 0
        
        quality_assessment = {
            'overall_score': round(quality_score, 2),
            'total_issues': total_issues,
            'total_ragas': total_ragas,
            'issue_rate': round(total_issues / total_ragas, 4) if total_ragas > 0 else 0,
            'grade': self.get_quality_grade(quality_score),
            'recommendations': self.generate_recommendations(reports)
        }
        
        return quality_assessment
    
    def get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def generate_recommendations(self, reports: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Completeness recommendations
        completeness = reports.get('completeness', {})
        if completeness.get('missing_fields'):
            recommendations.append("Fill in missing required fields for all ragas")
        
        # Tradition recommendations
        tradition = reports.get('tradition', {})
        if tradition.get('cross_tradition_ragas'):
            recommendations.append("Review and resolve cross-tradition raga classifications")
        
        # Musical recommendations
        musical = reports.get('musical', {})
        if musical.get('invalid_patterns'):
            recommendations.append("Fix invalid Arohana/Avarohana patterns")
        
        # Audio recommendations
        audio = reports.get('audio', {})
        if audio.get('total_with_audio', 0) < 10:
            recommendations.append("Increase audio coverage by adding more audio sources")
        
        return recommendations
    
    def save_validation_report(self, reports: Dict[str, Any], quality_score: Dict[str, Any]):
        """Save comprehensive validation report"""
        logger.info("Saving validation report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive report
        validation_report = {
            'validation_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_path': str(self.dataset_path),
                'validator_version': '1.0'
            },
            'quality_assessment': quality_score,
            'detailed_reports': reports,
            'summary': {
                'total_ragas': reports['completeness']['total_ragas'],
                'total_issues': quality_score['total_issues'],
                'quality_grade': quality_score['grade'],
                'overall_score': quality_score['overall_score']
            }
        }
        
        # Save main report
        report_path = self.output_path / f"validation_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        summary_path = self.output_path / f"validation_summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report['summary'], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to {report_path}")
        logger.info(f"Summary report saved to {summary_path}")
        
        return report_path, summary_path
    
    def run(self):
        """Run comprehensive dataset validation"""
        logger.info("🔍 Dataset Quality Validation")
        logger.info("=" * 50)
        logger.info("This will perform comprehensive validation")
        logger.info("of the final comprehensive dataset")
        logger.info("=" * 50)
        
        try:
            # Load dataset
            dataset = self.load_dataset()
            
            # Run validation checks
            reports = {
                'completeness': self.validate_data_completeness(dataset),
                'tradition': self.validate_tradition_consistency(dataset),
                'musical': self.validate_musical_consistency(dataset),
                'audio': self.validate_audio_metadata(dataset)
            }
            
            # Generate quality score
            quality_score = self.generate_quality_score(reports)
            
            # Save reports
            report_path, summary_path = self.save_validation_report(reports, quality_score)
            
            logger.info("✅ Dataset validation completed successfully!")
            logger.info(f"📊 Quality Score: {quality_score['overall_score']}/100 ({quality_score['grade']})")
            logger.info(f"🔍 Total Issues: {quality_score['total_issues']}")
            logger.info(f"📁 Reports saved to: {self.output_path}")
            
            return reports, quality_score
            
        except Exception as e:
            logger.error(f"❌ Error during validation: {e}")
            raise

def main():
    """Main function"""
    validator = DatasetValidator()
    validator.run()

if __name__ == "__main__":
    main()
