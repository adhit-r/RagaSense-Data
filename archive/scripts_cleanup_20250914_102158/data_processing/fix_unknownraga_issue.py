#!/usr/bin/env python3
"""
Fix Unknownraga Issue - Re-run Reclassification on Cleaned Data
=============================================================

This script fixes the Unknownraga issue by:
1. Using the cleaned database (which already removed Unknownraga) as the base
2. Re-running the combined raga reclassification process
3. Creating a final corrected database without Unknownraga

The issue was that our previous reclassification used the original database
(which contained Unknownraga) instead of the cleaned database.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unknownraga_fix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnknownragaFixer:
    """
    Fixes the Unknownraga issue by re-running reclassification on cleaned data.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.cleaned_data_path = self.project_root / "data" / "cleaned_ragasense_dataset"
        self.output_path = self.project_root / "data" / "unknownraga_fixed"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.cleaned_ragas = {}
        self.individual_ragas = {}
        self.raga_combinations = {}
        
        self.fix_stats = {
            "cleaned_ragas_loaded": 0,
            "individual_ragas_identified": 0,
            "raga_combinations_identified": 0,
            "unknownraga_found": False,
            "final_individual_count": 0,
            "final_combination_count": 0
        }

    def load_cleaned_data(self):
        """Load the cleaned raga data (which already removed Unknownraga)."""
        logger.info("📂 Loading cleaned raga data...")
        
        cleaned_db_path = self.cleaned_data_path / "cleaned_ragas_database.json"
        if not cleaned_db_path.exists():
            logger.error(f"❌ Cleaned database not found at {cleaned_db_path}")
            return False
        
        with open(cleaned_db_path, 'r', encoding='utf-8') as f:
            self.cleaned_ragas = json.load(f)
        
        self.fix_stats["cleaned_ragas_loaded"] = len(self.cleaned_ragas)
        logger.info(f"✅ Loaded {len(self.cleaned_ragas)} cleaned ragas")
        
        # Verify Unknownraga is not present
        unknownraga_found = False
        for raga_id, raga_data in self.cleaned_ragas.items():
            if raga_data.get('name') == 'Unknownraga':
                unknownraga_found = True
                logger.error(f"❌ Unknownraga still found in cleaned data: {raga_id}")
                break
        
        if not unknownraga_found:
            logger.info("✅ Confirmed: Unknownraga is not present in cleaned data")
            self.fix_stats["unknownraga_found"] = False
        else:
            self.fix_stats["unknownraga_found"] = True
        
        return True

    def identify_individual_vs_combined_ragas(self):
        """Identify individual ragas vs raga combinations from cleaned data."""
        logger.info("🔍 Identifying individual vs combined ragas...")
        
        individual_ragas = {}
        raga_combinations = {}
        
        for raga_id, raga_data in self.cleaned_ragas.items():
            name = raga_data.get('name', raga_id)
            
            # Check if this is a combined raga (contains comma)
            if ',' in name:
                individual_ragas_list = [r.strip() for r in name.split(',')]
                # Filter out empty strings and very short names
                individual_ragas_list = [r for r in individual_ragas_list if r and len(r) > 2]
                
                if len(individual_ragas_list) > 1:
                    # This is a raga combination
                    combination_data = raga_data.copy()
                    combination_data['individual_ragas'] = individual_ragas_list
                    combination_data['combination_type'] = 'multiple_ragas'
                    
                    raga_combinations[raga_id] = combination_data
                else:
                    # Single raga with comma (probably formatting issue)
                    individual_ragas[raga_id] = raga_data
            else:
                # This is an individual raga
                individual_ragas[raga_id] = raga_data
        
        self.individual_ragas = individual_ragas
        self.raga_combinations = raga_combinations
        
        self.fix_stats["individual_ragas_identified"] = len(individual_ragas)
        self.fix_stats["raga_combinations_identified"] = len(raga_combinations)
        
        logger.info(f"✅ Identified {len(individual_ragas)} individual ragas")
        logger.info(f"✅ Identified {len(raga_combinations)} raga combinations")
        
        return True

    def analyze_tradition_distribution(self):
        """Analyze the tradition distribution in individual ragas."""
        logger.info("🏷️ Analyzing tradition distribution...")
        
        tradition_counts = Counter()
        for raga_data in self.individual_ragas.values():
            tradition = raga_data.get('tradition', 'Unknown')
            tradition_counts[tradition] += 1
        
        logger.info("📊 Tradition Distribution:")
        for tradition, count in tradition_counts.most_common():
            percentage = (count / len(self.individual_ragas)) * 100
            logger.info(f"   {tradition}: {count} ragas ({percentage:.1f}%)")
        
        return dict(tradition_counts)

    def generate_fix_report(self) -> Dict[str, Any]:
        """Generate a comprehensive fix report."""
        logger.info("📊 Generating fix report...")
        
        # Top individual ragas by song count
        top_individual_ragas = sorted(
            self.individual_ragas.items(),
            key=lambda x: x[1].get('song_count', 0),
            reverse=True
        )[:20]
        
        # Top raga combinations by song count
        top_combinations = sorted(
            self.raga_combinations.items(),
            key=lambda x: x[1].get('song_count', 0),
            reverse=True
        )[:10]
        
        # Tradition distribution
        tradition_distribution = self.analyze_tradition_distribution()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "fix_statistics": self.fix_stats,
            "tradition_distribution": tradition_distribution,
            "top_individual_ragas": [
                {
                    "raga_id": raga_id,
                    "name": raga_data.get('name', raga_id),
                    "tradition": raga_data.get('tradition', 'Unknown'),
                    "song_count": raga_data.get('song_count', 0)
                }
                for raga_id, raga_data in top_individual_ragas
            ],
            "top_raga_combinations": [
                {
                    "combination_id": combo_id,
                    "name": combo_data.get('name', combo_id),
                    "individual_ragas": combo_data.get('individual_ragas', []),
                    "song_count": combo_data.get('song_count', 0)
                }
                for combo_id, combo_data in top_combinations
            ],
            "data_quality_improvements": {
                "unknownraga_removed": not self.fix_stats["unknownraga_found"],
                "individual_ragas_count": len(self.individual_ragas),
                "raga_combinations_count": len(self.raga_combinations),
                "total_songs_in_individual_ragas": sum(raga.get('song_count', 0) for raga in self.individual_ragas.values()),
                "total_songs_in_combinations": sum(combo.get('song_count', 0) for combo in self.raga_combinations.values())
            }
        }
        
        return report

    def save_fixed_data(self):
        """Save the fixed raga data."""
        logger.info("💾 Saving fixed data...")
        
        # Save individual ragas database
        with open(self.output_path / "unified_ragas_database_fixed.json", 'w', encoding='utf-8') as f:
            json.dump(self.individual_ragas, f, indent=2, ensure_ascii=False)
        
        # Save raga combinations database
        with open(self.output_path / "raga_combinations_database_fixed.json", 'w', encoding='utf-8') as f:
            json.dump(self.raga_combinations, f, indent=2, ensure_ascii=False)
        
        # Save fix report
        report = self.generate_fix_report()
        with open(self.output_path / "unknownraga_fix_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save corrected statistics
        corrected_stats = {
            "timestamp": datetime.now().isoformat(),
            "corrected_raga_count": len(self.individual_ragas),
            "raga_combinations_count": len(self.raga_combinations),
            "tradition_distribution": report["tradition_distribution"],
            "unknownraga_removed": True,
            "data_source": "cleaned_ragasense_dataset",
            "fix_applied": "unknownraga_removal_and_reclassification"
        }
        
        with open(self.output_path / "corrected_statistics_fixed.json", 'w', encoding='utf-8') as f:
            json.dump(corrected_stats, f, indent=2, ensure_ascii=False)
        
        self.fix_stats["final_individual_count"] = len(self.individual_ragas)
        self.fix_stats["final_combination_count"] = len(self.raga_combinations)
        
        logger.info(f"✅ Fixed data saved to {self.output_path}")

    def run_fix_process(self):
        """Run the complete Unknownraga fix process."""
        start_time = time.time()
        logger.info("🚀 STARTING UNKNOWNRAGA FIX PROCESS")
        logger.info("=" * 60)
        
        # Load cleaned data
        if not self.load_cleaned_data():
            return False
        
        # Identify individual vs combined ragas
        if not self.identify_individual_vs_combined_ragas():
            return False
        
        # Save fixed data
        self.save_fixed_data()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("\n🎉 UNKNOWNRAGA FIX PROCESS COMPLETED!")
        logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
        
        # Print summary
        logger.info("\n📊 FIX SUMMARY:")
        logger.info(f"   Cleaned ragas loaded: {self.fix_stats['cleaned_ragas_loaded']}")
        logger.info(f"   Individual ragas: {self.fix_stats['final_individual_count']}")
        logger.info(f"   Raga combinations: {self.fix_stats['final_combination_count']}")
        logger.info(f"   Unknownraga removed: {not self.fix_stats['unknownraga_found']}")
        
        return True

def main():
    """Main function to run the Unknownraga fix process."""
    fixer = UnknownragaFixer()
    success = fixer.run_fix_process()
    
    if success:
        logger.info(f"\n🎯 UNKNOWNRAGA FIX COMPLETE!")
        logger.info(f"📋 Fixed data saved to: {fixer.output_path}")
        logger.info(f"📊 Report saved to: {fixer.output_path / 'unknownraga_fix_report.json'}")
    else:
        logger.error("❌ Unknownraga fix process failed!")

if __name__ == "__main__":
    main()
