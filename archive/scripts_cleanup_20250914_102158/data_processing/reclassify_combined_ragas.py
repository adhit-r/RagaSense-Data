#!/usr/bin/env python3
"""
Reclassify Combined Ragas
========================

This script reclassifies combined raga names as "Raga Combinations" rather than
individual ragas, significantly reducing our raga count and providing more
accurate statistics.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('combined_raga_reclassification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CombinedRagaReclassifier:
    """
    Reclassifies combined raga names as raga combinations, not individual ragas.
    """
    
    def __init__(self, unified_dataset_path: Path, output_path: Path):
        self.unified_dataset_path = unified_dataset_path
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.reclassification_stats = {
            "total_entries_processed": 0,
            "individual_ragas": 0,
            "combined_ragas": 0,
            "raga_combinations_created": 0,
            "songs_reclassified": 0,
            "errors": 0
        }

    def load_unified_dataset(self):
        """Load the current unified dataset."""
        logger.info("📂 Loading unified dataset...")
        
        with open(self.unified_dataset_path / "unified_ragas_database.json", 'r') as f:
            self.ragas_data = json.load(f)
        
        logger.info(f"Loaded {len(self.ragas_data)} raga entries")

    def identify_combined_ragas(self):
        """Identify entries that are combined ragas vs individual ragas."""
        logger.info("🔍 Identifying combined vs individual ragas...")
        
        individual_ragas = {}
        combined_ragas = {}
        
        for raga_id, raga_info in self.ragas_data.items():
            name = raga_info["name"]
            
            # Check if this is a combined raga
            if ',' in name or ' and ' in name.lower() or '&' in name:
                combined_ragas[raga_id] = raga_info
                self.reclassification_stats["combined_ragas"] += 1
            else:
                individual_ragas[raga_id] = raga_info
                self.reclassification_stats["individual_ragas"] += 1
            
            self.reclassification_stats["total_entries_processed"] += 1
        
        logger.info(f"Found {len(individual_ragas)} individual ragas")
        logger.info(f"Found {len(combined_ragas)} combined ragas")
        
        return individual_ragas, combined_ragas

    def create_raga_combinations_database(self, combined_ragas):
        """Create a separate database for raga combinations."""
        logger.info("🎵 Creating raga combinations database...")
        
        raga_combinations_db = {}
        
        for raga_id, raga_info in combined_ragas.items():
            name = raga_info["name"]
            
            # Parse individual ragas from the combined name
            individual_ragas = self._parse_individual_ragas(name)
            
            raga_combinations_db[raga_id] = {
                "combination_id": raga_id,
                "name": name,
                "type": "raga_combination",
                "individual_ragas": individual_ragas,
                "individual_raga_count": len(individual_ragas),
                "tradition": raga_info["tradition"],
                "sources": raga_info["sources"],
                "song_count": raga_info["song_count"],
                "metadata": raga_info["metadata"],
                "youtube_links": raga_info["youtube_links"],
                "original_raga_id": raga_id,
                "description": f"Combination of {len(individual_ragas)} ragas: {', '.join(individual_ragas)}"
            }
            
            self.reclassification_stats["raga_combinations_created"] += 1
            self.reclassification_stats["songs_reclassified"] += raga_info["song_count"]
        
        return raga_combinations_db

    def _parse_individual_ragas(self, combined_name):
        """Parse individual raga names from a combined raga name."""
        # Split by comma, 'and', or '&'
        if ',' in combined_name:
            ragas = [raga.strip() for raga in combined_name.split(',')]
        elif ' and ' in combined_name.lower():
            ragas = [raga.strip() for raga in combined_name.split(' and ')]
        elif '&' in combined_name:
            ragas = [raga.strip() for raga in combined_name.split('&')]
        else:
            ragas = [combined_name.strip()]
        
        return ragas

    def calculate_corrected_statistics(self, individual_ragas, raga_combinations_db):
        """Calculate corrected statistics after reclassification."""
        logger.info("📊 Calculating corrected statistics...")
        
        # Count ragas by tradition
        tradition_counts = Counter(raga["tradition"] for raga in individual_ragas.values())
        
        # Count raga combinations by tradition
        combination_tradition_counts = Counter(combo["tradition"] for combo in raga_combinations_db.values())
        
        # Count ragas by source
        source_counts = Counter()
        for raga in individual_ragas.values():
            for source in raga["sources"]:
                source_counts[source] += 1
        
        # Top ragas by song count
        top_ragas = sorted(
            [(raga["name"], raga["song_count"]) for raga in individual_ragas.values()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # Top raga combinations by song count
        top_combinations = sorted(
            [(combo["name"], combo["song_count"]) for combo in raga_combinations_db.values()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        statistics = {
            "corrected_raga_count": len(individual_ragas),
            "raga_combinations_count": len(raga_combinations_db),
            "original_raga_count": len(individual_ragas) + len(raga_combinations_db),
            "reduction_percentage": (len(raga_combinations_db) / (len(individual_ragas) + len(raga_combinations_db))) * 100,
            "tradition_distribution": dict(tradition_counts),
            "combination_tradition_distribution": dict(combination_tradition_counts),
            "source_distribution": dict(source_counts),
            "top_ragas_by_songs": top_ragas,
            "top_raga_combinations_by_songs": top_combinations,
            "reclassification_statistics": self.reclassification_stats
        }
        
        return statistics

    def save_reclassified_databases(self, individual_ragas, raga_combinations_db, statistics):
        """Save all reclassified databases."""
        logger.info("💾 Saving reclassified databases...")
        
        # Save individual ragas database (this becomes our main raga database)
        with open(self.output_path / "unified_ragas_database_corrected.json", 'w', encoding='utf-8') as f:
            json.dump(individual_ragas, f, indent=2, ensure_ascii=False)
        
        # Save raga combinations database
        with open(self.output_path / "raga_combinations_database.json", 'w', encoding='utf-8') as f:
            json.dump(raga_combinations_db, f, indent=2, ensure_ascii=False)
        
        # Save corrected statistics
        with open(self.output_path / "corrected_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        # Save reclassification report
        report = {
            "timestamp": datetime.now().isoformat(),
            "description": "Combined raga reclassification report",
            "reclassification_statistics": self.reclassification_stats,
            "corrected_statistics": statistics,
            "changes_made": [
                "Reclassified combined raga names as raga combinations",
                "Reduced raga count by removing combinations from individual ragas",
                "Created separate raga combinations database",
                "Updated statistics to reflect accurate raga counts",
                "Preserved all musical information in appropriate categories"
            ],
            "impact": {
                "raga_count_reduction": f"{statistics['reduction_percentage']:.1f}%",
                "individual_ragas": statistics["corrected_raga_count"],
                "raga_combinations": statistics["raga_combinations_count"],
                "songs_reclassified": self.reclassification_stats["songs_reclassified"]
            }
        }
        
        with open(self.output_path / "combined_raga_reclassification_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Reclassified databases saved to {self.output_path}")

    def run_reclassification(self):
        """Run the complete combined raga reclassification process."""
        start_time = time.time()
        logger.info("🚀 STARTING COMBINED RAGA RECLASSIFICATION")
        logger.info("=" * 60)
        
        # Load unified dataset
        self.load_unified_dataset()
        
        # Identify combined vs individual ragas
        individual_ragas, combined_ragas = self.identify_combined_ragas()
        
        # Create raga combinations database
        raga_combinations_db = self.create_raga_combinations_database(combined_ragas)
        
        # Calculate corrected statistics
        statistics = self.calculate_corrected_statistics(individual_ragas, raga_combinations_db)
        
        # Save reclassified databases
        self.save_reclassified_databases(individual_ragas, raga_combinations_db, statistics)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("\n🎉 COMBINED RAGA RECLASSIFICATION COMPLETED!")
        logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
        
        # Print summary
        logger.info("\n📊 RECLASSIFICATION SUMMARY:")
        logger.info(f"   Original raga count: {statistics['original_raga_count']}")
        logger.info(f"   Corrected raga count: {statistics['corrected_raga_count']}")
        logger.info(f"   Raga combinations: {statistics['raga_combinations_count']}")
        logger.info(f"   Reduction: {statistics['reduction_percentage']:.1f}%")
        logger.info(f"   Songs reclassified: {self.reclassification_stats['songs_reclassified']}")
        
        logger.info("\n📊 CORRECTED TRADITION DISTRIBUTION:")
        for tradition, count in statistics['tradition_distribution'].items():
            percentage = (count / statistics['corrected_raga_count']) * 100
            logger.info(f"   {tradition}: {count} ragas ({percentage:.1f}%)")
        
        logger.info("\n📊 TOP 5 INDIVIDUAL RAGAS BY SONGS:")
        for i, (raga_name, count) in enumerate(statistics["top_ragas_by_songs"][:5], 1):
            logger.info(f"   {i}. {raga_name}: {count} songs")
        
        logger.info("\n📊 TOP 5 RAGA COMBINATIONS BY SONGS:")
        for i, (combo_name, count) in enumerate(statistics["top_raga_combinations_by_songs"][:5], 1):
            logger.info(f"   {i}. {combo_name}: {count} songs")
        
        return True

def main():
    """Main function to run combined raga reclassification."""
    project_root = Path(__file__).parent
    unified_dataset_path = project_root / "data" / "unified_ragasense_dataset"
    output_path = project_root / "data" / "combined_raga_reclassified"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize reclassifier
    reclassifier = CombinedRagaReclassifier(unified_dataset_path, output_path)
    
    # Run reclassification
    success = reclassifier.run_reclassification()
    
    if success:
        logger.info(f"\n🎯 COMBINED RAGA RECLASSIFICATION COMPLETE!")
        logger.info(f"📋 Corrected databases saved to: {output_path}")
        logger.info(f"📊 Report saved to: {output_path / 'combined_raga_reclassification_report.json'}")
    else:
        logger.error("❌ Combined raga reclassification failed!")

if __name__ == "__main__":
    main()
