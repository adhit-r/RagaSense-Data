#!/usr/bin/env python3
"""
Raga Data Cleaning and Deduplication Script
==========================================

This script addresses the critical data quality issues identified in our analysis:
1. Split combined ragas (e.g., "Maand, Bhatiyali" → "Maand" + "Bhatiyali")
2. Remove duplicates and ensure uniqueness
3. Assign traditions to ragas
4. Clean up placeholder/unknown entries

Features:
- Intelligent raga name splitting
- Duplicate detection and removal
- Tradition assignment based on patterns
- Quality scoring and validation
- Comprehensive reporting
"""

import json
import time
from datetime import datetime
from pathlib import Path
import sys
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import pandas as pd
from collections import defaultdict, Counter
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('raga_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RagaDataCleaner:
    """
    Cleans and deduplicates raga data to ensure accuracy and uniqueness.
    """
    
    def __init__(self, data_path: Path, output_path: Path):
        self.data_path = data_path
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.original_ragas = {}
        self.cleaned_ragas = {}
        self.combined_ragas = []
        self.duplicate_ragas = []
        self.unknown_ragas = []
        
        # Tradition patterns for classification
        self.carnatic_patterns = [
            'melakarta', 'janya', 'sankarabharanam', 'kalyani', 'todi', 'bhairavi',
            'mohanam', 'hindolam', 'kambhoji', 'kapi', 'madhyamavati', 'shanmukapriya',
            'natabhairavi', 'kharaharapriya', 'hanumatodi', 'dheerasankarabharanam'
        ]
        
        self.hindustani_patterns = [
            'thaat', 'bilawal', 'yaman', 'bhairavi', 'kafi', 'asavari', 'bhairav',
            'kalyan', 'khamaj', 'marwa', 'poorvi', 'todi', 'malkauns', 'bhoopali',
            'desh', 'khamaj', 'kafi', 'asavari'
        ]
        
        self.cleaning_stats = {
            "original_count": 0,
            "combined_ragas_found": 0,
            "individual_ragas_created": 0,
            "duplicates_removed": 0,
            "unknown_ragas_removed": 0,
            "final_unique_count": 0,
            "carnatic_assigned": 0,
            "hindustani_assigned": 0,
            "both_assigned": 0,
            "unassigned": 0
        }

    def load_original_data(self):
        """Load the original raga data."""
        logger.info("📂 Loading original raga data...")
        
        ragas_path = self.data_path / "unified_ragas_database.json"
        if ragas_path.exists():
            with open(ragas_path, 'r', encoding='utf-8') as f:
                self.original_ragas = json.load(f)
            logger.info(f"✅ Loaded {len(self.original_ragas)} original ragas")
        else:
            logger.error(f"❌ Raga database not found at {ragas_path}")
            return False
        
        self.cleaning_stats["original_count"] = len(self.original_ragas)
        return True

    def identify_combined_ragas(self) -> List[Dict[str, Any]]:
        """Identify ragas that are actually combinations of multiple ragas."""
        logger.info("🔍 Identifying combined ragas...")
        
        combined_ragas = []
        
        for raga_id, raga_data in self.original_ragas.items():
            name = raga_data.get('name', raga_id)
            
            # Check for comma-separated ragas
            if ',' in name:
                individual_ragas = [r.strip() for r in name.split(',')]
                # Filter out empty strings and very short names
                individual_ragas = [r for r in individual_ragas if r and len(r) > 2]
                
                if len(individual_ragas) > 1:
                    combined_ragas.append({
                        "original_id": raga_id,
                        "original_name": name,
                        "individual_ragas": individual_ragas,
                        "song_count": raga_data.get('song_count', 0),
                        "metadata": raga_data
                    })
        
        self.combined_ragas = combined_ragas
        self.cleaning_stats["combined_ragas_found"] = len(combined_ragas)
        
        logger.info(f"✅ Found {len(combined_ragas)} combined ragas")
        return combined_ragas

    def split_combined_ragas(self):
        """Split combined ragas into individual entries."""
        logger.info("✂️ Splitting combined ragas...")
        
        individual_ragas_created = 0
        
        for combined_raga in self.combined_ragas:
            original_metadata = combined_raga["metadata"]
            song_count = combined_raga["song_count"]
            
            # Distribute song count among individual ragas
            songs_per_raga = song_count // len(combined_raga["individual_ragas"])
            remaining_songs = song_count % len(combined_raga["individual_ragas"])
            
            for i, individual_raga_name in enumerate(combined_raga["individual_ragas"]):
                # Create unique ID for individual raga
                individual_id = f"{individual_raga_name.lower().replace(' ', '_')}"
                
                # Calculate song count for this raga
                individual_song_count = songs_per_raga
                if i < remaining_songs:  # Distribute remaining songs
                    individual_song_count += 1
                
                # Create individual raga entry
                individual_raga_data = {
                    "raga_id": individual_id,
                    "name": individual_raga_name,
                    "sanskrit_name": original_metadata.get('sanskrit_name', ''),
                    "tradition": "",  # Will be assigned later
                    "song_count": individual_song_count,
                    "sample_duration": original_metadata.get('sample_duration', ''),
                    "file_path": original_metadata.get('file_path', ''),
                    "cross_tradition_mapping": {
                        "type": "unique",
                        "mapping": None,
                        "confidence": "unknown"
                    },
                    "metadata": {
                        "source": "ramanarunachalam",
                        "last_updated": datetime.now().isoformat(),
                        "quality_score": original_metadata.get('metadata', {}).get('quality_score', 1.0),
                        "split_from": combined_raga["original_id"],
                        "original_combined_name": combined_raga["original_name"]
                    }
                }
                
                self.cleaned_ragas[individual_id] = individual_raga_data
                individual_ragas_created += 1
        
        self.cleaning_stats["individual_ragas_created"] = individual_ragas_created
        logger.info(f"✅ Created {individual_ragas_created} individual ragas from combined entries")

    def process_non_combined_ragas(self):
        """Process ragas that are not combined (single ragas)."""
        logger.info("🔄 Processing non-combined ragas...")
        
        combined_raga_ids = {cr["original_id"] for cr in self.combined_ragas}
        
        for raga_id, raga_data in self.original_ragas.items():
            if raga_id not in combined_raga_ids:
                name = raga_data.get('name', raga_id)
                
                # Skip unknown/placeholder ragas
                if 'unknown' in name.lower() or 'placeholder' in name.lower():
                    self.unknown_ragas.append({
                        "raga_id": raga_id,
                        "name": name,
                        "reason": "Unknown/placeholder entry"
                    })
                    continue
                
                # Create cleaned entry
                cleaned_raga_data = raga_data.copy()
                cleaned_raga_data["metadata"] = raga_data.get("metadata", {})
                cleaned_raga_data["metadata"]["last_updated"] = datetime.now().isoformat()
                
                self.cleaned_ragas[raga_id] = cleaned_raga_data
        
        self.cleaning_stats["unknown_ragas_removed"] = len(self.unknown_ragas)
        logger.info(f"✅ Processed {len(self.cleaned_ragas)} non-combined ragas")
        logger.info(f"⚠️ Removed {len(self.unknown_ragas)} unknown/placeholder ragas")

    def remove_duplicates(self):
        """Remove duplicate ragas based on name similarity."""
        logger.info("🔍 Identifying and removing duplicates...")
        
        # Group ragas by normalized name
        name_groups = defaultdict(list)
        
        for raga_id, raga_data in self.cleaned_ragas.items():
            name = raga_data.get('name', raga_id)
            # Normalize name for comparison
            normalized_name = re.sub(r'[^\w\s]', '', name.lower().strip())
            name_groups[normalized_name].append((raga_id, raga_data))
        
        # Process groups with multiple entries
        duplicates_removed = 0
        final_ragas = {}
        
        for normalized_name, raga_entries in name_groups.items():
            if len(raga_entries) == 1:
                # No duplicates, keep as is
                raga_id, raga_data = raga_entries[0]
                final_ragas[raga_id] = raga_data
            else:
                # Multiple entries with same name - merge them
                logger.info(f"🔄 Merging {len(raga_entries)} duplicates for '{normalized_name}'")
                
                # Choose the entry with highest song count or best metadata
                best_entry = max(raga_entries, key=lambda x: x[1].get('song_count', 0))
                best_raga_id, best_raga_data = best_entry
                
                # Merge song counts and metadata
                total_songs = sum(entry[1].get('song_count', 0) for entry in raga_entries)
                best_raga_data['song_count'] = total_songs
                
                # Add merge information to metadata
                merged_from = [entry[0] for entry in raga_entries if entry[0] != best_raga_id]
                best_raga_data['metadata']['merged_from'] = merged_from
                best_raga_data['metadata']['duplicate_count'] = len(raga_entries)
                
                final_ragas[best_raga_id] = best_raga_data
                duplicates_removed += len(raga_entries) - 1
                
                # Record duplicates
                for raga_id, raga_data in raga_entries:
                    if raga_id != best_raga_id:
                        self.duplicate_ragas.append({
                            "duplicate_id": raga_id,
                            "duplicate_name": raga_data.get('name', raga_id),
                            "merged_into": best_raga_id,
                            "song_count": raga_data.get('song_count', 0)
                        })
        
        self.cleaned_ragas = final_ragas
        self.cleaning_stats["duplicates_removed"] = duplicates_removed
        
        logger.info(f"✅ Removed {duplicates_removed} duplicate ragas")
        logger.info(f"✅ Final unique ragas: {len(self.cleaned_ragas)}")

    def assign_traditions(self):
        """Assign traditions to ragas based on patterns and context."""
        logger.info("🏷️ Assigning traditions to ragas...")
        
        carnatic_count = 0
        hindustani_count = 0
        both_count = 0
        unassigned_count = 0
        
        for raga_id, raga_data in self.cleaned_ragas.items():
            name = raga_data.get('name', raga_id).lower()
            tradition = ""
            
            # Check for explicit tradition indicators
            if any(pattern in name for pattern in self.carnatic_patterns):
                if any(pattern in name for pattern in self.hindustani_patterns):
                    tradition = "Both"
                    both_count += 1
                else:
                    tradition = "Carnatic"
                    carnatic_count += 1
            elif any(pattern in name for pattern in self.hindustani_patterns):
                tradition = "Hindustani"
                hindustani_count += 1
            else:
                # Default to Carnatic for now (most of our data is Carnatic)
                tradition = "Carnatic"
                carnatic_count += 1
            
            raga_data['tradition'] = tradition
        
        self.cleaning_stats["carnatic_assigned"] = carnatic_count
        self.cleaning_stats["hindustani_assigned"] = hindustani_count
        self.cleaning_stats["both_assigned"] = both_count
        self.cleaning_stats["unassigned"] = unassigned_count
        self.cleaning_stats["final_unique_count"] = len(self.cleaned_ragas)
        
        logger.info(f"✅ Tradition assignment complete:")
        logger.info(f"   Carnatic: {carnatic_count}")
        logger.info(f"   Hindustani: {hindustani_count}")
        logger.info(f"   Both: {both_count}")

    def generate_cleaning_report(self) -> Dict[str, Any]:
        """Generate a comprehensive cleaning report."""
        logger.info("📊 Generating cleaning report...")
        
        # Top ragas by song count
        top_ragas = sorted(
            self.cleaned_ragas.items(),
            key=lambda x: x[1].get('song_count', 0),
            reverse=True
        )[:20]
        
        # Tradition distribution
        tradition_distribution = Counter(raga.get('tradition', 'Unknown') for raga in self.cleaned_ragas.values())
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "cleaning_statistics": self.cleaning_stats,
            "top_ragas_by_songs": [
                {
                    "raga_id": raga_id,
                    "name": raga_data.get('name', raga_id),
                    "song_count": raga_data.get('song_count', 0),
                    "tradition": raga_data.get('tradition', 'Unknown')
                }
                for raga_id, raga_data in top_ragas
            ],
            "tradition_distribution": dict(tradition_distribution),
            "combined_ragas_processed": [
                {
                    "original_name": cr["original_name"],
                    "individual_ragas": cr["individual_ragas"],
                    "song_count": cr["song_count"]
                }
                for cr in self.combined_ragas[:10]  # Top 10 examples
            ],
            "duplicates_removed": [
                {
                    "duplicate_name": dr["duplicate_name"],
                    "merged_into": dr["merged_into"],
                    "song_count": dr["song_count"]
                }
                for dr in self.duplicate_ragas[:10]  # Top 10 examples
            ],
            "unknown_ragas_removed": [
                {
                    "raga_id": ur["raga_id"],
                    "name": ur["name"],
                    "reason": ur["reason"]
                }
                for ur in self.unknown_ragas
            ]
        }
        
        return report

    def save_cleaned_data(self):
        """Save the cleaned raga data and reports."""
        logger.info("💾 Saving cleaned data...")
        
        # Save cleaned ragas database
        with open(self.output_path / "cleaned_ragas_database.json", 'w', encoding='utf-8') as f:
            json.dump(self.cleaned_ragas, f, indent=2, ensure_ascii=False)
        
        # Save cleaning report
        report = self.generate_cleaning_report()
        with open(self.output_path / "raga_cleaning_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save CSV exports
        pd.DataFrame.from_dict(self.cleaned_ragas, orient='index').to_csv(
            self.output_path / "cleaned_ragas_database.csv", 
            index_label="raga_id"
        )
        
        logger.info(f"✅ Cleaned data saved to {self.output_path}")

    def run_cleaning_process(self):
        """Run the complete raga cleaning process."""
        start_time = time.time()
        logger.info("🚀 STARTING RAGA DATA CLEANING PROCESS")
        logger.info("=" * 60)
        
        # Load data
        if not self.load_original_data():
            return False
        
        # Identify and split combined ragas
        self.identify_combined_ragas()
        self.split_combined_ragas()
        
        # Process non-combined ragas
        self.process_non_combined_ragas()
        
        # Remove duplicates
        self.remove_duplicates()
        
        # Assign traditions
        self.assign_traditions()
        
        # Save results
        self.save_cleaned_data()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("\n🎉 RAGA CLEANING PROCESS COMPLETED!")
        logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
        
        # Print summary
        logger.info("\n📊 CLEANING SUMMARY:")
        logger.info(f"   Original ragas: {self.cleaning_stats['original_count']}")
        logger.info(f"   Combined ragas found: {self.cleaning_stats['combined_ragas_found']}")
        logger.info(f"   Individual ragas created: {self.cleaning_stats['individual_ragas_created']}")
        logger.info(f"   Duplicates removed: {self.cleaning_stats['duplicates_removed']}")
        logger.info(f"   Unknown ragas removed: {self.cleaning_stats['unknown_ragas_removed']}")
        logger.info(f"   Final unique ragas: {self.cleaning_stats['final_unique_count']}")
        
        logger.info("\n🏷️ TRADITION ASSIGNMENT:")
        logger.info(f"   Carnatic: {self.cleaning_stats['carnatic_assigned']}")
        logger.info(f"   Hindustani: {self.cleaning_stats['hindustani_assigned']}")
        logger.info(f"   Both: {self.cleaning_stats['both_assigned']}")
        
        return True

def main():
    """Main function to run the raga cleaning process."""
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "unified_ragasense_dataset"
    output_path = project_root / "data" / "cleaned_ragasense_dataset"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize cleaner
    cleaner = RagaDataCleaner(data_path, output_path)
    
    # Run cleaning process
    success = cleaner.run_cleaning_process()
    
    if success:
        logger.info(f"\n🎯 RAGA CLEANING COMPLETE!")
        logger.info(f"📋 Cleaned data saved to: {output_path}")
        logger.info(f"📊 Report saved to: {output_path / 'raga_cleaning_report.json'}")
    else:
        logger.error("❌ Raga cleaning process failed!")

if __name__ == "__main__":
    main()
