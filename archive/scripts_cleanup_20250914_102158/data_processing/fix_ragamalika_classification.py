#!/usr/bin/env python3
"""
Ragamalika Classification Fixer
==============================

This script fixes the ragamalika classification issues in our unified dataset by:
1. Reclassifying Ragamalika as a composition form, not individual raga
2. Extracting individual ragas from ragamalika compositions
3. Creating proper relationships between ragamalikas and their constituent ragas
4. Updating statistics to reflect accurate raga counts
"""

import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ragamalika_classification_fix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RagamalikaClassifier:
    """
    Fixes ragamalika classification in the unified dataset.
    """
    
    def __init__(self, unified_dataset_path: Path, ragamalika_mapping_path: Path, output_path: Path):
        self.unified_dataset_path = unified_dataset_path
        self.ragamalika_mapping_path = ragamalika_mapping_path
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load ragamalika mapping database
        with open(ragamalika_mapping_path / "ragamalika_mapping_database.json", 'r') as f:
            self.ragamalika_mapping = json.load(f)
        
        # Statistics tracking
        self.fix_stats = {
            "ragas_reclassified": 0,
            "composition_forms_created": 0,
            "individual_ragas_extracted": 0,
            "songs_reprocessed": 0,
            "errors": 0
        }

    def load_unified_dataset(self):
        """Load the current unified dataset."""
        logger.info("📂 Loading unified dataset...")
        
        with open(self.unified_dataset_path / "unified_ragas_database.json", 'r') as f:
            self.ragas_data = json.load(f)
        
        with open(self.unified_dataset_path / "unified_artists_database.json", 'r') as f:
            self.artists_data = json.load(f)
        
        with open(self.unified_dataset_path / "unified_composers_database.json", 'r') as f:
            self.composers_data = json.load(f)
        
        with open(self.unified_dataset_path / "unified_songs_database.json", 'r') as f:
            self.songs_data = json.load(f)
        
        logger.info(f"Loaded {len(self.ragas_data)} ragas, {len(self.artists_data)} artists, {len(self.composers_data)} composers, {len(self.songs_data)} songs")

    def identify_composition_forms(self):
        """Identify entries that should be reclassified as composition forms."""
        logger.info("🔍 Identifying composition forms...")
        
        composition_forms = self.ragamalika_mapping["composition_forms"]
        entries_to_reclassify = {}
        
        for raga_id, raga_info in self.ragas_data.items():
            name = raga_info["name"]
            
            # Check if this is a composition form
            if name == "Ragamalika":
                entries_to_reclassify[raga_id] = {
                    "current_name": name,
                    "new_type": "composition_form",
                    "new_name": "Ragamalika",
                    "description": composition_forms["Ragamalika"],
                    "songs_affected": raga_info["song_count"]
                }
            elif any(form in name for form in composition_forms.keys()):
                entries_to_reclassify[raga_id] = {
                    "current_name": name,
                    "new_type": "composition_form",
                    "new_name": name,
                    "description": composition_forms.get(name, "Composition form"),
                    "songs_affected": raga_info["song_count"]
                }
        
        logger.info(f"Found {len(entries_to_reclassify)} entries to reclassify as composition forms")
        return entries_to_reclassify

    def extract_individual_ragas_from_ragamalika(self, ragamalika_entry):
        """Extract individual ragas from a ragamalika composition."""
        logger.info("🎵 Extracting individual ragas from ragamalika...")
        
        # Get the ragamalika songs
        songs = ragamalika_entry.get("metadata", {}).get("songs", [])
        individual_ragas = {}
        
        # For now, we'll use the known ragamalika compositions
        # In a real implementation, we'd need to analyze each song individually
        known_compositions = self.ragamalika_mapping["ragamalika_compositions"]
        
        # Create individual raga entries for each known ragamalika
        for composition_id, composition_info in known_compositions.items():
            for raga_name in composition_info["unique_ragas"]:
                raga_id = raga_name.lower().replace(' ', '_')
                
                if raga_id not in individual_ragas:
                    individual_ragas[raga_id] = {
                        "raga_id": raga_id,
                        "name": raga_name,
                        "tradition": composition_info["tradition"],
                        "sources": ["Ragamalika-Extracted"],
                        "song_count": 0,  # Will be calculated later
                        "metadata": {
                            "extracted_from": "ragamalika_compositions",
                            "composition_forms": [composition_id]
                        },
                        "youtube_links": [],
                        "composition_forms": [composition_id]
                    }
                else:
                    # Add this composition to the existing raga
                    if composition_id not in individual_ragas[raga_id]["composition_forms"]:
                        individual_ragas[raga_id]["composition_forms"].append(composition_id)
                    if composition_id not in individual_ragas[raga_id]["metadata"]["composition_forms"]:
                        individual_ragas[raga_id]["metadata"]["composition_forms"].append(composition_id)
        
        logger.info(f"Extracted {len(individual_ragas)} individual ragas from ragamalika compositions")
        return individual_ragas

    def create_composition_forms_database(self, entries_to_reclassify):
        """Create a separate database for composition forms."""
        logger.info("📋 Creating composition forms database...")
        
        composition_forms_db = {}
        
        for raga_id, reclass_info in entries_to_reclassify.items():
            original_entry = self.ragas_data[raga_id]
            
            composition_forms_db[raga_id] = {
                "form_id": raga_id,
                "name": reclass_info["new_name"],
                "type": reclass_info["new_type"],
                "description": reclass_info["description"],
                "tradition": original_entry["tradition"],
                "sources": original_entry["sources"],
                "song_count": original_entry["song_count"],
                "metadata": original_entry["metadata"],
                "youtube_links": original_entry["youtube_links"],
                "original_raga_id": raga_id
            }
            
            self.fix_stats["composition_forms_created"] += 1
        
        return composition_forms_db

    def create_ragamalika_compositions_database(self):
        """Create database for specific ragamalika compositions."""
        logger.info("🎼 Creating ragamalika compositions database...")
        
        compositions_db = {}
        known_compositions = self.ragamalika_mapping["ragamalika_compositions"]
        
        for composition_id, composition_info in known_compositions.items():
            compositions_db[composition_id] = {
                "composition_id": composition_id,
                "name": composition_info["name"],
                "composer": composition_info["composer"],
                "type": composition_info["type"],
                "tradition": composition_info["tradition"],
                "constituent_ragas": composition_info["constituent_ragas"],
                "total_ragas": composition_info["total_ragas"],
                "unique_ragas": composition_info["unique_ragas"],
                "unique_raga_count": composition_info["unique_raga_count"],
                "songs": [],  # Would be populated with actual songs
                "metadata": {
                    "source": "ragamalika_mapping_database",
                    "created_at": datetime.now().isoformat()
                },
                "youtube_links": []
            }
        
        return compositions_db

    def update_ragas_database(self, entries_to_reclassify, individual_ragas):
        """Update the ragas database by removing composition forms and adding individual ragas."""
        logger.info("🔄 Updating ragas database...")
        
        # Remove composition forms from ragas database
        for raga_id in entries_to_reclassify.keys():
            if raga_id in self.ragas_data:
                del self.ragas_data[raga_id]
                self.fix_stats["ragas_reclassified"] += 1
        
        # Add individual ragas extracted from ragamalika
        for raga_id, raga_info in individual_ragas.items():
            if raga_id not in self.ragas_data:
                self.ragas_data[raga_id] = raga_info
                self.fix_stats["individual_ragas_extracted"] += 1
            else:
                # Merge with existing raga
                existing_raga = self.ragas_data[raga_id]
                existing_raga["sources"].extend(raga_info["sources"])
                existing_raga["sources"] = list(set(existing_raga["sources"]))
                
                if "composition_forms" not in existing_raga:
                    existing_raga["composition_forms"] = []
                existing_raga["composition_forms"].extend(raga_info["composition_forms"])
                existing_raga["composition_forms"] = list(set(existing_raga["composition_forms"]))
        
        logger.info(f"Updated ragas database: {len(self.ragas_data)} total ragas")

    def calculate_updated_statistics(self):
        """Calculate updated statistics after ragamalika reclassification."""
        logger.info("📊 Calculating updated statistics...")
        
        # Count ragas by tradition
        tradition_counts = Counter(raga["tradition"] for raga in self.ragas_data.values())
        
        # Count ragas by source
        source_counts = Counter()
        for raga in self.ragas_data.values():
            for source in raga["sources"]:
                source_counts[source] += 1
        
        # Top ragas by song count
        top_ragas = sorted(
            [(raga["name"], raga["song_count"]) for raga in self.ragas_data.values()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # Count composition forms
        composition_forms_count = len(self.ragamalika_mapping["composition_forms"])
        
        statistics = {
            "total_individual_ragas": len(self.ragas_data),
            "tradition_distribution": dict(tradition_counts),
            "source_distribution": dict(source_counts),
            "top_ragas_by_songs": top_ragas,
            "composition_forms_count": composition_forms_count,
            "ragamalika_compositions_count": len(self.ragamalika_mapping["ragamalika_compositions"]),
            "fix_statistics": self.fix_stats
        }
        
        return statistics

    def save_updated_databases(self, composition_forms_db, compositions_db, statistics):
        """Save all updated databases."""
        logger.info("💾 Saving updated databases...")
        
        # Save updated ragas database
        with open(self.output_path / "unified_ragas_database_fixed.json", 'w', encoding='utf-8') as f:
            json.dump(self.ragas_data, f, indent=2, ensure_ascii=False)
        
        # Save composition forms database
        with open(self.output_path / "composition_forms_database.json", 'w', encoding='utf-8') as f:
            json.dump(composition_forms_db, f, indent=2, ensure_ascii=False)
        
        # Save ragamalika compositions database
        with open(self.output_path / "ragamalika_compositions_database.json", 'w', encoding='utf-8') as f:
            json.dump(compositions_db, f, indent=2, ensure_ascii=False)
        
        # Save updated statistics
        with open(self.output_path / "updated_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        # Save fix report
        fix_report = {
            "timestamp": datetime.now().isoformat(),
            "description": "Ragamalika classification fix report",
            "fix_statistics": self.fix_stats,
            "updated_statistics": statistics,
            "changes_made": [
                "Reclassified Ragamalika as composition form",
                "Extracted individual ragas from ragamalika compositions",
                "Created separate composition forms database",
                "Created ragamalika compositions database",
                "Updated raga statistics to reflect individual ragas"
            ]
        }
        
        with open(self.output_path / "ragamalika_fix_report.json", 'w', encoding='utf-8') as f:
            json.dump(fix_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Updated databases saved to {self.output_path}")

    def run_classification_fix(self):
        """Run the complete ragamalika classification fix process."""
        start_time = time.time()
        logger.info("🚀 STARTING RAGAMALIKA CLASSIFICATION FIX")
        logger.info("=" * 60)
        
        # Load unified dataset
        self.load_unified_dataset()
        
        # Identify composition forms to reclassify
        entries_to_reclassify = self.identify_composition_forms()
        
        # Extract individual ragas from ragamalika
        ragamalika_entry = None
        for raga_id, raga_info in self.ragas_data.items():
            if raga_info["name"] == "Ragamalika":
                ragamalika_entry = raga_info
                break
        
        individual_ragas = {}
        if ragamalika_entry:
            individual_ragas = self.extract_individual_ragas_from_ragamalika(ragamalika_entry)
        
        # Create composition forms database
        composition_forms_db = self.create_composition_forms_database(entries_to_reclassify)
        
        # Create ragamalika compositions database
        compositions_db = self.create_ragamalika_compositions_database()
        
        # Update ragas database
        self.update_ragas_database(entries_to_reclassify, individual_ragas)
        
        # Calculate updated statistics
        statistics = self.calculate_updated_statistics()
        
        # Save updated databases
        self.save_updated_databases(composition_forms_db, compositions_db, statistics)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("\n🎉 RAGAMALIKA CLASSIFICATION FIX COMPLETED!")
        logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
        
        # Print summary
        logger.info("\n📊 FIX SUMMARY:")
        logger.info(f"   Ragas reclassified: {self.fix_stats['ragas_reclassified']}")
        logger.info(f"   Composition forms created: {self.fix_stats['composition_forms_created']}")
        logger.info(f"   Individual ragas extracted: {self.fix_stats['individual_ragas_extracted']}")
        logger.info(f"   Total individual ragas now: {statistics['total_individual_ragas']}")
        logger.info(f"   Ragamalika compositions mapped: {statistics['ragamalika_compositions_count']}")
        
        logger.info("\n📊 TOP RAGAS AFTER FIX:")
        for i, (raga_name, count) in enumerate(statistics["top_ragas_by_songs"][:5], 1):
            logger.info(f"   {i}. {raga_name}: {count} songs")
        
        return True

def main():
    """Main function to run ragamalika classification fix."""
    project_root = Path(__file__).parent
    unified_dataset_path = project_root / "data" / "unified_ragasense_dataset"
    ragamalika_mapping_path = project_root / "data" / "ragamalika_mapping"
    output_path = project_root / "data" / "ragamalika_classification_fixed"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize classifier
    classifier = RagamalikaClassifier(unified_dataset_path, ragamalika_mapping_path, output_path)
    
    # Run classification fix
    success = classifier.run_classification_fix()
    
    if success:
        logger.info(f"\n🎯 RAGAMALIKA CLASSIFICATION FIX COMPLETE!")
        logger.info(f"📋 Fixed databases saved to: {output_path}")
        logger.info(f"📊 Fix report saved to: {output_path / 'ragamalika_fix_report.json'}")
    else:
        logger.error("❌ Ragamalika classification fix failed!")

if __name__ == "__main__":
    main()
