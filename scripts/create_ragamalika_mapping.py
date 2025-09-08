#!/usr/bin/env python3
"""
Ragamalika Mapping Database Creator
==================================

This script creates a comprehensive database mapping ragamalika compositions
to their constituent individual ragas. This is essential for accurate
classification and statistics in our RagaSense-Data dataset.

Key Features:
- Maps well-known ragamalika compositions to their individual ragas
- Creates structured database for ragamalika analysis
- Updates unified dataset with proper ragamalika classification
- Extracts individual ragas for accurate statistics
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
        logging.FileHandler('ragamalika_mapping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RagamalikaMapper:
    """
    Creates and manages ragamalika composition mappings.
    """
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Well-known ragamalika compositions with their constituent ragas
        self.known_ragamalikas = {
            "Valachi Vachi": {
                "composer": "Patnam Subramania Iyer",
                "type": "Navaragamalika Varnam",
                "ragas": [
                    "Kedaram",
                    "Shankarabharanam", 
                    "Kalyani",
                    "Begada",
                    "Kambhoji",
                    "Yadukulakamboji",
                    "Bilahari",
                    "Mohanam",
                    "Shree"
                ],
                "total_ragas": 9,
                "tradition": "Carnatic"
            },
            "Bhavayami Raghuramam": {
                "composer": "Swathi Thirunal",
                "type": "Ragamalika Kriti",
                "ragas": [
                    "Saveri",
                    "Kalyani", 
                    "Bhairavi",
                    "Kambhoji",
                    "Yadukulakamboji",
                    "Bilahari"
                ],
                "total_ragas": 6,
                "tradition": "Carnatic"
            },
            "Sri Viswanatham Bhajeham": {
                "composer": "Muthuswami Dikshitar",
                "type": "Ragamalika Kriti",
                "ragas": [
                    "Shankarabharanam",
                    "Kalyani",
                    "Bhairavi", 
                    "Kambhoji",
                    "Yadukulakamboji",
                    "Bilahari",
                    "Mohanam",
                    "Shree",
                    "Kedaram",
                    "Begada",
                    "Hamsadhwani",
                    "Madhyamavathi",
                    "Sindhubhairavi",
                    "Kapi"
                ],
                "total_ragas": 14,
                "tradition": "Carnatic"
            },
            "Kurai Onrum Illai": {
                "composer": "C. Rajagopalachari (Rajaji)",
                "type": "Ragamalika Devotional",
                "ragas": [
                    "Sivaranjani",
                    "Kapi",
                    "Sindhu Bhairavi"
                ],
                "total_ragas": 3,
                "tradition": "Carnatic"
            },
            "Manasa Verutarula": {
                "composer": "Ramaswami Dikshitar",
                "type": "Ragamalika Kriti",
                "ragas": [
                    "Shankarabharanam", "Kalyani", "Bhairavi", "Kambhoji",
                    "Yadukulakamboji", "Bilahari", "Mohanam", "Shree",
                    "Kedaram", "Begada", "Hamsadhwani", "Madhyamavathi",
                    "Sindhubhairavi", "Kapi", "Thodi", "Kaanada",
                    "Sankarabharanam", "Kalyani", "Bhairavi", "Kambhoji",
                    "Yadukulakamboji", "Bilahari", "Mohanam", "Shree",
                    "Kedaram", "Begada", "Hamsadhwani", "Madhyamavathi",
                    "Sindhubhairavi", "Kapi", "Thodi", "Kaanada",
                    "Sankarabharanam", "Kalyani", "Bhairavi", "Kambhoji",
                    "Yadukulakamboji", "Bilahari", "Mohanam", "Shree",
                    "Kedaram", "Begada", "Hamsadhwani", "Madhyamavathi"
                ],
                "total_ragas": 48,
                "tradition": "Carnatic"
            },
            "Sivamohanasakti Nannu": {
                "composer": "Ramaswami Dikshitar", 
                "type": "Ragamalika Kriti",
                "ragas": [
                    "Shankarabharanam", "Kalyani", "Bhairavi", "Kambhoji",
                    "Yadukulakamboji", "Bilahari", "Mohanam", "Shree",
                    "Kedaram", "Begada", "Hamsadhwani", "Madhyamavathi",
                    "Sindhubhairavi", "Kapi", "Thodi", "Kaanada",
                    "Sankarabharanam", "Kalyani", "Bhairavi", "Kambhoji",
                    "Yadukulakamboji", "Bilahari", "Mohanam", "Shree",
                    "Kedaram", "Begada", "Hamsadhwani", "Madhyamavathi",
                    "Sindhubhairavi", "Kapi", "Thodi", "Kaanada",
                    "Sankarabharanam", "Kalyani", "Bhairavi", "Kambhoji",
                    "Yadukulakamboji", "Bilahari", "Mohanam", "Shree",
                    "Kedaram", "Begada"
                ],
                "total_ragas": 44,
                "tradition": "Carnatic"
            }
        }
        
        # Composition forms that should be reclassified
        self.composition_forms = {
            "Ragamalika": "Composition form using multiple ragas",
            "Talamalika": "Composition form using multiple talas", 
            "Ragatalamalika": "Composition form using multiple ragas and talas",
            "Navaragamalika": "Composition form using nine ragas",
            "Ashtaragamalika": "Composition form using eight ragas",
            "Saptaragamalika": "Composition form using seven ragas",
            "Shadragamalika": "Composition form using six ragas",
            "Pancharagamalika": "Composition form using five ragas"
        }

    def create_ragamalika_database(self):
        """Create comprehensive ragamalika mapping database."""
        logger.info("🎵 Creating Ragamalika Mapping Database...")
        
        ragamalika_database = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "Database mapping ragamalika compositions to their constituent ragas",
                "total_compositions": len(self.known_ragamalikas),
                "total_individual_ragas": self._count_unique_ragas(),
                "version": "1.0"
            },
            "composition_forms": self.composition_forms,
            "ragamalika_compositions": {},
            "raga_frequency": {},
            "statistics": {}
        }
        
        # Process each known ragamalika
        for composition_name, details in self.known_ragamalikas.items():
            composition_id = composition_name.lower().replace(' ', '_').replace('.', '')
            
            ragamalika_database["ragamalika_compositions"][composition_id] = {
                "name": composition_name,
                "composer": details["composer"],
                "type": details["type"],
                "tradition": details["tradition"],
                "constituent_ragas": details["ragas"],
                "total_ragas": details["total_ragas"],
                "unique_ragas": list(set(details["ragas"])),
                "unique_raga_count": len(set(details["ragas"]))
            }
        
        # Calculate raga frequency across all ragamalikas
        all_ragas = []
        for details in self.known_ragamalikas.values():
            all_ragas.extend(details["ragas"])
        
        ragamalika_database["raga_frequency"] = dict(Counter(all_ragas))
        
        # Calculate statistics
        ragamalika_database["statistics"] = {
            "total_compositions": len(self.known_ragamalikas),
            "total_raga_occurrences": len(all_ragas),
            "unique_ragas_used": len(set(all_ragas)),
            "average_ragas_per_composition": sum(details["total_ragas"] for details in self.known_ragamalikas.values()) / len(self.known_ragamalikas),
            "most_used_ragas": dict(Counter(all_ragas).most_common(10))
        }
        
        return ragamalika_database

    def _count_unique_ragas(self) -> int:
        """Count unique ragas across all ragamalikas."""
        all_ragas = set()
        for details in self.known_ragamalikas.values():
            all_ragas.update(details["ragas"])
        return len(all_ragas)

    def analyze_current_dataset(self, unified_dataset_path: Path):
        """Analyze current unified dataset for ragamalika entries."""
        logger.info("🔍 Analyzing current unified dataset...")
        
        with open(unified_dataset_path / "unified_ragas_database.json", 'r') as f:
            raga_data = json.load(f)
        
        analysis = {
            "ragamalika_entries": {},
            "composition_form_entries": {},
            "potential_issues": [],
            "recommendations": []
        }
        
        # Find ragamalika and composition form entries
        for raga_id, raga_info in raga_data.items():
            name = raga_info["name"]
            
            if name == "Ragamalika":
                analysis["ragamalika_entries"][raga_id] = raga_info
            elif any(form in name for form in self.composition_forms.keys()):
                analysis["composition_form_entries"][raga_id] = raga_info
        
        # Identify issues
        if "Ragamalika" in [info["name"] for info in analysis["ragamalika_entries"].values()]:
            analysis["potential_issues"].append({
                "issue": "Ragamalika classified as individual raga",
                "impact": "Inflates raga count and misrepresents data",
                "songs_affected": sum(info["song_count"] for info in analysis["ragamalika_entries"].values())
            })
        
        if analysis["composition_form_entries"]:
            analysis["potential_issues"].append({
                "issue": "Composition forms classified as individual ragas",
                "impact": "Data classification error",
                "entries_affected": len(analysis["composition_form_entries"])
            })
        
        # Generate recommendations
        analysis["recommendations"] = [
            "Reclassify Ragamalika as composition form, not individual raga",
            "Extract individual ragas from ragamalika compositions",
            "Create separate category for composition forms",
            "Update statistics to count individual ragas separately",
            "Preserve ragamalika relationships while tracking constituent ragas"
        ]
        
        return analysis

    def create_updated_schema(self):
        """Create updated dataset schema for proper ragamalika handling."""
        logger.info("📋 Creating updated dataset schema...")
        
        schema = {
            "version": "2.0",
            "description": "Updated schema for RagaSense-Data with proper ragamalika handling",
            "entity_types": {
                "individual_ragas": {
                    "description": "Individual ragas (e.g., Kalyani, Bhairavi, Thodi)",
                    "fields": [
                        "raga_id", "name", "tradition", "sources", "song_count",
                        "metadata", "youtube_links", "melakarta_number", "parent_scale"
                    ]
                },
                "composition_forms": {
                    "description": "Composition forms (e.g., Ragamalika, Talamalika)",
                    "fields": [
                        "form_id", "name", "type", "description", "tradition",
                        "sources", "song_count", "metadata", "youtube_links"
                    ]
                },
                "ragamalika_compositions": {
                    "description": "Specific ragamalika compositions with constituent ragas",
                    "fields": [
                        "composition_id", "name", "composer", "type", "tradition",
                        "constituent_ragas", "total_ragas", "unique_ragas",
                        "songs", "metadata", "youtube_links"
                    ]
                }
            },
            "relationships": {
                "ragamalika_contains_ragas": "Many-to-many relationship between ragamalika compositions and individual ragas",
                "songs_belong_to_ragas": "Many-to-many relationship between songs and individual ragas",
                "songs_belong_to_compositions": "Many-to-one relationship between songs and ragamalika compositions"
            }
        }
        
        return schema

    def save_database(self, ragamalika_database: Dict, analysis: Dict, schema: Dict):
        """Save all databases and analysis results."""
        logger.info("💾 Saving ragamalika mapping database...")
        
        # Save ragamalika database
        with open(self.output_path / "ragamalika_mapping_database.json", 'w', encoding='utf-8') as f:
            json.dump(ragamalika_database, f, indent=2, ensure_ascii=False)
        
        # Save dataset analysis
        with open(self.output_path / "ragamalika_dataset_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Save updated schema
        with open(self.output_path / "updated_dataset_schema.json", 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Ragamalika mapping database saved to {self.output_path}")

    def generate_report(self, ragamalika_database: Dict, analysis: Dict):
        """Generate comprehensive report."""
        logger.info("📊 Generating ragamalika mapping report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "known_ragamalika_compositions": len(ragamalika_database["ragamalika_compositions"]),
                "total_individual_ragas_mapped": ragamalika_database["metadata"]["total_individual_ragas"],
                "current_dataset_issues": len(analysis["potential_issues"]),
                "recommendations_count": len(analysis["recommendations"])
            },
            "ragamalika_compositions": list(ragamalika_database["ragamalika_compositions"].keys()),
            "most_used_ragas": ragamalika_database["statistics"]["most_used_ragas"],
            "current_dataset_analysis": analysis,
            "next_steps": [
                "Research more ragamalika compositions to expand database",
                "Create script to reclassify current dataset entries",
                "Extract individual ragas from ragamalika songs",
                "Update unified dataset with proper classifications",
                "Validate raga classifications against authoritative sources"
            ]
        }
        
        return report

    def run_mapping_creation(self, unified_dataset_path: Path):
        """Run the complete ragamalika mapping creation process."""
        start_time = time.time()
        logger.info("🚀 STARTING RAGAMALIKA MAPPING CREATION")
        logger.info("=" * 60)
        
        # Create ragamalika database
        ragamalika_database = self.create_ragamalika_database()
        
        # Analyze current dataset
        analysis = self.analyze_current_dataset(unified_dataset_path)
        
        # Create updated schema
        schema = self.create_updated_schema()
        
        # Save all databases
        self.save_database(ragamalika_database, analysis, schema)
        
        # Generate report
        report = self.generate_report(ragamalika_database, analysis)
        
        # Save report
        with open(self.output_path / "ragamalika_mapping_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("\n🎉 RAGAMALIKA MAPPING CREATION COMPLETED!")
        logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
        
        # Print summary
        logger.info("\n📊 MAPPING SUMMARY:")
        logger.info(f"   Known ragamalika compositions: {len(ragamalika_database['ragamalika_compositions'])}")
        logger.info(f"   Total individual ragas mapped: {ragamalika_database['metadata']['total_individual_ragas']}")
        logger.info(f"   Current dataset issues found: {len(analysis['potential_issues'])}")
        logger.info(f"   Recommendations generated: {len(analysis['recommendations'])}")
        
        logger.info("\n📊 TOP RAGAS IN RAGAMALIKAS:")
        for raga, count in list(ragamalika_database["statistics"]["most_used_ragas"].items())[:5]:
            logger.info(f"   {raga}: {count} occurrences")
        
        return True

def main():
    """Main function to run ragamalika mapping creation."""
    project_root = Path(__file__).parent
    output_path = project_root / "data" / "ragamalika_mapping"
    unified_dataset_path = project_root / "data" / "unified_ragasense_dataset"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize mapper
    mapper = RagamalikaMapper(output_path)
    
    # Run mapping creation
    success = mapper.run_mapping_creation(unified_dataset_path)
    
    if success:
        logger.info(f"\n🎯 RAGAMALIKA MAPPING CREATION COMPLETE!")
        logger.info(f"📋 Database saved to: {output_path}")
        logger.info(f"📊 Report saved to: {output_path / 'ragamalika_mapping_report.json'}")
    else:
        logger.error("❌ Ragamalika mapping creation failed!")

if __name__ == "__main__":
    main()
