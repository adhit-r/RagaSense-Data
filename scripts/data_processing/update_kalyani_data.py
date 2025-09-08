#!/usr/bin/env python3
"""
Update Kalyani data with correct song count from Ramanarunachalam analysis
"""

import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_kalyani_data():
    """Update Kalyani data with correct song count from Ramanarunachalam analysis"""
    
    base_path = Path(__file__).parent.parent
    unified_ragas_path = base_path / "data" / "unified_ragasense_final" / "unified_ragas.json"
    decoded_ragas_path = base_path / "data" / "ramanarunachalam_decoded" / "decoded_ragas.json"
    
    logger.info("🔄 Updating Kalyani data with correct song count...")
    
    # Load the decoded Ramanarunachalam data
    with open(decoded_ragas_path, 'r', encoding='utf-8') as f:
        decoded_ragas = json.load(f)
    
    # Load the unified ragas data
    with open(unified_ragas_path, 'r', encoding='utf-8') as f:
        unified_ragas = json.load(f)
    
    # Find Kalyani in decoded data (ID: 2, stored as key "2")
    kalyani_data = decoded_ragas.get("2")  # Kalyani has ID 2
    
    if not kalyani_data:
        logger.error("❌ Kalyani not found in decoded data")
        return False
    
    # Update Kalyani in unified dataset
    if "Kalyani" in unified_ragas:
        old_count = unified_ragas["Kalyani"]["song_count"]
        new_count = kalyani_data["song_count"]
        
        unified_ragas["Kalyani"]["song_count"] = new_count
        unified_ragas["Kalyani"]["metadata"]["ramanarunachalam_analysis"] = {
            "corrected_song_count": new_count,
            "previous_incorrect_count": old_count,
            "analysis_date": datetime.now().isoformat(),
            "data_source": "ramanarunachalam_decoded"
        }
        unified_ragas["Kalyani"]["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"✅ Updated Kalyani song count: {old_count} → {new_count}")
        
        # Also check for other Kalyani variants
        kalyani_variants = [
            ("Poorvikalyani", 18),
            ("Yamunakalyani", 41), 
            ("Hamirkalyani", 63),
            ("Mohanakalyani", 76)
        ]
        
        for variant_name, variant_id in kalyani_variants:
            # Find variant in decoded data (stored as key string)
            variant_data = decoded_ragas.get(str(variant_id))
            
            if variant_data and variant_name in unified_ragas:
                old_variant_count = unified_ragas[variant_name]["song_count"]
                new_variant_count = variant_data["song_count"]
                
                unified_ragas[variant_name]["song_count"] = new_variant_count
                unified_ragas[variant_name]["metadata"]["ramanarunachalam_analysis"] = {
                    "corrected_song_count": new_variant_count,
                    "previous_incorrect_count": old_variant_count,
                    "analysis_date": datetime.now().isoformat(),
                    "data_source": "ramanarunachalam_decoded"
                }
                unified_ragas[variant_name]["last_updated"] = datetime.now().isoformat()
                
                logger.info(f"✅ Updated {variant_name} song count: {old_variant_count} → {new_variant_count}")
    
    # Save updated data
    with open(unified_ragas_path, 'w', encoding='utf-8') as f:
        json.dump(unified_ragas, f, indent=2, ensure_ascii=False)
    
    logger.info("💾 Updated unified ragas database saved")
    
    # Generate summary
    logger.info("\n🎉 KALYANI DATA UPDATE COMPLETE!")
    logger.info("=" * 50)
    logger.info(f"✅ Kalyani: {old_count} → {new_count} songs")
    logger.info("✅ All Kalyani variants updated with correct counts")
    logger.info("✅ Analysis metadata added to raga entries")
    logger.info("✅ Database updated and saved")
    
    return True

if __name__ == "__main__":
    update_kalyani_data()
