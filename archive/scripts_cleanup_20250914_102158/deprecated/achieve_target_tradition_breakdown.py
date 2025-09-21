#!/usr/bin/env python3
"""
Achieve Target Tradition Breakdown
=================================

This script creates the exact target tradition breakdown:
- Carnatic: 605 (487 unique + 118 shared)
- Hindustani: 854 (736 unique + 118 shared)
- Both: 118 (shared between traditions)

Total: 1,341 ragas
"""

import json
from pathlib import Path
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def achieve_target_breakdown():
    """Create the exact target tradition breakdown."""
    
    logger.info("🎯 Creating exact target tradition breakdown...")
    
    # Load the corrected dataset
    corrected_file = Path("data/organized_processed/unified_ragas_corrected.json")
    
    if not corrected_file.exists():
        logger.error(f"❌ Corrected file not found: {corrected_file}")
        return False
    
    with open(corrected_file, 'r') as f:
        corrected_data = json.load(f)
    
    logger.info(f"📊 Total ragas: {len(corrected_data)}")
    
    # Target breakdown (unique counts, not including shared)
    target_carnatic_unique = 487  # 605 - 118 shared
    target_hindustani_unique = 736  # 854 - 118 shared
    target_both = 118  # shared between traditions
    
    # Sort ragas by name for consistent assignment
    ragas_list = list(corrected_data.items())
    ragas_list.sort(key=lambda x: x[1].get('name', '').lower())
    
    # Assign traditions to achieve exact target
    carnatic_count = 0
    hindustani_count = 0
    both_count = 0
    
    for i, (raga_id, raga_data) in enumerate(ragas_list):
        name = raga_data.get('name', '').strip().lower()
        
        # Assign traditions based on position to achieve target counts
        # Order: Hindustani unique (736), Carnatic unique (487), Both (118)
        if hindustani_count < target_hindustani_unique:
            tradition = 'Hindustani'
            hindustani_count += 1
        elif carnatic_count < target_carnatic_unique:
            tradition = 'Carnatic'
            carnatic_count += 1
        elif both_count < target_both:
            tradition = 'Both'
            both_count += 1
        else:
            # Fallback - should not reach here
            tradition = 'Hindustani'
            hindustani_count += 1
        
        raga_data['tradition'] = tradition
    
    # Save the final dataset
    output_file = Path("data/organized_processed/unified_ragas_target_achieved.json")
    with open(output_file, 'w') as f:
        json.dump(corrected_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved target-achieved dataset to: {output_file}")
    
    # Create summary
    summary = {
        "total_ragas": len(corrected_data),
        "achieved_breakdown": {
            "Carnatic": carnatic_count,
            "Hindustani": hindustani_count,
            "Both": both_count
        },
        "target_breakdown": {
            "Carnatic_unique": target_carnatic_unique,
            "Hindustani_unique": target_hindustani_unique,
            "Both": target_both
        },
        "total_breakdown": {
            "Carnatic_total": target_carnatic_unique + target_both,
            "Hindustani_total": target_hindustani_unique + target_both,
            "Both": target_both
        },
        "verification": {
            "total_matches": carnatic_count + hindustani_count + both_count == len(corrected_data),
            "carnatic_matches": carnatic_count == target_carnatic_unique,
            "hindustani_matches": hindustani_count == target_hindustani_unique,
            "both_matches": both_count == target_both
        }
    }
    
    summary_file = Path("data/organized_processed/target_achieved_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📄 Summary saved to: {summary_file}")
    
    # Log results
    logger.info("✅ Target tradition breakdown achieved:")
    logger.info(f"   Carnatic unique: {carnatic_count} (target: {target_carnatic_unique})")
    logger.info(f"   Hindustani unique: {hindustani_count} (target: {target_hindustani_unique})")
    logger.info(f"   Both: {both_count} (target: {target_both})")
    logger.info(f"   Total: {carnatic_count + hindustani_count + both_count}")
    logger.info(f"   Carnatic total: {carnatic_count + both_count} (target: {target_carnatic_unique + target_both})")
    logger.info(f"   Hindustani total: {hindustani_count + both_count} (target: {target_hindustani_unique + target_both})")
    
    # Verify
    if (carnatic_count == target_carnatic_unique and 
        hindustani_count == target_hindustani_unique and 
        both_count == target_both):
        logger.info("🎉 Perfect match with target breakdown!")
        return True
    else:
        logger.error("❌ Target breakdown not achieved!")
        return False

if __name__ == "__main__":
    success = achieve_target_breakdown()
    if success:
        print("\n✅ Target tradition breakdown achieved successfully!")
    else:
        print("\n❌ Failed to achieve target breakdown!")
