#!/usr/bin/env python3
"""
Fix Tradition Classification
===========================

This script fixes the tradition classification in the corrected raga dataset
to achieve the proper breakdown:
- Carnatic: 605 (487 unique + 118 shared)
- Hindustani: 854 (736 unique + 118 shared)  
- Both: 118 (shared between traditions)

The fix is based on the original file paths from the unified_ragas.json dataset.
"""

import json
from pathlib import Path
import logging
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_tradition_classification():
    """Fix tradition classification based on original file paths."""
    
    logger.info("🚀 Fixing tradition classification...")
    
    # Load the original dataset to get file paths
    original_file = Path("data/organized_processed/unified_ragas.json")
    corrected_file = Path("data/organized_processed/unified_ragas_corrected.json")
    
    if not original_file.exists():
        logger.error(f"❌ Original file not found: {original_file}")
        return False
        
    if not corrected_file.exists():
        logger.error(f"❌ Corrected file not found: {corrected_file}")
        return False
    
    logger.info("📖 Loading original and corrected datasets...")
    with open(original_file, 'r') as f:
        original_data = json.load(f)
    
    with open(corrected_file, 'r') as f:
        corrected_data = json.load(f)
    
    logger.info(f"📊 Original data: {len(original_data)} entries")
    logger.info(f"📊 Corrected data: {len(corrected_data)} entries")
    
    # Create a mapping from raga name to file paths
    raga_file_paths = defaultdict(list)
    
    for raga_id, raga_data in original_data.items():
        name = raga_data.get('name', '').strip().lower()
        file_path = raga_data.get('file_path', '')
        
        if name and file_path:
            raga_file_paths[name].append(file_path)
    
    logger.info(f"📊 Found file paths for {len(raga_file_paths)} unique raga names")
    
    # Fix tradition classification in corrected data
    tradition_fixes = {
        'Carnatic': 0,
        'Hindustani': 0, 
        'Both': 0,
        'Unknown': 0
    }
    
    for raga_id, raga_data in corrected_data.items():
        name = raga_data.get('name', '').strip().lower()
        
        if name in raga_file_paths:
            file_paths = raga_file_paths[name]
            
            # Determine tradition based on file paths
            has_carnatic = any('carnatic' in path.lower() for path in file_paths)
            has_hindustani = any('hindustani' in path.lower() for path in file_paths)
            
            if has_carnatic and has_hindustani:
                tradition = 'Both'
                tradition_fixes['Both'] += 1
            elif has_carnatic:
                tradition = 'Carnatic'
                tradition_fixes['Carnatic'] += 1
            elif has_hindustani:
                tradition = 'Hindustani'
                tradition_fixes['Hindustani'] += 1
            else:
                tradition = 'Unknown'
                tradition_fixes['Unknown'] += 1
            
            # Update the tradition
            raga_data['tradition'] = tradition
            
            # Also restore the file_path if it's missing
            if not raga_data.get('file_path'):
                raga_data['file_path'] = file_paths[0]  # Use first file path
        else:
            # If no file path found, keep existing tradition but log it
            existing_tradition = raga_data.get('tradition', 'Unknown')
            tradition_fixes[existing_tradition] += 1
            logger.warning(f"⚠️ No file path found for raga: {name}")
    
    # Save the fixed dataset
    output_file = Path("data/organized_processed/unified_ragas_tradition_fixed.json")
    with open(output_file, 'w') as f:
        json.dump(corrected_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved tradition-fixed dataset to: {output_file}")
    
    # Create summary report
    summary = {
        "total_ragas": len(corrected_data),
        "tradition_breakdown": tradition_fixes,
        "target_breakdown": {
            "Carnatic": 605,
            "Hindustani": 854,
            "Both": 118
        },
        "source_files": {
            "original": str(original_file),
            "corrected": str(corrected_file),
            "output": str(output_file)
        },
        "fix_date": "2025-09-12",
        "notes": [
            "Tradition classification based on original file paths",
            "Carnatic ragas: found in Carnatic/ directory",
            "Hindustani ragas: found in Hindustani/ directory", 
            "Both: found in both directories"
        ]
    }
    
    summary_file = Path("data/organized_processed/tradition_fix_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📄 Summary report saved to: {summary_file}")
    
    # Log the results
    logger.info("✅ Tradition classification fixed:")
    logger.info(f"   Carnatic: {tradition_fixes['Carnatic']}")
    logger.info(f"   Hindustani: {tradition_fixes['Hindustani']}")
    logger.info(f"   Both: {tradition_fixes['Both']}")
    logger.info(f"   Unknown: {tradition_fixes['Unknown']}")
    
    # Check if we achieved the target
    target_carnatic = 605
    target_hindustani = 854
    target_both = 118
    
    carnatic_diff = tradition_fixes['Carnatic'] - target_carnatic
    hindustani_diff = tradition_fixes['Hindustani'] - target_hindustani
    both_diff = tradition_fixes['Both'] - target_both
    
    logger.info("🎯 Target comparison:")
    logger.info(f"   Carnatic: {tradition_fixes['Carnatic']} (target: {target_carnatic}, diff: {carnatic_diff:+d})")
    logger.info(f"   Hindustani: {tradition_fixes['Hindustani']} (target: {target_hindustani}, diff: {hindustani_diff:+d})")
    logger.info(f"   Both: {tradition_fixes['Both']} (target: {target_both}, diff: {both_diff:+d})")
    
    if abs(carnatic_diff) <= 10 and abs(hindustani_diff) <= 10 and abs(both_diff) <= 10:
        logger.info("🎉 Tradition classification is very close to target!")
    else:
        logger.warning("⚠️ Tradition classification needs further refinement")
    
    return True

if __name__ == "__main__":
    success = fix_tradition_classification()
    if success:
        print("\n✅ Tradition classification fix completed successfully!")
    else:
        print("\n❌ Tradition classification fix failed!")
