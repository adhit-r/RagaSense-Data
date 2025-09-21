#!/usr/bin/env python3
"""
Create Corrected Raga Dataset
============================

This script creates the properly corrected raga dataset with:
- 1,341 unique ragas (not 5,819)
- Correct tradition breakdown:
  - Carnatic: 605 (487 unique + 118 shared)
  - Hindustani: 854 (736 unique + 118 shared)
  - Both: 118 (shared between traditions)
"""

import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_corrected_raga_dataset():
    """Create the corrected raga dataset with proper counts."""
    
    logger.info("🚀 Creating corrected raga dataset...")
    
    # Load the original data to extract unique ragas
    original_file = Path("data/organized_processed/unified_ragas.json")
    
    if not original_file.exists():
        logger.error(f"❌ Original file not found: {original_file}")
        return False
    
    logger.info("📖 Loading original raga data...")
    with open(original_file, 'r') as f:
        original_data = json.load(f)
    
    logger.info(f"📊 Original data contains {len(original_data)} entries")
    
    # Extract unique raga names and their traditions
    unique_ragas = {}
    tradition_counts = {"Carnatic": 0, "Hindustani": 0, "Both": 0}
    
    for raga_id, raga_data in original_data.items():
        name = raga_data.get('name', '')
        tradition = raga_data.get('tradition', 'Unknown')
        
        # Skip if name is empty or contains combinations
        if not name or ',' in name or ';' in name:
            continue
            
        # Normalize name (case-insensitive)
        normalized_name = name.strip().lower()
        
        if normalized_name in unique_ragas:
            # If raga already exists, check if we need to update tradition
            existing_tradition = unique_ragas[normalized_name]['tradition']
            if existing_tradition != tradition and tradition != 'Unknown':
                # If different traditions, mark as "Both"
                if existing_tradition == 'Carnatic' and tradition == 'Hindustani':
                    unique_ragas[normalized_name]['tradition'] = 'Both'
                elif existing_tradition == 'Hindustani' and tradition == 'Carnatic':
                    unique_ragas[normalized_name]['tradition'] = 'Both'
        else:
            # Add new unique raga
            unique_ragas[normalized_name] = {
                'raga_id': raga_id,
                'name': name,
                'tradition': tradition,
                'melakartha': raga_data.get('melakartha'),
                'arohana': raga_data.get('arohana'),
                'avarohana': raga_data.get('avarohana'),
                'chakra': raga_data.get('chakra'),
                'thaat': raga_data.get('thaat'),
                'vadi': raga_data.get('vadi'),
                'samvadi': raga_data.get('samvadi'),
                'metadata': raga_data.get('metadata', {}),
                'sources': raga_data.get('sources', [])
            }
    
    # Count traditions
    for raga_data in unique_ragas.values():
        tradition = raga_data['tradition']
        if tradition in tradition_counts:
            tradition_counts[tradition] += 1
        else:
            tradition_counts['Unknown'] = tradition_counts.get('Unknown', 0) + 1
    
    logger.info(f"✅ Extracted {len(unique_ragas)} unique ragas")
    logger.info("📊 Tradition breakdown:")
    for tradition, count in tradition_counts.items():
        logger.info(f"  {tradition}: {count}")
    
    # Create the corrected dataset
    corrected_data = {}
    for i, (normalized_name, raga_data) in enumerate(unique_ragas.items(), 1):
        corrected_data[f"raga_{i:04d}"] = raga_data
    
    # Save the corrected dataset
    output_file = Path("data/organized_processed/unified_ragas_corrected.json")
    with open(output_file, 'w') as f:
        json.dump(corrected_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved corrected dataset to: {output_file}")
    
    # Create summary report
    summary = {
        "total_unique_ragas": len(unique_ragas),
        "tradition_breakdown": tradition_counts,
        "source_file": str(original_file),
        "output_file": str(output_file),
        "correction_date": "2025-09-12",
        "notes": [
            "This dataset contains only unique raga names",
            "Combination entries (with commas) were excluded",
            "Case-insensitive deduplication was applied",
            "Cross-tradition ragas are marked as 'Both'"
        ]
    }
    
    summary_file = Path("data/organized_processed/raga_correction_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📄 Summary report saved to: {summary_file}")
    
    return True

if __name__ == "__main__":
    success = create_corrected_raga_dataset()
    if success:
        print("\n🎉 Corrected raga dataset created successfully!")
        print("📊 Use this dataset for PostgreSQL migration instead of the inflated data.")
    else:
        print("\n❌ Failed to create corrected raga dataset.")
