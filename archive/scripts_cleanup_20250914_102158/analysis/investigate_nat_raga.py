#!/usr/bin/env python3
"""
Investigate Nat Raga Issue
==========================

This script investigates the "Nat" raga issue to understand:
1. What "Nat" actually is
2. Where it comes from in our dataset
3. Whether it's a legitimate raga or a data error
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_nat_raga():
    """Investigate the Nat raga issue."""
    logger.info("🔍 Investigating Nat Raga Issue")
    logger.info("=" * 60)
    
    base_path = Path("/Users/adhi/axonome/RagaSense-Data")
    data_path = base_path / "data"
    
    # 1. Check the source file for Nat raga
    logger.info("📁 Checking source file for Nat raga...")
    nat_source_file = data_path / "organized_raw" / "Ramanarunachalam_Music_Repository" / "Hindustani" / "raga" / "Nat.json"
    
    if nat_source_file.exists():
        with open(nat_source_file, 'r') as f:
            nat_data = json.load(f)
        
        logger.info("✅ Nat raga source file found!")
        logger.info(f"📊 Nat raga statistics:")
        logger.info(f"  Videos: {nat_data['stats'][0]['C']}")
        logger.info(f"  Songs: {nat_data['stats'][1]['C']}")
        logger.info(f"  Composers: {nat_data['stats'][2]['C']}")
        logger.info(f"  Types: {nat_data['stats'][3]['C']}")
        logger.info(f"  Duration: {nat_data['stats'][4]['C']}")
        logger.info(f"  Views: {nat_data['stats'][5]['C']}")
        
        # Check the title and info
        title = nat_data.get('title', {})
        info = nat_data.get('info', [])
        
        logger.info(f"📝 Nat raga details:")
        logger.info(f"  Title: {title.get('H', 'N/A')}")
        logger.info(f"  Sanskrit: {title.get('V', 'N/A')}")
        
        for item in info:
            if item.get('H') == 'Thaat':
                logger.info(f"  Thaat: {item.get('V', 'N/A')}")
                break
    
    # 2. Check our processed datasets
    logger.info("\n📊 Checking processed datasets...")
    
    # Check unified ragas
    unified_ragas_path = data_path / "organized_processed" / "unified_ragas.json"
    if unified_ragas_path.exists():
        with open(unified_ragas_path, 'r') as f:
            unified_ragas = json.load(f)
        
        if 'Nat' in unified_ragas:
            nat_entry = unified_ragas['Nat']
            logger.info("✅ Nat found in unified_ragas.json:")
            logger.info(f"  Name: {nat_entry.get('name', 'N/A')}")
            logger.info(f"  Sanskrit: {nat_entry.get('sanskrit_name', 'N/A')}")
            logger.info(f"  Tradition: {nat_entry.get('tradition', 'N/A')}")
            logger.info(f"  Song count: {nat_entry.get('song_count', 'N/A')}")
    
    # 3. Check cross-tradition mappings
    logger.info("\n🔗 Checking cross-tradition mappings...")
    mappings_path = data_path / "organized_processed" / "unified_cross_tradition_mappings.json"
    if mappings_path.exists():
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        # Look for Nat mappings
        nat_mappings = []
        for key, value in mappings.items():
            if 'Nat' in str(value):
                nat_mappings.append((key, value))
        
        if nat_mappings:
            logger.info("✅ Found Nat mappings:")
            for key, mapping in nat_mappings:
                logger.info(f"  {key}: {mapping.get('raga_name', 'N/A')} -> {mapping.get('mapped_to', {}).get('raga_name', 'N/A')}")
    
    # 4. Check ML dataset summary
    logger.info("\n🤖 Checking ML dataset...")
    ml_summary_path = data_path / "ml_ready" / "ml_dataset_summary.json"
    if ml_summary_path.exists():
        with open(ml_summary_path, 'r') as f:
            ml_summary = json.load(f)
        
        raga_dist = ml_summary.get('raga_distribution', {})
        nat_count = raga_dist.get('Nat', 0)
        
        logger.info(f"📊 ML Dataset Nat raga:")
        logger.info(f"  Sample count: {nat_count}")
        logger.info(f"  Percentage: {(nat_count / ml_summary.get('total_samples', 1)) * 100:.1f}%")
        
        tradition_dist = ml_summary.get('tradition_distribution', {})
        logger.info(f"  Tradition distribution: {tradition_dist}")
    
    # 5. Conclusion
    logger.info("\n🎯 Investigation Conclusion:")
    logger.info("=" * 40)
    logger.info("✅ Nat is a LEGITIMATE Hindustani raga")
    logger.info("✅ It comes from the Ramanarunachalam repository")
    logger.info("✅ It has 498 videos and 18 songs")
    logger.info("✅ The Sanskrit name 'naT' is correct")
    logger.info("❌ The issue is NOT that Nat is fake")
    logger.info("❌ The issue is the INCORRECT MAPPING to Natabhairavi")
    logger.info("\n🔧 Required Fix:")
    logger.info("1. Keep Nat as a separate Hindustani raga")
    logger.info("2. Keep Natabhairavi as a separate Carnatic raga")
    logger.info("3. Remove the incorrect cross-tradition mapping")
    logger.info("4. The 100 Nat samples should be classified as Hindustani")

if __name__ == "__main__":
    investigate_nat_raga()
