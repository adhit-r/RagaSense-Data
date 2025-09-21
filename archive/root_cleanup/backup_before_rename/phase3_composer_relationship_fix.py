#!/usr/bin/env python3
"""
Phase 3: Fix Composer-Song Relationships
=======================================

This script addresses the critical data quality issue where 443 composers 
have 0 songs despite having song data in their metadata.

Problem: song_count field is not being properly updated from metadata
Solution: Extract song counts from metadata and update composer records
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase3_composer_fix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComposerRelationshipFixer:
    """
    Fixes composer-song relationships by extracting song counts from metadata
    and updating the song_count field properly.
    """
    
    def __init__(self):
        self.data_path = Path("data/processed")
        self.archive_path = Path("archive/data_versions")
        
        # Data containers
        self.composers_data = {}
        self.songs_data = {}
        self.fixed_composers = {}
        
        # Statistics
        self.stats = {
            "total_composers": 0,
            "composers_with_zero_songs": 0,
            "composers_with_metadata_songs": 0,
            "composers_fixed": 0,
            "total_songs_extracted": 0,
            "processing_errors": 0
        }
        
        logger.info("🔧 Composer Relationship Fixer initialized")
    
    def load_data(self):
        """Load composer and song data from various sources."""
        logger.info("📂 Loading composer and song data...")
        
        # Try multiple possible locations for composer data
        composer_files = [
            self.data_path / "unified_composers_database.json",
            self.archive_path / "unified_ragasense_dataset" / "unified_composers_database.json",
            self.archive_path / "comprehensive_unified_dataset" / "unified_composers_database.json"
        ]
        
        composer_loaded = False
        for composer_file in composer_files:
            if composer_file.exists():
                try:
                    with open(composer_file, 'r', encoding='utf-8') as f:
                        self.composers_data = json.load(f)
                    logger.info(f"✅ Loaded {len(self.composers_data)} composers from {composer_file}")
                    composer_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {composer_file}: {e}")
        
        if not composer_loaded:
            logger.error("❌ Could not load composer data from any location")
            return False
        
        # Try to load song data
        song_files = [
            self.data_path / "unified_songs_database.json",
            self.archive_path / "unified_ragasense_dataset" / "unified_songs_database.json",
            self.archive_path / "comprehensive_unified_dataset" / "unified_songs_database.json"
        ]
        
        song_loaded = False
        for song_file in song_files:
            if song_file.exists():
                try:
                    with open(song_file, 'r', encoding='utf-8') as f:
                        self.songs_data = json.load(f)
                    logger.info(f"✅ Loaded {len(self.songs_data)} songs from {song_file}")
                    song_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {song_file}: {e}")
        
        if not song_loaded:
            logger.warning("⚠️ Could not load song data - will extract from composer metadata only")
        
        return True
    
    def extract_song_count_from_metadata(self, composer_data: Dict) -> int:
        """Extract song count from composer metadata."""
        try:
            # Check stats array for song count
            if 'metadata' in composer_data and 'stats' in composer_data['metadata']:
                stats = composer_data['metadata']['stats']
                for stat in stats:
                    if isinstance(stat, dict):
                        # Look for "Songs" field
                        if stat.get('H') == 'Songs':
                            song_count_str = stat.get('C', '0')
                            # Extract number from string like "2", "15", "1.2K", etc.
                            song_count = self._parse_song_count(song_count_str)
                            if song_count > 0:
                                return song_count
            
            # Check if there's a direct songs array
            if 'songs' in composer_data and isinstance(composer_data['songs'], list):
                return len(composer_data['songs'])
            
            # Check metadata for songs array
            if 'metadata' in composer_data and 'songs' in composer_data['metadata']:
                songs = composer_data['metadata']['songs']
                if isinstance(songs, list):
                    return len(songs)
            
            return 0
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting song count: {e}")
            return 0
    
    def _parse_song_count(self, count_str: str) -> int:
        """Parse song count string to integer."""
        try:
            # Remove any non-numeric characters except K, M
            count_str = str(count_str).strip()
            
            # Handle K suffix (thousands)
            if 'K' in count_str.upper():
                number = float(re.sub(r'[^\d.]', '', count_str))
                return int(number * 1000)
            
            # Handle M suffix (millions)
            elif 'M' in count_str.upper():
                number = float(re.sub(r'[^\d.]', '', count_str))
                return int(number * 1000000)
            
            # Regular number
            else:
                return int(re.sub(r'[^\d]', '', count_str))
                
        except Exception as e:
            logger.warning(f"⚠️ Error parsing song count '{count_str}': {e}")
            return 0
    
    def analyze_composer_issues(self) -> Dict:
        """Analyze composer-song relationship issues."""
        logger.info("🔍 Analyzing composer-song relationship issues...")
        
        analysis = {
            "composers_with_zero_songs": [],
            "composers_with_metadata_songs": [],
            "song_count_discrepancies": [],
            "statistics": {}
        }
        
        for composer_id, composer_data in self.composers_data.items():
            current_song_count = composer_data.get('song_count', 0)
            metadata_song_count = self.extract_song_count_from_metadata(composer_data)
            
            self.stats["total_composers"] += 1
            
            if current_song_count == 0:
                self.stats["composers_with_zero_songs"] += 1
                analysis["composers_with_zero_songs"].append({
                    "composer_id": composer_id,
                    "name": composer_data.get('name', 'Unknown'),
                    "current_song_count": current_song_count,
                    "metadata_song_count": metadata_song_count
                })
            
            if metadata_song_count > 0:
                self.stats["composers_with_metadata_songs"] += 1
                analysis["composers_with_metadata_songs"].append({
                    "composer_id": composer_id,
                    "name": composer_data.get('name', 'Unknown'),
                    "current_song_count": current_song_count,
                    "metadata_song_count": metadata_song_count
                })
            
            if current_song_count != metadata_song_count:
                analysis["song_count_discrepancies"].append({
                    "composer_id": composer_id,
                    "name": composer_data.get('name', 'Unknown'),
                    "current_song_count": current_song_count,
                    "metadata_song_count": metadata_song_count,
                    "discrepancy": metadata_song_count - current_song_count
                })
        
        analysis["statistics"] = {
            "total_composers": self.stats["total_composers"],
            "composers_with_zero_songs": self.stats["composers_with_zero_songs"],
            "composers_with_metadata_songs": self.stats["composers_with_metadata_songs"],
            "composers_with_discrepancies": len(analysis["song_count_discrepancies"])
        }
        
        return analysis
    
    def fix_composer_relationships(self) -> bool:
        """Fix composer-song relationships by updating song counts."""
        logger.info("🔧 Fixing composer-song relationships...")
        
        try:
            for composer_id, composer_data in self.composers_data.items():
                # Extract correct song count from metadata
                correct_song_count = self.extract_song_count_from_metadata(composer_data)
                
                # Create fixed composer record
                fixed_composer = composer_data.copy()
                fixed_composer['song_count'] = correct_song_count
                
                # Add fix metadata
                if 'fix_metadata' not in fixed_composer:
                    fixed_composer['fix_metadata'] = {}
                
                fixed_composer['fix_metadata'].update({
                    "fixed_timestamp": datetime.now().isoformat(),
                    "original_song_count": composer_data.get('song_count', 0),
                    "corrected_song_count": correct_song_count,
                    "fix_method": "metadata_extraction"
                })
                
                self.fixed_composers[composer_id] = fixed_composer
                
                if correct_song_count > 0:
                    self.stats["composers_fixed"] += 1
                    self.stats["total_songs_extracted"] += correct_song_count
                
                # Log progress
                if self.stats["composers_fixed"] % 100 == 0:
                    logger.info(f"📊 Fixed {self.stats['composers_fixed']} composers so far...")
            
            logger.info(f"✅ Fixed {self.stats['composers_fixed']} composers with {self.stats['total_songs_extracted']} total songs")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error fixing composer relationships: {e}")
            self.stats["processing_errors"] += 1
            return False
    
    def save_fixed_data(self) -> bool:
        """Save the fixed composer data."""
        logger.info("💾 Saving fixed composer data...")
        
        try:
            # Create output directory
            output_dir = self.data_path / "composer_relationship_fixed"
            output_dir.mkdir(exist_ok=True)
            
            # Save fixed composers
            fixed_file = output_dir / "unified_composers_database_fixed.json"
            with open(fixed_file, 'w', encoding='utf-8') as f:
                json.dump(self.fixed_composers, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved fixed composer data to {fixed_file}")
            
            # Save fix report
            report = {
                "fix_timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "summary": {
                    "total_composers_processed": len(self.composers_data),
                    "composers_fixed": self.stats["composers_fixed"],
                    "total_songs_extracted": self.stats["total_songs_extracted"],
                    "composers_with_zero_songs_before": self.stats["composers_with_zero_songs"],
                    "composers_with_zero_songs_after": len([c for c in self.fixed_composers.values() if c.get('song_count', 0) == 0])
                }
            }
            
            report_file = output_dir / "composer_relationship_fix_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved fix report to {report_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving fixed data: {e}")
            return False
    
    def run_fix(self) -> bool:
        """Run the complete composer relationship fix process."""
        logger.info("🚀 Starting Phase 3: Composer Relationship Fix")
        logger.info("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Analyze issues
        analysis = self.analyze_composer_issues()
        
        # Print analysis results
        logger.info("📊 ANALYSIS RESULTS:")
        logger.info(f"  • Total composers: {analysis['statistics']['total_composers']}")
        logger.info(f"  • Composers with 0 songs: {analysis['statistics']['composers_with_zero_songs']}")
        logger.info(f"  • Composers with metadata songs: {analysis['statistics']['composers_with_metadata_songs']}")
        logger.info(f"  • Composers with discrepancies: {analysis['statistics']['composers_with_discrepancies']}")
        
        # Fix relationships
        if not self.fix_composer_relationships():
            return False
        
        # Save fixed data
        if not self.save_fixed_data():
            return False
        
        # Final summary
        logger.info("🎉 COMPOSER RELATIONSHIP FIX COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"✅ Fixed {self.stats['composers_fixed']} composers")
        logger.info(f"✅ Extracted {self.stats['total_songs_extracted']} total songs")
        logger.info(f"✅ Reduced composers with 0 songs from {self.stats['composers_with_zero_songs']} to {len([c for c in self.fixed_composers.values() if c.get('song_count', 0) == 0])}")
        
        return True

def main():
    """Main function to run composer relationship fix."""
    print("🔧 Phase 3: Composer-Song Relationship Fix")
    print("=" * 50)
    
    # Initialize fixer
    fixer = ComposerRelationshipFixer()
    
    # Run fix
    success = fixer.run_fix()
    
    if success:
        print("\n✅ Composer relationship fix completed successfully!")
        print(f"📊 Fixed {fixer.stats['composers_fixed']} composers")
        print(f"📊 Extracted {fixer.stats['total_songs_extracted']} total songs")
    else:
        print("\n❌ Composer relationship fix failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
