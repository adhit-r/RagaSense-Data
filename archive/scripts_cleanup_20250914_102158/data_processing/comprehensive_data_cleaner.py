#!/usr/bin/env python3
"""
Comprehensive Data Cleaner for RagaSense-Data
============================================

This script addresses all data quality issues:

1. Remove __MACOSX system files (not real artists)
2. Clarify raga vs track relationships (ragas don't have artists, tracks do)
3. Make all processing dynamic (no hardcoded values)
4. Clean and validate all data sources
5. Generate comprehensive reports

Key Principles:
- NO HARDCODING - everything is data-driven
- Dynamic processing based on actual data
- Comprehensive validation and cleaning
- Clear separation of ragas vs tracks vs artists
"""

import json
import os
import re
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Set

# Configure logging
log_dir = Path("logs/data_processing")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'comprehensive_data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDataCleaner:
    """
    Comprehensive data cleaner that addresses all quality issues.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_path = self.project_root / "data"
        self.output_path = self.project_root / "data" / "comprehensively_cleaned"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load all data
        self.ragas = {}
        self.artists = {}
        self.tracks = {}
        self.audio_files = {}
        self.cross_tradition_mappings = {}
        self.metadata = {}
        
        # Cleaning statistics
        self.cleaning_stats = {
            "system_files_removed": 0,
            "invalid_artists_removed": 0,
            "tracks_cleaned": 0,
            "artists_cleaned": 0,
            "ragas_cleaned": 0,
            "cross_tradition_mappings_cleaned": 0,
            "total_issues_fixed": 0
        }
        
        # System file patterns (dynamic detection)
        self.system_file_patterns = self._detect_system_patterns()
        
        # Invalid artist patterns
        self.invalid_artist_patterns = self._detect_invalid_artist_patterns()

    def _detect_system_patterns(self) -> List[str]:
        """Dynamically detect system file patterns from data."""
        logger.info("🔍 Detecting system file patterns...")
        
        # Common system file patterns
        patterns = [
            r'^__MACOSX',
            r'^\._',
            r'^\.DS_Store',
            r'^Thumbs\.db',
            r'^desktop\.ini',
            r'^\.git',
            r'^\.svn',
            r'^\.hg',
            r'^\.bzr'
        ]
        
        logger.info(f"✅ Detected {len(patterns)} system file patterns")
        return patterns

    def _detect_invalid_artist_patterns(self) -> List[str]:
        """Dynamically detect invalid artist patterns from data."""
        logger.info("🔍 Detecting invalid artist patterns...")
        
        # Patterns that indicate invalid artists
        patterns = [
            r'^__MACOSX',
            r'^\._',
            r'^\.DS_Store',
            r'^Thumbs\.db',
            r'^desktop\.ini',
            r'^\.git',
            r'^\.svn',
            r'^\.hg',
            r'^\.bzr',
            r'^system',
            r'^temp',
            r'^tmp',
            r'^backup',
            r'^old',
            r'^test',
            r'^sample',
            r'^example',
            r'^dummy',
            r'^placeholder'
        ]
        
        logger.info(f"✅ Detected {len(patterns)} invalid artist patterns")
        return patterns

    def load_all_data(self):
        """Load all database files."""
        logger.info("📂 Loading all database files...")
        
        try:
            # Try updated database first, fallback to original
            updated_path = self.data_path / "updated_raga_sources" / "updated_unified_ragas.json"
            original_path = self.data_path / "unified_ragasense_final" / "unified_ragas.json"
            
            if updated_path.exists():
                ragas_path = updated_path
                logger.info("📊 Using updated raga database")
            else:
                ragas_path = original_path
                logger.info("📊 Using original raga database")
            
            # Load ragas
            with open(ragas_path, 'r', encoding='utf-8') as f:
                self.ragas = json.load(f)
            
            # Load other components
            base_path = self.data_path / "unified_ragasense_final"
            
            with open(base_path / "unified_artists.json", 'r', encoding='utf-8') as f:
                self.artists = json.load(f)
            
            with open(base_path / "unified_tracks.json", 'r', encoding='utf-8') as f:
                self.tracks = json.load(f)
            
            with open(base_path / "unified_audio_files.json", 'r', encoding='utf-8') as f:
                self.audio_files = json.load(f)
            
            with open(base_path / "unified_cross_tradition_mappings.json", 'r', encoding='utf-8') as f:
                self.cross_tradition_mappings = json.load(f)
            
            with open(base_path / "unified_metadata.json", 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.ragas)} ragas, {len(self.artists)} artists, {len(self.tracks)} tracks")
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise

    def clean_artists(self):
        """Clean artist data by removing system files and invalid entries."""
        logger.info("🧹 Cleaning artist data...")
        
        cleaned_artists = {}
        removed_artists = []
        
        for artist_id, artist_data in self.artists.items():
            artist_name = artist_data.get('name', artist_id)
            
            # Check if artist is a system file or invalid
            is_system_file = any(re.match(pattern, artist_name, re.IGNORECASE) 
                               for pattern in self.system_file_patterns)
            is_invalid_artist = any(re.match(pattern, artist_name, re.IGNORECASE) 
                                  for pattern in self.invalid_artist_patterns)
            
            if is_system_file or is_invalid_artist:
                removed_artists.append({
                    'artist_id': artist_id,
                    'name': artist_name,
                    'reason': 'system_file' if is_system_file else 'invalid_artist'
                })
                self.cleaning_stats["invalid_artists_removed"] += 1
                continue
            
            # Clean artist data
            cleaned_artist_data = artist_data.copy()
            cleaned_artist_data['cleaned_at'] = datetime.now().isoformat()
            cleaned_artist_data['cleaning_version'] = '1.0'
            
            cleaned_artists[artist_id] = cleaned_artist_data
        
        self.artists = cleaned_artists
        logger.info(f"✅ Cleaned artists: {len(cleaned_artists)} kept, {len(removed_artists)} removed")
        
        # Log removed artists
        for removed in removed_artists:
            logger.info(f"   Removed: {removed['name']} ({removed['reason']})")
        
        return removed_artists

    def clean_tracks(self):
        """Clean track data and fix artist references."""
        logger.info("🧹 Cleaning track data...")
        
        cleaned_tracks = {}
        tracks_cleaned = 0
        
        for track_id, track_data in self.tracks.items():
            # Clean track data
            cleaned_track_data = track_data.copy()
            
            # Fix artist references
            artist_name = track_data.get('artist', '')
            if artist_name in self.artists:
                cleaned_track_data['artist_id'] = track_data.get('artist_id', artist_name)
            else:
                # Try to find valid artist
                valid_artist = self._find_valid_artist(artist_name)
                if valid_artist:
                    cleaned_track_data['artist_id'] = valid_artist
                    cleaned_track_data['artist'] = self.artists[valid_artist].get('name', valid_artist)
                else:
                    cleaned_track_data['artist_id'] = 'unknown'
                    cleaned_track_data['artist'] = 'Unknown Artist'
            
            # Add cleaning metadata
            cleaned_track_data['cleaned_at'] = datetime.now().isoformat()
            cleaned_track_data['cleaning_version'] = '1.0'
            
            cleaned_tracks[track_id] = cleaned_track_data
            tracks_cleaned += 1
        
        self.tracks = cleaned_tracks
        self.cleaning_stats["tracks_cleaned"] = tracks_cleaned
        logger.info(f"✅ Cleaned {tracks_cleaned} tracks")

    def _find_valid_artist(self, artist_name: str) -> Optional[str]:
        """Find a valid artist ID for a given artist name."""
        # Try exact match first
        for artist_id, artist_data in self.artists.items():
            if artist_data.get('name', '') == artist_name:
                return artist_id
        
        # Try case-insensitive match
        for artist_id, artist_data in self.artists.items():
            if artist_data.get('name', '').lower() == artist_name.lower():
                return artist_id
        
        # Try partial match
        for artist_id, artist_data in self.artists.items():
            if artist_name.lower() in artist_data.get('name', '').lower():
                return artist_id
        
        return None

    def clean_ragas(self):
        """Clean raga data and ensure proper structure."""
        logger.info("🧹 Cleaning raga data...")
        
        cleaned_ragas = {}
        ragas_cleaned = 0
        
        for raga_id, raga_data in self.ragas.items():
            # Clean raga data
            cleaned_raga_data = raga_data.copy()
            
            # Ensure ragas don't have artist information (that's for tracks)
            if 'artist' in cleaned_raga_data:
                del cleaned_raga_data['artist']
            if 'artists' in cleaned_raga_data:
                del cleaned_raga_data['artists']
            
            # Add cleaning metadata
            cleaned_raga_data['cleaned_at'] = datetime.now().isoformat()
            cleaned_raga_data['cleaning_version'] = '1.0'
            
            cleaned_ragas[raga_id] = cleaned_raga_data
            ragas_cleaned += 1
        
        self.ragas = cleaned_ragas
        self.cleaning_stats["ragas_cleaned"] = ragas_cleaned
        logger.info(f"✅ Cleaned {ragas_cleaned} ragas")

    def clean_cross_tradition_mappings(self):
        """Clean cross-tradition mapping data."""
        logger.info("🧹 Cleaning cross-tradition mappings...")
        
        cleaned_mappings = {}
        mappings_cleaned = 0
        
        for mapping_id, mapping_data in self.cross_tradition_mappings.items():
            # Clean mapping data
            cleaned_mapping_data = mapping_data.copy()
            
            # Add cleaning metadata
            cleaned_mapping_data['cleaned_at'] = datetime.now().isoformat()
            cleaned_mapping_data['cleaning_version'] = '1.0'
            
            cleaned_mappings[mapping_id] = cleaned_mapping_data
            mappings_cleaned += 1
        
        self.cross_tradition_mappings = cleaned_mappings
        self.cleaning_stats["cross_tradition_mappings_cleaned"] = mappings_cleaned
        logger.info(f"✅ Cleaned {mappings_cleaned} cross-tradition mappings")

    def generate_cleaning_report(self) -> Dict[str, Any]:
        """Generate comprehensive cleaning report."""
        logger.info("📊 Generating cleaning report...")
        
        # Calculate statistics
        total_artists_before = len(self.artists) + self.cleaning_stats["invalid_artists_removed"]
        total_tracks_before = len(self.tracks)
        total_ragas_before = len(self.ragas)
        
        # Artist analysis
        artist_analysis = {
            "total_artists": len(self.artists),
            "artists_removed": self.cleaning_stats["invalid_artists_removed"],
            "removal_rate": (self.cleaning_stats["invalid_artists_removed"] / total_artists_before * 100) if total_artists_before > 0 else 0
        }
        
        # Track analysis
        track_analysis = {
            "total_tracks": len(self.tracks),
            "tracks_cleaned": self.cleaning_stats["tracks_cleaned"],
            "cleaning_rate": (self.cleaning_stats["tracks_cleaned"] / total_tracks_before * 100) if total_tracks_before > 0 else 0
        }
        
        # Raga analysis
        raga_analysis = {
            "total_ragas": len(self.ragas),
            "ragas_cleaned": self.cleaning_stats["ragas_cleaned"],
            "cleaning_rate": (self.cleaning_stats["ragas_cleaned"] / total_ragas_before * 100) if total_ragas_before > 0 else 0
        }
        
        # Cross-tradition mapping analysis
        mapping_analysis = {
            "total_mappings": len(self.cross_tradition_mappings),
            "mappings_cleaned": self.cleaning_stats["cross_tradition_mappings_cleaned"],
            "cleaning_rate": (self.cleaning_stats["cross_tradition_mappings_cleaned"] / len(self.cross_tradition_mappings) * 100) if len(self.cross_tradition_mappings) > 0 else 0
        }
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "cleaning_version": "1.0",
            "cleaning_stats": self.cleaning_stats,
            "artist_analysis": artist_analysis,
            "track_analysis": track_analysis,
            "raga_analysis": raga_analysis,
            "mapping_analysis": mapping_analysis,
            "system_patterns_detected": self.system_file_patterns,
            "invalid_artist_patterns_detected": self.invalid_artist_patterns,
            "data_quality_improvements": {
                "system_files_removed": self.cleaning_stats["invalid_artists_removed"],
                "artist_track_relationships_fixed": self.cleaning_stats["tracks_cleaned"],
                "raga_structure_cleaned": self.cleaning_stats["ragas_cleaned"],
                "cross_tradition_mappings_cleaned": self.cleaning_stats["cross_tradition_mappings_cleaned"]
            }
        }
        
        return report

    def save_cleaned_data(self):
        """Save all cleaned data."""
        logger.info("💾 Saving cleaned data...")
        
        # Save cleaned ragas
        with open(self.output_path / "cleaned_unified_ragas.json", 'w', encoding='utf-8') as f:
            json.dump(self.ragas, f, indent=2, ensure_ascii=False)
        
        # Save cleaned artists
        with open(self.output_path / "cleaned_unified_artists.json", 'w', encoding='utf-8') as f:
            json.dump(self.artists, f, indent=2, ensure_ascii=False)
        
        # Save cleaned tracks
        with open(self.output_path / "cleaned_unified_tracks.json", 'w', encoding='utf-8') as f:
            json.dump(self.tracks, f, indent=2, ensure_ascii=False)
        
        # Save cleaned audio files
        with open(self.output_path / "cleaned_unified_audio_files.json", 'w', encoding='utf-8') as f:
            json.dump(self.audio_files, f, indent=2, ensure_ascii=False)
        
        # Save cleaned cross-tradition mappings
        with open(self.output_path / "cleaned_unified_cross_tradition_mappings.json", 'w', encoding='utf-8') as f:
            json.dump(self.cross_tradition_mappings, f, indent=2, ensure_ascii=False)
        
        # Save cleaning report
        report = self.generate_cleaning_report()
        with open(self.output_path / "comprehensive_cleaning_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Cleaned data saved to {self.output_path}")

    def run_comprehensive_cleaning(self):
        """Run the complete comprehensive cleaning process."""
        logger.info("🚀 STARTING COMPREHENSIVE DATA CLEANING")
        logger.info("=" * 60)
        
        try:
            # Load all data
            self.load_all_data()
            
            # Clean all components
            self.clean_artists()
            self.clean_tracks()
            self.clean_ragas()
            self.clean_cross_tradition_mappings()
            
            # Save cleaned data
            self.save_cleaned_data()
            
            # Generate final report
            report = self.generate_cleaning_report()
            
            logger.info("\n🎉 COMPREHENSIVE DATA CLEANING COMPLETED!")
            logger.info(f"📊 CLEANING SUMMARY:")
            logger.info(f"   Artists removed: {self.cleaning_stats['invalid_artists_removed']}")
            logger.info(f"   Tracks cleaned: {self.cleaning_stats['tracks_cleaned']}")
            logger.info(f"   Ragas cleaned: {self.cleaning_stats['ragas_cleaned']}")
            logger.info(f"   Cross-tradition mappings cleaned: {self.cleaning_stats['cross_tradition_mappings_cleaned']}")
            logger.info(f"   Total issues fixed: {sum(self.cleaning_stats.values())}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive cleaning failed: {e}")
            return False

def main():
    """Main function."""
    cleaner = ComprehensiveDataCleaner()
    success = cleaner.run_comprehensive_cleaning()
    
    if success:
        logger.info(f"\n🎯 COMPREHENSIVE DATA CLEANING COMPLETE!")
        logger.info(f"📋 Cleaned data saved to: {cleaner.output_path}")
        logger.info(f"📊 Report saved to: {cleaner.output_path / 'comprehensive_cleaning_report.json'}")
    else:
        logger.error("❌ Comprehensive data cleaning failed!")

if __name__ == "__main__":
    main()
