#!/usr/bin/env python3
"""
Integrate All Datasets into Unified RagaSense-Data
================================================

This script integrates all our data sources into a unified RagaSense-Data dataset:

1. Ramanarunachalam (1,340 individual ragas) - already processed
2. Saraga 1.5 Carnatic (1,982 tracks, 2 artists) - metadata extracted
3. Saraga 1.5 Hindustani (216 tracks, 2 artists) - metadata extracted
4. Saraga Carnatic Melody Synth (2,460 tracks, 16 artists, 339 audio files) - processed

Features:
- Unified raga database with all sources
- Comprehensive artist database
- Complete track database with audio file references
- Cross-tradition mappings
- Data quality validation
- Comprehensive reporting
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetIntegrator:
    """
    Integrates all datasets into a unified RagaSense-Data database.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_path = self.project_root / "data"
        self.output_path = self.project_root / "data" / "unified_ragasense_final"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Data sources
        self.data_sources = {
            "ramanarunachalam": {
                "path": self.data_path / "unknownraga_fixed" / "unified_ragas_database_fixed.json",
                "type": "ragas",
                "description": "Individual ragas after Unknownraga removal"
            },
            "saraga_processed": {
                "path": self.data_path / "saraga_processed" / "saraga_all_processed_data.json",
                "type": "artists_tracks",
                "description": "All Saraga datasets (1.5 Carnatic, 1.5 Hindustani, Melody Synth)"
            },
            "cross_tradition": {
                "path": self.data_path / "cross_tradition_corrected" / "unified_ragas_database_cross_tradition_corrected.json",
                "type": "ragas_with_mappings",
                "description": "Ragas with corrected cross-tradition mappings"
            }
        }
        
        # Unified database structure
        self.unified_database = {
            "ragas": {},
            "artists": {},
            "tracks": {},
            "audio_files": {},
            "cross_tradition_mappings": {},
            "metadata": {
                "integration_timestamp": datetime.now().isoformat(),
                "data_sources": list(self.data_sources.keys()),
                "total_ragas": 0,
                "total_artists": 0,
                "total_tracks": 0,
                "total_audio_files": 0,
                "tradition_distribution": {},
                "source_distribution": {}
            }
        }
        
        self.integration_stats = {
            "ragas_loaded": 0,
            "artists_loaded": 0,
            "tracks_loaded": 0,
            "audio_files_loaded": 0,
            "cross_tradition_mappings_loaded": 0,
            "duplicates_removed": 0,
            "integration_time": 0
        }

    def load_ramanarunachalam_data(self):
        """Load the processed Ramanarunachalam raga data."""
        logger.info("📂 Loading Ramanarunachalam raga data...")
        
        raga_file = self.data_sources["ramanarunachalam"]["path"]
        if not raga_file.exists():
            logger.error(f"❌ Ramanarunachalam data not found: {raga_file}")
            return False
        
        with open(raga_file, 'r', encoding='utf-8') as f:
            raga_data = json.load(f)
        
        # Integrate ragas
        for raga_id, raga_info in raga_data.items():
            # Add source information
            raga_info["sources"] = raga_info.get("sources", [])
            if "ramanarunachalam" not in raga_info["sources"]:
                raga_info["sources"].append("ramanarunachalam")
            
            raga_info["source_priority"] = "primary"  # Ramanarunachalam is our primary source
            raga_info["last_updated"] = datetime.now().isoformat()
            
            self.unified_database["ragas"][raga_id] = raga_info
        
        self.integration_stats["ragas_loaded"] = len(raga_data)
        logger.info(f"✅ Loaded {len(raga_data)} ragas from Ramanarunachalam")
        return True

    def load_saraga_data(self):
        """Load the processed Saraga data (artists, tracks, audio files)."""
        logger.info("📂 Loading Saraga processed data...")
        
        saraga_file = self.data_sources["saraga_processed"]["path"]
        if not saraga_file.exists():
            logger.error(f"❌ Saraga processed data not found: {saraga_file}")
            return False
        
        with open(saraga_file, 'r', encoding='utf-8') as f:
            saraga_data = json.load(f)
        
        # Integrate artists
        for artist_id, artist_info in saraga_data.get("artists", {}).items():
            artist_info["sources"] = artist_info.get("sources", [])
            if "saraga" not in artist_info["sources"]:
                artist_info["sources"].append("saraga")
            
            artist_info["source_priority"] = "secondary"
            artist_info["last_updated"] = datetime.now().isoformat()
            
            self.unified_database["artists"][artist_id] = artist_info
        
        # Integrate tracks
        for track_id, track_info in saraga_data.get("tracks", {}).items():
            track_info["sources"] = track_info.get("sources", [])
            if "saraga" not in track_info["sources"]:
                track_info["sources"].append("saraga")
            
            track_info["source_priority"] = "secondary"
            track_info["last_updated"] = datetime.now().isoformat()
            
            # Add audio file reference if available
            if track_info.get("audio_file"):
                audio_file_id = f"audio_{track_id}"
                self.unified_database["audio_files"][audio_file_id] = {
                    "audio_file_id": audio_file_id,
                    "track_id": track_id,
                    "filename": track_info["audio_file"],
                    "file_type": track_info.get("file_type", "wav"),
                    "source": "saraga",
                    "dataset": track_info.get("dataset", "unknown"),
                    "last_updated": datetime.now().isoformat()
                }
            
            self.unified_database["tracks"][track_id] = track_info
        
        self.integration_stats["artists_loaded"] = len(saraga_data.get("artists", {}))
        self.integration_stats["tracks_loaded"] = len(saraga_data.get("tracks", {}))
        self.integration_stats["audio_files_loaded"] = len(self.unified_database["audio_files"])
        
        logger.info(f"✅ Loaded {len(saraga_data.get('artists', {}))} artists, {len(saraga_data.get('tracks', {}))} tracks from Saraga")
        return True

    def load_cross_tradition_mappings(self):
        """Load the corrected cross-tradition mappings."""
        logger.info("📂 Loading cross-tradition mappings...")
        
        cross_tradition_file = self.data_sources["cross_tradition"]["path"]
        if not cross_tradition_file.exists():
            logger.warning(f"⚠️ Cross-tradition mappings not found: {cross_tradition_file}")
            return True  # Not critical, continue without mappings
        
        with open(cross_tradition_file, 'r', encoding='utf-8') as f:
            cross_tradition_data = json.load(f)
        
        # Extract cross-tradition mappings from raga data
        mappings_count = 0
        for raga_id, raga_info in cross_tradition_data.items():
            cross_mapping = raga_info.get("cross_tradition_mapping", {})
            if cross_mapping and cross_mapping.get("mapping"):
                mapping_id = f"mapping_{raga_id}"
                self.unified_database["cross_tradition_mappings"][mapping_id] = {
                    "mapping_id": mapping_id,
                    "raga_id": raga_id,
                    "raga_name": raga_info.get("name", raga_id),
                    "tradition": raga_info.get("tradition", "Unknown"),
                    "mapped_to": cross_mapping["mapping"],
                    "equivalence_type": cross_mapping.get("type", "unknown"),
                    "confidence": cross_mapping.get("confidence", "unknown"),
                    "score": cross_mapping.get("score", 0),
                    "evidence": cross_mapping.get("evidence", ""),
                    "sources": cross_mapping.get("sources", []),
                    "last_updated": datetime.now().isoformat()
                }
                mappings_count += 1
        
        self.integration_stats["cross_tradition_mappings_loaded"] = mappings_count
        logger.info(f"✅ Loaded {mappings_count} cross-tradition mappings")
        return True

    def deduplicate_and_merge(self):
        """Deduplicate entries and merge data from different sources."""
        logger.info("🔄 Deduplicating and merging data...")
        
        # Deduplicate artists by name
        artist_name_map = {}
        duplicates_removed = 0
        
        for artist_id, artist_info in list(self.unified_database["artists"].items()):
            artist_name = artist_info.get("name", artist_id)
            
            if artist_name in artist_name_map:
                # Merge artist data
                existing_artist = self.unified_database["artists"][artist_name_map[artist_name]]
                
                # Combine sources
                existing_sources = set(existing_artist.get("sources", []))
                new_sources = set(artist_info.get("sources", []))
                existing_artist["sources"] = list(existing_sources.union(new_sources))
                
                # Update track count
                existing_artist["total_tracks"] = existing_artist.get("total_tracks", 0) + artist_info.get("total_tracks", 0)
                
                # Remove duplicate
                del self.unified_database["artists"][artist_id]
                duplicates_removed += 1
            else:
                artist_name_map[artist_name] = artist_id
        
        # Update track references to use merged artist IDs
        for track_id, track_info in self.unified_database["tracks"].items():
            artist_name = track_info.get("artist", "Unknown")
            if artist_name in artist_name_map:
                track_info["artist_id"] = artist_name_map[artist_name]
        
        self.integration_stats["duplicates_removed"] = duplicates_removed
        logger.info(f"✅ Removed {duplicates_removed} duplicate artists")

    def calculate_statistics(self):
        """Calculate comprehensive statistics for the unified database."""
        logger.info("📊 Calculating unified database statistics...")
        
        # Basic counts
        self.unified_database["metadata"]["total_ragas"] = len(self.unified_database["ragas"])
        self.unified_database["metadata"]["total_artists"] = len(self.unified_database["artists"])
        self.unified_database["metadata"]["total_tracks"] = len(self.unified_database["tracks"])
        self.unified_database["metadata"]["total_audio_files"] = len(self.unified_database["audio_files"])
        
        # Tradition distribution
        raga_traditions = Counter(raga.get("tradition", "Unknown") for raga in self.unified_database["ragas"].values())
        artist_traditions = Counter(artist.get("tradition", "Unknown") for artist in self.unified_database["artists"].values())
        track_traditions = Counter(track.get("tradition", "Unknown") for track in self.unified_database["tracks"].values())
        
        self.unified_database["metadata"]["tradition_distribution"] = {
            "ragas": dict(raga_traditions),
            "artists": dict(artist_traditions),
            "tracks": dict(track_traditions)
        }
        
        # Source distribution
        raga_sources = Counter()
        for raga in self.unified_database["ragas"].values():
            for source in raga.get("sources", []):
                raga_sources[source] += 1
        
        artist_sources = Counter()
        for artist in self.unified_database["artists"].values():
            for source in artist.get("sources", []):
                artist_sources[source] += 1
        
        track_sources = Counter()
        for track in self.unified_database["tracks"].values():
            for source in track.get("sources", []):
                track_sources[source] += 1
        
        self.unified_database["metadata"]["source_distribution"] = {
            "ragas": dict(raga_sources),
            "artists": dict(artist_sources),
            "tracks": dict(track_sources)
        }
        
        logger.info("✅ Statistics calculated")

    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate a comprehensive integration report."""
        logger.info("📊 Generating integration report...")
        
        # Top ragas by song count
        top_ragas = sorted(
            self.unified_database["ragas"].items(),
            key=lambda x: x[1].get("song_count", 0),
            reverse=True
        )[:20]
        
        # Top artists by track count
        top_artists = sorted(
            self.unified_database["artists"].items(),
            key=lambda x: x[1].get("total_tracks", 0),
            reverse=True
        )[:20]
        
        # Cross-tradition mapping summary
        mapping_summary = Counter(
            mapping.get("equivalence_type", "unknown") 
            for mapping in self.unified_database["cross_tradition_mappings"].values()
        )
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "integration_statistics": self.integration_stats,
            "unified_database_summary": self.unified_database["metadata"],
            "top_ragas": [
                {
                    "raga_id": raga_id,
                    "name": raga_data.get("name", raga_id),
                    "tradition": raga_data.get("tradition", "Unknown"),
                    "song_count": raga_data.get("song_count", 0),
                    "sources": raga_data.get("sources", [])
                }
                for raga_id, raga_data in top_ragas
            ],
            "top_artists": [
                {
                    "artist_id": artist_id,
                    "name": artist_data.get("name", artist_id),
                    "tradition": artist_data.get("tradition", "Unknown"),
                    "total_tracks": artist_data.get("total_tracks", 0),
                    "sources": artist_data.get("sources", [])
                }
                for artist_id, artist_data in top_artists
            ],
            "cross_tradition_mappings": {
                "total_mappings": len(self.unified_database["cross_tradition_mappings"]),
                "equivalence_types": dict(mapping_summary)
            },
            "data_quality": {
                "ragas_with_sources": len([r for r in self.unified_database["ragas"].values() if r.get("sources")]),
                "artists_with_tracks": len([a for a in self.unified_database["artists"].values() if a.get("total_tracks", 0) > 0]),
                "tracks_with_audio": len([t for t in self.unified_database["tracks"].values() if t.get("audio_file")]),
                "cross_tradition_coverage": len(self.unified_database["cross_tradition_mappings"])
            }
        }
        
        return report

    def save_unified_database(self):
        """Save the unified database and reports."""
        logger.info("💾 Saving unified database...")
        
        # Save main unified database
        with open(self.output_path / "unified_ragasense_database.json", 'w', encoding='utf-8') as f:
            json.dump(self.unified_database, f, indent=2, ensure_ascii=False)
        
        # Save individual components
        for component in ["ragas", "artists", "tracks", "audio_files", "cross_tradition_mappings"]:
            with open(self.output_path / f"unified_{component}.json", 'w', encoding='utf-8') as f:
                json.dump(self.unified_database[component], f, indent=2, ensure_ascii=False)
        
        # Save integration report
        report = self.generate_integration_report()
        with open(self.output_path / "integration_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save metadata summary
        with open(self.output_path / "unified_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.unified_database["metadata"], f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Unified database saved to {self.output_path}")

    def run_integration(self):
        """Run the complete dataset integration process."""
        logger.info("🚀 STARTING UNIFIED DATASET INTEGRATION")
        logger.info("=" * 60)
        start_time = time.time()
        
        # Load all data sources
        if not self.load_ramanarunachalam_data():
            return False
        
        if not self.load_saraga_data():
            return False
        
        if not self.load_cross_tradition_mappings():
            return False
        
        # Deduplicate and merge
        self.deduplicate_and_merge()
        
        # Calculate statistics
        self.calculate_statistics()
        
        # Save unified database
        self.save_unified_database()
        
        end_time = time.time()
        self.integration_stats["integration_time"] = end_time - start_time
        
        logger.info("\n🎉 UNIFIED DATASET INTEGRATION COMPLETED!")
        logger.info(f"⏱️ Integration time: {self.integration_stats['integration_time']:.1f} seconds")
        
        # Print summary
        logger.info("\n📊 INTEGRATION SUMMARY:")
        logger.info(f"   Ragas loaded: {self.integration_stats['ragas_loaded']}")
        logger.info(f"   Artists loaded: {self.integration_stats['artists_loaded']}")
        logger.info(f"   Tracks loaded: {self.integration_stats['tracks_loaded']}")
        logger.info(f"   Audio files loaded: {self.integration_stats['audio_files_loaded']}")
        logger.info(f"   Cross-tradition mappings: {self.integration_stats['cross_tradition_mappings_loaded']}")
        logger.info(f"   Duplicates removed: {self.integration_stats['duplicates_removed']}")
        
        logger.info("\n📊 UNIFIED DATABASE SUMMARY:")
        metadata = self.unified_database["metadata"]
        logger.info(f"   Total ragas: {metadata['total_ragas']}")
        logger.info(f"   Total artists: {metadata['total_artists']}")
        logger.info(f"   Total tracks: {metadata['total_tracks']}")
        logger.info(f"   Total audio files: {metadata['total_audio_files']}")
        
        return True

def main():
    """Main function to run the dataset integration."""
    integrator = DatasetIntegrator()
    success = integrator.run_integration()
    
    if success:
        logger.info(f"\n🎯 UNIFIED DATASET INTEGRATION COMPLETE!")
        logger.info(f"📋 Unified database saved to: {integrator.output_path}")
        logger.info(f"📊 Report saved to: {integrator.output_path / 'integration_report.json'}")
    else:
        logger.error("❌ Dataset integration failed!")

if __name__ == "__main__":
    main()
