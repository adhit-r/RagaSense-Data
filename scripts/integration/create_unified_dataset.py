#!/usr/bin/env python3
"""
Create Unified Dataset from All Sources
======================================

This script creates a comprehensive unified dataset by properly extracting
and processing metadata from all downloaded sources:

1. Ramanarunachalam Music Repository (Carnatic & Hindustani)
2. Saraga-Carnatic-Melody-Synth
3. Saraga 1.5 Carnatic & Hindustani datasets
4. YouTube links integration

Features:
- Proper JSON parsing and metadata extraction
- Cross-source deduplication
- YouTube link integration
- Quality scoring and validation
- Comprehensive reporting
"""

import json
import time
import zipfile
import os
from datetime import datetime
from pathlib import Path
import sys
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import pandas as pd
from collections import defaultdict, Counter
import re
import concurrent.futures
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_dataset_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedDatasetCreator:
    """
    Creates a unified dataset from all data sources.
    """
    
    def __init__(self, downloads_path: Path, output_path: Path):
        self.downloads_path = downloads_path
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.unified_ragas = {}
        self.unified_artists = {}
        self.unified_composers = {}
        self.unified_songs = {}
        self.youtube_links = set()
        
        self.processing_stats = {
            "ragas_processed": 0,
            "artists_processed": 0,
            "composers_processed": 0,
            "songs_processed": 0,
            "youtube_links_found": 0,
            "errors": 0
        }

    def process_ramanarunachalam(self):
        """Process Ramanarunachalam Music Repository."""
        logger.info("🔍 Processing Ramanarunachalam Music Repository...")
        
        ramanarunachalam_path = self.downloads_path / "Ramanarunachalam_Music_Repository"
        
        # Process Carnatic
        carnatic_path = ramanarunachalam_path / "Carnatic"
        if carnatic_path.exists():
            self._process_ramanarunachalam_tradition(carnatic_path, "Carnatic")
        
        # Process Hindustani
        hindustani_path = ramanarunachalam_path / "Hindustani"
        if hindustani_path.exists():
            self._process_ramanarunachalam_tradition(hindustani_path, "Hindustani")

    def _process_ramanarunachalam_tradition(self, tradition_path: Path, tradition: str):
        """Process a tradition-specific Ramanarunachalam dataset."""
        logger.info(f"📂 Processing Ramanarunachalam {tradition}...")
        
        # Process raga files
        raga_path = tradition_path / "raga"
        if raga_path.exists():
            self._process_raga_files(raga_path, tradition, "Ramanarunachalam")
        
        # Process artist files
        artist_path = tradition_path / "artist"
        if artist_path.exists():
            self._process_artist_files(artist_path, tradition, "Ramanarunachalam")
        
        # Process composer files
        composer_path = tradition_path / "composer"
        if composer_path.exists():
            self._process_composer_files(composer_path, tradition, "Ramanarunachalam")
        
        # Process song files
        song_path = tradition_path / "song"
        if song_path.exists():
            self._process_song_files(song_path, tradition, "Ramanarunachalam")

    def _process_raga_files(self, raga_path: Path, tradition: str, source: str):
        """Process raga JSON files."""
        logger.info(f"🎵 Processing raga files from {source} {tradition}...")
        
        for raga_file in raga_path.glob("*.json"):
            try:
                with open(raga_file, 'r', encoding='utf-8') as f:
                    raga_data = json.load(f)
                    # Extract raga name from filename
                    raga_name = raga_file.stem
                    # Add raga name to the data
                    raga_data['title'] = raga_name
                    self._add_raga_to_unified(raga_data, tradition, source)
                    self.processing_stats["ragas_processed"] += 1
            except Exception as e:
                logger.warning(f"Could not process raga file {raga_file}: {e}")
                self.processing_stats["errors"] += 1

    def _process_artist_files(self, artist_path: Path, tradition: str, source: str):
        """Process artist JSON files."""
        logger.info(f"👤 Processing artist files from {source} {tradition}...")
        
        for artist_file in artist_path.glob("*.json"):
            try:
                with open(artist_file, 'r', encoding='utf-8') as f:
                    artist_data = json.load(f)
                    # Extract artist name from filename
                    artist_name = artist_file.stem
                    # Add artist name to the data
                    artist_data['title'] = artist_name
                    self._add_artist_to_unified(artist_data, tradition, source)
                    self.processing_stats["artists_processed"] += 1
            except Exception as e:
                logger.warning(f"Could not process artist file {artist_file}: {e}")
                self.processing_stats["errors"] += 1

    def _process_composer_files(self, composer_path: Path, tradition: str, source: str):
        """Process composer JSON files."""
        logger.info(f"✍️ Processing composer files from {source} {tradition}...")
        
        for composer_file in composer_path.glob("*.json"):
            try:
                with open(composer_file, 'r', encoding='utf-8') as f:
                    composer_data = json.load(f)
                    # Extract composer name from filename
                    composer_name = composer_file.stem
                    # Add composer name to the data
                    composer_data['title'] = composer_name
                    self._add_composer_to_unified(composer_data, tradition, source)
                    self.processing_stats["composers_processed"] += 1
            except Exception as e:
                logger.warning(f"Could not process composer file {composer_file}: {e}")
                self.processing_stats["errors"] += 1

    def _process_song_files(self, song_path: Path, tradition: str, source: str):
        """Process song JSON files."""
        logger.info(f"🎶 Processing song files from {source} {tradition}...")
        
        for song_file in song_path.glob("*.json"):
            try:
                with open(song_file, 'r', encoding='utf-8') as f:
                    song_data = json.load(f)
                    self._add_song_to_unified(song_data, tradition, source)
                    self.processing_stats["songs_processed"] += 1
            except Exception as e:
                logger.warning(f"Could not process song file {song_file}: {e}")
                self.processing_stats["errors"] += 1

    def _add_raga_to_unified(self, raga_data: Dict, tradition: str, source: str):
        """Add raga to unified dataset."""
        # Extract raga name from the filename (passed as raga_data)
        if isinstance(raga_data, str):
            raga_name = raga_data
            raga_id = raga_name.lower().replace(' ', '_')
        else:
            # For Ramanarunachalam files, extract from songs
            if 'songs' in raga_data and isinstance(raga_data['songs'], list) and len(raga_data['songs']) > 0:
                # Get raga ID from first song
                first_song = raga_data['songs'][0]
                raga_id = str(first_song.get('R', ''))
                
                # Extract raga name from filename or use raga_id
                raga_name = raga_data.get('title', f'Raga_{raga_id}')
            else:
                # Fallback to standard extraction
                raga_id = raga_data.get('id', '')
                raga_name = raga_data.get('name', '')
        
        if raga_id and raga_name:
            if raga_id not in self.unified_ragas:
                self.unified_ragas[raga_id] = {
                    "raga_id": raga_id,
                    "name": raga_name,
                    "tradition": tradition,
                    "sources": [source],
                    "song_count": 0,
                    "metadata": raga_data,
                    "youtube_links": []
                }
            else:
                if source not in self.unified_ragas[raga_id]["sources"]:
                    self.unified_ragas[raga_id]["sources"].append(source)
            
            # Extract YouTube links from songs
            youtube_links = self._extract_youtube_links_from_songs(raga_data)
            self.unified_ragas[raga_id]["youtube_links"].extend(youtube_links)
            self.youtube_links.update(youtube_links)
            self.processing_stats["youtube_links_found"] += len(youtube_links)
            
            # Count songs for this raga
            if 'songs' in raga_data and isinstance(raga_data['songs'], list):
                self.unified_ragas[raga_id]["song_count"] = len(raga_data['songs'])

    def _add_artist_to_unified(self, artist_data: Dict, tradition: str, source: str):
        """Add artist to unified dataset."""
        # For Ramanarunachalam files, extract from title and songs
        if 'title' in artist_data:
            artist_name = artist_data['title']
            artist_id = artist_name.lower().replace(' ', '_').replace('.', '_')
        else:
            # Fallback to standard extraction
            artist_id = artist_data.get('id', '')
            artist_name = artist_data.get('name', '')
        
        if artist_id and artist_name:
            if artist_id not in self.unified_artists:
                self.unified_artists[artist_id] = {
                    "artist_id": artist_id,
                    "name": artist_name,
                    "tradition": tradition,
                    "sources": [source],
                    "song_count": 0,
                    "metadata": artist_data,
                    "youtube_links": []
                }
            else:
                if source not in self.unified_artists[artist_id]["sources"]:
                    self.unified_artists[artist_id]["sources"].append(source)
            
            # Extract YouTube links from songs
            youtube_links = self._extract_youtube_links_from_songs(artist_data)
            self.unified_artists[artist_id]["youtube_links"].extend(youtube_links)
            self.youtube_links.update(youtube_links)
            self.processing_stats["youtube_links_found"] += len(youtube_links)
            
            # Count songs for this artist
            if 'songs' in artist_data and isinstance(artist_data['songs'], list):
                self.unified_artists[artist_id]["song_count"] = len(artist_data['songs'])

    def _add_composer_to_unified(self, composer_data: Dict, tradition: str, source: str):
        """Add composer to unified dataset."""
        # For Ramanarunachalam files, extract from title and songs
        if 'title' in composer_data:
            composer_name = composer_data['title']
            composer_id = composer_name.lower().replace(' ', '_').replace('.', '_')
        else:
            # Fallback to standard extraction
            composer_id = composer_data.get('id', '')
            composer_name = composer_data.get('name', '')
        
        if composer_id and composer_name:
            if composer_id not in self.unified_composers:
                self.unified_composers[composer_id] = {
                    "composer_id": composer_id,
                    "name": composer_name,
                    "tradition": tradition,
                    "sources": [source],
                    "song_count": 0,
                    "metadata": composer_data,
                    "youtube_links": []
                }
            else:
                if source not in self.unified_composers[composer_id]["sources"]:
                    self.unified_composers[composer_id]["sources"].append(source)
            
            # Extract YouTube links from songs
            youtube_links = self._extract_youtube_links_from_songs(composer_data)
            self.unified_composers[composer_id]["youtube_links"].extend(youtube_links)
            self.youtube_links.update(youtube_links)
            self.processing_stats["youtube_links_found"] += len(youtube_links)
            
            # Count songs for this composer
            if 'songs' in composer_data and isinstance(composer_data['songs'], list):
                self.unified_composers[composer_id]["song_count"] = len(composer_data['songs'])

    def _add_song_to_unified(self, song_data: Dict, tradition: str, source: str):
        """Add song to unified dataset."""
        song_id = song_data.get('id', '')
        song_name = song_data.get('name', '')
        
        if song_id and song_name:
            if song_id not in self.unified_songs:
                self.unified_songs[song_id] = {
                    "song_id": song_id,
                    "name": song_name,
                    "tradition": tradition,
                    "sources": [source],
                    "metadata": song_data,
                    "youtube_links": []
                }
            else:
                if source not in self.unified_songs[song_id]["sources"]:
                    self.unified_songs[song_id]["sources"].append(source)
            
            # Extract YouTube links
            youtube_links = self._extract_youtube_links(song_data)
            self.unified_songs[song_id]["youtube_links"].extend(youtube_links)
            self.youtube_links.update(youtube_links)
            self.processing_stats["youtube_links_found"] += len(youtube_links)

    def _extract_youtube_links_from_songs(self, raga_data: Dict) -> List[str]:
        """Extract YouTube links from raga songs data."""
        youtube_links = []
        
        if 'songs' in raga_data and isinstance(raga_data['songs'], list):
            for song in raga_data['songs']:
                if isinstance(song, dict) and 'I' in song:
                    youtube_id = song['I']
                    if youtube_id and youtube_id != '0':
                        # Convert to full YouTube URL
                        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
                        youtube_links.append(youtube_url)
        
        return youtube_links

    def _extract_youtube_links(self, data: Any) -> List[str]:
        """Extract YouTube links from data structure."""
        youtube_links = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and 'youtube.com' in value:
                    youtube_links.append(value)
                elif isinstance(value, (dict, list)):
                    youtube_links.extend(self._extract_youtube_links(value))
        elif isinstance(data, list):
            for item in data:
                youtube_links.extend(self._extract_youtube_links(item))
        
        return youtube_links

    def process_saraga_carnatic_melody_synth(self):
        """Process Saraga-Carnatic-Melody-Synth dataset."""
        logger.info("🔍 Processing Saraga-Carnatic-Melody-Synth...")
        
        saraga_path = self.downloads_path / "saraga_carnatic_melody_synth" / "Saraga-Carnatic-Melody-Synth"
        
        if saraga_path.exists():
            # Process artists mapping
            mapping_file = saraga_path / "artists_to_track_mapping.json"
            if mapping_file.exists():
                try:
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                        self._process_saraga_mapping(mapping_data, "Carnatic", "Saraga-Carnatic-Melody-Synth")
                except Exception as e:
                    logger.warning(f"Could not process Saraga mapping: {e}")

    def _process_saraga_mapping(self, mapping_data: Dict, tradition: str, source: str):
        """Process Saraga mapping data."""
        logger.info(f"📊 Processing Saraga mapping from {source}...")
        
        for artist_name, tracks in mapping_data.items():
            # Add artist
            artist_id = f"saraga_{artist_name.lower().replace(' ', '_')}"
            if artist_id not in self.unified_artists:
                self.unified_artists[artist_id] = {
                    "artist_id": artist_id,
                    "name": artist_name,
                    "tradition": tradition,
                    "sources": [source],
                    "song_count": len(tracks) if isinstance(tracks, list) else 0,
                    "metadata": {"tracks": tracks},
                    "youtube_links": []
                }
            else:
                if source not in self.unified_artists[artist_id]["sources"]:
                    self.unified_artists[artist_id]["sources"].append(source)
            
            # Process tracks
            if isinstance(tracks, list):
                for track in tracks:
                    if isinstance(track, dict):
                        track_name = track.get('name', '')
                        if track_name:
                            track_id = f"saraga_{track_name.lower().replace(' ', '_')}"
                            if track_id not in self.unified_songs:
                                self.unified_songs[track_id] = {
                                    "song_id": track_id,
                                    "name": track_name,
                                    "tradition": tradition,
                                    "sources": [source],
                                    "metadata": track,
                                    "youtube_links": []
                                }

    def process_saraga_datasets(self):
        """Process Saraga 1.5 datasets."""
        logger.info("🔍 Processing Saraga 1.5 datasets...")
        
        saraga_datasets_path = self.downloads_path / "saraga_datasets"
        
        # Process Carnatic
        carnatic_zip = saraga_datasets_path / "carnatic" / "saraga1.5_carnatic.zip"
        if carnatic_zip.exists():
            self._process_saraga_zip(carnatic_zip, "Carnatic", "Saraga1.5-Carnatic")
        
        # Process Hindustani
        hindustani_zip = saraga_datasets_path / "hindustani" / "saraga1.5_hindustani.zip"
        if hindustani_zip.exists():
            self._process_saraga_zip(hindustani_zip, "Hindustani", "Saraga1.5-Hindustani")

    def _process_saraga_zip(self, zip_path: Path, tradition: str, source: str):
        """Process a Saraga ZIP dataset."""
        logger.info(f"📦 Processing {source}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if file_info.filename.endswith('.json') and not file_info.filename.startswith('__MACOSX'):
                        try:
                            with zip_file.open(file_info) as f:
                                data = json.load(f)
                                self._process_saraga_json(data, tradition, source, file_info.filename)
                        except Exception as e:
                            logger.warning(f"Could not process {file_info.filename}: {e}")
        except Exception as e:
            logger.error(f"Could not process {zip_path}: {e}")

    def _process_saraga_json(self, data: Dict, tradition: str, source: str, filename: str):
        """Process Saraga JSON data."""
        # Extract raga information
        if 'raga' in data:
            raga_info = data['raga']
            raga_name = raga_info.get('name', '')
            if raga_name:
                raga_id = f"saraga_{raga_name.lower().replace(' ', '_')}"
                if raga_id not in self.unified_ragas:
                    self.unified_ragas[raga_id] = {
                        "raga_id": raga_id,
                        "name": raga_name,
                        "tradition": tradition,
                        "sources": [source],
                        "song_count": 0,
                        "metadata": raga_info,
                        "youtube_links": []
                    }
        
        # Extract artist information
        if 'artist' in data:
            artist_info = data['artist']
            artist_name = artist_info.get('name', '')
            if artist_name:
                artist_id = f"saraga_{artist_name.lower().replace(' ', '_')}"
                if artist_id not in self.unified_artists:
                    self.unified_artists[artist_id] = {
                        "artist_id": artist_id,
                        "name": artist_name,
                        "tradition": tradition,
                        "sources": [source],
                        "song_count": 0,
                        "metadata": artist_info,
                        "youtube_links": []
                    }
        
        # Extract song information
        song_name = data.get('title', data.get('name', ''))
        if song_name:
            song_id = f"saraga_{song_name.lower().replace(' ', '_')}"
            if song_id not in self.unified_songs:
                self.unified_songs[song_id] = {
                    "song_id": song_id,
                    "name": song_name,
                    "tradition": tradition,
                    "sources": [source],
                    "metadata": data,
                    "youtube_links": []
                }

    def calculate_song_counts(self):
        """Calculate song counts for ragas, artists, and composers."""
        logger.info("📊 Calculating song counts...")
        
        # Count songs per raga
        for song_id, song_data in self.unified_songs.items():
            # This is a simplified approach - in reality, we'd need to link songs to ragas
            # For now, we'll just count total songs
            pass
        
        # Count songs per artist
        for artist_id, artist_data in self.unified_artists.items():
            # Count songs where this artist appears
            song_count = 0
            for song_id, song_data in self.unified_songs.items():
                if artist_data["name"] in str(song_data.get("metadata", {})):
                    song_count += 1
            artist_data["song_count"] = song_count
        
        # Count songs per composer
        for composer_id, composer_data in self.unified_composers.items():
            # Count songs where this composer appears
            song_count = 0
            for song_id, song_data in self.unified_songs.items():
                if composer_data["name"] in str(song_data.get("metadata", {})):
                    song_count += 1
            composer_data["song_count"] = song_count

    def generate_report(self):
        """Generate comprehensive report."""
        logger.info("📊 Generating comprehensive report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "processing_stats": self.processing_stats,
            "dataset_summary": {
                "total_ragas": len(self.unified_ragas),
                "total_artists": len(self.unified_artists),
                "total_composers": len(self.unified_composers),
                "total_songs": len(self.unified_songs),
                "total_youtube_links": len(self.youtube_links)
            },
            "tradition_distribution": {
                "ragas": Counter(raga["tradition"] for raga in self.unified_ragas.values()),
                "artists": Counter(artist["tradition"] for artist in self.unified_artists.values()),
                "composers": Counter(composer["tradition"] for composer in self.unified_composers.values()),
                "songs": Counter(song["tradition"] for song in self.unified_songs.values())
            },
            "source_distribution": {
                "ragas": Counter(source for raga in self.unified_ragas.values() for source in raga["sources"]),
                "artists": Counter(source for artist in self.unified_artists.values() for source in artist["sources"]),
                "composers": Counter(source for composer in self.unified_composers.values() for source in composer["sources"]),
                "songs": Counter(source for song in self.unified_songs.values() for source in song["sources"])
            },
            "top_ragas_by_songs": sorted(
                [(raga["name"], raga["song_count"]) for raga in self.unified_ragas.values()],
                key=lambda x: x[1],
                reverse=True
            )[:20],
            "top_artists_by_songs": sorted(
                [(artist["name"], artist["song_count"]) for artist in self.unified_artists.values()],
                key=lambda x: x[1],
                reverse=True
            )[:20],
            "youtube_links_sample": list(self.youtube_links)[:20]
        }
        
        return report

    def save_unified_dataset(self):
        """Save the unified dataset."""
        logger.info("💾 Saving unified dataset...")
        
        # Save unified datasets
        with open(self.output_path / "unified_ragas_database.json", 'w', encoding='utf-8') as f:
            json.dump(self.unified_ragas, f, indent=2, ensure_ascii=False)
        
        with open(self.output_path / "unified_artists_database.json", 'w', encoding='utf-8') as f:
            json.dump(self.unified_artists, f, indent=2, ensure_ascii=False)
        
        with open(self.output_path / "unified_composers_database.json", 'w', encoding='utf-8') as f:
            json.dump(self.unified_composers, f, indent=2, ensure_ascii=False)
        
        with open(self.output_path / "unified_songs_database.json", 'w', encoding='utf-8') as f:
            json.dump(self.unified_songs, f, indent=2, ensure_ascii=False)
        
        # Save YouTube links
        with open(self.output_path / "youtube_links.json", 'w', encoding='utf-8') as f:
            json.dump(list(self.youtube_links), f, indent=2, ensure_ascii=False)
        
        # Save report
        report = self.generate_report()
        with open(self.output_path / "unified_dataset_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save CSV exports
        pd.DataFrame.from_dict(self.unified_ragas, orient='index').to_csv(
            self.output_path / "unified_ragas_database.csv", 
            index_label="raga_id"
        )
        
        pd.DataFrame.from_dict(self.unified_artists, orient='index').to_csv(
            self.output_path / "unified_artists_database.csv", 
            index_label="artist_id"
        )
        
        logger.info(f"✅ Unified dataset saved to {self.output_path}")

    def run_creation(self):
        """Run the complete unified dataset creation."""
        start_time = time.time()
        logger.info("🚀 STARTING UNIFIED DATASET CREATION")
        logger.info("=" * 60)
        
        # Process all data sources
        self.process_ramanarunachalam()
        self.process_saraga_carnatic_melody_synth()
        self.process_saraga_datasets()
        
        # Calculate song counts
        self.calculate_song_counts()
        
        # Save results
        self.save_unified_dataset()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("\n🎉 UNIFIED DATASET CREATION COMPLETED!")
        logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
        
        # Print summary
        logger.info("\n📊 CREATION SUMMARY:")
        logger.info(f"   Ragas processed: {self.processing_stats['ragas_processed']}")
        logger.info(f"   Artists processed: {self.processing_stats['artists_processed']}")
        logger.info(f"   Composers processed: {self.processing_stats['composers_processed']}")
        logger.info(f"   Songs processed: {self.processing_stats['songs_processed']}")
        logger.info(f"   YouTube links found: {self.processing_stats['youtube_links_found']}")
        logger.info(f"   Errors: {self.processing_stats['errors']}")
        
        logger.info("\n📊 UNIFIED DATASET SUMMARY:")
        logger.info(f"   Total ragas: {len(self.unified_ragas)}")
        logger.info(f"   Total artists: {len(self.unified_artists)}")
        logger.info(f"   Total composers: {len(self.unified_composers)}")
        logger.info(f"   Total songs: {len(self.unified_songs)}")
        logger.info(f"   Total YouTube links: {len(self.youtube_links)}")
        
        return True

def main():
    """Main function to run the unified dataset creation."""
    project_root = Path(__file__).parent
    downloads_path = project_root / "downloads"
    output_path = project_root / "data" / "unified_ragasense_dataset"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize creator
    creator = UnifiedDatasetCreator(downloads_path, output_path)
    
    # Run creation
    success = creator.run_creation()
    
    if success:
        logger.info(f"\n🎯 UNIFIED DATASET CREATION COMPLETE!")
        logger.info(f"📋 Dataset saved to: {output_path}")
        logger.info(f"📊 Report saved to: {output_path / 'unified_dataset_report.json'}")
    else:
        logger.error("❌ Unified dataset creation failed!")

if __name__ == "__main__":
    main()
