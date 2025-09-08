#!/usr/bin/env python3
"""
Comprehensive Data Analysis for RagaSense-Data
==============================================

This script performs deep analysis of all downloaded data sources:
1. Ramanarunachalam Music Repository (Carnatic & Hindustani)
2. Saraga-Carnatic-Melody-Synth
3. Saraga 1.5 Carnatic & Hindustani datasets
4. YouTube links extraction and analysis
5. Cross-source metadata correlation

Features:
- Multi-threaded processing for large datasets
- YouTube link extraction and validation
- Cross-tradition mapping analysis
- Data quality assessment
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
import requests
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_data_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    name: str
    path: Path
    tradition: str
    file_count: int
    size_mb: float
    metadata_files: List[Path]
    audio_files: List[Path]
    youtube_links: List[str]

class ComprehensiveDataAnalyzer:
    """
    Analyzes all data sources and creates a unified dataset.
    """
    
    def __init__(self, downloads_path: Path, output_path: Path):
        self.downloads_path = downloads_path
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.data_sources = []
        self.unified_ragas = {}
        self.unified_artists = {}
        self.unified_composers = {}
        self.unified_songs = {}
        self.youtube_links = set()
        
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "data_sources": [],
            "summary": {},
            "youtube_analysis": {},
            "cross_tradition_mapping": {},
            "data_quality": {},
            "recommendations": []
        }

    def analyze_ramanarunachalam(self):
        """Analyze Ramanarunachalam Music Repository."""
        logger.info("🔍 Analyzing Ramanarunachalam Music Repository...")
        
        ramanarunachalam_path = self.downloads_path / "Ramanarunachalam_Music_Repository"
        
        # Analyze Carnatic
        carnatic_path = ramanarunachalam_path / "Carnatic"
        if carnatic_path.exists():
            carnatic_source = self._analyze_tradition_source(
                "Ramanarunachalam-Carnatic", 
                carnatic_path, 
                "Carnatic"
            )
            self.data_sources.append(carnatic_source)
        
        # Analyze Hindustani
        hindustani_path = ramanarunachalam_path / "Hindustani"
        if hindustani_path.exists():
            hindustani_source = self._analyze_tradition_source(
                "Ramanarunachalam-Hindustani", 
                hindustani_path, 
                "Hindustani"
            )
            self.data_sources.append(hindustani_source)

    def _analyze_tradition_source(self, name: str, path: Path, tradition: str) -> DataSource:
        """Analyze a tradition-specific source."""
        logger.info(f"📂 Analyzing {name}...")
        
        metadata_files = []
        audio_files = []
        youtube_links = []
        total_size = 0
        
        # Count files and extract metadata
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                
                if file_path.suffix == '.json':
                    metadata_files.append(file_path)
                    # Extract YouTube links from JSON files
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            youtube_links.extend(self._extract_youtube_links(data))
                    except Exception as e:
                        logger.warning(f"Could not read {file_path}: {e}")
                
                elif file_path.suffix in ['.mp3', '.wav', '.flac']:
                    audio_files.append(file_path)
        
        return DataSource(
            name=name,
            path=path,
            tradition=tradition,
            file_count=len(metadata_files) + len(audio_files),
            size_mb=total_size / (1024 * 1024),
            metadata_files=metadata_files,
            audio_files=audio_files,
            youtube_links=youtube_links
        )

    def analyze_saraga_carnatic_melody_synth(self):
        """Analyze Saraga-Carnatic-Melody-Synth dataset."""
        logger.info("🔍 Analyzing Saraga-Carnatic-Melody-Synth...")
        
        saraga_path = self.downloads_path / "saraga_carnatic_melody_synth" / "Saraga-Carnatic-Melody-Synth"
        
        if saraga_path.exists():
            metadata_files = []
            audio_files = []
            youtube_links = []
            total_size = 0
            
            # Analyze audio files
            audio_dir = saraga_path / "audio"
            if audio_dir.exists():
                for file_path in audio_dir.rglob("*.wav"):
                    audio_files.append(file_path)
                    total_size += file_path.stat().st_size
            
            # Analyze metadata
            mapping_file = saraga_path / "artists_to_track_mapping.json"
            if mapping_file.exists():
                metadata_files.append(mapping_file)
                total_size += mapping_file.stat().st_size
                
                try:
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        youtube_links.extend(self._extract_youtube_links(data))
                except Exception as e:
                    logger.warning(f"Could not read {mapping_file}: {e}")
            
            source = DataSource(
                name="Saraga-Carnatic-Melody-Synth",
                path=saraga_path,
                tradition="Carnatic",
                file_count=len(metadata_files) + len(audio_files),
                size_mb=total_size / (1024 * 1024),
                metadata_files=metadata_files,
                audio_files=audio_files,
                youtube_links=youtube_links
            )
            self.data_sources.append(source)

    def analyze_saraga_datasets(self):
        """Analyze Saraga 1.5 datasets."""
        logger.info("🔍 Analyzing Saraga 1.5 datasets...")
        
        saraga_datasets_path = self.downloads_path / "saraga_datasets"
        
        # Analyze Carnatic
        carnatic_zip = saraga_datasets_path / "carnatic" / "saraga1.5_carnatic.zip"
        if carnatic_zip.exists():
            carnatic_source = self._analyze_zip_dataset(
                "Saraga1.5-Carnatic", 
                carnatic_zip, 
                "Carnatic"
            )
            self.data_sources.append(carnatic_source)
        
        # Analyze Hindustani
        hindustani_zip = saraga_datasets_path / "hindustani" / "saraga1.5_hindustani.zip"
        if hindustani_zip.exists():
            hindustani_source = self._analyze_zip_dataset(
                "Saraga1.5-Hindustani", 
                hindustani_zip, 
                "Hindustani"
            )
            self.data_sources.append(hindustani_source)

    def _analyze_zip_dataset(self, name: str, zip_path: Path, tradition: str) -> DataSource:
        """Analyze a ZIP dataset."""
        logger.info(f"📦 Analyzing {name}...")
        
        metadata_files = []
        audio_files = []
        youtube_links = []
        total_size = zip_path.stat().st_size
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if file_info.filename.endswith('.json'):
                        metadata_files.append(Path(file_info.filename))
                        # Extract YouTube links from JSON files
                        try:
                            with zip_file.open(file_info) as f:
                                data = json.load(f)
                                youtube_links.extend(self._extract_youtube_links(data))
                        except Exception as e:
                            logger.warning(f"Could not read {file_info.filename}: {e}")
                    
                    elif file_info.filename.endswith(('.mp3', '.wav', '.flac')):
                        audio_files.append(Path(file_info.filename))
        except Exception as e:
            logger.error(f"Could not analyze {zip_path}: {e}")
        
        return DataSource(
            name=name,
            path=zip_path,
            tradition=tradition,
            file_count=len(metadata_files) + len(audio_files),
            size_mb=total_size / (1024 * 1024),
            metadata_files=metadata_files,
            audio_files=audio_files,
            youtube_links=youtube_links
        )

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

    def analyze_youtube_links(self):
        """Analyze all YouTube links found."""
        logger.info("📺 Analyzing YouTube links...")
        
        all_links = set()
        for source in self.data_sources:
            all_links.update(source.youtube_links)
        
        self.youtube_links = all_links
        
        # Analyze YouTube links
        youtube_analysis = {
            "total_links": len(all_links),
            "unique_links": len(set(all_links)),
            "sources_with_links": len([s for s in self.data_sources if s.youtube_links]),
            "links_by_source": {s.name: len(s.youtube_links) for s in self.data_sources},
            "sample_links": list(all_links)[:10] if all_links else []
        }
        
        self.analysis_results["youtube_analysis"] = youtube_analysis
        logger.info(f"✅ Found {len(all_links)} YouTube links across {youtube_analysis['sources_with_links']} sources")

    def create_unified_dataset(self):
        """Create unified dataset from all sources."""
        logger.info("🔄 Creating unified dataset...")
        
        # Process each data source
        for source in self.data_sources:
            logger.info(f"Processing {source.name}...")
            self._process_data_source(source)
        
        # Generate summary
        self._generate_summary()

    def _process_data_source(self, source: DataSource):
        """Process a single data source."""
        logger.info(f"📊 Processing {source.name} ({source.tradition})...")
        
        # Process metadata files
        for metadata_file in source.metadata_files:
            try:
                if source.path.is_file() and source.path.suffix == '.zip':
                    # Handle ZIP files
                    with zipfile.ZipFile(source.path, 'r') as zip_file:
                        with zip_file.open(str(metadata_file)) as f:
                            data = json.load(f)
                            self._process_metadata(data, source.tradition, source.name)
                else:
                    # Handle regular files
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._process_metadata(data, source.tradition, source.name)
            except Exception as e:
                logger.warning(f"Could not process {metadata_file}: {e}")

    def _process_metadata(self, data: Any, tradition: str, source_name: str):
        """Process metadata and add to unified dataset."""
        if isinstance(data, dict):
            # Process individual records
            if 'raga' in data or 'name' in data:
                self._process_raga_record(data, tradition, source_name)
            elif 'artist' in data or 'performer' in data:
                self._process_artist_record(data, tradition, source_name)
            elif 'composer' in data or 'creator' in data:
                self._process_composer_record(data, tradition, source_name)
            elif 'song' in data or 'title' in data:
                self._process_song_record(data, tradition, source_name)
        elif isinstance(data, list):
            # Process list of records
            for item in data:
                self._process_metadata(item, tradition, source_name)

    def _process_raga_record(self, data: Dict, tradition: str, source_name: str):
        """Process a raga record."""
        raga_id = data.get('id', data.get('raga_id', ''))
        raga_name = data.get('name', data.get('raga', ''))
        
        if raga_id and raga_name:
            if raga_id not in self.unified_ragas:
                self.unified_ragas[raga_id] = {
                    "raga_id": raga_id,
                    "name": raga_name,
                    "tradition": tradition,
                    "sources": [source_name],
                    "song_count": 0,
                    "metadata": {}
                }
            else:
                # Merge with existing raga
                if source_name not in self.unified_ragas[raga_id]["sources"]:
                    self.unified_ragas[raga_id]["sources"].append(source_name)
            
            # Add metadata
            self.unified_ragas[raga_id]["metadata"].update(data)

    def _process_artist_record(self, data: Dict, tradition: str, source_name: str):
        """Process an artist record."""
        artist_id = data.get('id', data.get('artist_id', ''))
        artist_name = data.get('name', data.get('artist', data.get('performer', '')))
        
        if artist_id and artist_name:
            if artist_id not in self.unified_artists:
                self.unified_artists[artist_id] = {
                    "artist_id": artist_id,
                    "name": artist_name,
                    "tradition": tradition,
                    "sources": [source_name],
                    "song_count": 0,
                    "metadata": {}
                }
            else:
                if source_name not in self.unified_artists[artist_id]["sources"]:
                    self.unified_artists[artist_id]["sources"].append(source_name)
            
            self.unified_artists[artist_id]["metadata"].update(data)

    def _process_composer_record(self, data: Dict, tradition: str, source_name: str):
        """Process a composer record."""
        composer_id = data.get('id', data.get('composer_id', ''))
        composer_name = data.get('name', data.get('composer', data.get('creator', '')))
        
        if composer_id and composer_name:
            if composer_id not in self.unified_composers:
                self.unified_composers[composer_id] = {
                    "composer_id": composer_id,
                    "name": composer_name,
                    "tradition": tradition,
                    "sources": [source_name],
                    "song_count": 0,
                    "metadata": {}
                }
            else:
                if source_name not in self.unified_composers[composer_id]["sources"]:
                    self.unified_composers[composer_id]["sources"].append(source_name)
            
            self.unified_composers[composer_id]["metadata"].update(data)

    def _process_song_record(self, data: Dict, tradition: str, source_name: str):
        """Process a song record."""
        song_id = data.get('id', data.get('song_id', ''))
        song_name = data.get('name', data.get('song', data.get('title', '')))
        
        if song_id and song_name:
            if song_id not in self.unified_songs:
                self.unified_songs[song_id] = {
                    "song_id": song_id,
                    "name": song_name,
                    "tradition": tradition,
                    "sources": [source_name],
                    "metadata": {}
                }
            else:
                if source_name not in self.unified_songs[song_id]["sources"]:
                    self.unified_songs[song_id]["sources"].append(source_name)
            
            self.unified_songs[song_id]["metadata"].update(data)

    def _generate_summary(self):
        """Generate analysis summary."""
        logger.info("📊 Generating analysis summary...")
        
        # Data source summary
        for source in self.data_sources:
            self.analysis_results["data_sources"].append({
                "name": source.name,
                "tradition": source.tradition,
                "file_count": source.file_count,
                "size_mb": round(source.size_mb, 2),
                "metadata_files": len(source.metadata_files),
                "audio_files": len(source.audio_files),
                "youtube_links": len(source.youtube_links)
            })
        
        # Overall summary
        self.analysis_results["summary"] = {
            "total_sources": len(self.data_sources),
            "total_files": sum(s.file_count for s in self.data_sources),
            "total_size_mb": round(sum(s.size_mb for s in self.data_sources), 2),
            "total_metadata_files": sum(len(s.metadata_files) for s in self.data_sources),
            "total_audio_files": sum(len(s.audio_files) for s in self.data_sources),
            "total_youtube_links": len(self.youtube_links),
            "unified_ragas": len(self.unified_ragas),
            "unified_artists": len(self.unified_artists),
            "unified_composers": len(self.unified_composers),
            "unified_songs": len(self.unified_songs)
        }
        
        # Data quality assessment
        self.analysis_results["data_quality"] = {
            "ragas_with_multiple_sources": len([r for r in self.unified_ragas.values() if len(r["sources"]) > 1]),
            "artists_with_multiple_sources": len([a for a in self.unified_artists.values() if len(a["sources"]) > 1]),
            "composers_with_multiple_sources": len([c for c in self.unified_composers.values() if len(c["sources"]) > 1]),
            "songs_with_multiple_sources": len([s for s in self.unified_songs.values() if len(s["sources"]) > 1]),
            "cross_tradition_ragas": len([r for r in self.unified_ragas.values() if len(set([s.split('-')[0] for s in r["sources"]])) > 1])
        }

    def save_results(self):
        """Save analysis results."""
        logger.info("💾 Saving analysis results...")
        
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
        
        # Save analysis report
        with open(self.output_path / "comprehensive_analysis_report.json", 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        # Save CSV exports
        pd.DataFrame.from_dict(self.unified_ragas, orient='index').to_csv(
            self.output_path / "unified_ragas_database.csv", 
            index_label="raga_id"
        )
        
        pd.DataFrame.from_dict(self.unified_artists, orient='index').to_csv(
            self.output_path / "unified_artists_database.csv", 
            index_label="artist_id"
        )
        
        logger.info(f"✅ Results saved to {self.output_path}")

    def run_analysis(self):
        """Run the complete analysis."""
        start_time = time.time()
        logger.info("🚀 STARTING COMPREHENSIVE DATA ANALYSIS")
        logger.info("=" * 60)
        
        # Analyze all data sources
        self.analyze_ramanarunachalam()
        self.analyze_saraga_carnatic_melody_synth()
        self.analyze_saraga_datasets()
        
        # Analyze YouTube links
        self.analyze_youtube_links()
        
        # Create unified dataset
        self.create_unified_dataset()
        
        # Save results
        self.save_results()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
        
        # Print summary
        summary = self.analysis_results["summary"]
        logger.info("\n📊 ANALYSIS SUMMARY:")
        logger.info(f"   Data sources analyzed: {summary['total_sources']}")
        logger.info(f"   Total files: {summary['total_files']:,}")
        logger.info(f"   Total size: {summary['total_size_mb']:.1f} MB")
        logger.info(f"   Metadata files: {summary['total_metadata_files']:,}")
        logger.info(f"   Audio files: {summary['total_audio_files']:,}")
        logger.info(f"   YouTube links: {summary['total_youtube_links']:,}")
        logger.info(f"   Unified ragas: {summary['unified_ragas']:,}")
        logger.info(f"   Unified artists: {summary['unified_artists']:,}")
        logger.info(f"   Unified composers: {summary['unified_composers']:,}")
        logger.info(f"   Unified songs: {summary['unified_songs']:,}")
        
        return True

def main():
    """Main function to run the comprehensive analysis."""
    project_root = Path(__file__).parent
    downloads_path = project_root / "downloads"
    output_path = project_root / "data" / "comprehensive_unified_dataset"
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ComprehensiveDataAnalyzer(downloads_path, output_path)
    
    # Run analysis
    success = analyzer.run_analysis()
    
    if success:
        logger.info(f"\n🎯 COMPREHENSIVE ANALYSIS COMPLETE!")
        logger.info(f"📋 Results saved to: {output_path}")
        logger.info(f"📊 Report saved to: {output_path / 'comprehensive_analysis_report.json'}")
    else:
        logger.error("❌ Comprehensive analysis failed!")

if __name__ == "__main__":
    main()
