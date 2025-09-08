#!/usr/bin/env python3
"""
Real Data Analysis for RagaSense-Data
=====================================

This script analyzes the actual downloaded data to provide accurate statistics
and correct cross-tradition mappings based on real data rather than assumptions.

Features:
- Analyzes actual raga data for duplicates and quality issues
- Provides accurate statistics after cleaning
- Corrects cross-tradition mappings based on real data
- Generates accurate reports for website
"""

import json
import time
from datetime import datetime
from pathlib import Path
import sys
import logging
from typing import Dict, Any, List, Optional, Set
import pandas as pd
from collections import defaultdict, Counter
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_data_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealDataAnalyzer:
    """
    Analyzes the actual RagaSense-Data to provide accurate statistics.
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.ragas_data = {}
        self.artists_data = {}
        self.composers_data = {}
        self.songs_data = {}
        
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "cleaned_statistics": {},
            "data_quality_issues": {},
            "accurate_cross_tradition_mappings": {},
            "recommendations": {}
        }

    def load_data(self):
        """Load all the unified dataset files."""
        logger.info("📂 Loading unified dataset files...")
        
        try:
            # Load ragas
            ragas_path = self.data_path / "unified_ragas_database.json"
            if ragas_path.exists():
                with open(ragas_path, 'r', encoding='utf-8') as f:
                    self.ragas_data = json.load(f)
                logger.info(f"✅ Loaded {len(self.ragas_data)} ragas")
            
            # Load artists
            artists_path = self.data_path / "unified_artists_database.json"
            if artists_path.exists():
                with open(artists_path, 'r', encoding='utf-8') as f:
                    self.artists_data = json.load(f)
                logger.info(f"✅ Loaded {len(self.artists_data)} artists")
            
            # Load composers
            composers_path = self.data_path / "unified_composers_database.json"
            if composers_path.exists():
                with open(composers_path, 'r', encoding='utf-8') as f:
                    self.composers_data = json.load(f)
                logger.info(f"✅ Loaded {len(self.composers_data)} composers")
            
            # Load songs
            songs_path = self.data_path / "unified_songs_database.json"
            if songs_path.exists():
                with open(songs_path, 'r', encoding='utf-8') as f:
                    self.songs_data = json.load(f)
                logger.info(f"✅ Loaded {len(self.songs_data)} songs")
                
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")

    def analyze_raga_quality_issues(self) -> Dict[str, Any]:
        """Analyzes raga data for quality issues and duplicates."""
        logger.info("🔍 Analyzing raga data quality...")
        
        issues = {
            "duplicate_ragas": [],
            "combined_ragas": [],
            "empty_tradition": [],
            "unknown_ragas": [],
            "low_quality_ragas": [],
            "statistics": {}
        }
        
        # Analyze raga names for issues
        raga_names = list(self.ragas_data.keys())
        
        # Find combined ragas (containing commas)
        combined_ragas = [name for name in raga_names if ',' in name]
        issues["combined_ragas"] = combined_ragas
        
        # Find ragas with empty tradition
        empty_tradition = []
        for raga_id, raga_data in self.ragas_data.items():
            if not raga_data.get('tradition') or raga_data.get('tradition') == '':
                empty_tradition.append(raga_id)
        issues["empty_tradition"] = empty_tradition
        
        # Find unknown/placeholder ragas
        unknown_ragas = [name for name in raga_names if 'unknown' in name.lower() or 'placeholder' in name.lower()]
        issues["unknown_ragas"] = unknown_ragas
        
        # Find ragas with very high song counts (likely data issues)
        high_count_ragas = []
        for raga_id, raga_data in self.ragas_data.items():
            song_count = raga_data.get('song_count', 0)
            if song_count > 10000:  # Suspiciously high
                high_count_ragas.append({
                    "raga_id": raga_id,
                    "song_count": song_count,
                    "name": raga_data.get('name', 'Unknown')
                })
        issues["low_quality_ragas"] = high_count_ragas
        
        # Calculate statistics
        issues["statistics"] = {
            "total_ragas": len(raga_names),
            "combined_ragas_count": len(combined_ragas),
            "empty_tradition_count": len(empty_tradition),
            "unknown_ragas_count": len(unknown_ragas),
            "high_count_ragas_count": len(high_count_ragas)
        }
        
        logger.info(f"📊 Raga quality analysis:")
        logger.info(f"   Total ragas: {issues['statistics']['total_ragas']}")
        logger.info(f"   Combined ragas: {issues['statistics']['combined_ragas_count']}")
        logger.info(f"   Empty tradition: {issues['statistics']['empty_tradition_count']}")
        logger.info(f"   Unknown ragas: {issues['statistics']['unknown_ragas_count']}")
        logger.info(f"   High count ragas: {issues['statistics']['high_count_ragas_count']}")
        
        return issues

    def clean_raga_data(self) -> Dict[str, Any]:
        """Cleans raga data and provides accurate statistics."""
        logger.info("🧹 Cleaning raga data...")
        
        cleaned_data = {
            "unique_ragas": set(),
            "carnatic_ragas": set(),
            "hindustani_ragas": set(),
            "combined_ragas": set(),
            "unknown_ragas": set(),
            "statistics": {}
        }
        
        for raga_id, raga_data in self.ragas_data.items():
            name = raga_data.get('name', raga_id)
            tradition = raga_data.get('tradition', '')
            song_count = raga_data.get('song_count', 0)
            
            # Skip unknown/placeholder ragas
            if 'unknown' in name.lower() or 'placeholder' in name.lower():
                cleaned_data["unknown_ragas"].add(raga_id)
                continue
            
            # Handle combined ragas (containing commas)
            if ',' in name:
                cleaned_data["combined_ragas"].add(raga_id)
                # Split and add individual ragas
                individual_ragas = [r.strip() for r in name.split(',')]
                for individual_raga in individual_ragas:
                    if individual_raga and individual_raga.lower() != 'unknown':
                        cleaned_data["unique_ragas"].add(individual_raga)
            else:
                cleaned_data["unique_ragas"].add(name)
            
            # Categorize by tradition
            if tradition.lower() == 'carnatic':
                cleaned_data["carnatic_ragas"].add(name)
            elif tradition.lower() == 'hindustani':
                cleaned_data["hindustani_ragas"].add(name)
        
        # Calculate statistics
        cleaned_data["statistics"] = {
            "unique_ragas_count": len(cleaned_data["unique_ragas"]),
            "carnatic_ragas_count": len(cleaned_data["carnatic_ragas"]),
            "hindustani_ragas_count": len(cleaned_data["hindustani_ragas"]),
            "combined_ragas_count": len(cleaned_data["combined_ragas"]),
            "unknown_ragas_count": len(cleaned_data["unknown_ragas"]),
            "total_original_ragas": len(self.ragas_data)
        }
        
        logger.info(f"✅ Cleaned raga statistics:")
        logger.info(f"   Unique ragas: {cleaned_data['statistics']['unique_ragas_count']}")
        logger.info(f"   Carnatic ragas: {cleaned_data['statistics']['carnatic_ragas_count']}")
        logger.info(f"   Hindustani ragas: {cleaned_data['statistics']['hindustani_ragas_count']}")
        logger.info(f"   Combined ragas: {cleaned_data['statistics']['combined_ragas_count']}")
        logger.info(f"   Unknown ragas: {cleaned_data['statistics']['unknown_ragas_count']}")
        
        return cleaned_data

    def analyze_artist_data(self) -> Dict[str, Any]:
        """Analyzes artist data for quality and accuracy."""
        logger.info("🎭 Analyzing artist data...")
        
        artist_analysis = {
            "total_artists": len(self.artists_data),
            "carnatic_artists": 0,
            "hindustani_artists": 0,
            "unknown_tradition": 0,
            "top_artists": [],
            "statistics": {}
        }
        
        # Analyze artists by tradition
        for artist_id, artist_data in self.artists_data.items():
            tradition = artist_data.get('tradition', '').lower()
            song_count = artist_data.get('song_count', 0)
            
            if tradition == 'carnatic':
                artist_analysis["carnatic_artists"] += 1
            elif tradition == 'hindustani':
                artist_analysis["hindustani_artists"] += 1
            else:
                artist_analysis["unknown_tradition"] += 1
        
        # Get top artists by song count
        artists_with_counts = []
        for artist_id, artist_data in self.artists_data.items():
            song_count = artist_data.get('song_count', 0)
            if song_count > 0:
                artists_with_counts.append({
                    "artist_id": artist_id,
                    "name": artist_data.get('name', artist_id),
                    "song_count": song_count,
                    "tradition": artist_data.get('tradition', 'Unknown')
                })
        
        # Sort by song count and get top 20
        artists_with_counts.sort(key=lambda x: x['song_count'], reverse=True)
        artist_analysis["top_artists"] = artists_with_counts[:20]
        
        artist_analysis["statistics"] = {
            "total_artists": artist_analysis["total_artists"],
            "carnatic_artists": artist_analysis["carnatic_artists"],
            "hindustani_artists": artist_analysis["hindustani_artists"],
            "unknown_tradition": artist_analysis["unknown_tradition"],
            "artists_with_songs": len(artists_with_counts)
        }
        
        logger.info(f"✅ Artist analysis:")
        logger.info(f"   Total artists: {artist_analysis['statistics']['total_artists']}")
        logger.info(f"   Carnatic artists: {artist_analysis['statistics']['carnatic_artists']}")
        logger.info(f"   Hindustani artists: {artist_analysis['statistics']['hindustani_artists']}")
        logger.info(f"   Artists with songs: {artist_analysis['statistics']['artists_with_songs']}")
        
        return artist_analysis

    def analyze_composer_data(self) -> Dict[str, Any]:
        """Analyzes composer data for quality and accuracy."""
        logger.info("🎼 Analyzing composer data...")
        
        composer_analysis = {
            "total_composers": len(self.composers_data),
            "composers_with_songs": 0,
            "composers_without_songs": 0,
            "top_composers": [],
            "statistics": {}
        }
        
        # Analyze composers
        composers_with_counts = []
        for composer_id, composer_data in self.composers_data.items():
            song_count = composer_data.get('song_count', 0)
            
            if song_count > 0:
                composer_analysis["composers_with_songs"] += 1
                composers_with_counts.append({
                    "composer_id": composer_id,
                    "name": composer_data.get('name', composer_id),
                    "song_count": song_count,
                    "tradition": composer_data.get('tradition', 'Unknown')
                })
            else:
                composer_analysis["composers_without_songs"] += 1
        
        # Sort by song count and get top 20
        composers_with_counts.sort(key=lambda x: x['song_count'], reverse=True)
        composer_analysis["top_composers"] = composers_with_counts[:20]
        
        composer_analysis["statistics"] = {
            "total_composers": composer_analysis["total_composers"],
            "composers_with_songs": composer_analysis["composers_with_songs"],
            "composers_without_songs": composer_analysis["composers_without_songs"]
        }
        
        logger.info(f"✅ Composer analysis:")
        logger.info(f"   Total composers: {composer_analysis['statistics']['total_composers']}")
        logger.info(f"   Composers with songs: {composer_analysis['statistics']['composers_with_songs']}")
        logger.info(f"   Composers without songs: {composer_analysis['statistics']['composers_without_songs']}")
        
        return composer_analysis

    def generate_accurate_cross_tradition_mappings(self) -> Dict[str, Any]:
        """Generates accurate cross-tradition mappings based on real data."""
        logger.info("🔗 Generating accurate cross-tradition mappings...")
        
        # Based on your corrections, here are the accurate mappings
        accurate_mappings = {
            "structural_equivalents": [
                {
                    "carnatic": "Kalyani",
                    "hindustani": "Yaman",
                    "relationship": "Same scale (Lydian mode with tivra Ma)",
                    "confidence": "high",
                    "notes": "Both use Lydian mode. Differences are stylistic (gamaka in Carnatic, Re/Pa treatment in Hindustani)"
                },
                {
                    "carnatic": "Shankarabharanam",
                    "hindustani": "Bilawal",
                    "relationship": "Same scale (Major/Ionian scale)",
                    "confidence": "high",
                    "notes": "Both use Major/Ionian scale. Very close equivalents."
                },
                {
                    "carnatic": "Mohanam",
                    "hindustani": "Bhoopali (Bhoop)",
                    "relationship": "Same scale (Major pentatonic)",
                    "confidence": "high",
                    "notes": "Both use S R2 G3 P D2 S. Practically identical."
                }
            ],
            "mood_equivalents": [
                {
                    "carnatic": "Bhairavi",
                    "hindustani": "Bhairavi",
                    "relationship": "Mood-equivalent (different scales)",
                    "confidence": "medium",
                    "notes": "Carnatic Bhairavi = janya of Natabhairavi, Hindustani Bhairavi = Bhairavi thaat (all komal swaras). Different structures, both evoke pathos/devotion."
                },
                {
                    "carnatic": "Todi (Hanumatodi)",
                    "hindustani": "Miyan ki Todi",
                    "relationship": "Mood-equivalent (different scales)",
                    "confidence": "medium",
                    "notes": "Carnatic Todi uses R1 G2 M1 D1 N2, Hindustani Todi uses komal Re, Ga, Dha, Ni with tivra Ma. Emotional overlap but different swaras."
                },
                {
                    "carnatic": "Hindolam",
                    "hindustani": "Malkauns",
                    "relationship": "Mood-equivalent (different pentatonic scales)",
                    "confidence": "medium",
                    "notes": "Both are pentatonic, but Hindolam = S G2 M1 D1 N2; Malkauns = S g M d n. Different note sets, similar introspective/serious mood."
                }
            ],
            "summary": {
                "structural_equivalents_count": 3,
                "mood_equivalents_count": 3,
                "total_mappings": 6,
                "high_confidence": 3,
                "medium_confidence": 3
            }
        }
        
        logger.info(f"✅ Accurate cross-tradition mappings:")
        logger.info(f"   Structural equivalents: {accurate_mappings['summary']['structural_equivalents_count']}")
        logger.info(f"   Mood equivalents: {accurate_mappings['summary']['mood_equivalents_count']}")
        logger.info(f"   High confidence: {accurate_mappings['summary']['high_confidence']}")
        logger.info(f"   Medium confidence: {accurate_mappings['summary']['medium_confidence']}")
        
        return accurate_mappings

    def generate_accurate_statistics(self) -> Dict[str, Any]:
        """Generates accurate statistics after cleaning the data."""
        logger.info("📊 Generating accurate statistics...")
        
        # Clean the data first
        cleaned_ragas = self.clean_raga_data()
        artist_analysis = self.analyze_artist_data()
        composer_analysis = self.analyze_composer_data()
        
        accurate_stats = {
            "ragas": {
                "unique_ragas": cleaned_ragas["statistics"]["unique_ragas_count"],
                "carnatic_ragas": cleaned_ragas["statistics"]["carnatic_ragas_count"],
                "hindustani_ragas": cleaned_ragas["statistics"]["hindustani_ragas_count"],
                "combined_ragas": cleaned_ragas["statistics"]["combined_ragas_count"],
                "unknown_ragas": cleaned_ragas["statistics"]["unknown_ragas_count"],
                "original_total": cleaned_ragas["statistics"]["total_original_ragas"]
            },
            "artists": {
                "total_artists": artist_analysis["statistics"]["total_artists"],
                "carnatic_artists": artist_analysis["statistics"]["carnatic_artists"],
                "hindustani_artists": artist_analysis["statistics"]["hindustani_artists"],
                "artists_with_songs": artist_analysis["statistics"]["artists_with_songs"]
            },
            "composers": {
                "total_composers": composer_analysis["statistics"]["total_composers"],
                "composers_with_songs": composer_analysis["statistics"]["composers_with_songs"],
                "composers_without_songs": composer_analysis["statistics"]["composers_without_songs"]
            },
            "songs": {
                "total_songs": len(self.songs_data),
                "songs_with_youtube": sum(1 for song in self.songs_data.values() if song.get('youtube_link')),
                "total_youtube_views": sum(song.get('views', 0) for song in self.songs_data.values()),
                "total_duration_hours": sum(song.get('duration_minutes', 0) for song in self.songs_data.values()) / 60
            },
            "data_quality": {
                "ragas_with_issues": cleaned_ragas["statistics"]["combined_ragas_count"] + cleaned_ragas["statistics"]["unknown_ragas_count"],
                "artists_without_tradition": artist_analysis["statistics"]["unknown_tradition"],
                "composers_without_songs": composer_analysis["statistics"]["composers_without_songs"]
            }
        }
        
        logger.info(f"✅ Accurate statistics:")
        logger.info(f"   Unique ragas: {accurate_stats['ragas']['unique_ragas']}")
        logger.info(f"   Total artists: {accurate_stats['artists']['total_artists']}")
        logger.info(f"   Composers with songs: {accurate_stats['composers']['composers_with_songs']}")
        logger.info(f"   Total songs: {accurate_stats['songs']['total_songs']}")
        logger.info(f"   YouTube videos: {accurate_stats['songs']['songs_with_youtube']}")
        
        return accurate_stats

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Runs the complete analysis of real data."""
        start_time = time.time()
        logger.info("🚀 STARTING REAL DATA ANALYSIS")
        logger.info("=" * 60)
        
        # Load data
        self.load_data()
        
        # Analyze data quality issues
        raga_issues = self.analyze_raga_quality_issues()
        
        # Generate accurate statistics
        accurate_stats = self.generate_accurate_statistics()
        
        # Generate accurate cross-tradition mappings
        accurate_mappings = self.generate_accurate_cross_tradition_mappings()
        
        # Compile results
        self.analysis_results.update({
            "cleaned_statistics": accurate_stats,
            "data_quality_issues": raga_issues,
            "accurate_cross_tradition_mappings": accurate_mappings,
            "recommendations": {
                "data_cleaning_needed": True,
                "ragas_to_split": raga_issues["combined_ragas"],
                "ragas_to_remove": raga_issues["unknown_ragas"],
                "tradition_assignment_needed": raga_issues["empty_tradition"]
            }
        })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("\n🎉 REAL DATA ANALYSIS COMPLETED!")
        logger.info(f"⏱️ Analysis time: {processing_time:.1f} seconds")
        
        # Print summary
        logger.info("\n📊 ACCURATE STATISTICS SUMMARY:")
        logger.info(f"   🎼 Unique ragas: {accurate_stats['ragas']['unique_ragas']}")
        logger.info(f"   🎭 Total artists: {accurate_stats['artists']['total_artists']}")
        logger.info(f"   🎼 Composers with songs: {accurate_stats['composers']['composers_with_songs']}")
        logger.info(f"   🎵 Total songs: {accurate_stats['songs']['total_songs']}")
        logger.info(f"   🎥 YouTube videos: {accurate_stats['songs']['songs_with_youtube']}")
        logger.info(f"   👀 Total views: {accurate_stats['songs']['total_youtube_views']:,}")
        logger.info(f"   ⏱️ Total duration: {accurate_stats['songs']['total_duration_hours']:.1f} hours")
        
        logger.info("\n🔗 ACCURATE CROSS-TRADITION MAPPINGS:")
        logger.info(f"   ✅ Structural equivalents: {accurate_mappings['summary']['structural_equivalents_count']}")
        logger.info(f"   🎭 Mood equivalents: {accurate_mappings['summary']['mood_equivalents_count']}")
        logger.info(f"   🎯 High confidence: {accurate_mappings['summary']['high_confidence']}")
        logger.info(f"   ⚠️ Medium confidence: {accurate_mappings['summary']['medium_confidence']}")
        
        return self.analysis_results

    def save_analysis_results(self, output_path: Path):
        """Saves the complete analysis results to JSON."""
        logger.info("💾 Saving real data analysis results...")
        
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(item) for item in obj]
            return obj
        
        serializable_results = convert_sets(self.analysis_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Real data analysis results saved to {output_path}")

def main():
    """Main function to run the real data analysis."""
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "unified_ragasense_dataset"
    output_path = project_root / "data" / "real_data_analysis_results.json"
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = RealDataAnalyzer(data_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Save results
    analyzer.save_analysis_results(output_path)
    
    logger.info(f"\n🎯 REAL DATA ANALYSIS COMPLETE!")
    logger.info(f"📋 Results saved to: {output_path}")

if __name__ == "__main__":
    main()
