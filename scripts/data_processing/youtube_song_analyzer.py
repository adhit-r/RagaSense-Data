#!/usr/bin/env python3
"""
YouTube Song Analyzer for Ramanarunachalam Repository
====================================================

This script extracts all YouTube links, song details, and builds a comprehensive
database of:
- YouTube video IDs and links
- Song-Raga-Artist-Composer mappings
- View counts and durations
- Multi-language song information
- Complete metadata for each song

Features:
- Extracts all YouTube video IDs
- Builds song database with full metadata
- Analyzes view counts and popularity
- Creates raga-song-artist-composer relationships
- Exports to multiple formats (JSON, CSV, Neo4j-ready)
"""

import json
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
import sys
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import pandas as pd
from collections import defaultdict, Counter
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_song_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YouTubeSongAnalyzer:
    """
    Analyzes the Ramanarunachalam repository to extract YouTube links,
    song details, and build comprehensive music database.
    """
    
    def __init__(self, repo_path: Path, max_workers: int = 8):
        self.repo_path = repo_path
        self.output_path = Path("data") / "youtube_song_analysis"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # Data storage
        self.songs = {}  # song_id -> song_data
        self.youtube_videos = {}  # video_id -> video_data
        self.raga_songs = defaultdict(list)  # raga_id -> [song_ids]
        self.artist_songs = defaultdict(list)  # artist_id -> [song_ids]
        self.composer_songs = defaultdict(list)  # composer_id -> [song_ids]
        self.song_metadata = {}  # song_id -> metadata
        
        # Statistics
        self.stats = {
            'total_songs_found': 0,
            'youtube_videos_found': 0,
            'unique_ragas': set(),
            'unique_artists': set(),
            'unique_composers': set(),
            'total_views': 0,
            'total_duration_minutes': 0,
            'processing_time': 0
        }

    def _extract_songs_from_file(self, file_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all songs from a JSON file."""
        songs = []
        data = file_data.get('data', {})
        
        if 'songs' in data and isinstance(data['songs'], list):
            for song in data['songs']:
                if isinstance(song, dict) and 'I' in song:  # YouTube ID present
                    song_info = {
                        'youtube_id': song.get('I', ''),
                        'song_id': song.get('S', ''),
                        'raga_id': song.get('R', ''),
                        'composer_id': song.get('C', ''),
                        'artist_id': song.get('A', ''),
                        'duration': song.get('D', ''),
                        'views': song.get('V', ''),
                        'language_type': song.get('J', ''),
                        'type_id': song.get('T', ''),
                        'file_path': file_data.get('file_path', ''),
                        'file_type': file_data.get('file_type', '')
                    }
                    songs.append(song_info)
        
        return songs

    def _process_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single JSON file to extract song information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Classify file type
            relative_path = str(file_path.relative_to(self.repo_path))
            file_type = self._classify_file_type(relative_path)
            
            file_data = {
                'file_path': relative_path,
                'file_type': file_type,
                'data': data,
                'processed_at': datetime.now().isoformat()
            }
            
            # Extract songs from this file
            songs = self._extract_songs_from_file(file_data)
            
            return {
                'file_data': file_data,
                'songs': songs,
                'song_count': len(songs)
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return {'file_data': {'file_path': str(file_path.relative_to(self.repo_path))}, 'songs': [], 'song_count': 0}

    def _classify_file_type(self, file_path: str) -> str:
        """Classify the type of file based on its path."""
        if '/raga/' in file_path:
            return 'raga'
        elif '/artist/' in file_path:
            return 'artist'
        elif '/composer/' in file_path:
            return 'composer'
        elif '/type/' in file_path:
            return 'type'
        elif 'raga.json' in file_path:
            return 'raga_index'
        elif 'artist.json' in file_path:
            return 'artist_index'
        elif 'composer.json' in file_path:
            return 'composer_index'
        else:
            return 'other'

    def _parse_views(self, views_str: str) -> int:
        """Parse view count string to integer."""
        if not views_str:
            return 0
        
        views_str = str(views_str).replace(',', '').replace('K', '000').replace('M', '000000')
        try:
            return int(views_str)
        except:
            return 0

    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string to minutes."""
        if not duration_str or duration_str == '0:00':
            return 0.0
        
        try:
            # Handle formats like "7:58", "1:02:19", "59:44"
            parts = duration_str.split(':')
            if len(parts) == 2:  # MM:SS
                minutes = int(parts[0]) + int(parts[1]) / 60.0
            elif len(parts) == 3:  # HH:MM:SS
                hours = int(parts[0])
                minutes = int(parts[1]) + int(parts[2]) / 60.0
                minutes += hours * 60
            else:
                return 0.0
            return minutes
        except:
            return 0.0

    def _build_youtube_url(self, video_id: str) -> str:
        """Build YouTube URL from video ID."""
        if not video_id:
            return ''
        
        # Handle video IDs with additional parameters
        clean_id = video_id.split('&')[0]
        return f"https://www.youtube.com/watch?v={clean_id}"

    def analyze_repository(self):
        """Analyze the entire repository to extract YouTube links and song data."""
        logger.info("🎥 YOUTUBE SONG ANALYSIS - RAMANARUNACHALAM REPOSITORY")
        logger.info("=" * 60)
        logger.info(f"🧵 Max workers: {self.max_workers}")
        
        # Find all JSON files
        json_files = list(self.repo_path.rglob('*.json'))
        logger.info(f"📊 Found {len(json_files)} JSON files to analyze")
        
        start_time = time.time()
        all_songs = []
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self._process_json_file, file_path): file_path 
                            for file_path in json_files}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    all_songs.extend(result['songs'])
                    
                except Exception as exc:
                    logger.error(f"❌ {file_path} generated an exception: {exc}")
                
                # Progress reporting
                if (i + 1) % 100 == 0 or (i + 1) == len(json_files):
                    progress = ((i + 1) / len(json_files)) * 100
                    logger.info(f"📊 Progress: {progress:.1f}% ({i+1}/{len(json_files)}) - Songs found: {len(all_songs)}")
        
        end_time = time.time()
        self.stats['processing_time'] = end_time - start_time
        
        # Process all extracted songs
        self._process_extracted_songs(all_songs)
        
        # Generate comprehensive analysis
        self._generate_analysis_report()
        
        logger.info("\n🎉 YOUTUBE SONG ANALYSIS COMPLETED!")
        logger.info(f"⏱️ Processing time: {self.stats['processing_time']:.1f} seconds")
        logger.info(f"🎵 Total songs found: {self.stats['total_songs_found']}")
        logger.info(f"🎥 YouTube videos found: {self.stats['youtube_videos_found']}")
        logger.info(f"🎼 Unique ragas: {len(self.stats['unique_ragas'])}")
        logger.info(f"🎭 Unique artists: {len(self.stats['unique_artists'])}")
        logger.info(f"🎼 Unique composers: {len(self.stats['unique_composers'])}")
        logger.info(f"👀 Total views: {self.stats['total_views']:,}")
        logger.info(f"⏱️ Total duration: {self.stats['total_duration_minutes']:.1f} minutes")
        
        return self.stats

    def _process_extracted_songs(self, all_songs: List[Dict[str, Any]]):
        """Process all extracted songs and build comprehensive database."""
        logger.info(f"🔄 Processing {len(all_songs)} extracted songs...")
        
        for song in all_songs:
            song_id = song.get('song_id', '')
            youtube_id = song.get('youtube_id', '')
            raga_id = song.get('raga_id', '')
            artist_id = song.get('artist_id', '')
            composer_id = song.get('composer_id', '')
            
            if not song_id or not youtube_id:
                continue
            
            # Parse views and duration
            views = self._parse_views(song.get('views', ''))
            duration_minutes = self._parse_duration(song.get('duration', ''))
            
            # Build YouTube URL
            youtube_url = self._build_youtube_url(youtube_id)
            
            # Store song data
            song_data = {
                'song_id': song_id,
                'youtube_id': youtube_id,
                'youtube_url': youtube_url,
                'raga_id': raga_id,
                'artist_id': artist_id,
                'composer_id': composer_id,
                'duration': song.get('duration', ''),
                'duration_minutes': duration_minutes,
                'views': views,
                'language_type': song.get('language_type', ''),
                'type_id': song.get('type_id', ''),
                'file_path': song.get('file_path', ''),
                'file_type': song.get('file_type', '')
            }
            
            self.songs[song_id] = song_data
            self.youtube_videos[youtube_id] = song_data
            
            # Build relationships
            if raga_id:
                self.raga_songs[raga_id].append(song_id)
                self.stats['unique_ragas'].add(raga_id)
            
            if artist_id:
                self.artist_songs[artist_id].append(song_id)
                self.stats['unique_artists'].add(artist_id)
            
            if composer_id:
                self.composer_songs[composer_id].append(song_id)
                self.stats['unique_composers'].add(composer_id)
            
            # Update statistics
            self.stats['total_songs_found'] += 1
            self.stats['youtube_videos_found'] += 1
            self.stats['total_views'] += views
            self.stats['total_duration_minutes'] += duration_minutes

    def _generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        logger.info("📋 Generating YouTube song analysis report...")
        
        # Create analysis summary
        analysis_summary = {
            "timestamp": datetime.now().isoformat(),
            "processing_stats": {
                'total_songs_found': self.stats['total_songs_found'],
                'youtube_videos_found': self.stats['youtube_videos_found'],
                'unique_ragas_count': len(self.stats['unique_ragas']),
                'unique_artists_count': len(self.stats['unique_artists']),
                'unique_composers_count': len(self.stats['unique_composers']),
                'total_views': self.stats['total_views'],
                'total_duration_minutes': self.stats['total_duration_minutes'],
                'processing_time': self.stats['processing_time']
            },
            "summary": {
                "total_songs": self.stats['total_songs_found'],
                "youtube_videos": self.stats['youtube_videos_found'],
                "unique_ragas": len(self.stats['unique_ragas']),
                "unique_artists": len(self.stats['unique_artists']),
                "unique_composers": len(self.stats['unique_composers']),
                "total_views": self.stats['total_views'],
                "total_duration_hours": self.stats['total_duration_minutes'] / 60,
                "average_views_per_song": self.stats['total_views'] / max(self.stats['total_songs_found'], 1),
                "average_duration_minutes": self.stats['total_duration_minutes'] / max(self.stats['total_songs_found'], 1)
            },
            "top_ragas_by_songs": self._get_top_ragas_by_songs(),
            "top_artists_by_songs": self._get_top_artists_by_songs(),
            "top_composers_by_songs": self._get_top_composers_by_songs(),
            "most_viewed_songs": self._get_most_viewed_songs(),
            "longest_songs": self._get_longest_songs()
        }
        
        # Save comprehensive analysis
        analysis_file = self.output_path / "youtube_song_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
        
        # Save individual datasets
        self._save_songs_dataset()
        self._save_youtube_videos_dataset()
        self._save_relationships_dataset()
        
        # Generate CSV exports
        self._export_to_csv()
        
        logger.info(f"📋 YouTube song analysis saved: {analysis_file}")

    def _get_top_ragas_by_songs(self, top_n: int = 20) -> List[Dict]:
        """Get top ragas by number of songs."""
        raga_counts = [(raga_id, len(songs)) for raga_id, songs in self.raga_songs.items()]
        raga_counts.sort(key=lambda x: x[1], reverse=True)
        return [{"raga_id": raga_id, "song_count": count} for raga_id, count in raga_counts[:top_n]]

    def _get_top_artists_by_songs(self, top_n: int = 20) -> List[Dict]:
        """Get top artists by number of songs."""
        artist_counts = [(artist_id, len(songs)) for artist_id, songs in self.artist_songs.items()]
        artist_counts.sort(key=lambda x: x[1], reverse=True)
        return [{"artist_id": artist_id, "song_count": count} for artist_id, count in artist_counts[:top_n]]

    def _get_top_composers_by_songs(self, top_n: int = 20) -> List[Dict]:
        """Get top composers by number of songs."""
        composer_counts = [(composer_id, len(songs)) for composer_id, songs in self.composer_songs.items()]
        composer_counts.sort(key=lambda x: x[1], reverse=True)
        return [{"composer_id": composer_id, "song_count": count} for composer_id, count in composer_counts[:top_n]]

    def _get_most_viewed_songs(self, top_n: int = 20) -> List[Dict]:
        """Get most viewed songs."""
        songs_with_views = [(song_id, song_data['views']) for song_id, song_data in self.songs.items() if song_data['views'] > 0]
        songs_with_views.sort(key=lambda x: x[1], reverse=True)
        return [{"song_id": song_id, "views": views, "youtube_url": self.songs[song_id]['youtube_url']} 
                for song_id, views in songs_with_views[:top_n]]

    def _get_longest_songs(self, top_n: int = 20) -> List[Dict]:
        """Get longest songs by duration."""
        songs_with_duration = [(song_id, song_data['duration_minutes']) for song_id, song_data in self.songs.items() if song_data['duration_minutes'] > 0]
        songs_with_duration.sort(key=lambda x: x[1], reverse=True)
        return [{"song_id": song_id, "duration_minutes": duration, "youtube_url": self.songs[song_id]['youtube_url']} 
                for song_id, duration in songs_with_duration[:top_n]]

    def _save_songs_dataset(self):
        """Save songs dataset to JSON file."""
        songs_file = self.output_path / "songs_dataset.json"
        with open(songs_file, 'w', encoding='utf-8') as f:
            json.dump(self.songs, f, indent=2, ensure_ascii=False)
        logger.info(f"🎵 Songs dataset saved: {songs_file}")

    def _save_youtube_videos_dataset(self):
        """Save YouTube videos dataset to JSON file."""
        videos_file = self.output_path / "youtube_videos_dataset.json"
        with open(videos_file, 'w', encoding='utf-8') as f:
            json.dump(self.youtube_videos, f, indent=2, ensure_ascii=False)
        logger.info(f"🎥 YouTube videos dataset saved: {videos_file}")

    def _save_relationships_dataset(self):
        """Save relationships dataset to JSON file."""
        relationships = {
            "raga_songs": dict(self.raga_songs),
            "artist_songs": dict(self.artist_songs),
            "composer_songs": dict(self.composer_songs)
        }
        relationships_file = self.output_path / "relationships_dataset.json"
        with open(relationships_file, 'w', encoding='utf-8') as f:
            json.dump(relationships, f, indent=2, ensure_ascii=False)
        logger.info(f"🔗 Relationships dataset saved: {relationships_file}")

    def _export_to_csv(self):
        """Export datasets to CSV format."""
        logger.info("📊 Exporting datasets to CSV format...")
        
        # Export songs to CSV
        if self.songs:
            songs_df = pd.DataFrame([
                {
                    'song_id': song_data['song_id'],
                    'youtube_id': song_data['youtube_id'],
                    'youtube_url': song_data['youtube_url'],
                    'raga_id': song_data['raga_id'],
                    'artist_id': song_data['artist_id'],
                    'composer_id': song_data['composer_id'],
                    'duration': song_data['duration'],
                    'duration_minutes': song_data['duration_minutes'],
                    'views': song_data['views'],
                    'language_type': song_data['language_type'],
                    'type_id': song_data['type_id']
                }
                for song_data in self.songs.values()
            ])
            songs_df.to_csv(self.output_path / "songs_dataset.csv", index=False)
        
        logger.info("📊 CSV exports completed")

def main():
    """Main execution function."""
    repo_path = Path("downloads") / "Ramanarunachalam_Music_Repository"
    
    if not repo_path.exists():
        logger.error(f"❌ Repository not found: {repo_path}")
        return
    
    analyzer = YouTubeSongAnalyzer(repo_path, max_workers=8)
    stats = analyzer.analyze_repository()
    
    logger.info("\n🎯 YOUTUBE SONG ANALYSIS SUMMARY:")
    logger.info(f"   🎵 Total songs: {stats['total_songs_found']:,}")
    logger.info(f"   🎥 YouTube videos: {stats['youtube_videos_found']:,}")
    logger.info(f"   🎼 Unique ragas: {len(stats['unique_ragas']):,}")
    logger.info(f"   🎭 Unique artists: {len(stats['unique_artists']):,}")
    logger.info(f"   🎼 Unique composers: {len(stats['unique_composers']):,}")
    logger.info(f"   👀 Total views: {stats['total_views']:,}")
    logger.info(f"   ⏱️ Total duration: {stats['total_duration_minutes']/60:.1f} hours")
    logger.info(f"   ⏱️ Processing time: {stats['processing_time']:.1f} seconds")

if __name__ == "__main__":
    main()
