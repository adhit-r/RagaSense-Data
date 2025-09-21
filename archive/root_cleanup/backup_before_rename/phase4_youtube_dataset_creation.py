#!/usr/bin/env python3
"""
Phase 4: YouTube Dataset Creation for ML Training
================================================

This script creates a comprehensive audio dataset by downloading YouTube videos
from the 470,557 validated links in the RagaSense-Data project.

Strategy:
1. Extract and organize YouTube links by raga/tradition
2. Download audio from videos (not video files to save space)
3. Process audio for ML training (segmentation, feature extraction)
4. Create organized dataset structure for training
5. Generate metadata and quality reports

Estimated Dataset Size: ~50-100 TB (based on 470K videos)
Processing Time: 2-4 weeks with parallel processing
"""

import json
import os
import time
import re
import subprocess
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import concurrent.futures
from threading import Lock
import hashlib
import sqlite3
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase4_youtube_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoInfo:
    """Information about a YouTube video."""
    url: str
    video_id: str
    title: str
    duration: int
    raga: str
    tradition: str
    artist: str
    composer: str
    quality: str
    file_size: int
    download_path: str
    status: str
    error: Optional[str] = None

class YouTubeDatasetCreator:
    """
    Creates a comprehensive audio dataset from YouTube videos.
    """
    
    def __init__(self, output_dir: str = "data/youtube_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.audio_dir = self.output_dir / "audio"
        self.metadata_dir = self.output_dir / "metadata"
        self.logs_dir = self.output_dir / "logs"
        self.temp_dir = self.output_dir / "temp"
        
        for dir_path in [self.audio_dir, self.metadata_dir, self.logs_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Data containers
        self.youtube_links = set()
        self.video_info = {}  # video_id -> VideoInfo
        self.download_queue = []
        self.completed_downloads = set()
        self.failed_downloads = set()
        
        # Statistics
        self.stats = {
            "total_links": 0,
            "unique_videos": 0,
            "videos_downloaded": 0,
            "videos_failed": 0,
            "total_duration_hours": 0,
            "total_size_gb": 0,
            "processing_time": 0,
            "rate_limited_requests": 0
        }
        
        # Rate limiting
        self.request_lock = Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        
        # Database for tracking
        self.db_path = self.metadata_dir / "youtube_dataset.db"
        self._init_database()
        
        logger.info("🎥 YouTube Dataset Creator initialized")
    
    def _init_database(self):
        """Initialize SQLite database for tracking downloads."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT,
                duration INTEGER,
                raga TEXT,
                tradition TEXT,
                artist TEXT,
                composer TEXT,
                quality TEXT,
                file_size INTEGER,
                download_path TEXT,
                status TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS download_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                action TEXT,
                status TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_youtube_links(self) -> bool:
        """Load YouTube links from the validated dataset."""
        logger.info("🔍 Loading YouTube links from validated dataset...")
        
        # Load from validation results
        validation_file = Path("data/processed/youtube_validation/youtube_validation_results_sample.json")
        if validation_file.exists():
            with open(validation_file, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
            
            for url, result in validation_data.items():
                if result.get("overall_valid", False):
                    self.youtube_links.add(url)
        
        # Also load from the full dataset (if available)
        # This would be the full 470,557 links
        logger.info(f"✅ Loaded {len(self.youtube_links)} validated YouTube links")
        self.stats["total_links"] = len(self.youtube_links)
        
        return True
    
    def extract_video_info(self, url: str) -> Optional[VideoInfo]:
        """Extract video information using yt-dlp."""
        try:
            # Rate limiting
            with self.request_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last)
                self.last_request_time = time.time()
            
            # Extract video ID
            video_id = self._extract_video_id(url)
            if not video_id:
                return None
            
            # Get video info using yt-dlp
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Failed to get info for {url}: {result.stderr}")
                return None
            
            # Parse JSON output
            video_data = json.loads(result.stdout)
            
            # Extract information
            title = video_data.get('title', 'Unknown')
            duration = video_data.get('duration', 0)
            
            # Determine quality (prefer audio-only)
            formats = video_data.get('formats', [])
            audio_format = None
            for fmt in formats:
                if fmt.get('acodec') != 'none' and fmt.get('vcodec') == 'none':
                    audio_format = fmt
                    break
            
            if not audio_format:
                # Fallback to any format with audio
                for fmt in formats:
                    if fmt.get('acodec') != 'none':
                        audio_format = fmt
                        break
            
            if not audio_format:
                logger.warning(f"⚠️ No audio format found for {url}")
                return None
            
            # Create VideoInfo object
            video_info = VideoInfo(
                url=url,
                video_id=video_id,
                title=title,
                duration=duration,
                raga="Unknown",  # Will be filled from metadata
                tradition="Unknown",  # Will be filled from metadata
                artist="Unknown",  # Will be filled from metadata
                composer="Unknown",  # Will be filled from metadata
                quality=audio_format.get('format_id', 'unknown'),
                file_size=audio_format.get('filesize', 0),
                download_path="",
                status="pending"
            )
            
            return video_info
            
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Timeout getting info for {url}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error getting info for {url}: {e}")
            return None
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'https?://(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'https?://(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def download_audio(self, video_info: VideoInfo) -> bool:
        """Download audio from YouTube video."""
        try:
            # Create output path
            raga_dir = self.audio_dir / video_info.tradition / video_info.raga
            raga_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            safe_title = re.sub(r'[^\w\s-]', '', video_info.title)
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            filename = f"{video_info.video_id}_{safe_title}.%(ext)s"
            output_path = raga_dir / filename
            
            # Download command
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'wav',
                '--audio-quality', '0',  # Best quality
                '--output', str(output_path),
                '--no-playlist',
                video_info.url
            ]
            
            # Execute download
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Find the actual downloaded file
                for file_path in raga_dir.glob(f"{video_info.video_id}_*"):
                    if file_path.is_file():
                        video_info.download_path = str(file_path)
                        video_info.file_size = file_path.stat().st_size
                        video_info.status = "completed"
                        
                        # Update database
                        self._update_video_info(video_info)
                        
                        logger.info(f"✅ Downloaded: {video_info.title}")
                        return True
            
            # Download failed
            video_info.status = "failed"
            video_info.error = result.stderr
            self._update_video_info(video_info)
            
            logger.warning(f"❌ Failed to download: {video_info.title}")
            return False
            
        except subprocess.TimeoutExpired:
            video_info.status = "failed"
            video_info.error = "Download timeout"
            self._update_video_info(video_info)
            logger.warning(f"⏰ Download timeout: {video_info.title}")
            return False
        except Exception as e:
            video_info.status = "failed"
            video_info.error = str(e)
            self._update_video_info(video_info)
            logger.warning(f"❌ Download error: {video_info.title}: {e}")
            return False
    
    def _update_video_info(self, video_info: VideoInfo):
        """Update video information in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO videos 
            (video_id, url, title, duration, raga, tradition, artist, composer, 
             quality, file_size, download_path, status, error, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            video_info.video_id, video_info.url, video_info.title, video_info.duration,
            video_info.raga, video_info.tradition, video_info.artist, video_info.composer,
            video_info.quality, video_info.file_size, video_info.download_path,
            video_info.status, video_info.error
        ))
        
        conn.commit()
        conn.close()
    
    def process_sample_dataset(self, sample_size: int = 100) -> bool:
        """Process a sample of videos for testing."""
        logger.info(f"🧪 Processing sample dataset ({sample_size} videos)...")
        
        # Load links
        if not self.load_youtube_links():
            return False
        
        # Take sample
        sample_links = list(self.youtube_links)[:sample_size]
        logger.info(f"📊 Processing {len(sample_links)} sample videos")
        
        # Process videos
        start_time = time.time()
        successful_downloads = 0
        
        for i, url in enumerate(sample_links):
            logger.info(f"📥 Processing {i+1}/{len(sample_links)}: {url}")
            
            # Extract video info
            video_info = self.extract_video_info(url)
            if not video_info:
                continue
            
            # Download audio
            if self.download_audio(video_info):
                successful_downloads += 1
                self.stats["videos_downloaded"] += 1
                self.stats["total_duration_hours"] += video_info.duration / 3600
                self.stats["total_size_gb"] += video_info.file_size / (1024**3)
            else:
                self.stats["videos_failed"] += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"📊 Progress: {i+1}/{len(sample_links)} processed, {successful_downloads} successful")
        
        # Final statistics
        self.stats["processing_time"] = time.time() - start_time
        self.stats["unique_videos"] = len(sample_links)
        
        # Generate report
        self._generate_sample_report()
        
        logger.info("🎉 Sample dataset processing complete!")
        return True
    
    def _generate_sample_report(self):
        """Generate a report for the sample dataset."""
        report = {
            "sample_processing_timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "summary": {
                "total_videos_processed": self.stats["unique_videos"],
                "successful_downloads": self.stats["videos_downloaded"],
                "failed_downloads": self.stats["videos_failed"],
                "success_rate": (self.stats["videos_downloaded"] / self.stats["unique_videos"]) * 100 if self.stats["unique_videos"] > 0 else 0,
                "total_duration_hours": self.stats["total_duration_hours"],
                "total_size_gb": self.stats["total_size_gb"],
                "processing_time_minutes": self.stats["processing_time"] / 60
            }
        }
        
        report_file = self.metadata_dir / "sample_dataset_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Sample report saved to {report_file}")
    
    def estimate_full_dataset(self) -> Dict:
        """Estimate the size and requirements for the full dataset."""
        logger.info("📊 Estimating full dataset requirements...")
        
        # Load sample statistics
        sample_report_file = self.metadata_dir / "sample_dataset_report.json"
        if not sample_report_file.exists():
            logger.warning("⚠️ No sample report found. Run sample processing first.")
            return {}
        
        with open(sample_report_file, 'r', encoding='utf-8') as f:
            sample_report = json.load(f)
        
        sample_stats = sample_report["statistics"]
        total_links = 470557  # From validation results
        
        # Calculate estimates
        success_rate = sample_stats["videos_downloaded"] / sample_stats["unique_videos"] if sample_stats["unique_videos"] > 0 else 0
        avg_duration_hours = sample_stats["total_duration_hours"] / sample_stats["videos_downloaded"] if sample_stats["videos_downloaded"] > 0 else 0
        avg_size_gb = sample_stats["total_size_gb"] / sample_stats["videos_downloaded"] if sample_stats["videos_downloaded"] > 0 else 0
        avg_processing_time = sample_stats["processing_time"] / sample_stats["unique_videos"] if sample_stats["unique_videos"] > 0 else 0
        
        estimates = {
            "total_youtube_links": total_links,
            "estimated_successful_downloads": int(total_links * success_rate),
            "estimated_total_duration_hours": total_links * avg_duration_hours,
            "estimated_total_size_gb": total_links * avg_size_gb,
            "estimated_total_size_tb": (total_links * avg_size_gb) / 1024,
            "estimated_processing_time_hours": (total_links * avg_processing_time) / 3600,
            "estimated_processing_time_days": (total_links * avg_processing_time) / (3600 * 24),
            "storage_requirements": {
                "minimum_gb": (total_links * avg_size_gb) * 0.8,  # 80% success rate
                "recommended_gb": (total_links * avg_size_gb) * 1.2,  # 20% buffer
                "minimum_tb": ((total_links * avg_size_gb) * 0.8) / 1024,
                "recommended_tb": ((total_links * avg_size_gb) * 1.2) / 1024
            },
            "processing_requirements": {
                "parallel_workers": 10,
                "estimated_days_with_10_workers": ((total_links * avg_processing_time) / (3600 * 24)) / 10,
                "estimated_days_with_20_workers": ((total_links * avg_processing_time) / (3600 * 24)) / 20
            }
        }
        
        return estimates

def main():
    """Main function to run YouTube dataset creation."""
    print("🎥 Phase 4: YouTube Dataset Creation for ML Training")
    print("=" * 60)
    
    # Initialize creator
    creator = YouTubeDatasetCreator()
    
    # Process sample dataset
    print("🧪 Processing sample dataset (100 videos)...")
    success = creator.process_sample_dataset(sample_size=100)
    
    if success:
        print("\n✅ Sample dataset processing completed!")
        
        # Generate estimates
        estimates = creator.estimate_full_dataset()
        if estimates:
            print("\n📊 FULL DATASET ESTIMATES:")
            print("=" * 40)
            print(f"Total YouTube links: {estimates['total_youtube_links']:,}")
            print(f"Estimated successful downloads: {estimates['estimated_successful_downloads']:,}")
            print(f"Estimated total duration: {estimates['estimated_total_duration_hours']:.1f} hours")
            print(f"Estimated total size: {estimates['estimated_total_size_tb']:.1f} TB")
            print(f"Estimated processing time: {estimates['estimated_processing_time_days']:.1f} days")
            print(f"Storage requirement: {estimates['storage_requirements']['recommended_tb']:.1f} TB")
            print(f"Processing with 20 workers: {estimates['processing_requirements']['estimated_days_with_20_workers']:.1f} days")
    else:
        print("\n❌ Sample dataset processing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
