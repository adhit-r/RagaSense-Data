#!/usr/bin/env python3
"""
Resumable Zenodo Downloader for RagaSense-Data
Supports resuming interrupted downloads using HTTP Range requests
"""

import requests
import json
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resumable_zenodo_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResumableZenodoDownloader:
    """
    Resumable Zenodo Downloader using HTTP Range requests
    """
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://zenodo.org/api"
        self.downloads_path = Path("downloads")
        self.downloads_path.mkdir(exist_ok=True)
        
        # Set up headers with access token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        logger.info("🔑 Resumable Zenodo Downloader initialized with access token")
    
    def get_record_info(self, record_id: str) -> Optional[Dict]:
        """Get record information from Zenodo API"""
        try:
            url = f"{self.base_url}/records/{record_id}"
            logger.info(f"📡 Fetching record info for ID: {record_id}")
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✅ Record found: {data.get('metadata', {}).get('title', 'Unknown')}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to fetch record {record_id}: {e}")
            return None
    
    def list_record_files(self, record_id: str) -> List[Dict]:
        """List all files in a Zenodo record"""
        try:
            # Use the correct endpoint for files
            url = f"{self.base_url}/records/{record_id}"
            logger.info(f"📁 Fetching record with files for: {record_id}")
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            record_data = response.json()
            files = record_data.get('files', [])
            logger.info(f"📊 Found {len(files)} files in record")
            
            # Log file information
            for file_info in files:
                if isinstance(file_info, dict):
                    filename = file_info.get('key', 'Unknown')
                    size = file_info.get('size', 0)
                    logger.info(f"   📄 {filename} ({size} bytes)")
                else:
                    logger.info(f"   📄 File: {file_info}")
            
            return files
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to list files for record {record_id}: {e}")
            return []
    
    def get_file_size(self, record_id: str, file_key: str) -> int:
        """Get the total size of a file without downloading it"""
        try:
            url = f"https://zenodo.org/api/records/{record_id}/files/{file_key}/content"
            response = requests.head(url)
            response.raise_for_status()
            return int(response.headers.get('content-length', 0))
        except:
            return 0
    
    def download_file_resumable(self, record_id: str, file_key: str, output_path: Path) -> bool:
        """Download a file with resume capability"""
        try:
            url = f"https://zenodo.org/api/records/{record_id}/files/{file_key}/content"
            logger.info(f"⬇️ Downloading: {file_key}")
            logger.info(f"   URL: {url}")
            
            # Check if file already exists and get its size
            resume_pos = 0
            if output_path.exists():
                resume_pos = output_path.stat().st_size
                logger.info(f"🔄 Resuming download from {resume_pos} bytes")
            
            # Get total file size
            total_size = self.get_file_size(record_id, file_key)
            if total_size == 0:
                # Fallback: try to get size from HEAD request
                response = requests.head(url)
                total_size = int(response.headers.get('content-length', 0))
            
            if total_size > 0:
                logger.info(f"   Size: {total_size} bytes ({total_size / (1024*1024):.1f} MB)")
                
                # If we already have the complete file, skip download
                if resume_pos >= total_size:
                    logger.info(f"✅ File already complete: {output_path}")
                    return True
                
                # If we have partial file, show progress
                if resume_pos > 0:
                    progress = (resume_pos / total_size) * 100
                    logger.info(f"   Resuming at {progress:.1f}% ({resume_pos // (1024*1024)}MB)")
            
            # Set up headers for range request if resuming
            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'
                logger.info(f"   Using Range header: bytes={resume_pos}-")
            
            # Download with resume capability
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Open file in append mode if resuming, write mode if starting fresh
            mode = 'ab' if resume_pos > 0 else 'wb'
            downloaded = resume_pos
            last_log_time = time.time()
            
            with open(output_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 30 seconds or every 10MB
                        current_time = time.time()
                        if (current_time - last_log_time >= 30) or (downloaded % (10 * 1024 * 1024) == 0):
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                speed_mb = (downloaded - resume_pos) / (1024**2) / (current_time - last_log_time + 1)
                                eta_seconds = (total_size - downloaded) / ((downloaded - resume_pos) / (current_time - last_log_time + 1)) if (downloaded - resume_pos) > 0 else 0
                                eta_hours = eta_seconds / 3600
                                
                                logger.info(f"   Progress: {progress:.1f}% ({downloaded // (1024**2)}MB) - Speed: {speed_mb:.1f} MB/s - ETA: {eta_hours:.1f}h")
                            else:
                                logger.info(f"   Downloaded: {downloaded // (1024**2)}MB")
                            last_log_time = current_time
            
            logger.info(f"✅ Downloaded successfully: {output_path}")
            logger.info(f"   Final size: {downloaded} bytes ({downloaded / (1024*1024):.1f} MB)")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download {file_key}: {e}")
            return False
        except KeyboardInterrupt:
            logger.info("⏹️ Download interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error downloading {file_key}: {e}")
            return False
    
    def download_record_resumable(self, record_id: str, output_dir: Optional[Path] = None) -> bool:
        """Download all files from a Zenodo record with resume capability"""
        if output_dir is None:
            output_dir = self.downloads_path / f"zenodo_{record_id}"
        
        output_dir.mkdir(exist_ok=True)
        
        # Get record info
        record_info = self.get_record_info(record_id)
        if not record_info:
            return False
        
        # Get files list
        files = self.list_record_files(record_id)
        if not files:
            logger.warning(f"⚠️ No files found in record {record_id}")
            return False
        
        # Download each file with resume capability
        success_count = 0
        for file_info in files:
            if isinstance(file_info, dict):
                file_key = file_info.get('key', '')
            else:
                # If file_info is a string, use it as the key
                file_key = str(file_info)
            
            if not file_key:
                continue
            
            output_path = output_dir / file_key
            if self.download_file_resumable(record_id, file_key, output_path):
                success_count += 1
        
        logger.info(f"🎉 Download completed: {success_count}/{len(files)} files downloaded")
        return success_count > 0

def main():
    """Main function to test resumable Zenodo downloader"""
    
    # Your access token
    ACCESS_TOKEN = "HRm7CO4Pab11m1mX75zMX12FY3Ga2SP06zzU53FAVMLF0tj5cBsvFzNXDsO8"
    
    # Initialize downloader
    downloader = ResumableZenodoDownloader(ACCESS_TOKEN)
    
    # Test with Saraga dataset (record ID: 4301737)
    saraga_record_id = "4301737"
    
    logger.info("🎵 RESUMABLE DOWNLOAD OF SARAGA DATASET")
    logger.info("=" * 60)
    
    # Get record info first
    record_info = downloader.get_record_info(saraga_record_id)
    if record_info:
        title = record_info.get('metadata', {}).get('title', 'Unknown')
        logger.info(f"📋 Record Title: {title}")
        
        # List files
        files = downloader.list_record_files(saraga_record_id)
        
        # Download the record with resume capability
        success = downloader.download_record_resumable(saraga_record_id)
        
        if success:
            logger.info("🎉 Saraga dataset download completed successfully!")
        else:
            logger.error("❌ Saraga dataset download failed")

if __name__ == "__main__":
    main()
