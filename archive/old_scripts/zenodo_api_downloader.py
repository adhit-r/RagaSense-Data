#!/usr/bin/env python3
"""
Zenodo API Downloader for RagaSense-Data
Uses Zenodo REST API with access token for reliable downloads
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
        logging.FileHandler('zenodo_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ZenodoAPIDownloader:
    """
    Zenodo API Downloader using REST API with access token
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
        
        logger.info("🔑 Zenodo API Downloader initialized with access token")
    
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
    
    def download_file(self, record_id: str, file_key: str, output_path: Path) -> bool:
        """Download a specific file from Zenodo record"""
        try:
            # Use the correct Zenodo content URL format
            url = f"https://zenodo.org/api/records/{record_id}/files/{file_key}/content"
            logger.info(f"⬇️ Downloading: {file_key}")
            logger.info(f"   URL: {url}")
            
            # Don't use headers for download (Zenodo doesn't require auth for public downloads)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            logger.info(f"   Size: {total_size} bytes ({total_size / (1024*1024):.1f} MB)")
            
            # Download with progress tracking
            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"   Progress: {progress:.1f}% ({downloaded // (1024*1024)}MB)")
            
            logger.info(f"✅ Downloaded successfully: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download {file_key}: {e}")
            return False
    
    def download_record(self, record_id: str, output_dir: Optional[Path] = None) -> bool:
        """Download all files from a Zenodo record"""
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
        
        # Download each file
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
            if self.download_file(record_id, file_key, output_path):
                success_count += 1
        
        logger.info(f"🎉 Download completed: {success_count}/{len(files)} files downloaded")
        return success_count > 0
    
    def search_records(self, query: str, size: int = 10) -> List[Dict]:
        """Search for records using Zenodo API"""
        try:
            url = f"{self.base_url}/records"
            params = {
                "q": query,
                "size": size,
                "type": "dataset"
            }
            
            logger.info(f"🔍 Searching for: {query}")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            hits = data.get('hits', {}).get('hits', [])
            
            logger.info(f"📊 Found {len(hits)} records")
            for hit in hits:
                title = hit.get('metadata', {}).get('title', 'Unknown')
                record_id = hit.get('id', 'Unknown')
                logger.info(f"   📄 {title} (ID: {record_id})")
            
            return hits
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Search failed: {e}")
            return []

def main():
    """Main function to test Zenodo API downloader"""
    
    # Your access token
    ACCESS_TOKEN = "HRm7CO4Pab11m1mX75zMX12FY3Ga2SP06zzU53FAVMLF0tj5cBsvFzNXDsO8"
    
    # Initialize downloader
    downloader = ZenodoAPIDownloader(ACCESS_TOKEN)
    
    # Test with Saraga dataset (record ID: 4301737)
    saraga_record_id = "4301737"
    
    logger.info("🎵 DOWNLOADING SARAGA DATASET VIA ZENODO API")
    logger.info("=" * 60)
    
    # Get record info first
    record_info = downloader.get_record_info(saraga_record_id)
    if record_info:
        title = record_info.get('metadata', {}).get('title', 'Unknown')
        logger.info(f"📋 Record Title: {title}")
        
        # List files
        files = downloader.list_record_files(saraga_record_id)
        
        # Download the record
        success = downloader.download_record(saraga_record_id)
        
        if success:
            logger.info("🎉 Saraga dataset download completed successfully!")
        else:
            logger.error("❌ Saraga dataset download failed")
    
    # Test search functionality
    logger.info("\n🔍 TESTING SEARCH FUNCTIONALITY")
    logger.info("=" * 40)
    
    search_results = downloader.search_records("carnatic music", size=5)
    if search_results:
        logger.info(f"✅ Search successful - found {len(search_results)} records")
    else:
        logger.info("⚠️ No search results found")

if __name__ == "__main__":
    main()
