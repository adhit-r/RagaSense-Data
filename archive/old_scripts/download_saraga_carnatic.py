#!/usr/bin/env python3
"""
Download Saraga Carnatic Dataset (14 GB) using Zenodo API
"""

import requests
import os
import sys
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('saraga_carnatic_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_saraga_carnatic():
    """Download the Saraga Carnatic dataset (14 GB)"""
    
    # Saraga record ID and file info
    record_id = "4301737"
    filename = "saraga1.5_carnatic.zip"
    file_size_gb = 14.4  # Approximate size
    
    # Download URL
    download_url = f"https://zenodo.org/records/{record_id}/files/{filename}/download"
    
    # Output path
    output_path = Path("downloads") / filename
    output_path.parent.mkdir(exist_ok=True)
    
    logger.info("🎵 DOWNLOADING SARAGA CARNATIC DATASET")
    logger.info("=" * 50)
    logger.info(f"📁 File: {filename}")
    logger.info(f"📊 Size: ~{file_size_gb} GB")
    logger.info(f"🔗 URL: {download_url}")
    logger.info(f"💾 Output: {output_path}")
    logger.info("")
    
    try:
        logger.info("⬇️ Starting download...")
        start_time = time.time()
        
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0:
            logger.info(f"📊 Actual file size: {total_size / (1024**3):.2f} GB")
        
        downloaded = 0
        last_log_time = time.time()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 30 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= 30:
                        progress = (downloaded / total_size * 100) if total_size > 0 else 0
                        speed_mb = (downloaded / (1024**2)) / (current_time - start_time)
                        eta_seconds = (total_size - downloaded) / (downloaded / (current_time - start_time)) if downloaded > 0 else 0
                        eta_hours = eta_seconds / 3600
                        
                        logger.info(f"📊 Progress: {progress:.1f}% ({downloaded // (1024**2)}MB) - Speed: {speed_mb:.1f} MB/s - ETA: {eta_hours:.1f}h")
                        last_log_time = current_time
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_speed = (downloaded / (1024**2)) / total_time
        
        logger.info("🎉 DOWNLOAD COMPLETED!")
        logger.info(f"⏱️ Total time: {total_time/3600:.2f} hours")
        logger.info(f"📊 Average speed: {avg_speed:.1f} MB/s")
        logger.info(f"💾 File saved: {output_path}")
        logger.info(f"📏 File size: {downloaded / (1024**3):.2f} GB")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Download failed: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("⏹️ Download interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = download_saraga_carnatic()
    if success:
        logger.info("✅ Saraga Carnatic dataset download completed successfully!")
    else:
        logger.error("❌ Saraga Carnatic dataset download failed!")
        sys.exit(1)
