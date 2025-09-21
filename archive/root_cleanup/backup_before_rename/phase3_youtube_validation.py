#!/usr/bin/env python3
"""
Phase 3: YouTube Link Validation
===============================

This script validates YouTube links in the RagaSense-Data dataset to identify
broken, dead, or invalid links. With 470,544 links to check, this requires
efficient batch processing and rate limiting.

Strategy:
1. Extract all YouTube links from the dataset
2. Validate link format and structure
3. Check link accessibility (with rate limiting)
4. Generate validation report with statistics
"""

import json
import os
import time
import re
import requests
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from urllib.parse import urlparse, parse_qs
import concurrent.futures
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase3_youtube_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YouTubeLinkValidator:
    """
    Validates YouTube links in the RagaSense-Data dataset.
    """
    
    def __init__(self):
        self.data_path = Path("data/processed")
        self.archive_path = Path("archive/data_versions")
        
        # Data containers
        self.all_links = set()
        self.link_sources = defaultdict(list)  # Track where each link comes from
        self.validation_results = {}
        
        # Statistics
        self.stats = {
            "total_links_found": 0,
            "unique_links": 0,
            "valid_format_links": 0,
            "invalid_format_links": 0,
            "accessible_links": 0,
            "broken_links": 0,
            "rate_limited_requests": 0,
            "validation_errors": 0
        }
        
        # Rate limiting
        self.request_lock = Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # YouTube URL patterns
        self.youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'https?://(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'https?://(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})'
        ]
        
        logger.info("🔗 YouTube Link Validator initialized")
    
    def extract_youtube_links(self) -> bool:
        """Extract all YouTube links from the dataset."""
        logger.info("🔍 Extracting YouTube links from dataset...")
        
        # Define search paths
        search_paths = [
            self.data_path,
            self.archive_path / "unified_ragasense_dataset",
            self.archive_path / "comprehensive_unified_dataset"
        ]
        
        # File patterns to search
        file_patterns = [
            "*.json",
            "unified_*_database.json",
            "*_database.json"
        ]
        
        total_files_processed = 0
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for pattern in file_patterns:
                for file_path in search_path.glob(pattern):
                    if file_path.is_file():
                        try:
                            self._extract_links_from_file(file_path)
                            total_files_processed += 1
                            
                            if total_files_processed % 10 == 0:
                                logger.info(f"📂 Processed {total_files_processed} files, found {len(self.all_links)} unique links")
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Error processing {file_path}: {e}")
        
        self.stats["total_links_found"] = len(self.all_links)
        self.stats["unique_links"] = len(self.all_links)
        
        logger.info(f"✅ Extracted {len(self.all_links)} unique YouTube links from {total_files_processed} files")
        return True
    
    def _extract_links_from_file(self, file_path: Path):
        """Extract YouTube links from a single JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Recursively search for YouTube links in the data
            self._search_links_recursive(data, str(file_path))
            
        except Exception as e:
            logger.warning(f"⚠️ Error reading {file_path}: {e}")
    
    def _search_links_recursive(self, data: Any, source: str):
        """Recursively search for YouTube links in data structures."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and self._is_youtube_link(value):
                    self.all_links.add(value)
                    self.link_sources[value].append(source)
                elif isinstance(value, (dict, list)):
                    self._search_links_recursive(value, source)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str) and self._is_youtube_link(item):
                    self.all_links.add(item)
                    self.link_sources[item].append(source)
                elif isinstance(item, (dict, list)):
                    self._search_links_recursive(item, source)
    
    def _is_youtube_link(self, text: str) -> bool:
        """Check if text contains a YouTube link."""
        if not isinstance(text, str):
            return False
        
        # Check for any YouTube URL pattern
        for pattern in self.youtube_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def validate_link_format(self, url: str) -> Dict:
        """Validate YouTube link format and extract video ID."""
        result = {
            "url": url,
            "is_valid_format": False,
            "video_id": None,
            "format_type": None,
            "error": None
        }
        
        try:
            # Check each YouTube pattern
            for i, pattern in enumerate(self.youtube_patterns):
                match = re.search(pattern, url)
                if match:
                    result["is_valid_format"] = True
                    result["video_id"] = match.group(1)
                    result["format_type"] = ["watch", "short", "embed", "v"][i]
                    break
            
            if not result["is_valid_format"]:
                result["error"] = "Invalid YouTube URL format"
                
        except Exception as e:
            result["error"] = f"Format validation error: {e}"
        
        return result
    
    def validate_link_accessibility(self, url: str) -> Dict:
        """Validate if YouTube link is accessible (with rate limiting)."""
        result = {
            "url": url,
            "is_accessible": False,
            "status_code": None,
            "response_time": None,
            "error": None
        }
        
        try:
            # Rate limiting
            with self.request_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last)
                self.last_request_time = time.time()
            
            # Make request with timeout
            start_time = time.time()
            response = requests.head(url, timeout=10, allow_redirects=True)
            response_time = time.time() - start_time
            
            result["status_code"] = response.status_code
            result["response_time"] = response_time
            
            # Consider 200-299 as accessible
            if 200 <= response.status_code < 300:
                result["is_accessible"] = True
            else:
                result["error"] = f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            result["error"] = "Request timeout"
        except requests.exceptions.ConnectionError:
            result["error"] = "Connection error"
        except requests.exceptions.RequestException as e:
            result["error"] = f"Request error: {e}"
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
        
        return result
    
    def validate_links_batch(self, links: List[str], max_workers: int = 5) -> Dict:
        """Validate a batch of links using concurrent processing."""
        logger.info(f"🔍 Validating {len(links)} links with {max_workers} workers...")
        
        validation_results = {}
        
        def validate_single_link(link):
            # Format validation
            format_result = self.validate_link_format(link)
            
            if format_result["is_valid_format"]:
                # Accessibility validation
                access_result = self.validate_link_accessibility(link)
                return {
                    "url": link,
                    "format_validation": format_result,
                    "accessibility_validation": access_result,
                    "overall_valid": format_result["is_valid_format"] and access_result["is_accessible"]
                }
            else:
                return {
                    "url": link,
                    "format_validation": format_result,
                    "accessibility_validation": None,
                    "overall_valid": False
                }
        
        # Use ThreadPoolExecutor for concurrent validation
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_link = {executor.submit(validate_single_link, link): link for link in links}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_link):
                link = future_to_link[future]
                try:
                    result = future.result()
                    validation_results[link] = result
                    completed += 1
                    
                    # Update statistics
                    if result["format_validation"]["is_valid_format"]:
                        self.stats["valid_format_links"] += 1
                    else:
                        self.stats["invalid_format_links"] += 1
                    
                    if result.get("accessibility_validation") and result["accessibility_validation"]["is_accessible"]:
                        self.stats["accessible_links"] += 1
                    elif result.get("accessibility_validation"):
                        self.stats["broken_links"] += 1
                    
                    # Log progress
                    if completed % 100 == 0:
                        logger.info(f"📊 Validated {completed}/{len(links)} links...")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error validating {link}: {e}")
                    self.stats["validation_errors"] += 1
        
        return validation_results
    
    def run_sample_validation(self, sample_size: int = 1000) -> bool:
        """Run validation on a sample of links for testing."""
        logger.info(f"🧪 Running sample validation on {sample_size} links...")
        
        # Extract links first
        if not self.extract_youtube_links():
            return False
        
        # Take a sample
        sample_links = list(self.all_links)[:sample_size]
        logger.info(f"📊 Sample size: {len(sample_links)} links")
        
        # Validate sample
        sample_results = self.validate_links_batch(sample_links, max_workers=3)
        
        # Save sample results
        self._save_validation_results(sample_results, "sample")
        
        # Print summary
        self._print_validation_summary(sample_results)
        
        return True
    
    def _save_validation_results(self, results: Dict, suffix: str = ""):
        """Save validation results to file."""
        try:
            output_dir = self.data_path / "youtube_validation"
            output_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            results_file = output_dir / f"youtube_validation_results_{suffix}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save summary statistics
            summary = {
                "validation_timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "summary": {
                    "total_links_validated": len(results),
                    "valid_format_links": self.stats["valid_format_links"],
                    "invalid_format_links": self.stats["invalid_format_links"],
                    "accessible_links": self.stats["accessible_links"],
                    "broken_links": self.stats["broken_links"],
                    "validation_errors": self.stats["validation_errors"]
                }
            }
            
            summary_file = output_dir / f"youtube_validation_summary_{suffix}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved validation results to {results_file}")
            logger.info(f"✅ Saved validation summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving validation results: {e}")
    
    def _print_validation_summary(self, results: Dict):
        """Print validation summary."""
        logger.info("📊 YOUTUBE LINK VALIDATION SUMMARY:")
        logger.info("=" * 50)
        logger.info(f"  • Total links validated: {len(results)}")
        logger.info(f"  • Valid format links: {self.stats['valid_format_links']}")
        logger.info(f"  • Invalid format links: {self.stats['invalid_format_links']}")
        logger.info(f"  • Accessible links: {self.stats['accessible_links']}")
        logger.info(f"  • Broken links: {self.stats['broken_links']}")
        logger.info(f"  • Validation errors: {self.stats['validation_errors']}")
        
        if len(results) > 0:
            success_rate = (self.stats['accessible_links'] / len(results)) * 100
            logger.info(f"  • Success rate: {success_rate:.1f}%")

def main():
    """Main function to run YouTube link validation."""
    print("🔗 Phase 3: YouTube Link Validation")
    print("=" * 40)
    
    # Initialize validator
    validator = YouTubeLinkValidator()
    
    # Run sample validation first
    print("🧪 Running sample validation (1000 links)...")
    success = validator.run_sample_validation(sample_size=1000)
    
    if success:
        print("\n✅ Sample validation completed successfully!")
        print("📊 Check the results in data/processed/youtube_validation/")
        print("\n💡 For full validation of all 470,544 links, run with larger sample size")
    else:
        print("\n❌ Sample validation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
