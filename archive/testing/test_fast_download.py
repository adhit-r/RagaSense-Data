#!/usr/bin/env python3
"""
Test the fast downloader with a small file first
"""

import requests
import time
from pathlib import Path

def test_download_speed():
    """Test download speed with a small file"""
    print("🚀 TESTING FAST DOWNLOADER")
    print("=" * 40)
    
    # Test with a small file (1MB test file)
    test_url = "https://httpbin.org/bytes/1048576"  # 1MB test file
    test_file = Path("test_download.bin")
    
    print("📥 Testing single-threaded download...")
    start_time = time.time()
    
    try:
        response = requests.get(test_url, stream=True)
        response.raise_for_status()
        
        with open(test_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        elapsed = time.time() - start_time
        file_size = test_file.stat().st_size
        speed = file_size / elapsed / (1024 * 1024)  # MB/s
        
        print(f"✅ Single-threaded: {speed:.1f} MB/s")
        
        # Clean up
        test_file.unlink()
        
        return speed
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 0

def show_performance_estimate():
    """Show performance estimates"""
    print("\n📊 PERFORMANCE ESTIMATES")
    print("=" * 40)
    
    # Saraga dataset info
    saraga_size_gb = 7.6  # 7.6 GB from Zenodo
    saraga_size_mb = saraga_size_gb * 1024
    
    print(f"📦 Saraga dataset size: {saraga_size_gb:.1f} GB")
    
    # Estimate download times
    speeds = [1, 5, 10, 20, 50]  # MB/s
    
    print("\n⏱️ Estimated download times:")
    for speed in speeds:
        time_hours = saraga_size_mb / speed / 3600
        print(f"   {speed:2d} MB/s: {time_hours:.1f} hours")
    
    print("\n🚀 With multi-threading (8 workers):")
    for speed in speeds:
        threaded_speed = speed * 4  # Rough estimate
        time_hours = saraga_size_mb / threaded_speed / 3600
        print(f"   {speed:2d} MB/s base → {threaded_speed:2d} MB/s: {time_hours:.1f} hours")

def main():
    """Main function"""
    # Test current speed
    current_speed = test_download_speed()
    
    # Show estimates
    show_performance_estimate()
    
    print(f"\n🎯 RECOMMENDATION:")
    if current_speed > 10:
        print("✅ Your connection is fast! Multi-threading will help significantly.")
        print("🚀 Run: python fast_downloader.py --workers 8")
    elif current_speed > 5:
        print("✅ Your connection is good. Multi-threading will provide moderate improvement.")
        print("🚀 Run: python fast_downloader.py --workers 6")
    else:
        print("⚠️ Your connection is slower. Multi-threading will help but download will take time.")
        print("🚀 Run: python fast_downloader.py --workers 4")
    
    print(f"\n💡 Current speed: {current_speed:.1f} MB/s")
    print("📝 Note: Actual speeds may vary based on server response and network conditions")

if __name__ == "__main__":
    main()
