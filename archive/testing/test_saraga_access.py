#!/usr/bin/env python3
"""
Test Saraga API access and show what's available
"""

import requests
import json
from pathlib import Path

def test_saraga_api():
    """Test basic Saraga API access"""
    print("🔍 TESTING SARAGA API ACCESS")
    print("=" * 40)
    
    # Test public endpoints (no auth required)
    base_url = "https://compmusic.upf.edu/dunya-api"
    
    try:
        # Test basic API
        print("📡 Testing basic API access...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Basic API access successful")
            print(f"   Response: {response.text[:200]}...")
        else:
            print(f"❌ Basic API access failed: {response.status_code}")
            return False
        
        # Test traditions endpoint
        print("\n📡 Testing traditions endpoint...")
        response = requests.get(f"{base_url}/traditions/")
        if response.status_code == 200:
            traditions = response.json()
            print("✅ Traditions endpoint accessible")
            print(f"   Available traditions: {len(traditions)}")
            for tradition in traditions:
                print(f"   - {tradition.get('name', 'Unknown')} ({tradition.get('slug', 'no-slug')})")
        else:
            print(f"❌ Traditions endpoint failed: {response.status_code}")
        
        # Test recordings endpoint (might require auth)
        print("\n📡 Testing recordings endpoint...")
        response = requests.get(f"{base_url}/recordings/")
        if response.status_code == 200:
            recordings = response.json()
            print("✅ Recordings endpoint accessible")
            print(f"   Total recordings: {len(recordings)}")
        elif response.status_code == 401:
            print("🔐 Recordings endpoint requires authentication")
            print("   This is expected - you need an API token")
        else:
            print(f"❌ Recordings endpoint failed: {response.status_code}")
        
        print("\n📋 SUMMARY:")
        print("✅ Saraga API is accessible")
        print("🔐 Authentication required for full access")
        print("📝 To get full access:")
        print("   1. Go to https://compmusic.upf.edu/dunya-api")
        print("   2. Create an account")
        print("   3. Get your API token")
        print("   4. Use: python saraga_downloader.py --token YOUR_TOKEN")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Saraga API: {e}")
        return False

def show_alternative_download():
    """Show alternative download options"""
    print("\n🔄 ALTERNATIVE DOWNLOAD OPTIONS")
    print("=" * 40)
    
    print("📦 Direct download from Zenodo:")
    print("   https://zenodo.org/record/1256127#.X3VXcZMzZhE")
    print("   This is the recommended approach for bulk download")
    
    print("\n📚 GitHub repository:")
    print("   https://github.com/mtg/saraga")
    print("   Contains the official download scripts and documentation")
    
    print("\n🎵 What we already have:")
    print("   ✅ Ramanarunachalam repository (20K+ files, 3.7GB)")
    print("   ✅ Comprehensive raga metadata")
    print("   ✅ Multi-language support")
    print("   ✅ Both Carnatic and Hindustani data")

if __name__ == "__main__":
    success = test_saraga_api()
    show_alternative_download()
    
    if success:
        print("\n🎉 Saraga API test completed successfully!")
    else:
        print("\n❌ Saraga API test failed")
