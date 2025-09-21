#!/usr/bin/env python3
"""
Quick Dataset Explorer - Command Line Interface
=============================================

Fast command-line exploration of RagaSense dataset
"""

import json
import os
from pathlib import Path

def load_dataset_info():
    """Load basic dataset information"""
    info = {}
    
    # Dataset manifest
    try:
        with open('data/DATASET_MANIFEST.json', 'r') as f:
            info['manifest'] = json.load(f)
    except:
        info['manifest'] = {}
    
    # Integration summary
    try:
        with open('data/03_processed/metadata/integration_summary_20250921_112230.json', 'r') as f:
            info['integration'] = json.load(f)
    except:
        info['integration'] = {}
    
    # ML dataset summary
    try:
        with open('data/04_ml_datasets/final_ml_dataset_summary_20250914_115657.json', 'r') as f:
            info['ml_summary'] = json.load(f)
    except:
        info['ml_summary'] = {}
    
    return info

def show_overview():
    """Show dataset overview"""
    info = load_dataset_info()
    
    print("🎵 RagaSense Dataset Overview")
    print("=" * 50)
    
    # Version and basic info
    version = info['manifest'].get('version', 'Unknown')
    print(f"📅 Version: {version}")
    
    # File statistics
    stats = info['manifest'].get('statistics', {})
    total_files = sum(tier.get('files', 0) for tier in stats.values())
    print(f"📁 Total Files: {total_files:,}")
    
    print("\n📊 Data Structure:")
    for tier, data in stats.items():
        files = data.get('files', 0)
        dirs = data.get('directories', 0)
        print(f"   {tier:15} {files:6,} files, {dirs:3,} dirs")
    
    # Integration results
    integration = info['integration']
    if integration:
        print(f"\n🔗 Integration Results:")
        print(f"   Saraga Ragas: {integration.get('saraga_ragas_processed', 'N/A')}")
        print(f"   Ramanarunachalam Ragas: {integration.get('ramanarunachalam_ragas_total', 'N/A')}")
        print(f"   Match Rate: {integration.get('match_rate', 0)*100:.1f}%")
    
    # ML dataset info
    ml_info = info['ml_summary']
    if ml_info:
        print(f"\n🤖 ML Dataset Summary:")
        print(f"   Total Samples: {ml_info.get('total_samples', 'N/A'):,}")
        print(f"   Unique Ragas: {ml_info.get('unique_ragas', 'N/A')}")
        print(f"   Traditions: {ml_info.get('unique_traditions', 'N/A')}")
        print(f"   Duration: {ml_info.get('total_duration_minutes', 'N/A')} minutes")

def show_file_structure():
    """Show detailed file structure"""
    print("📂 Detailed File Structure")
    print("=" * 50)
    
    base_path = Path('data')
    
    for tier_dir in sorted(base_path.iterdir()):
        if tier_dir.is_dir() and tier_dir.name.startswith(('01_', '02_', '03_', '04_', '05_', '99_')):
            print(f"\n📁 {tier_dir.name}/")
            
            # Count files and show samples
            all_files = list(tier_dir.rglob('*'))
            json_files = [f for f in all_files if f.suffix == '.json']
            other_files = [f for f in all_files if f != json_files and f.is_file()]
            
            print(f"   📄 JSON files: {len(json_files)}")
            print(f"   📄 Other files: {len(other_files)}")
            
            # Show sample files
            sample_files = (json_files + other_files)[:5]
            for f in sample_files:
                rel_path = f.relative_to(tier_dir)
                size_kb = f.stat().st_size / 1024 if f.is_file() else 0
                print(f"      📄 {rel_path} ({size_kb:.1f} KB)")
            
            if len(all_files) > 5:
                print(f"      ... and {len(all_files) - 5} more files")

def show_raga_samples():
    """Show sample ragas from different sources"""
    print("🎵 Sample Ragas")
    print("=" * 50)
    
    # Sample from Ramanarunachalam
    print("\n📚 Ramanarunachalam Samples:")
    carnatic_dir = Path('data/01_source/ramanarunachalam/Carnatic/raga')
    if carnatic_dir.exists():
        sample_files = list(carnatic_dir.glob('*.json'))[:5]
        for f in sample_files:
            raga_name = f.stem
            print(f"   🎵 {raga_name} (Carnatic)")
    
    hindustani_dir = Path('data/01_source/ramanarunachalam/Hindustani/raga')
    if hindustani_dir.exists():
        sample_files = list(hindustani_dir.glob('*.json'))[:5]
        for f in sample_files:
            raga_name = f.stem
            print(f"   🎵 {raga_name} (Hindustani)")
    
    # Sample from Saraga
    print("\n📀 Saraga Dataset Samples:")
    try:
        with open('data/02_raw/extracted_saraga_metadata/carnatic_metadata_extracted.json', 'r') as f:
            carnatic_data = json.load(f)
            sample_ragas = list(carnatic_data.keys())[:5]
            for raga in sample_ragas:
                print(f"   🎵 {raga} (Carnatic)")
    except:
        print("   ⚠️ Carnatic metadata not found")
    
    try:
        with open('data/02_raw/extracted_saraga_metadata/hindustani_metadata_extracted.json', 'r') as f:
            hindustani_data = json.load(f)
            sample_ragas = list(hindustani_data.keys())[:5]
            for raga in sample_ragas:
                print(f"   🎵 {raga} (Hindustani)")
    except:
        print("   ⚠️ Hindustani metadata not found")

def show_web_apps():
    """Show available web applications"""
    print("🌐 Available Web Applications")
    print("=" * 50)
    
    apps = [
        {
            'name': 'Dataset Explorer',
            'file': 'dataset_explorer.py',
            'port': '5002',
            'description': 'Modern search and exploration interface',
            'features': ['Advanced search', 'Statistics dashboard', 'Detailed raga views']
        },
        {
            'name': 'Modern Raga Mapper',
            'file': 'modern_raga_mapper.py',
            'port': '5001',
            'description': 'Cross-tradition raga mapping tool',
            'features': ['Smart matching', 'Confidence scoring', 'Session management']
        },
        {
            'name': 'Collaborative Mapper',
            'file': 'app.py',
            'port': '5000',
            'description': 'Team-based mapping with voting',
            'features': ['Multi-user support', 'Voting system', 'Progress tracking']
        }
    ]
    
    for app in apps:
        print(f"\n🚀 {app['name']}")
        print(f"   📁 File: {app['file']}")
        print(f"   🌐 URL: http://localhost:{app['port']}")
        print(f"   📝 Description: {app['description']}")
        print(f"   ✨ Features: {', '.join(app['features'])}")
        print(f"   🚀 Command: python {app['file']}")

def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = 'overview'
    
    print("🎵 RagaSense Quick Explorer")
    print("=" * 50)
    
    if command == 'overview':
        show_overview()
    elif command == 'structure':
        show_file_structure()
    elif command == 'ragas':
        show_raga_samples()
    elif command == 'apps':
        show_web_apps()
    elif command == 'help':
        print("Usage: python quick_explorer.py [command]")
        print("\nCommands:")
        print("  overview   - Dataset overview and statistics (default)")
        print("  structure  - Detailed file structure")
        print("  ragas      - Sample ragas from different sources")
        print("  apps       - Available web applications")
        print("  help       - Show this help message")
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python quick_explorer.py help' for usage information")

if __name__ == '__main__':
    main()