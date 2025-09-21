#!/usr/bin/env python3
"""
Analyze Saraga data structure and find missing matches in Ramanarunachalam
"""

import sys
import os
import json

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant

def analyze_saraga_data_structure():
    """Analyze how Saraga data is structured"""
    print("🔍 SARAGA DATA STRUCTURE ANALYSIS")
    print("=" * 80)
    
    assistant = DataUnificationAssistant()
    
    print(f"📊 Total Saraga ragas loaded: {len(assistant.saraga_data)}")
    
    # Get a few sample entries
    sample_ragas = list(assistant.saraga_data.items())[:5]
    
    for i, (key, entity) in enumerate(sample_ragas, 1):
        print(f"\n{'='*50}")
        print(f"SAMPLE {i}: {key}")
        print(f"{'='*50}")
        print(f"Original name: {entity.get('original_name', 'N/A')}")
        print(f"Tradition: {entity.get('tradition', 'N/A')}")
        print(f"Source: {entity.get('source', 'N/A')}")
        
        data = entity.get('data', {})
        print(f"\nData structure keys: {list(data.keys())}")
        
        # Show full structure for first sample
        if i == 1:
            print(f"\nFull data structure:")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:1000] + "..." if len(str(data)) > 1000 else json.dumps(data, indent=2, ensure_ascii=False))

def find_saraga_ragas_missing_ramanarunachalam():
    """Find Saraga ragas that don't have matches in Ramanarunachalam"""
    print("\n\n🔍 FINDING MISSING MATCHES")
    print("=" * 80)
    
    assistant = DataUnificationAssistant()
    
    missing_matches = []
    partial_matches = []
    perfect_matches = []
    
    print("Analyzing each Saraga raga...")
    
    for key, entity in assistant.saraga_data.items():
        original_name = entity.get('original_name', key)
        tradition = entity.get('tradition', '').lower()
        
        # Try to find in Ramanarunachalam
        result = assistant.unify_entity_with_musical_theory(original_name, tradition_filter=tradition)
        
        if result.get('ramanarunachalam_found'):
            confidence = result.get('confidence', 0)
            if confidence >= 95:
                perfect_matches.append({
                    'saraga_name': original_name,
                    'confidence': confidence,
                    'has_theory': bool(result.get('merged_data', {}).get('arohana'))
                })
            else:
                partial_matches.append({
                    'saraga_name': original_name,
                    'confidence': confidence,
                    'reason': result.get('confidence_reason', 'Unknown')
                })
        else:
            missing_matches.append({
                'saraga_name': original_name,
                'normalized': assistant.normalize_name(original_name),
                'tradition': tradition
            })
    
    print(f"\n📊 MATCHING RESULTS:")
    print(f"   ✅ Perfect matches (95%+): {len(perfect_matches)}")
    print(f"   ⚠️ Partial matches (<95%): {len(partial_matches)}")
    print(f"   ❌ Missing matches: {len(missing_matches)}")
    
    print(f"\n❌ MISSING MATCHES (Saraga ragas not found in Ramanarunachalam):")
    for i, item in enumerate(missing_matches[:10], 1):  # Show first 10
        print(f"   {i}. {item['saraga_name']} → {item['normalized']} ({item['tradition']})")
    
    if len(missing_matches) > 10:
        print(f"   ... and {len(missing_matches) - 10} more")
    
    print(f"\n⚠️ PARTIAL MATCHES (Low confidence):")
    for i, item in enumerate(partial_matches[:10], 1):  # Show first 10
        print(f"   {i}. {item['saraga_name']} - {item['confidence']:.1f}% ({item['reason']})")
    
    if len(partial_matches) > 10:
        print(f"   ... and {len(partial_matches) - 10} more")
    
    print(f"\n✅ PERFECT MATCHES (with musical theory):")
    theory_count = sum(1 for m in perfect_matches if m['has_theory'])
    print(f"   Total: {len(perfect_matches)}")
    print(f"   With musical theory: {theory_count}")
    print(f"   Missing theory: {len(perfect_matches) - theory_count}")
    
    return missing_matches, partial_matches, perfect_matches

def test_specific_name_variations():
    """Test specific name variations to understand the pattern"""
    print("\n\n🔍 TESTING SPECIFIC NAME VARIATIONS")
    print("=" * 80)
    
    assistant = DataUnificationAssistant()
    
    # Test cases based on common variations
    test_cases = [
        ("Suraṭi", "SuruTTi"),  # We know this works
        ("Kamās", "Kambhoji"),  # Common variation
        ("Pūrṇacandrika", "Poornachandrika"),  # Diacritic variation
        ("Śudda sāvēri", "Shuddha saveri"),  # Unicode variation
        ("Mōhanaṁ", "Mohanam"),  # Simple variation
    ]
    
    print("Testing known variation patterns:")
    
    for saraga_name, expected_rama_name in test_cases:
        print(f"\n🔍 Testing: {saraga_name}")
        print(f"   Expected in Ramanarunachalam: {expected_rama_name}")
        
        # Normalize both
        norm_saraga = assistant.normalize_name(saraga_name)
        norm_rama = assistant.normalize_name(expected_rama_name)
        
        print(f"   Normalized Saraga: '{norm_saraga}'")
        print(f"   Normalized Rama: '{norm_rama}'")
        print(f"   Match: {'✅' if norm_saraga == norm_rama else '❌'}")
        
        # Check if they exist
        saraga_exists = norm_saraga in assistant.saraga_data
        rama_exists = norm_rama in assistant.ramanarunachalam_data
        
        print(f"   In Saraga: {saraga_exists}")
        print(f"   In Ramanarunachalam: {rama_exists}")
        
        # Try unification
        result = assistant.unify_entity_with_musical_theory(saraga_name, tradition_filter="carnatic")
        print(f"   Unification confidence: {result.get('confidence', 0):.1f}%")

def analyze_ramanarunachalam_video_structure():
    """Analyze how videos/YouTube links are structured in Ramanarunachalam"""
    print("\n\n🎥 RAMANARUNACHALAM VIDEO STRUCTURE ANALYSIS")
    print("=" * 80)
    
    assistant = DataUnificationAssistant()
    
    # Find a raga with videos
    sample_raga = None
    for key, entity in assistant.ramanarunachalam_data.items():
        data = entity.get('data', {})
        if data.get('songs') and len(data.get('songs', [])) > 0:
            sample_raga = (key, entity)
            break
    
    if sample_raga:
        key, entity = sample_raga
        data = entity.get('data', {})
        print(f"📹 Sample raga with videos: {entity.get('original_name', key)}")
        
        # Check for video-related fields
        video_fields = ['songs', 'videos', 'stats', 'info']
        for field in video_fields:
            if field in data:
                print(f"\n🔍 {field.upper()} structure:")
                value = data[field]
                if isinstance(value, list) and len(value) > 0:
                    print(f"   Type: List with {len(value)} items")
                    print(f"   Sample item: {json.dumps(value[0], indent=2, ensure_ascii=False)}")
                else:
                    print(f"   Value: {value}")
        
        # Look for YouTube ID patterns
        songs = data.get('songs', [])
        if songs:
            print(f"\n🎬 YouTube Video Analysis:")
            print(f"   Total videos: {len(songs)}")
            
            # Sample a few videos
            for i, song in enumerate(songs[:3], 1):
                if isinstance(song, dict):
                    youtube_id = song.get('I', '')
                    if youtube_id:
                        print(f"   Video {i}: https://youtube.com/watch?v={youtube_id}")
                        print(f"      Duration: {song.get('D', 'N/A')}")
                        print(f"      Views: {song.get('V', 'N/A')}")

if __name__ == "__main__":
    analyze_saraga_data_structure()
    missing, partial, perfect = find_saraga_ragas_missing_ramanarunachalam()
    test_specific_name_variations()
    analyze_ramanarunachalam_video_structure()