#!/usr/bin/env python3
"""
Comprehensive fix for Saraga-Ramanarunachalam matching issues
- Enhanced name normalization with complete mappings
- YouTube video integration from Ramanarunachalam
- Musical theory preservation
- Proper confidence scoring
"""

import sys
import os
import json
import re
from datetime import datetime

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant

class ComprehensiveMatchingFixer:
    def __init__(self):
        self.assistant = DataUnificationAssistant()
        
        # Comprehensive name mappings based on analysis
        self.comprehensive_mappings = {
            # Diacritic and transliteration variations
            'purnacandrika': 'poornachandrika',
            'pūrṇacandrika': 'poornachandrika', 
            'purna candrika': 'poornachandrika',
            
            'surati': 'surati',
            'suraṭi': 'surati',
            'surutti': 'surati',
            
            'sudda saveri': 'shuddha saveri',
            'śudda sāvēri': 'shuddha saveri',
            'shudda saveri': 'shuddha saveri',
            
            'mohanam': 'mohanam',
            'mōhanaṁ': 'mohanam',
            'mohana': 'mohanam',
            
            'todi': 'todi',
            'tōḍī': 'todi',
            'thodi': 'todi',
            
            'kambhoji': 'kambhoji',
            'kāṁbhōji': 'kambhoji',
            'kamas': 'kambhoji',  # Kamās → Kambhoji
            'kamās': 'kambhoji',
            
            'hamsadhvani': 'hamsadhvani',
            'hamsa dhvani': 'hamsadhvani',
            
            'lalita': 'lalita',
            'lalit': 'lalita',
            
            'latangi': 'latangi',
            'latāngi': 'latangi',
        }
    
    def enhanced_normalize_name(self, name):
        """Enhanced name normalization with comprehensive mappings"""
        if not name:
            return ""
        
        # Basic normalization from original assistant
        normalized = self.assistant.normalize_name(name)
        
        # Apply comprehensive mappings
        if normalized in self.comprehensive_mappings:
            mapped_name = self.comprehensive_mappings[normalized]
            print(f"    📝 Name mapping: '{normalized}' → '{mapped_name}'")
            return mapped_name
        
        return normalized
    
    def extract_youtube_videos(self, ramanarunachalam_data):
        """Extract YouTube videos from Ramanarunachalam data structure"""
        videos = []
        
        data = ramanarunachalam_data.get('data', {})
        songs = data.get('songs', [])
        
        for song in songs:
            if isinstance(song, dict) and 'I' in song:
                video_id = song['I']
                video_info = {
                    'video_id': video_id,
                    'youtube_url': f"https://youtube.com/watch?v={video_id}",
                    'duration': song.get('D', ''),
                    'views': song.get('V', ''),
                    'rating': song.get('R', 0),
                    'age_days': song.get('A', 0)
                }
                videos.append(video_info)
        
        return videos
    
    def create_enhanced_unified_entity(self, entity_name, tradition_filter=None):
        """Create enhanced unified entity with all fixes applied"""
        normalized_entity = self.enhanced_normalize_name(entity_name)
        
        print(f"\\n🎵 Enhanced Unification for: {entity_name}")
        print(f"   Normalized name: {normalized_entity}")
        if tradition_filter:
            print(f"   Tradition filter: {tradition_filter}")
        
        # Find matches with enhanced normalization
        saraga_matches = []
        ramanarunachalam_matches = []
        
        # Search Saraga data with enhanced normalization
        for key, entity in self.assistant.saraga_data.items():
            entity_norm = self.enhanced_normalize_name(entity.get('original_name', ''))
            if entity_norm == normalized_entity:
                if not tradition_filter or entity.get('tradition', '').lower() == tradition_filter.lower():
                    saraga_matches.append(entity)
        
        # Search Ramanarunachalam data with enhanced normalization
        for key, entity in self.assistant.ramanarunachalam_data.items():
            entity_norm = self.enhanced_normalize_name(entity.get('original_name', ''))
            if entity_norm == normalized_entity:
                if not tradition_filter or entity.get('tradition', '').lower() == tradition_filter.lower():
                    ramanarunachalam_matches.append(entity)
        
        if not saraga_matches and not ramanarunachalam_matches:
            return {
                "entity": entity_name,
                "status": "not_found",
                "confidence": 0.0,
                "message": f"Entity not found in either dataset after enhanced normalization"
            }
        
        # Calculate enhanced confidence
        confidence = 0.0
        confidence_reason = "No match"
        
        if saraga_matches and ramanarunachalam_matches:
            # Perfect match with both datasets - very high confidence
            confidence = 0.95
            confidence_reason = "Perfect match in both datasets with enhanced normalization"
        elif saraga_matches or ramanarunachalam_matches:
            # Found in one dataset
            confidence = 0.85
            confidence_reason = "Found in one dataset with enhanced normalization"
        
        # Get best matches
        best_saraga = saraga_matches[0] if saraga_matches else None
        best_ramanarunachalam = ramanarunachalam_matches[0] if ramanarunachalam_matches else None
        
        # Extract YouTube videos
        youtube_videos = []
        if best_ramanarunachalam:
            youtube_videos = self.extract_youtube_videos(best_ramanarunachalam)
        
        # Create merged data
        merged_data = {}
        if best_saraga:
            # Add Saraga data (audio, recording info)
            saraga_data = best_saraga.get('data', {})
            merged_data.update({
                'recording_info': saraga_data.get('recording_info'),
                'audio_file': saraga_data.get('audio_file'),
                'artist': saraga_data.get('artist'),
                'form': saraga_data.get('form'),
                'taala': saraga_data.get('taala'),
                'duration': saraga_data.get('duration'),
                'metadata_file': saraga_data.get('metadata_file')
            })
        
        if best_ramanarunachalam:
            # Add Ramanarunachalam data (musical theory, videos)
            rama_data = best_ramanarunachalam.get('data', {})
            merged_data.update({
                'arohana': rama_data.get('arohana'),
                'avarohana': rama_data.get('avarohana'),
                'melakartha': rama_data.get('melakartha'),
                'songs': rama_data.get('songs', []),
                'stats': rama_data.get('stats', []),
                'info': rama_data.get('info', [])
            })
        
        # Add YouTube videos
        merged_data['youtube_videos'] = youtube_videos
        merged_data['total_videos'] = len(youtube_videos)
        
        # Add source information
        merged_data['_sources'] = {
            'saraga': bool(best_saraga),
            'ramanarunachalam': bool(best_ramanarunachalam),
            'enhanced_normalization': True,
            'merged_at': datetime.now().isoformat()
        }
        
        return {
            "entity": entity_name,
            "normalized_name": normalized_entity,
            "status": "unified",
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "tradition": tradition_filter,
            "saraga_found": bool(best_saraga),
            "ramanarunachalam_found": bool(best_ramanarunachalam),
            "youtube_videos_count": len(youtube_videos),
            "merged_data": merged_data,
            "saraga_matches": saraga_matches,
            "ramanarunachalam_matches": ramanarunachalam_matches,
            "unification_timestamp": datetime.now().isoformat()
        }

def main():
    print("🔧 COMPREHENSIVE MATCHING FIX")
    print("=" * 80)
    
    fixer = ComprehensiveMatchingFixer()
    
    # Test enhanced matching on known problematic cases
    test_cases = [
        ("Suraṭi", "carnatic"),
        ("Pūrṇacandrika", "carnatic"),
        ("Kamās", "carnatic"),
        ("Śudda sāvēri", "carnatic"),
        ("Mōhanaṁ", "carnatic"),
        ("Tōḍī", "carnatic"),
        ("Kāṁbhōji", "carnatic"),
        ("Hamsadhvani", "carnatic"),
        ("Lalita", "carnatic"),
        ("Latāngi", "carnatic")
    ]
    
    results = []
    
    for raga_name, tradition in test_cases:
        print(f"\\n{'='*60}")
        result = fixer.create_enhanced_unified_entity(raga_name, tradition)
        results.append(result)
        
        print(f"✅ Result for {raga_name}:")
        print(f"   Status: {result['status']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Saraga Found: {result['saraga_found']}")
        print(f"   Ramanarunachalam Found: {result['ramanarunachalam_found']}")
        print(f"   YouTube Videos: {result['youtube_videos_count']}")
    
    # Summary statistics
    print(f"\\n\\n📊 SUMMARY STATISTICS")
    print("=" * 80)
    
    perfect_matches = sum(1 for r in results if r['confidence'] >= 0.95)
    high_confidence = sum(1 for r in results if r['confidence'] >= 0.85)
    total_with_videos = sum(1 for r in results if r['youtube_videos_count'] > 0)
    
    print(f"   Perfect matches (95%+): {perfect_matches}/{len(results)}")
    print(f"   High confidence (85%+): {high_confidence}/{len(results)}")
    print(f"   With YouTube videos: {total_with_videos}/{len(results)}")
    
    # Save detailed results
    output_file = f"enhanced_matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_results': results,
            'summary': {
                'perfect_matches': perfect_matches,
                'high_confidence': high_confidence,
                'total_with_videos': total_with_videos,
                'total_tested': len(results)
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()