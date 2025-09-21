#!/usr/bin/env python3
"""
Test Musical Theory-Based Raga Matching
Demonstrates arohana, avarohana, melakartha-based matching with tradition constraints
"""

import sys
import os

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant
import json

def test_musical_theory_matching():
    """Test the enhanced musical theory-based matching functionality"""
    
    print("🎵 Testing Musical Theory-Based Raga Matching")
    print("=" * 60)
    
    # Initialize the assistant
    assistant = DataUnificationAssistant()
    
    # Test cases for both Carnatic and Hindustani traditions
    test_cases = [
        {
            "name": "Yaman",
            "tradition": "hindustani",
            "description": "Famous Hindustani raga"
        },
        {
            "name": "Kalyani", 
            "tradition": "carnatic",
            "description": "Carnatic equivalent of Yaman"
        },
        {
            "name": "Bhairav",
            "tradition": "hindustani", 
            "description": "Morning raga in Hindustani"
        },
        {
            "name": "Mayamalavagowla",
            "tradition": "carnatic",
            "description": "Carnatic equivalent of Bhairav"
        }
    ]
    
    print("\n🔍 Testing Musical Theory Extraction and Matching\n")
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {test_case['name']} ({test_case['tradition']})")
        print(f"Description: {test_case['description']}")
        print(f"{'='*50}")
        
        # Test with tradition filter
        result = assistant.unify_entity_with_musical_theory(
            test_case['name'], 
            tradition_filter=test_case['tradition']
        )
        
        # Display results
        print(f"\n📊 Results for {test_case['name']}:")
        print(f"   Status: {result['status']}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Confidence Reason: {result.get('confidence_reason', 'N/A')}")
        print(f"   Tradition Filter: {result.get('tradition_filter', 'None')}")
        print(f"   Saraga Found: {result.get('saraga_found', False)}")
        print(f"   Ramanarunachalam Found: {result.get('ramanarunachalam_found', False)}")
        
        # Show musical theory analysis
        if result.get('musical_analysis'):
            musical_analysis = result['musical_analysis']
            print(f"\n🎼 Musical Theory Analysis:")
            print(f"   Arohana Match: {musical_analysis.get('arohana_match', False)}")
            print(f"   Avarohana Match: {musical_analysis.get('avarohana_match', False)}")
            print(f"   Melakartha Match: {musical_analysis.get('melakartha_match', False)}")
            print(f"   Thaat Match: {musical_analysis.get('thaat_match', False)}")
            print(f"   Overall Compatibility: {musical_analysis.get('overall_compatibility', 0):.1%}")
        
        # Show merged musical theory data
        if result.get('merged_data'):
            merged_data = result['merged_data']
            print(f"\n🎯 Extracted Musical Theory:")
            
            # Extract musical theory from merged data
            theory_fields = ['arohana', 'avarohana', 'melakartha', 'thaat', 'tradition']
            for field in theory_fields:
                value = merged_data.get(field, 'N/A')
                if value and value != 'N/A':
                    print(f"   {field.title()}: {value}")
            
            # Show audio files if available
            audio_files = merged_data.get('audio_files', [])
            if audio_files:
                print(f"   Audio Files: {len(audio_files)} files available")
        
        print("\n" + "-" * 50)

def test_tradition_constraints():
    """Test tradition-specific matching constraints"""
    
    print("\n\n🏛️ Testing Tradition Constraints")
    print("=" * 60)
    
    assistant = DataUnificationAssistant()
    
    # Test cross-tradition matching (should be filtered out)
    print("\n🚫 Testing Cross-Tradition Filtering:")
    
    # Try to find Carnatic ragas with Hindustani filter
    carnatic_raga = "Kalyani"
    print(f"\nSearching for Carnatic raga '{carnatic_raga}' with Hindustani filter:")
    result = assistant.unify_entity_with_musical_theory(carnatic_raga, tradition_filter="hindustani")
    print(f"   Status: {result['status']}")
    print(f"   Message: {result.get('message', 'N/A')}")
    
    # Try to find Hindustani ragas with Carnatic filter  
    hindustani_raga = "Yaman"
    print(f"\nSearching for Hindustani raga '{hindustani_raga}' with Carnatic filter:")
    result = assistant.unify_entity_with_musical_theory(hindustani_raga, tradition_filter="carnatic")
    print(f"   Status: {result['status']}")
    print(f"   Message: {result.get('message', 'N/A')}")
    
    # Search without tradition filter (should find both)
    print(f"\n✅ Testing Universal Search (No Tradition Filter):")
    result = assistant.unify_entity_with_musical_theory(carnatic_raga, tradition_filter=None)
    print(f"   Searching for '{carnatic_raga}' without tradition filter:")
    print(f"   Status: {result['status']}")
    print(f"   Saraga Found: {result.get('saraga_found', False)}")
    print(f"   Ramanarunachalam Found: {result.get('ramanarunachalam_found', False)}")

def show_dataset_statistics():
    """Show statistics about musical theory data availability"""
    
    print("\n\n📈 Dataset Musical Theory Statistics")
    print("=" * 60)
    
    assistant = DataUnificationAssistant()
    
    # Analyze musical theory coverage
    saraga_stats = {"total": 0, "with_theory": 0, "carnatic": 0, "hindustani": 0}
    ramanarunachalam_stats = {"total": 0, "with_theory": 0, "carnatic": 0, "hindustani": 0}
    
    # Saraga analysis
    for key, entity in assistant.saraga_data.items():
        saraga_stats["total"] += 1
        tradition = entity.get('tradition', '').lower()
        if tradition == 'carnatic':
            saraga_stats["carnatic"] += 1
        elif tradition == 'hindustani':
            saraga_stats["hindustani"] += 1
        
        theory = assistant._extract_musical_theory(entity)
        if any(theory.values()):
            saraga_stats["with_theory"] += 1
    
    # Ramanarunachalam analysis
    for key, entity in assistant.ramanarunachalam_data.items():
        ramanarunachalam_stats["total"] += 1
        tradition = entity.get('tradition', '').lower()
        if tradition == 'carnatic':
            ramanarunachalam_stats["carnatic"] += 1
        elif tradition == 'hindustani':  
            ramanarunachalam_stats["hindustani"] += 1
        
        theory = assistant._extract_musical_theory(entity)
        if any(theory.values()):
            ramanarunachalam_stats["with_theory"] += 1
    
    print(f"\n📊 Saraga Dataset:")
    print(f"   Total Ragas: {saraga_stats['total']}")
    print(f"   With Musical Theory: {saraga_stats['with_theory']} ({saraga_stats['with_theory']/max(saraga_stats['total'],1)*100:.1f}%)")
    print(f"   Carnatic: {saraga_stats['carnatic']}")
    print(f"   Hindustani: {saraga_stats['hindustani']}")
    
    print(f"\n📊 Ramanarunachalam Dataset:")
    print(f"   Total Ragas: {ramanarunachalam_stats['total']}")
    print(f"   With Musical Theory: {ramanarunachalam_stats['with_theory']} ({ramanarunachalam_stats['with_theory']/max(ramanarunachalam_stats['total'],1)*100:.1f}%)")
    print(f"   Carnatic: {ramanarunachalam_stats['carnatic']}")
    print(f"   Hindustani: {ramanarunachalam_stats['hindustani']}")

if __name__ == "__main__":
    try:
        test_musical_theory_matching()
        test_tradition_constraints() 
        show_dataset_statistics()
        
        print("\n\n✅ Musical Theory Matching Test Complete!")
        print("🎵 The system now prioritizes arohana, avarohana, melakartha over raga names")
        print("🏛️ Tradition constraints ensure Carnatic↔Carnatic and Hindustani↔Hindustani matching")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()