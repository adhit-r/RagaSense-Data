#!/usr/bin/env python3
"""
Test Surati/SuruTTi unification issue
"""

import sys
import os

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant

def test_surati_unification():
    """Test the specific Surati/SuruTTi case"""
    print("🔍 Testing Surati/SuruTTi Unification Issue")
    print("=" * 60)
    
    assistant = DataUnificationAssistant()
    
    # Test normalization
    saraga_name = "Suraṭi"
    ramanarunachalam_name = "SuruTTi"
    
    print(f"📝 Original Names:")
    print(f"   Saraga: {saraga_name}")
    print(f"   Ramanarunachalam: {ramanarunachalam_name}")
    
    # Test normalization
    normalized_saraga = assistant.normalize_name(saraga_name)
    normalized_ramanarunachalam = assistant.normalize_name(ramanarunachalam_name)
    
    print(f"\\n🔄 Normalized Names:")
    print(f"   Saraga normalized: '{normalized_saraga}'")
    print(f"   Ramanarunachalam normalized: '{normalized_ramanarunachalam}'")
    print(f"   Match: {normalized_saraga == normalized_ramanarunachalam}")
    
    # Check if they exist in datasets
    print(f"\\n🔍 Dataset Search:")
    saraga_exists = normalized_saraga in assistant.saraga_data
    ramanarunachalam_exists = normalized_ramanarunachalam in assistant.ramanarunachalam_data
    
    print(f"   '{normalized_saraga}' in Saraga: {saraga_exists}")
    print(f"   '{normalized_ramanarunachalam}' in Ramanarunachalam: {ramanarunachalam_exists}")
    
    # Test unification
    print(f"\\n🎵 Testing Unification:")
    
    # Try both names
    for test_name in [saraga_name, ramanarunachalam_name, "Surati", "surutti"]:
        print(f"\\n   Testing with: '{test_name}'")
        result = assistant.unify_entity_with_musical_theory(test_name, tradition_filter="carnatic")
        print(f"   Status: {result['status']}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Saraga Found: {result.get('saraga_found', False)}")
        print(f"   Ramanarunachalam Found: {result.get('ramanarunachalam_found', False)}")
        
        if result.get('merged_data'):
            theory = result['merged_data']
            if theory.get('arohana'):
                print(f"   🎼 Arohana: {theory['arohana']}")
            if theory.get('melakartha'):
                print(f"   🎼 Melakartha: {theory['melakartha']}")

def test_name_variations():
    """Test various name normalization patterns"""
    print(f"\\n\\n🔤 Testing Name Normalization Patterns")
    print("=" * 60)
    
    assistant = DataUnificationAssistant()
    
    test_cases = [
        ("Suraṭi", "SuruTTi", "Should match - same raga"),
        ("Kalyāṇi", "Kalyani", "Should match - diacritic difference"),
        ("Bhairav", "Bhairavi", "Should match - known variation"),
        ("Kambhoji", "Kamboji", "Should match - spelling variation"),
        ("Yaman", "Yaman kalyāṇ", "Should match - compound name"),
    ]
    
    for name1, name2, description in test_cases:
        norm1 = assistant.normalize_name(name1)
        norm2 = assistant.normalize_name(name2)
        match = norm1 == norm2
        
        print(f"\\n📝 {description}")
        print(f"   '{name1}' → '{norm1}'")
        print(f"   '{name2}' → '{norm2}'")
        print(f"   Match: {'✅' if match else '❌'} {match}")

if __name__ == "__main__":
    test_surati_unification()
    test_name_variations()