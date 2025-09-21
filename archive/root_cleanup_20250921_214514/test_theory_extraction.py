#!/usr/bin/env python3
"""
Quick test to verify musical theory extraction is working
"""

import sys
import os

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant
import json

def test_musical_theory_extraction():
    """Test if we can extract musical theory from sample data"""
    
    print("🔍 Testing Musical Theory Extraction")
    print("=" * 50)
    
    assistant = DataUnificationAssistant()
    
    # Test with a known raga that should have musical theory
    test_raga = "Kalyani"
    
    print(f"\n🎵 Testing extraction for: {test_raga}")
    
    # Look for Kalyani in both datasets
    normalized_name = assistant.normalize_name(test_raga)
    print(f"Normalized name: {normalized_name}")
    
    # Check in Ramanarunachalam data
    if normalized_name in assistant.ramanarunachalam_data:
        ramanarunachalam_entity = assistant.ramanarunachalam_data[normalized_name]
        print(f"\n📚 Found in Ramanarunachalam:")
        print(f"   Original name: {ramanarunachalam_entity.get('original_name')}")
        print(f"   Tradition: {ramanarunachalam_entity.get('tradition')}")
        
        # Extract musical theory
        data = ramanarunachalam_entity.get('data', {})
        theory = assistant._extract_musical_theory(data)
        
        print(f"\n🎼 Extracted Musical Theory:")
        for key, value in theory.items():
            if value:
                print(f"   {key.title()}: {value}")
        
        print(f"\n📄 Raw data structure:")
        print(f"   Keys in data: {list(data.keys())}")
        
        # Show first few fields from raw data
        for key in ['arohana', 'avarohana', 'melakarta', 'melakartha']:
            if key in data:
                print(f"   {key}: {data[key]}")
    
    # Check in Saraga data  
    if normalized_name in assistant.saraga_data:
        saraga_entity = assistant.saraga_data[normalized_name]
        print(f"\n🎼 Found in Saraga:")
        print(f"   Original name: {saraga_entity.get('original_name')}")
        print(f"   Tradition: {saraga_entity.get('tradition')}")
        
        # Extract musical theory
        data = saraga_entity.get('data', {})
        theory = assistant._extract_musical_theory(data)
        
        print(f"\n🎼 Extracted Musical Theory:")
        for key, value in theory.items():
            if value:
                print(f"   {key.title()}: {value}")
            else:
                print(f"   {key.title()}: (empty)")

if __name__ == "__main__":
    test_musical_theory_extraction()