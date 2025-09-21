#!/usr/bin/env python3
"""
Test the complete Surati unified result
"""

import sys
import os
import json

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant

def test_complete_surati_unification():
    """Test the complete unified Surati data"""
    print("🎵 Complete Surati Unification Test")
    print("=" * 60)
    
    assistant = DataUnificationAssistant()
    
    # Test the unified result
    result = assistant.unify_entity_with_musical_theory("Surati", tradition_filter="carnatic")
    
    print(f"📊 Unification Results:")
    print(f"   Status: {result['status']}")
    print(f"   Confidence: {result.get('confidence', 0):.1%}")
    print(f"   Confidence Reason: {result.get('confidence_reason', 'N/A')}")
    print(f"   Saraga Found: {result.get('saraga_found', False)}")
    print(f"   Ramanarunachalam Found: {result.get('ramanarunachalam_found', False)}")
    
    merged_data = result.get('merged_data', {})
    
    print(f"\n🎼 Musical Theory Data:")
    if merged_data.get('arohana'):
        print(f"   ✅ Arohana: {merged_data['arohana']}")
    if merged_data.get('avarohana'):
        print(f"   ✅ Avarohana: {merged_data['avarohana']}")
    if merged_data.get('melakartha'):
        print(f"   ✅ Melakartha: {merged_data['melakartha']}")
    if merged_data.get('description'):
        print(f"   📝 Description: {merged_data['description']}")
    
    print(f"\n📁 Data Sources:")
    sources = merged_data.get('_sources', {})
    if sources:
        print(f"   Saraga: {sources.get('saraga', False)}")
        print(f"   Ramanarunachalam: {sources.get('ramanarunachalam', False)}")
        print(f"   Merged at: {sources.get('merged_at', 'N/A')}")
    
    # Check musical analysis
    musical_analysis = result.get('musical_analysis', {})
    if musical_analysis:
        print(f"\n🎯 Musical Analysis:")
        print(f"   Arohana Match: {musical_analysis.get('arohana_match', False)}")
        print(f"   Avarohana Match: {musical_analysis.get('avarohana_match', False)}")
        print(f"   Melakartha Match: {musical_analysis.get('melakartha_match', False)}")
        print(f"   Overall Compatibility: {musical_analysis.get('overall_compatibility', 0):.1%}")
    
    # Show complete merged data structure
    print(f"\n🗂️ Complete Merged Data Structure:")
    print(json.dumps(merged_data, indent=2, ensure_ascii=False)[:1000] + "..." if len(str(merged_data)) > 1000 else json.dumps(merged_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_complete_surati_unification()