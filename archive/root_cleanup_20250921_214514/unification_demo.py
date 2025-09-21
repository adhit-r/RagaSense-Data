#!/usr/bin/env python3
"""
Interactive Data Unification Demo
================================

Simple command-line demo of the data unification assistant
"""

import json
import os
from pathlib import Path

def demo_unification():
    """Run an interactive demo of data unification"""
    
    print("🎵" + "="*60 + "🎵")
    print("    RAGASENSE DATA UNIFICATION ASSISTANT DEMO")
    print("🎵" + "="*60 + "🎵")
    print()
    
    # Check if results exist
    results_dir = Path("data/03_processed/unification")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        if result_files:
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            print(f"📄 Loading latest results: {latest_file.name}")
            
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                display_batch_results(results)
                
                # Interactive exploration
                while True:
                    print("\n" + "="*50)
                    choice = input("🔍 Enter entity name to explore (or 'quit' to exit): ").strip()
                    
                    if choice.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    display_entity_details(results, choice)
                    
            except Exception as e:
                print(f"❌ Error loading results: {e}")
        else:
            print("⚠️ No unification results found. Run the unification assistant first:")
            print("   python scripts/utilities/data_unification_assistant.py 'Bhairavi' 'Yaman'")
    else:
        print("⚠️ Unification results directory not found.")
        print("   Run the data unification assistant first to generate results.")

def display_batch_results(results):
    """Display batch unification results summary"""
    
    if "total_entities" in results:
        # Batch results
        summary = results.get("summary", {})
        
        print(f"\n📊 BATCH UNIFICATION SUMMARY")
        print("-" * 40)
        print(f"   Total Entities: {results['total_entities']}")
        print(f"   Found in Both: {summary.get('found_in_both', 0)}")
        print(f"   Saraga Only: {summary.get('found_in_saraga_only', 0)}")
        print(f"   Ramanarunachalam Only: {summary.get('found_in_ramanarunachalam_only', 0)}")
        print(f"   Not Found: {summary.get('not_found', 0)}")
        print(f"   High Confidence: {summary.get('high_confidence_matches', 0)}")
        
        print(f"\n🎵 PROCESSED ENTITIES:")
        for i, entity in enumerate(results.get("unified_entities", []), 1):
            status_icon = get_status_icon(entity)
            confidence_str = f"{entity['confidence']:.1%}" if entity['confidence'] > 0 else "N/A"
            print(f"   {i:2d}. {status_icon} {entity['entity']:15} ({confidence_str})")
    
    elif "entity" in results:
        # Single entity result
        print(f"\n🎵 SINGLE ENTITY RESULT")
        print("-" * 40)
        display_entity_summary(results)

def display_entity_details(results, entity_name):
    """Display detailed information for a specific entity"""
    
    entity_name_lower = entity_name.lower()
    
    # Find matching entity
    found_entity = None
    
    if "unified_entities" in results:
        # Batch results
        for entity in results["unified_entities"]:
            if entity["entity"].lower() == entity_name_lower:
                found_entity = entity
                break
    elif "entity" in results:
        # Single entity result
        if results["entity"].lower() == entity_name_lower:
            found_entity = results
    
    if not found_entity:
        print(f"❌ Entity '{entity_name}' not found in results.")
        return
    
    print(f"\n🎵 DETAILED ANALYSIS: {found_entity['entity']}")
    print("="*50)
    
    display_entity_summary(found_entity)
    display_field_analysis(found_entity)
    display_source_details(found_entity)

def display_entity_summary(entity):
    """Display entity summary information"""
    
    status_icon = get_status_icon(entity)
    confidence_str = f"{entity['confidence']:.1%}" if entity['confidence'] > 0 else "N/A"
    
    print(f"   Status: {status_icon} {entity['status']}")
    print(f"   Confidence: {confidence_str}")
    print(f"   Reason: {entity.get('confidence_reason', 'N/A')}")
    print(f"   Saraga Found: {'✅' if entity.get('saraga_found') else '❌'}")
    print(f"   Ramanarunachalam Found: {'✅' if entity.get('ramanarunachalam_found') else '❌'}")

def display_field_analysis(entity):
    """Display field analysis information"""
    
    field_analysis = entity.get("field_analysis", {})
    
    if not field_analysis:
        return
    
    print(f"\n📊 FIELD ANALYSIS:")
    print(f"   Field Overlap: {field_analysis.get('overlap_percentage', 0):.1f}%")
    print(f"   Total Fields A: {field_analysis.get('total_fields_a', 0)}")
    print(f"   Total Fields B: {field_analysis.get('total_fields_b', 0)}")
    
    common_fields = field_analysis.get("common_fields", [])
    missing_from_a = field_analysis.get("missing_from_a", [])
    missing_from_b = field_analysis.get("missing_from_b", [])
    
    if common_fields:
        print(f"   Common Fields ({len(common_fields)}): {', '.join(common_fields[:5])}")
        if len(common_fields) > 5:
            print(f"      ... and {len(common_fields) - 5} more")
    
    if missing_from_a:
        print(f"   Missing from Saraga ({len(missing_from_a)}): {', '.join(missing_from_a[:5])}")
        if len(missing_from_a) > 5:
            print(f"      ... and {len(missing_from_a) - 5} more")
    
    if missing_from_b:
        print(f"   Missing from Ramanarunachalam ({len(missing_from_b)}): {', '.join(missing_from_b[:5])}")
        if len(missing_from_b) > 5:
            print(f"      ... and {len(missing_from_b) - 5} more")

def display_source_details(entity):
    """Display source-specific details"""
    
    merged_data = entity.get("merged_data", {})
    
    if not merged_data:
        return
    
    sources = merged_data.get("sources", [])
    traditions = merged_data.get("traditions", [])
    unified_fields = merged_data.get("unified_fields", {})
    
    print(f"\n🔗 UNIFIED DATA:")
    print(f"   Sources: {', '.join(sources)}")
    print(f"   Traditions: {', '.join(traditions)}")
    print(f"   Unified Fields: {len(unified_fields)}")
    
    # Show sample fields
    sample_fields = list(unified_fields.items())[:3]
    for field_name, field_data in sample_fields:
        value = field_data.get("value", "")
        if isinstance(value, list):
            value_str = f"[{len(value)} items]"
        elif isinstance(value, str) and len(value) > 50:
            value_str = value[:47] + "..."
        else:
            value_str = str(value)
        
        confidence = field_data.get("confidence", "unknown")
        field_sources = field_data.get("sources", [])
        
        print(f"   📄 {field_name}: {value_str}")
        print(f"      Sources: {', '.join(field_sources)} | Confidence: {confidence}")

def get_status_icon(entity):
    """Get status icon for entity"""
    
    if entity.get("saraga_found") and entity.get("ramanarunachalam_found"):
        if entity.get("confidence", 0) >= 0.8:
            return "🟢"  # High confidence match
        else:
            return "🟡"  # Both found but low confidence
    elif entity.get("saraga_found") or entity.get("ramanarunachalam_found"):
        return "🔵"  # Found in one dataset
    else:
        return "🔴"  # Not found in either

def main():
    """Main demo function"""
    try:
        demo_unification()
    except KeyboardInterrupt:
        print("\n\n👋 Demo ended by user.")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    
    print("\n🎵 Thank you for using the RagaSense Data Unification Assistant!")

if __name__ == "__main__":
    main()