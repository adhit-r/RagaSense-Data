#!/usr/bin/env python3
"""
Raga Classification by Musical Tradition
Classifies ragas into Carnatic and Hindustani traditions with detailed analysis
"""

import sys
import os

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant
import json

def classify_ragas_by_tradition():
    """Classify and analyze ragas by musical tradition"""
    
    print("🏛️ RAGA CLASSIFICATION BY MUSICAL TRADITION")
    print("=" * 70)
    
    assistant = DataUnificationAssistant()
    
    # Initialize classification containers
    classification = {
        'carnatic': {
            'ragas': [],
            'total_count': 0,
            'with_musical_theory': 0,
            'sources': {'saraga': 0, 'ramanarunachalam': 0}
        },
        'hindustani': {
            'ragas': [],
            'total_count': 0,
            'with_musical_theory': 0,
            'sources': {'saraga': 0, 'ramanarunachalam': 0}
        },
        'unknown': {
            'ragas': [],
            'total_count': 0,
            'sources': {'saraga': 0, 'ramanarunachalam': 0}
        }
    }
    
    print("\n🔍 Analyzing Saraga Dataset...")
    
    # Classify Saraga data
    for key, entity in assistant.saraga_data.items():
        tradition = entity.get('tradition', '').lower()
        source = entity.get('source', 'unknown')
        
        if tradition in ['carnatic', 'hindustani']:
            classification[tradition]['ragas'].append({
                'name': entity.get('original_name', key),
                'normalized_name': key,
                'source': source,
                'data': entity.get('data', {})
            })
            classification[tradition]['total_count'] += 1
            classification[tradition]['sources']['saraga'] += 1
        else:
            classification['unknown']['ragas'].append({
                'name': entity.get('original_name', key),
                'normalized_name': key,
                'source': source,
                'tradition': tradition,
                'data': entity.get('data', {})
            })
            classification['unknown']['total_count'] += 1
            classification['unknown']['sources']['saraga'] += 1
    
    print("🔍 Analyzing Ramanarunachalam Dataset...")
    
    # Classify Ramanarunachalam data
    for key, entity in assistant.ramanarunachalam_data.items():
        tradition = entity.get('tradition', '').lower()
        source = entity.get('source', 'unknown')
        
        if tradition in ['carnatic', 'hindustani']:
            # Check for musical theory
            data = entity.get('data', {})
            theory = assistant._extract_musical_theory(data)
            has_theory = any([theory.get('arohana'), theory.get('avarohana'), theory.get('melakartha'), theory.get('thaat')])
            
            classification[tradition]['ragas'].append({
                'name': entity.get('original_name', key),
                'normalized_name': key,
                'source': source,
                'data': data,
                'has_musical_theory': has_theory,
                'theory': theory
            })
            classification[tradition]['total_count'] += 1
            classification[tradition]['sources']['ramanarunachalam'] += 1
            
            if has_theory:
                classification[tradition]['with_musical_theory'] += 1
        else:
            classification['unknown']['ragas'].append({
                'name': entity.get('original_name', key),
                'normalized_name': key,
                'source': source,
                'tradition': tradition,
                'data': entity.get('data', {})
            })
            classification['unknown']['total_count'] += 1
            classification['unknown']['sources']['ramanarunachalam'] += 1
    
    # Display classification results
    print(f"\n📊 CLASSIFICATION RESULTS")
    print("=" * 70)
    
    # Carnatic Classification
    print(f"\n🎼 CARNATIC TRADITION")
    print(f"   Total Ragas: {classification['carnatic']['total_count']}")
    print(f"   From Saraga: {classification['carnatic']['sources']['saraga']}")
    print(f"   From Ramanarunachalam: {classification['carnatic']['sources']['ramanarunachalam']}")
    print(f"   With Musical Theory: {classification['carnatic']['with_musical_theory']}")
    if classification['carnatic']['total_count'] > 0:
        theory_percentage = (classification['carnatic']['with_musical_theory'] / classification['carnatic']['total_count']) * 100
        print(f"   Musical Theory Coverage: {theory_percentage:.1f}%")
    
    # Hindustani Classification
    print(f"\n🎵 HINDUSTANI TRADITION")
    print(f"   Total Ragas: {classification['hindustani']['total_count']}")
    print(f"   From Saraga: {classification['hindustani']['sources']['saraga']}")
    print(f"   From Ramanarunachalam: {classification['hindustani']['sources']['ramanarunachalam']}")
    print(f"   With Musical Theory: {classification['hindustani']['with_musical_theory']}")
    if classification['hindustani']['total_count'] > 0:
        theory_percentage = (classification['hindustani']['with_musical_theory'] / classification['hindustani']['total_count']) * 100
        print(f"   Musical Theory Coverage: {theory_percentage:.1f}%")
    
    # Unknown Classification
    if classification['unknown']['total_count'] > 0:
        print(f"\n❓ UNKNOWN/UNCLASSIFIED")
        print(f"   Total Ragas: {classification['unknown']['total_count']}")
        print(f"   From Saraga: {classification['unknown']['sources']['saraga']}")
        print(f"   From Ramanarunachalam: {classification['unknown']['sources']['ramanarunachalam']}")
    
    # Sample ragas from each tradition
    print(f"\n🎯 SAMPLE RAGAS BY TRADITION")
    print("=" * 70)
    
    # Carnatic samples
    carnatic_samples = classification['carnatic']['ragas'][:10]
    print(f"\n🎼 Sample Carnatic Ragas:")
    for i, raga in enumerate(carnatic_samples, 1):
        theory_indicator = "🎵" if raga.get('has_musical_theory', False) else "📝"
        print(f"   {i:2d}. {theory_indicator} {raga['name']} ({raga['source']})")
    
    # Hindustani samples
    hindustani_samples = classification['hindustani']['ragas'][:10]
    print(f"\n🎵 Sample Hindustani Ragas:")
    for i, raga in enumerate(hindustani_samples, 1):
        theory_indicator = "🎵" if raga.get('has_musical_theory', False) else "📝"
        print(f"   {i:2d}. {theory_indicator} {raga['name']} ({raga['source']})")
    
    # Musical theory examples
    print(f"\n🎼 MUSICAL THEORY EXAMPLES")
    print("=" * 70)
    
    # Find Carnatic ragas with musical theory
    carnatic_with_theory = [r for r in classification['carnatic']['ragas'] if r.get('has_musical_theory', False)]
    if carnatic_with_theory:
        example = carnatic_with_theory[0]
        print(f"\n🎼 Carnatic Example: {example['name']}")
        theory = example.get('theory', {})
        if theory.get('arohana'):
            print(f"   🔗 Arohana: {theory['arohana']}")
        if theory.get('avarohana'):
            print(f"   🔗 Avarohana: {theory['avarohana']}")
        if theory.get('melakartha'):
            print(f"   🔗 Melakartha: {theory['melakartha']}")
    
    # Find Hindustani ragas with musical theory
    hindustani_with_theory = [r for r in classification['hindustani']['ragas'] if r.get('has_musical_theory', False)]
    if hindustani_with_theory:
        example = hindustani_with_theory[0]
        print(f"\n🎵 Hindustani Example: {example['name']}")
        theory = example.get('theory', {})
        if theory.get('thaat'):
            print(f"   🔗 Thaat: {theory['thaat']}")
        if theory.get('arohana'):
            print(f"   🔗 Arohana: {theory['arohana']}")
        if theory.get('avarohana'):
            print(f"   🔗 Avarohana: {theory['avarohana']}")
    
    # Save classification results
    output_file = 'data/03_processed/metadata/raga_classification_by_tradition.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Prepare output data (remove large data objects for file size)
    output_data = {
        'classification_summary': {
            'carnatic': {
                'total_count': classification['carnatic']['total_count'],
                'with_musical_theory': classification['carnatic']['with_musical_theory'],
                'sources': classification['carnatic']['sources']
            },
            'hindustani': {
                'total_count': classification['hindustani']['total_count'],
                'with_musical_theory': classification['hindustani']['with_musical_theory'],
                'sources': classification['hindustani']['sources']
            },
            'unknown': {
                'total_count': classification['unknown']['total_count'],
                'sources': classification['unknown']['sources']
            }
        },
        'carnatic_ragas': [{'name': r['name'], 'source': r['source'], 'has_theory': r.get('has_musical_theory', False)} for r in classification['carnatic']['ragas']],
        'hindustani_ragas': [{'name': r['name'], 'source': r['source'], 'has_theory': r.get('has_musical_theory', False)} for r in classification['hindustani']['ragas']],
        'generated_at': '2025-09-21'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Classification results saved to: {output_file}")
    
    # Final summary
    total_ragas = classification['carnatic']['total_count'] + classification['hindustani']['total_count']
    total_with_theory = classification['carnatic']['with_musical_theory'] + classification['hindustani']['with_musical_theory']
    
    print(f"\n✅ CLASSIFICATION COMPLETE!")
    print("=" * 70)
    print(f"📊 Total Classified Ragas: {total_ragas}")
    print(f"🎼 Carnatic: {classification['carnatic']['total_count']} ragas")
    print(f"🎵 Hindustani: {classification['hindustani']['total_count']} ragas")
    print(f"🎯 With Musical Theory: {total_with_theory} ragas ({(total_with_theory/max(total_ragas,1)*100):.1f}%)")
    
    return classification

def create_tradition_specific_reports():
    """Create detailed reports for each tradition"""
    
    print(f"\n\n📋 CREATING TRADITION-SPECIFIC REPORTS")
    print("=" * 70)
    
    classification = classify_ragas_by_tradition()
    
    # Create Carnatic-specific report
    carnatic_report = {
        'tradition': 'Carnatic',
        'total_ragas': classification['carnatic']['total_count'],
        'ragas_with_theory': classification['carnatic']['with_musical_theory'],
        'sources': classification['carnatic']['sources'],
        'sample_ragas': []
    }
    
    # Add sample ragas with musical theory
    for raga in classification['carnatic']['ragas'][:20]:
        if raga.get('has_musical_theory', False):
            theory = raga.get('theory', {})
            carnatic_report['sample_ragas'].append({
                'name': raga['name'],
                'arohana': theory.get('arohana', ''),
                'avarohana': theory.get('avarohana', ''),
                'melakartha': theory.get('melakartha', ''),
                'source': raga['source']
            })
    
    # Create Hindustani-specific report
    hindustani_report = {
        'tradition': 'Hindustani',
        'total_ragas': classification['hindustani']['total_count'],
        'ragas_with_theory': classification['hindustani']['with_musical_theory'],
        'sources': classification['hindustani']['sources'],
        'sample_ragas': []
    }
    
    # Add sample ragas with musical theory
    for raga in classification['hindustani']['ragas'][:20]:
        if raga.get('has_musical_theory', False):
            theory = raga.get('theory', {})
            hindustani_report['sample_ragas'].append({
                'name': raga['name'],
                'thaat': theory.get('thaat', ''),
                'arohana': theory.get('arohana', ''),
                'avarohana': theory.get('avarohana', ''),
                'source': raga['source']
            })
    
    # Save reports
    carnatic_file = 'data/03_processed/metadata/carnatic_ragas_report.json'
    hindustani_file = 'data/03_processed/metadata/hindustani_ragas_report.json'
    
    with open(carnatic_file, 'w', encoding='utf-8') as f:
        json.dump(carnatic_report, f, indent=2, ensure_ascii=False)
    
    with open(hindustani_file, 'w', encoding='utf-8') as f:
        json.dump(hindustani_report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Carnatic report saved to: {carnatic_file}")
    print(f"📄 Hindustani report saved to: {hindustani_file}")
    
    print(f"\n🎼 Carnatic Report Summary:")
    print(f"   Total Ragas: {carnatic_report['total_ragas']}")
    print(f"   With Theory: {carnatic_report['ragas_with_theory']}")
    print(f"   Sample Theory Ragas: {len(carnatic_report['sample_ragas'])}")
    
    print(f"\n🎵 Hindustani Report Summary:")
    print(f"   Total Ragas: {hindustani_report['total_ragas']}")
    print(f"   With Theory: {hindustani_report['ragas_with_theory']}")
    print(f"   Sample Theory Ragas: {len(hindustani_report['sample_ragas'])}")

if __name__ == "__main__":
    try:
        classify_ragas_by_tradition()
        create_tradition_specific_reports()
        
        print(f"\n\n🏛️ TRADITION CLASSIFICATION COMPLETE!")
        print("🎼 Carnatic and Hindustani ragas are now classified and analyzed")
        print("📊 Detailed reports generated for each tradition")
        print("🎯 Musical theory coverage analyzed for both traditions")
        
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()