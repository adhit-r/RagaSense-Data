#!/usr/bin/env python3
"""
Correct Cross-Tradition Mappings
===============================

This script corrects and enhances the cross-tradition raga mappings based on
authoritative musicological research, particularly addressing the Bhairavi
confusion and providing accurate structural vs mood-equivalent classifications.
"""

import json
from datetime import datetime
from typing import Dict, List, Any

def create_corrected_mappings():
    """Create corrected cross-tradition mappings based on research."""
    
    corrected_mappings = {
        "timestamp": datetime.now().isoformat(),
        "description": "Corrected cross-tradition raga mappings based on authoritative musicological research",
        "structural_equivalents": [
            {
                "carnatic": "Kalyani",
                "hindustani": "Yaman",
                "relationship": "Structurally Identical",
                "confidence": "high",
                "notes": "Both use Lydian mode (S R2 G3 M2 P D2 N3 S). Differences are stylistic (gamaka in Carnatic, Re/Pa treatment in Hindustani)",
                "scale_structure": "S R2 G3 M2 P D2 N3 S",
                "melakarta": "65th Melakarta (Kalyani)",
                "thaat": "Kalyan"
            },
            {
                "carnatic": "Shankarabharanam", 
                "hindustani": "Bilawal",
                "relationship": "Structurally Identical",
                "confidence": "high",
                "notes": "Both use Major/Ionian scale. Very close structural equivalents.",
                "scale_structure": "S R2 G3 M1 P D2 N3 S",
                "melakarta": "29th Melakarta (Shankarabharanam)",
                "thaat": "Bilawal"
            },
            {
                "carnatic": "Mohanam",
                "hindustani": "Bhoopali (Bhoop)",
                "relationship": "Structurally Identical", 
                "confidence": "high",
                "notes": "Both use Major pentatonic scale. Practically identical structures.",
                "scale_structure": "S R2 G3 P D2 S",
                "melakarta": "28th Melakarta (Harikambhoji) - janya",
                "thaat": "Bilawal - janya"
            },
            {
                "carnatic": "Hanumatodi",
                "hindustani": "Bhairavi", 
                "relationship": "Structurally Identical",
                "confidence": "high",
                "notes": "Both use all komal (flat) notes. Hindustani Bhairavi corresponds to Carnatic Hanumatodi, not Carnatic Bhairavi.",
                "scale_structure": "S r1 g2 m1 P d1 n2 S",
                "melakarta": "8th Melakarta (Hanumatodi)",
                "thaat": "Bhairavi"
            }
        ],
        "mood_equivalents": [
            {
                "carnatic": "Bhairavi",
                "hindustani": "Bhairavi",
                "relationship": "Mood-Equivalent (Different Scales)",
                "confidence": "medium",
                "notes": "IMPORTANT: Despite same name, they have different scale structures. Carnatic Bhairavi = janya of Natabhairavi (uses both D1 and D2), Hindustani Bhairavi = Bhairavi thaat (all komal swaras). Both evoke pathos/devotion but are structurally different.",
                "carnatic_scale": "S R2 G2 M1 P D1 D2 N2 S (Bhashanga raga)",
                "hindustani_scale": "S r1 g2 m1 P d1 n2 S",
                "melakarta": "20th Melakarta (Natabhairavi) - janya",
                "thaat": "Bhairavi",
                "correction_note": "This was previously marked as 'identical' - INCORRECT. They are mood-equivalent but structurally different."
            },
            {
                "carnatic": "Todi (Hanumatodi)",
                "hindustani": "Miyan ki Todi",
                "relationship": "Mood-Equivalent (Different Scales)",
                "confidence": "medium", 
                "notes": "Carnatic Todi uses R1 G2 M1 D1 N2, Hindustani Todi uses komal Re, Ga, Dha, Ni with tivra Ma. Emotional overlap but different swaras.",
                "carnatic_scale": "S R1 G2 M1 P D1 N2 S",
                "hindustani_scale": "S r1 g2 M2 P d1 n2 S",
                "melakarta": "8th Melakarta (Hanumatodi)",
                "thaat": "Todi"
            },
            {
                "carnatic": "Hindolam",
                "hindustani": "Malkauns", 
                "relationship": "Mood-Equivalent (Different Pentatonic Scales)",
                "confidence": "medium",
                "notes": "Both are pentatonic, but Hindolam = S G2 M1 D1 N2; Malkauns = S g M d n. Different note sets, similar introspective/serious mood.",
                "carnatic_scale": "S G2 M1 D1 N2 S",
                "hindustani_scale": "S g M d n S",
                "melakarta": "20th Melakarta (Natabhairavi) - janya",
                "thaat": "Bhairavi - janya"
            }
        ],
        "additional_corrections": [
            {
                "issue": "Bhairavi Confusion",
                "explanation": "The raga name 'Bhairavi' exists in both traditions but refers to different musical structures. This is a common source of confusion in cross-tradition analysis.",
                "correct_mapping": "Hindustani Bhairavi ↔ Carnatic Hanumatodi (structurally identical)",
                "incorrect_assumption": "Hindustani Bhairavi ↔ Carnatic Bhairavi (mood-equivalent only)"
            },
            {
                "issue": "Scale Structure Differences",
                "explanation": "Carnatic Bhairavi is a Bhashanga raga (uses notes foreign to parent scale) with both D1 and D2, while Hindustani Bhairavi uses all komal notes in a different structure.",
                "carnatic_characteristics": "Bhashanga raga, uses both D1 and D2, janya of Natabhairavi",
                "hindustani_characteristics": "All komal notes, Bhairavi thaat, corresponds to Carnatic Hanumatodi"
            }
        ],
        "summary": {
            "structural_equivalents_count": 4,
            "mood_equivalents_count": 3, 
            "total_mappings": 7,
            "high_confidence": 4,
            "medium_confidence": 3,
            "corrections_made": [
                "Added Hanumatodi ↔ Bhairavi as structurally identical",
                "Clarified Bhairavi ↔ Bhairavi as mood-equivalent only",
                "Added detailed scale structures and explanations",
                "Included melakarta and thaat information"
            ]
        }
    }
    
    return corrected_mappings

def main():
    """Main function to create corrected mappings."""
    print("🔍 Creating Corrected Cross-Tradition Mappings...")
    
    # Create corrected mappings
    corrected_mappings = create_corrected_mappings()
    
    # Save to file
    output_file = "data/corrected_cross_tradition_mappings.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corrected_mappings, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Corrected mappings saved to {output_file}")
    
    # Print summary
    print("\n📊 CORRECTED MAPPINGS SUMMARY:")
    print(f"   Structural Equivalents: {corrected_mappings['summary']['structural_equivalents_count']}")
    print(f"   Mood Equivalents: {corrected_mappings['summary']['mood_equivalents_count']}")
    print(f"   Total Mappings: {corrected_mappings['summary']['total_mappings']}")
    
    print("\n🎵 KEY CORRECTIONS:")
    print("   ✅ Hindustani Bhairavi ↔ Carnatic Hanumatodi (Structurally Identical)")
    print("   ✅ Carnatic Bhairavi ↔ Hindustani Bhairavi (Mood-Equivalent Only)")
    print("   ✅ Added detailed scale structures and explanations")
    print("   ✅ Clarified Bhairavi confusion with authoritative research")
    
    print("\n🔍 BHairavi Analysis:")
    print("   • Same name, different structures")
    print("   • Carnatic Bhairavi: Bhashanga raga with D1 and D2")
    print("   • Hindustani Bhairavi: All komal notes, corresponds to Hanumatodi")
    print("   • Both evoke devotion/pathos but are structurally different")

if __name__ == "__main__":
    main()
