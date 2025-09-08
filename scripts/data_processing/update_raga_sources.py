#!/usr/bin/env python3
"""
Update Raga Sources with Saraga Data
====================================

This script updates the raga database to include additional sources from Saraga datasets.
It identifies ragas found in both Ramanarunachalam and Saraga datasets and updates
the source attribution accordingly.

Key findings:
- Kalyani: Found in both Ramanarunachalam (6,244 songs) and Saraga (22 tracks)
- Yaman: Found in Saraga Hindustani (2 tracks) - equivalent to Kalyani
- Hameerkalyani: Found in Saraga (22 tracks) - variant of Kalyani
"""

import json
import os
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('update_raga_sources.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RagaSourceUpdater:
    """
    Updates raga database with additional sources from Saraga datasets.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_path = self.project_root / "data" / "unified_ragasense_final"
        self.output_path = self.project_root / "data" / "updated_raga_sources"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.ragas = {}
        self.tracks = {}
        self.updated_ragas = {}
        self.source_updates = {
            "ragas_updated": 0,
            "new_sources_added": 0,
            "cross_tradition_links": 0
        }

    def load_data(self):
        """Load the unified raga and track databases."""
        logger.info("📂 Loading unified databases...")
        
        # Load ragas
        with open(self.data_path / "unified_ragas.json", 'r', encoding='utf-8') as f:
            self.ragas = json.load(f)
        
        # Load tracks
        with open(self.data_path / "unified_tracks.json", 'r', encoding='utf-8') as f:
            self.tracks = json.load(f)
        
        logger.info(f"✅ Loaded {len(self.ragas)} ragas and {len(self.tracks)} tracks")

    def identify_raga_tracks(self):
        """Identify tracks that correspond to known ragas."""
        logger.info("🔍 Identifying raga tracks from Saraga datasets...")
        
        raga_track_mapping = defaultdict(list)
        
        for track_id, track_data in self.tracks.items():
            name = track_data.get('name', '').lower()
            audio_file = track_data.get('audio_file', '').lower()
            dataset = track_data.get('dataset', '')
            
            # Skip if not from Saraga
            if not dataset.startswith('saraga') and dataset != 'melody_synth':
                continue
            
            # Extract raga from filename
            raga_name = None
            
            # Known raga patterns
            raga_patterns = [
                'kalyani', 'yaman', 'bhairavi', 'thodi', 'kambhoji', 'mohanam',
                'hamsadhwani', 'kapi', 'shree', 'bageshri', 'lalit', 'puriya',
                'dhani', 'bhimpalasi', 'kalyan', 'bageshree', 'kedar', 'jog',
                'marwa', 'bilaskhani', 'madhukauns', 'desh', 'miyan', 'bhairav',
                'abhogi', 'khamaj', 'hameerkalyani'
            ]
            
            for pattern in raga_patterns:
                if pattern in name or pattern in audio_file:
                    raga_name = pattern
                    break
            
            if raga_name:
                raga_track_mapping[raga_name].append({
                    'track_id': track_id,
                    'name': track_data.get('name', ''),
                    'artist': track_data.get('artist', ''),
                    'dataset': dataset,
                    'tradition': track_data.get('tradition', ''),
                    'audio_file': track_data.get('audio_file', '')
                })
        
        logger.info(f"✅ Identified {len(raga_track_mapping)} ragas in Saraga tracks")
        return raga_track_mapping

    def update_raga_sources(self, raga_track_mapping):
        """Update raga database with additional sources."""
        logger.info("🔄 Updating raga sources...")
        
        self.updated_ragas = self.ragas.copy()
        
        for raga_name, tracks in raga_track_mapping.items():
            # Find matching raga in database
            matching_raga_id = None
            for raga_id, raga_data in self.ragas.items():
                if raga_data.get('name', '').lower() == raga_name:
                    matching_raga_id = raga_id
                    break
            
            if matching_raga_id:
                # Update existing raga
                raga_data = self.updated_ragas[matching_raga_id]
                
                # Add Saraga as source
                sources = raga_data.get('sources', [])
                if 'saraga' not in sources:
                    sources.append('saraga')
                    raga_data['sources'] = sources
                    self.source_updates["new_sources_added"] += 1
                
                # Add track count from Saraga
                saraga_track_count = len(tracks)
                raga_data['saraga_track_count'] = raga_data.get('saraga_track_count', 0) + saraga_track_count
                
                # Add metadata about Saraga tracks
                if 'saraga_metadata' not in raga_data:
                    raga_data['saraga_metadata'] = {
                        'total_tracks': 0,
                        'datasets': set(),
                        'artists': set(),
                        'traditions': set()
                    }
                
                raga_data['saraga_metadata']['total_tracks'] += saraga_track_count
                raga_data['saraga_metadata']['datasets'].update(track['dataset'] for track in tracks)
                raga_data['saraga_metadata']['artists'].update(track['artist'] for track in tracks)
                raga_data['saraga_metadata']['traditions'].update(track['tradition'] for track in tracks)
                
                # Convert sets to lists for JSON serialization
                raga_data['saraga_metadata']['datasets'] = list(raga_data['saraga_metadata']['datasets'])
                raga_data['saraga_metadata']['artists'] = list(raga_data['saraga_metadata']['artists'])
                raga_data['saraga_metadata']['traditions'] = list(raga_data['saraga_metadata']['traditions'])
                
                raga_data['last_updated'] = datetime.now().isoformat()
                self.source_updates["ragas_updated"] += 1
                
                logger.info(f"✅ Updated {raga_name}: +{saraga_track_count} Saraga tracks")
            
            else:
                # Create new raga entry for Saraga-only ragas
                if raga_name == 'yaman':
                    # Yaman is Hindustani equivalent of Kalyani
                    new_raga_id = 'Yaman'
                    new_raga_data = {
                        'raga_id': new_raga_id,
                        'name': 'Yaman',
                        'sanskrit_name': 'yaman',
                        'tradition': 'Hindustani',
                        'song_count': len(tracks),
                        'saraga_track_count': len(tracks),
                        'sources': ['saraga'],
                        'source_priority': 'secondary',
                        'cross_tradition_mapping': {
                            'type': 'similar',
                            'mapping': {
                                'raga_name': 'Yaman',
                                'carnatic_name': 'Kalyani',
                                'hindustani_name': 'Yaman',
                                'confidence': 'high',
                                'evidence': 'expert_knowledge'
                            },
                            'confidence': 'high'
                        },
                        'saraga_metadata': {
                            'total_tracks': len(tracks),
                            'datasets': list(set(track['dataset'] for track in tracks)),
                            'artists': list(set(track['artist'] for track in tracks)),
                            'traditions': list(set(track['tradition'] for track in tracks))
                        },
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    self.updated_ragas[new_raga_id] = new_raga_data
                    self.source_updates["cross_tradition_links"] += 1
                    logger.info(f"✅ Created new raga: {new_raga_id} (Hindustani equivalent of Kalyani)")

    def save_updated_database(self):
        """Save the updated raga database."""
        logger.info("💾 Saving updated raga database...")
        
        # Save updated ragas
        with open(self.output_path / "updated_unified_ragas.json", 'w', encoding='utf-8') as f:
            json.dump(self.updated_ragas, f, indent=2, ensure_ascii=False)
        
        # Save update report
        report = {
            "timestamp": datetime.now().isoformat(),
            "update_summary": self.source_updates,
            "total_ragas": len(self.updated_ragas),
            "ragas_with_saraga_sources": len([r for r in self.updated_ragas.values() if 'saraga' in r.get('sources', [])]),
            "cross_tradition_ragas": len([r for r in self.updated_ragas.values() if r.get('tradition') == 'Both'])
        }
        
        with open(self.output_path / "source_update_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Updated database saved to {self.output_path}")

    def generate_summary(self):
        """Generate a summary of the updates."""
        logger.info("\n📊 SOURCE UPDATE SUMMARY:")
        logger.info(f"   Ragas updated: {self.source_updates['ragas_updated']}")
        logger.info(f"   New sources added: {self.source_updates['new_sources_added']}")
        logger.info(f"   Cross-tradition links: {self.source_updates['cross_tradition_links']}")
        logger.info(f"   Total ragas: {len(self.updated_ragas)}")
        
        # Show ragas with multiple sources
        multi_source_ragas = [r for r in self.updated_ragas.values() if len(r.get('sources', [])) > 1]
        logger.info(f"   Ragas with multiple sources: {len(multi_source_ragas)}")
        
        if multi_source_ragas:
            logger.info("\n🔗 RAGAS WITH MULTIPLE SOURCES:")
            for raga in multi_source_ragas[:10]:  # Show top 10
                sources = raga.get('sources', [])
                saraga_tracks = raga.get('saraga_track_count', 0)
                logger.info(f"   {raga.get('name', 'Unknown')}: {sources} (+{saraga_tracks} Saraga tracks)")

    def run_update_process(self):
        """Run the complete source update process."""
        logger.info("🚀 STARTING RAGA SOURCE UPDATE")
        logger.info("=" * 60)
        
        self.load_data()
        raga_track_mapping = self.identify_raga_tracks()
        self.update_raga_sources(raga_track_mapping)
        self.save_updated_database()
        self.generate_summary()
        
        logger.info("\n🎉 RAGA SOURCE UPDATE COMPLETED!")
        return True

def main():
    """Main function."""
    updater = RagaSourceUpdater()
    success = updater.run_update_process()
    
    if success:
        logger.info(f"\n🎯 RAGA SOURCE UPDATE COMPLETE!")
        logger.info(f"📋 Updated database saved to: {updater.output_path}")
        logger.info(f"📊 Report saved to: {updater.output_path / 'source_update_report.json'}")
    else:
        logger.error("❌ Raga source update failed!")

if __name__ == "__main__":
    main()
