#!/usr/bin/env python3
"""
Proper Ramanarunachalam Data Decoder
Decodes the actual numeric ID system and extracts real song counts
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RamanarunachalamProperDecoder:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.ramana_path = base_path / "downloads" / "Ramanarunachalam_Music_Repository" / "Carnatic"
        
        # Data containers
        self.raga_mappings = {}
        self.composer_mappings = {}
        self.concert_data = {}
        self.decoded_ragas = {}
        self.decoded_composers = {}
        
        logger.info(f"🔍 Initializing proper Ramanarunachalam decoder for {self.ramana_path}")

    def load_mappings(self):
        """Load the mapping files to understand the numeric ID system"""
        logger.info("📂 Loading mapping files...")
        
        try:
            # Load raga mappings (letters to numeric IDs)
            raga_path = self.ramana_path / "raga.json"
            with open(raga_path, 'r', encoding='utf-8') as f:
                raga_data = json.load(f)
            
            # The structure is: {"letters": {"english": {"A": "47,2397,2474,..."}, ...}}
            self.raga_mappings = raga_data.get('letters', {})
            logger.info(f"✅ Loaded raga mappings for {len(self.raga_mappings)} languages")
            
            # Load concert data (YouTube videos to songs)
            concert_path = self.ramana_path / "concert.json"
            with open(concert_path, 'r', encoding='utf-8') as f:
                self.concert_data = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.concert_data)} concerts")
            
            # Load composer mappings if available
            composer_path = self.ramana_path / "composer.json"
            if composer_path.exists():
                with open(composer_path, 'r', encoding='utf-8') as f:
                    composer_data = json.load(f)
                self.composer_mappings = composer_data.get('letters', {})
                logger.info(f"✅ Loaded composer mappings for {len(self.composer_mappings)} languages")
            else:
                logger.warning("⚠️ Composer mappings not found")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading mappings: {e}")
            return False

    def decode_raga_ids(self):
        """Decode raga numeric IDs to actual raga names"""
        logger.info("🔍 Decoding raga IDs...")
        
        # Use English mappings as primary
        english_mappings = self.raga_mappings.get('english', {})
        
        for letter, id_string in english_mappings.items():
            if letter == '?':  # Skip unknown
                continue
                
            # Split comma-separated IDs
            raga_ids = [int(id_str.strip()) for id_str in id_string.split(',') if id_str.strip()]
            
            for raga_id in raga_ids:
                # Create a raga entry
                raga_name = f"Raga_{raga_id}"  # We'll need to find the actual names
                self.decoded_ragas[raga_id] = {
                    'id': raga_id,
                    'letter': letter,
                    'name': raga_name,
                    'songs': []
                }
        
        logger.info(f"✅ Decoded {len(self.decoded_ragas)} raga IDs")
        return True

    def analyze_concert_data(self):
        """Analyze concert data to understand song-raga relationships"""
        logger.info("🎵 Analyzing concert data...")
        
        raga_song_counts = defaultdict(int)
        total_songs = 0
        
        for video_id, songs in self.concert_data.items():
            for song in songs:
                raga_id = song.get('R')  # Raga ID
                song_id = song.get('S')  # Song ID
                composer_id = song.get('C')  # Composer ID
                
                if raga_id and raga_id in self.decoded_ragas:
                    raga_song_counts[raga_id] += 1
                    self.decoded_ragas[raga_id]['songs'].append({
                        'song_id': song_id,
                        'composer_id': composer_id,
                        'video_id': video_id
                    })
                
                total_songs += 1
        
        logger.info(f"✅ Analyzed {total_songs} songs across {len(raga_song_counts)} ragas")
        
        # Update raga song counts
        for raga_id, count in raga_song_counts.items():
            self.decoded_ragas[raga_id]['song_count'] = count
        
        return raga_song_counts

    def find_raga_names_from_files(self):
        """Try to find actual raga names from individual raga files"""
        logger.info("🔍 Searching for raga names in individual files...")
        
        raga_dir = self.ramana_path / "raga"
        if not raga_dir.exists():
            logger.warning("⚠️ Raga directory not found")
            return False
        
        raga_files = list(raga_dir.glob("*.json"))
        logger.info(f"📁 Found {len(raga_files)} raga files")
        
        for raga_file in raga_files:
            try:
                with open(raga_file, 'r', encoding='utf-8') as f:
                    raga_data = json.load(f)
                
                # Extract raga name from title
                title_info = raga_data.get('title', {})
                if isinstance(title_info, dict):
                    raga_name = title_info.get('H', '')  # H seems to be the raga name
                    if raga_name:
                        # Try to find matching raga ID
                        # This is tricky - we need to match by some other means
                        # For now, let's just collect the names
                        logger.info(f"📝 Found raga: {raga_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error reading {raga_file}: {e}")
        
        return True

    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        logger.info("📊 Generating analysis report...")
        
        # Sort ragas by song count
        sorted_ragas = sorted(
            self.decoded_ragas.items(),
            key=lambda x: x[1].get('song_count', 0),
            reverse=True
        )
        
        report = {
            'total_ragas': len(self.decoded_ragas),
            'total_concerts': len(self.concert_data),
            'total_songs': sum(raga.get('song_count', 0) for raga in self.decoded_ragas.values()),
            'top_ragas': [],
            'raga_distribution': {},
            'data_structure_analysis': {
                'raga_mappings_languages': list(self.raga_mappings.keys()),
                'composer_mappings_languages': list(self.composer_mappings.keys()) if self.composer_mappings else [],
                'concert_structure': 'YouTube video ID -> List of songs with Raga/Composer/Song IDs'
            }
        }
        
        # Top 20 ragas by song count
        for raga_id, raga_data in sorted_ragas[:20]:
            report['top_ragas'].append({
                'raga_id': raga_id,
                'raga_name': raga_data['name'],
                'song_count': raga_data.get('song_count', 0),
                'letter': raga_data['letter']
            })
        
        # Song count distribution
        song_counts = [raga_data.get('song_count', 0) for raga_data in self.decoded_ragas.values()]
        report['raga_distribution'] = {
            'min_songs': min(song_counts) if song_counts else 0,
            'max_songs': max(song_counts) if song_counts else 0,
            'avg_songs': sum(song_counts) / len(song_counts) if song_counts else 0,
            'ragas_with_1000_plus_songs': len([c for c in song_counts if c >= 1000]),
            'ragas_with_100_plus_songs': len([c for c in song_counts if c >= 100]),
            'ragas_with_10_plus_songs': len([c for c in song_counts if c >= 10])
        }
        
        return report

    def save_results(self, report):
        """Save the decoded results"""
        logger.info("💾 Saving results...")
        
        output_dir = self.base_path / "data" / "ramanarunachalam_decoded"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save decoded ragas
        with open(output_dir / "decoded_ragas.json", 'w', encoding='utf-8') as f:
            json.dump(self.decoded_ragas, f, indent=2, ensure_ascii=False)
        
        # Save analysis report
        with open(output_dir / "analysis_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Results saved to {output_dir}")

    def run_analysis(self):
        """Run the complete analysis"""
        logger.info("🚀 STARTING PROPER RAMANARUNACHALAM ANALYSIS")
        logger.info("=" * 60)
        
        if not self.load_mappings():
            logger.error("❌ Failed to load mappings")
            return False
        
        if not self.decode_raga_ids():
            logger.error("❌ Failed to decode raga IDs")
            return False
        
        if not self.analyze_concert_data():
            logger.error("❌ Failed to analyze concert data")
            return False
        
        self.find_raga_names_from_files()
        
        report = self.generate_analysis_report()
        self.save_results(report)
        
        logger.info("\n🎉 PROPER RAMANARUNACHALAM ANALYSIS COMPLETED!")
        logger.info(f"📊 ANALYSIS SUMMARY:")
        logger.info(f"   Total Ragas: {report['total_ragas']}")
        logger.info(f"   Total Concerts: {report['total_concerts']}")
        logger.info(f"   Total Songs: {report['total_songs']}")
        logger.info(f"   Ragas with 1000+ songs: {report['raga_distribution']['ragas_with_1000_plus_songs']}")
        logger.info(f"   Ragas with 100+ songs: {report['raga_distribution']['ragas_with_100_plus_songs']}")
        
        logger.info("\n🏆 TOP 10 RAGAS BY SONG COUNT:")
        for i, raga in enumerate(report['top_ragas'][:10], 1):
            logger.info(f"   {i:2d}. Raga {raga['raga_id']} ({raga['letter']}): {raga['song_count']} songs")
        
        return True

if __name__ == "__main__":
    decoder = RamanarunachalamProperDecoder(Path(__file__).parent.parent)
    decoder.run_analysis()
