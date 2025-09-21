#!/usr/bin/env python3
"""
Comprehensive Ramanarunachalam Data Decoder
Properly decodes the numeric ID system and extracts accurate song counts
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RamanarunachalamDecoder:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.ramana_path = base_path / "downloads" / "Ramanarunachalam_Music_Repository" / "Carnatic"
        
        # Data containers
        self.raga_mappings = {}
        self.composer_mappings = {}
        self.language_mappings = {}
        self.concert_data = {}
        self.decoded_ragas = {}
        
        logger.info(f"🔍 Initializing Ramanarunachalam decoder for {self.ramana_path}")

    def load_mapping_files(self):
        """Load all the mapping files"""
        logger.info("📂 Loading mapping files...")
        
        # Load raga mappings
        raga_file = self.ramana_path / "raga.json"
        if raga_file.exists():
            with open(raga_file, 'r', encoding='utf-8') as f:
                raga_data = json.load(f)
                self.raga_mappings = raga_data.get('letters', {})
                logger.info(f"✅ Loaded raga mappings for {len(self.raga_mappings)} languages")
        
        # Load concert data
        concert_file = self.ramana_path / "concert.json"
        if concert_file.exists():
            with open(concert_file, 'r', encoding='utf-8') as f:
                self.concert_data = json.load(f)
                logger.info(f"✅ Loaded concert data for {len(self.concert_data)} concerts")
        
        # Try to find composer mappings (might be in a similar file)
        composer_file = self.ramana_path / "composer.json"
        if composer_file.exists():
            with open(composer_file, 'r', encoding='utf-8') as f:
                composer_data = json.load(f)
                self.composer_mappings = composer_data.get('letters', {})
                logger.info(f"✅ Loaded composer mappings")
        
        return True

    def create_reverse_mappings(self):
        """Create reverse mappings from numeric IDs to names"""
        logger.info("🔄 Creating reverse mappings...")
        
        self.id_to_raga = {}
        self.id_to_composer = {}
        
        # Create raga ID to name mapping (using English as primary)
        if 'English' in self.raga_mappings:
            for name, id_val in self.raga_mappings['English'].items():
                if isinstance(id_val, (int, str)):
                    self.id_to_raga[str(id_val)] = name
        
        # Also try other languages for more complete mapping
        for lang, mappings in self.raga_mappings.items():
            for name, id_val in mappings.items():
                if isinstance(id_val, (int, str)):
                    id_str = str(id_val)
                    if id_str not in self.id_to_raga:
                        self.id_to_raga[id_str] = name
        
        # Create composer mappings similarly
        if 'English' in self.composer_mappings:
            for name, id_val in self.composer_mappings['English'].items():
                if isinstance(id_val, (int, str)):
                    self.id_to_composer[str(id_val)] = name
        
        logger.info(f"✅ Created reverse mappings: {len(self.id_to_raga)} ragas, {len(self.id_to_composer)} composers")

    def analyze_individual_raga_files(self):
        """Analyze individual raga JSON files"""
        logger.info("📊 Analyzing individual raga files...")
        
        raga_dir = self.ramana_path / "raga"
        if not raga_dir.exists():
            logger.error(f"Raga directory not found: {raga_dir}")
            return
        
        raga_stats = {}
        
        for raga_file in raga_dir.glob("*.json"):
            raga_name = raga_file.stem
            
            try:
                with open(raga_file, 'r', encoding='utf-8') as f:
                    raga_data = json.load(f)
                
                songs = raga_data.get('songs', [])
                
                # Analyze song quality
                unique_songs = set()
                valid_songs = 0
                unknown_songs = 0
                
                for song in songs:
                    # Extract identifiers
                    song_id = song.get('I', '')  # YouTube ID or identifier
                    title_id = song.get('T', '')  # Title ID
                    
                    if song_id and song_id != 'Unknown':
                        unique_songs.add(song_id)
                        valid_songs += 1
                    else:
                        unknown_songs += 1
                
                raga_stats[raga_name] = {
                    'total_entries': len(songs),
                    'unique_songs': len(unique_songs),
                    'valid_songs': valid_songs,
                    'unknown_songs': unknown_songs,
                    'quality_score': (valid_songs / len(songs)) if songs else 0,
                    'stats': raga_data.get('stats', []),
                    'info': raga_data.get('info', [])
                }
                
            except Exception as e:
                logger.error(f"Error processing {raga_file}: {e}")
                continue
        
        # Sort by unique song count
        sorted_ragas = sorted(raga_stats.items(), key=lambda x: x[1]['unique_songs'], reverse=True)
        
        logger.info(f"📊 RAGA ANALYSIS RESULTS:")
        logger.info(f"   Total ragas analyzed: {len(raga_stats)}")
        logger.info(f"   Top 10 ragas by unique song count:")
        
        for i, (raga_name, stats) in enumerate(sorted_ragas[:10]):
            logger.info(f"   {i+1:2d}. {raga_name}: {stats['unique_songs']} unique songs "
                       f"({stats['total_entries']} entries, {stats['quality_score']:.1%} quality)")
        
        # Special focus on Kalyani
        if 'Kalyani' in raga_stats:
            kalyani_stats = raga_stats['Kalyani']
            logger.info(f"\n🎵 KALYANI DETAILED ANALYSIS:")
            logger.info(f"   Total entries: {kalyani_stats['total_entries']}")
            logger.info(f"   Unique songs: {kalyani_stats['unique_songs']}")
            logger.info(f"   Valid songs: {kalyani_stats['valid_songs']}")
            logger.info(f"   Unknown songs: {kalyani_stats['unknown_songs']}")
            logger.info(f"   Quality score: {kalyani_stats['quality_score']:.1%}")
            
            # Check the stats section
            if kalyani_stats['stats']:
                logger.info(f"   Stats from file:")
                for stat in kalyani_stats['stats']:
                    if isinstance(stat, dict):
                        logger.info(f"     {stat.get('H', 'Unknown')}: {stat.get('C', 'N/A')}")
        
        return raga_stats

    def cross_reference_with_concerts(self, raga_stats):
        """Cross-reference raga files with concert data"""
        logger.info("🔍 Cross-referencing with concert data...")
        
        # Count actual songs per raga from concert data
        raga_song_counts = defaultdict(set)
        
        for video_id, songs in self.concert_data.items():
            for song in songs:
                raga_id = str(song.get('R', ''))
                if raga_id in self.id_to_raga:
                    raga_name = self.id_to_raga[raga_id]
                    raga_song_counts[raga_name].add(video_id + "_" + str(song.get('S', '')))
        
        logger.info(f"📊 CROSS-REFERENCE RESULTS:")
        
        # Compare with file-based counts
        for raga_name, actual_count in sorted(
            [(k, len(v)) for k, v in raga_song_counts.items()], 
            key=lambda x: x[1], reverse=True)[:10]:
            
            file_stats = raga_stats.get(raga_name, {})
            file_count = file_stats.get('unique_songs', 0)
            
            logger.info(f"   {raga_name}: Concert data: {actual_count}, File data: {file_count}")
        
        # Check Kalyani specifically
        kalyani_actual = len(raga_song_counts.get('Kalyani', set()))
        kalyani_file = raga_stats.get('Kalyani', {}).get('unique_songs', 0)
        
        logger.info(f"\n🎵 KALYANI CROSS-REFERENCE:")
        logger.info(f"   Concert data songs: {kalyani_actual}")
        logger.info(f"   File data songs: {kalyani_file}")
        logger.info(f"   Data consistency: {'✅ GOOD' if abs(kalyani_actual - kalyani_file) < 10 else '❌ POOR'}")
        
        return raga_song_counts

    def generate_corrected_dataset(self, raga_song_counts):
        """Generate corrected dataset with proper song counts"""
        logger.info("📝 Generating corrected dataset...")
        
        corrected_ragas = {}
        
        for raga_name, song_ids in raga_song_counts.items():
            corrected_ragas[raga_name] = {
                'name': raga_name,
                'tradition': 'carnatic',
                'verified_song_count': len(song_ids),
                'source': 'ramanarunachalam_corrected',
                'data_quality': 'verified_from_concert_data',
                'songs': list(song_ids)
            }
        
        # Save corrected dataset
        output_path = self.base_path / "data" / "ramanarunachalam_corrected.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(corrected_ragas, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Corrected dataset saved to {output_path}")
        logger.info(f"   Total ragas: {len(corrected_ragas)}")
        logger.info(f"   Total songs: {sum(len(data['songs']) for data in corrected_ragas.values())}")
        
        return corrected_ragas

    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        logger.info("🚀 STARTING RAMANARUNACHALAM DATA DECODE")
        logger.info("=" * 60)
        
        try:
            if not self.load_mapping_files():
                logger.error("Failed to load mapping files")
                return False
            
            self.create_reverse_mappings()
            raga_stats = self.analyze_individual_raga_files()
            raga_song_counts = self.cross_reference_with_concerts(raga_stats)
            corrected_dataset = self.generate_corrected_dataset(raga_song_counts)
            
            logger.info("\n🎉 RAMANARUNACHALAM DECODE COMPLETED!")
            logger.info("📊 KEY FINDINGS:")
            logger.info(f"   Total ragas with data: {len(corrected_dataset)}")
            
            # Top 5 ragas by song count
            top_ragas = sorted(corrected_dataset.items(), 
                             key=lambda x: x[1]['verified_song_count'], reverse=True)[:5]
            
            for i, (name, data) in enumerate(top_ragas):
                logger.info(f"   {i+1}. {name}: {data['verified_song_count']} songs")
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return False

if __name__ == "__main__":
    decoder = RamanarunachalamDecoder(Path(__file__).parent.parent)
    decoder.run_full_analysis()
