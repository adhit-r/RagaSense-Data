#!/usr/bin/env python3
"""
Cleaned RagaSense-Data Explorer
==============================

This explorer uses the comprehensively cleaned data and properly separates:
- Ragas (musical scales/modes) - don't have artists
- Tracks (compositions) - have artists
- Artists (performers) - perform tracks

Key Features:
- Uses cleaned data (no __MACOSX, no hardcoded values)
- Proper separation of ragas vs tracks vs artists
- Dynamic data processing
- Comprehensive validation
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

class CleanedRagaSenseExplorer:
    """
    Explorer for comprehensively cleaned RagaSense-Data.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / "data"
        
        # Try to load cleaned database first, fallback to others
        cleaned_path = self.data_path / "comprehensively_cleaned"
        updated_path = self.data_path / "updated_raga_sources" / "updated_unified_ragas.json"
        original_path = self.data_path / "unified_ragasense_final" / "unified_ragas.json"
        
        if cleaned_path.exists():
            self.database_path = cleaned_path
            self.database_version = "comprehensively_cleaned"
        elif updated_path.exists():
            self.database_path = updated_path.parent
            self.database_version = "updated"
        else:
            self.database_path = original_path.parent
            self.database_version = "original"
        
        self.ragas = {}
        self.artists = {}
        self.tracks = {}
        self.audio_files = {}
        self.cross_tradition_mappings = {}
        self.metadata = {}
        
        self.load_all_data()

    def load_all_data(self):
        """Load all database files."""
        try:
            if self.database_version == "comprehensively_cleaned":
                # Load cleaned data
                with open(self.database_path / "cleaned_unified_ragas.json", 'r', encoding='utf-8') as f:
                    self.ragas = json.load(f)
                
                with open(self.database_path / "cleaned_unified_artists.json", 'r', encoding='utf-8') as f:
                    self.artists = json.load(f)
                
                with open(self.database_path / "cleaned_unified_tracks.json", 'r', encoding='utf-8') as f:
                    self.tracks = json.load(f)
                
                with open(self.database_path / "cleaned_unified_audio_files.json", 'r', encoding='utf-8') as f:
                    self.audio_files = json.load(f)
                
                with open(self.database_path / "cleaned_unified_cross_tradition_mappings.json", 'r', encoding='utf-8') as f:
                    self.cross_tradition_mappings = json.load(f)
                
                # Load metadata from original location
                with open(self.data_path / "unified_ragasense_final" / "unified_metadata.json", 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                # Load from other versions
                with open(self.database_path / "unified_ragas.json", 'r', encoding='utf-8') as f:
                    self.ragas = json.load(f)
                
                with open(self.database_path / "unified_artists.json", 'r', encoding='utf-8') as f:
                    self.artists = json.load(f)
                
                with open(self.database_path / "unified_tracks.json", 'r', encoding='utf-8') as f:
                    self.tracks = json.load(f)
                
                with open(self.database_path / "unified_audio_files.json", 'r', encoding='utf-8') as f:
                    self.audio_files = json.load(f)
                
                with open(self.database_path / "unified_cross_tradition_mappings.json", 'r', encoding='utf-8') as f:
                    self.cross_tradition_mappings = json.load(f)
                
                with open(self.database_path / "unified_metadata.json", 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            print(f"✅ Loaded {len(self.ragas)} ragas, {len(self.artists)} artists, {len(self.tracks)} tracks")
            print(f"📊 Database version: {self.database_version}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)

    def search_ragas(self, query: str) -> List[Dict[str, Any]]:
        """Search ragas by name (case-insensitive)."""
        query = query.lower()
        results = []
        
        for raga_id, raga_data in self.ragas.items():
            name = raga_data.get('name', '').lower()
            sanskrit_name = raga_data.get('sanskrit_name', '').lower()
            
            if query in name or query in sanskrit_name or query in raga_id.lower():
                results.append({
                    'raga_id': raga_id,
                    'name': raga_data.get('name', ''),
                    'sanskrit_name': raga_data.get('sanskrit_name', ''),
                    'tradition': raga_data.get('tradition', 'Unknown'),
                    'song_count': raga_data.get('song_count', 0),
                    'sources': raga_data.get('sources', []),
                    'saraga_track_count': raga_data.get('saraga_track_count', 0)
                })
        
        return sorted(results, key=lambda x: x['song_count'], reverse=True)

    def get_tracks_for_raga(self, raga_name: str) -> List[Dict[str, Any]]:
        """Get all tracks that use a specific raga."""
        raga_tracks = []
        
        for track_id, track_data in self.tracks.items():
            track_raga = track_data.get('raga', '').lower()
            if raga_name.lower() in track_raga or track_raga in raga_name.lower():
                raga_tracks.append({
                    'track_id': track_id,
                    'name': track_data.get('name', ''),
                    'artist': track_data.get('artist', 'Unknown'),
                    'artist_id': track_data.get('artist_id', 'unknown'),
                    'tradition': track_data.get('tradition', 'Unknown'),
                    'dataset': track_data.get('dataset', 'Unknown'),
                    'audio_file': track_data.get('audio_file', '')
                })
        
        return sorted(raga_tracks, key=lambda x: x['name'])

    def get_artist_tracks(self, artist_name: str) -> List[Dict[str, Any]]:
        """Get all tracks by a specific artist."""
        artist_tracks = []
        
        for track_id, track_data in self.tracks.items():
            track_artist = track_data.get('artist', '').lower()
            if artist_name.lower() in track_artist or track_artist in artist_name.lower():
                artist_tracks.append({
                    'track_id': track_id,
                    'name': track_data.get('name', ''),
                    'raga': track_data.get('raga', 'Unknown'),
                    'tradition': track_data.get('tradition', 'Unknown'),
                    'dataset': track_data.get('dataset', 'Unknown'),
                    'audio_file': track_data.get('audio_file', '')
                })
        
        return sorted(artist_tracks, key=lambda x: x['name'])

    def get_raga_details(self, raga_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific raga."""
        for raga_id, raga_data in self.ragas.items():
            if raga_data.get('name', '').lower() == raga_name.lower():
                return raga_data
        return None

    def get_artist_details(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific artist."""
        for artist_id, artist_data in self.artists.items():
            if artist_data.get('name', '').lower() == artist_name.lower():
                return artist_data
        return None

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        # Calculate dynamic statistics
        multi_source_count = len([r for r in self.ragas.values() if len(r.get('sources', [])) > 1])
        saraga_ragas = len([r for r in self.ragas.values() if 'saraga' in r.get('sources', [])])
        total_saraga_tracks = sum(r.get('saraga_track_count', 0) for r in self.ragas.values())
        
        # Artist statistics
        artist_track_counts = {}
        for track in self.tracks.values():
            artist = track.get('artist', 'Unknown')
            artist_track_counts[artist] = artist_track_counts.get(artist, 0) + 1
        
        # Raga statistics
        raga_track_counts = {}
        for track in self.tracks.values():
            raga = track.get('raga', 'Unknown')
            raga_track_counts[raga] = raga_track_counts.get(raga, 0) + 1
        
        return {
            'total_ragas': len(self.ragas),
            'total_artists': len(self.artists),
            'total_tracks': len(self.tracks),
            'total_audio_files': len(self.audio_files),
            'total_cross_tradition_mappings': len(self.cross_tradition_mappings),
            'multi_source_ragas': multi_source_count,
            'saraga_ragas': saraga_ragas,
            'total_saraga_tracks': total_saraga_tracks,
            'database_version': self.database_version,
            'top_artists_by_tracks': dict(sorted(artist_track_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_ragas_by_tracks': dict(sorted(raga_track_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'tradition_distribution': self.metadata.get('tradition_distribution', {}),
            'source_distribution': self.metadata.get('source_distribution', {})
        }

    def print_search_results(self, results: List[Dict[str, Any]], title: str = "Search Results"):
        """Print search results in a formatted way."""
        print(f"\n🔍 {title}")
        print("=" * 60)
        
        if not results:
            print("No results found.")
            return
        
        for i, raga in enumerate(results, 1):
            sources_str = ", ".join(raga['sources'])
            saraga_info = f" (+{raga['saraga_track_count']} Saraga)" if raga['saraga_track_count'] > 0 else ""
            
            print(f"{i:2d}. {raga['name']} ({raga['tradition']})")
            print(f"    Sanskrit: {raga['sanskrit_name']}")
            print(f"    Songs: {raga['song_count']:,}{saraga_info}")
            print(f"    Sources: {sources_str}")
            print()

    def print_raga_details(self, raga_data: Dict[str, Any]):
        """Print detailed raga information."""
        print(f"\n🎵 Raga Details: {raga_data.get('name', 'Unknown')}")
        print("=" * 60)
        
        # Basic info
        print(f"Name: {raga_data.get('name', 'Unknown')}")
        print(f"Sanskrit: {raga_data.get('sanskrit_name', 'N/A')}")
        print(f"Tradition: {raga_data.get('tradition', 'Unknown')}")
        print(f"Song Count: {raga_data.get('song_count', 0):,}")
        print(f"Sources: {', '.join(raga_data.get('sources', []))}")
        
        # Saraga info if available
        if raga_data.get('saraga_track_count', 0) > 0:
            print(f"Saraga Tracks: {raga_data['saraga_track_count']}")
            saraga_meta = raga_data.get('saraga_metadata', {})
            if saraga_meta:
                print(f"Saraga Datasets: {', '.join(saraga_meta.get('datasets', []))}")
                print(f"Saraga Artists: {', '.join(saraga_meta.get('artists', []))}")
        
        # Cross-tradition mapping
        cross_mapping = raga_data.get('cross_tradition_mapping', {})
        if cross_mapping and cross_mapping.get('mapping'):
            print(f"\nCross-Tradition Mapping:")
            mapping = cross_mapping['mapping']
            print(f"  Carnatic: {mapping.get('carnatic_name', 'N/A')}")
            print(f"  Hindustani: {mapping.get('hindustani_name', 'N/A')}")
            print(f"  Confidence: {cross_mapping.get('confidence', 'N/A')}")
        
        # Show tracks that use this raga
        raga_name = raga_data.get('name', '')
        tracks = self.get_tracks_for_raga(raga_name)
        if tracks:
            print(f"\nTracks using this raga ({len(tracks)} total):")
            for track in tracks[:5]:  # Show first 5
                print(f"  • {track['name']} by {track['artist']}")
            if len(tracks) > 5:
                print(f"  ... and {len(tracks) - 5} more")

    def print_artist_details(self, artist_data: Dict[str, Any]):
        """Print detailed artist information."""
        print(f"\n🎤 Artist Details: {artist_data.get('name', 'Unknown')}")
        print("=" * 60)
        
        # Basic info
        print(f"Name: {artist_data.get('name', 'Unknown')}")
        print(f"Tradition: {artist_data.get('tradition', 'Unknown')}")
        print(f"Total Tracks: {artist_data.get('total_tracks', 0)}")
        print(f"Sources: {', '.join(artist_data.get('sources', []))}")
        
        # Show tracks by this artist
        artist_name = artist_data.get('name', '')
        tracks = self.get_artist_tracks(artist_name)
        if tracks:
            print(f"\nTracks by this artist ({len(tracks)} total):")
            for track in tracks[:5]:  # Show first 5
                print(f"  • {track['name']} (Raga: {track['raga']})")
            if len(tracks) > 5:
                print(f"  ... and {len(tracks) - 5} more")

    def print_database_stats(self, stats: Dict[str, Any]):
        """Print database statistics."""
        print(f"\n📊 Database Statistics ({stats['database_version']} version)")
        print("=" * 60)
        
        print(f"Total Ragas: {stats['total_ragas']:,}")
        print(f"Total Artists: {stats['total_artists']:,}")
        print(f"Total Tracks: {stats['total_tracks']:,}")
        print(f"Total Audio Files: {stats['total_audio_files']:,}")
        print(f"Cross-Tradition Mappings: {stats['total_cross_tradition_mappings']:,}")
        print()
        
        print("Multi-Source Information:")
        print(f"  Ragas with multiple sources: {stats['multi_source_ragas']:,}")
        print(f"  Ragas with Saraga data: {stats['saraga_ragas']:,}")
        print(f"  Total Saraga tracks: {stats['total_saraga_tracks']:,}")
        print()
        
        print("Top Artists by Track Count:")
        for artist, count in list(stats['top_artists_by_tracks'].items())[:5]:
            print(f"  {artist}: {count} tracks")
        print()
        
        print("Top Ragas by Track Count:")
        for raga, count in list(stats['top_ragas_by_tracks'].items())[:5]:
            print(f"  {raga}: {count} tracks")
        print()
        
        print("Tradition Distribution:")
        for tradition, count in stats['tradition_distribution'].get('ragas', {}).items():
            print(f"  {tradition}: {count:,} ragas")

    def interactive_mode(self):
        """Start interactive exploration mode."""
        print("\n🎵 Welcome to Cleaned RagaSense-Data Explorer!")
        print("=" * 60)
        print("Commands:")
        print("  search <query>     - Search ragas by name")
        print("  raga <name>        - Show raga details and tracks")
        print("  artist <name>      - Show artist details and tracks")
        print("  tracks <raga>      - Show tracks for a raga")
        print("  stats              - Show database statistics")
        print("  help               - Show this help")
        print("  quit               - Exit")
        print()
        
        while True:
            try:
                command = input("Cleaned RagaSense> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    print("Goodbye! 🎵")
                    break
                elif cmd == 'help':
                    print("Available commands: search, raga, artist, tracks, stats, help, quit")
                elif cmd == 'search' and len(command) > 1:
                    query = ' '.join(command[1:])
                    results = self.search_ragas(query)
                    self.print_search_results(results, f"Search Results for '{query}'")
                elif cmd == 'raga' and len(command) > 1:
                    raga_name = ' '.join(command[1:])
                    raga_data = self.get_raga_details(raga_name)
                    if raga_data:
                        self.print_raga_details(raga_data)
                    else:
                        print(f"Raga '{raga_name}' not found.")
                elif cmd == 'artist' and len(command) > 1:
                    artist_name = ' '.join(command[1:])
                    artist_data = self.get_artist_details(artist_name)
                    if artist_data:
                        self.print_artist_details(artist_data)
                    else:
                        print(f"Artist '{artist_name}' not found.")
                elif cmd == 'tracks' and len(command) > 1:
                    raga_name = ' '.join(command[1:])
                    tracks = self.get_tracks_for_raga(raga_name)
                    if tracks:
                        print(f"\n🎵 Tracks for Raga: {raga_name} ({len(tracks)} total)")
                        print("=" * 60)
                        for track in tracks[:10]:  # Show first 10
                            print(f"• {track['name']} by {track['artist']} ({track['tradition']})")
                        if len(tracks) > 10:
                            print(f"... and {len(tracks) - 10} more")
                    else:
                        print(f"No tracks found for raga '{raga_name}'.")
                elif cmd == 'stats':
                    stats = self.get_database_stats()
                    self.print_database_stats(stats)
                else:
                    print("Invalid command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye! 🎵")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Cleaned RagaSense-Data Exploration Tool')
    parser.add_argument('command', nargs='?', help='Command to execute')
    parser.add_argument('args', nargs='*', help='Command arguments')
    
    args = parser.parse_args()
    
    explorer = CleanedRagaSenseExplorer()
    
    if not args.command:
        explorer.interactive_mode()
        return
    
    command = args.command.lower()
    
    if command == 'search' and args.args:
        query = ' '.join(args.args)
        results = explorer.search_ragas(query)
        explorer.print_search_results(results, f"Search Results for '{query}'")
    
    elif command == 'raga' and args.args:
        raga_name = ' '.join(args.args)
        raga_data = explorer.get_raga_details(raga_name)
        if raga_data:
            explorer.print_raga_details(raga_data)
        else:
            print(f"Raga '{raga_name}' not found.")
    
    elif command == 'artist' and args.args:
        artist_name = ' '.join(args.args)
        artist_data = explorer.get_artist_details(artist_name)
        if artist_data:
            explorer.print_artist_details(artist_data)
        else:
            print(f"Artist '{artist_name}' not found.")
    
    elif command == 'tracks' and args.args:
        raga_name = ' '.join(args.args)
        tracks = explorer.get_tracks_for_raga(raga_name)
        if tracks:
            print(f"\n🎵 Tracks for Raga: {raga_name} ({len(tracks)} total)")
            print("=" * 60)
            for track in tracks[:10]:
                print(f"• {track['name']} by {track['artist']} ({track['tradition']})")
            if len(tracks) > 10:
                print(f"... and {len(tracks) - 10} more")
        else:
            print(f"No tracks found for raga '{raga_name}'.")
    
    elif command == 'stats':
        stats = explorer.get_database_stats()
        explorer.print_database_stats(stats)
    
    elif command == 'interactive':
        explorer.interactive_mode()
    
    else:
        print("Invalid command. Use 'python3 cleaned_ragasense_explorer.py' for interactive mode.")
        print("Available commands: search, raga, artist, tracks, stats, interactive")

if __name__ == "__main__":
    main()

