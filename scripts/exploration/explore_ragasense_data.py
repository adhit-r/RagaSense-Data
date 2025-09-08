#!/usr/bin/env python3
"""
RagaSense-Data Exploration Tool
==============================

This tool provides comprehensive exploration capabilities for the RagaSense-Data database.
You can search, filter, and analyze the 1,340 unique ragas and related data.

Usage:
    python3 explore_ragasense_data.py [command] [options]

Commands:
    search <query>           - Search ragas by name
    filter <tradition>       - Filter by tradition (Carnatic/Hindustani/Both)
    top <n>                  - Show top N ragas by song count
    cross-tradition          - Show cross-tradition mappings
    stats                    - Show database statistics
    raga <name>              - Show detailed info for a specific raga
    artists                  - Show artist information
    tracks                   - Show track information
    interactive              - Start interactive exploration mode
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

class RagaSenseExplorer:
    """
    Comprehensive exploration tool for RagaSense-Data database.
    """
    
    def __init__(self):
        self.data_path = Path("data/unified_ragasense_final")
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
            # Load ragas
            with open(self.data_path / "unified_ragas.json", 'r', encoding='utf-8') as f:
                self.ragas = json.load(f)
            
            # Load artists
            with open(self.data_path / "unified_artists.json", 'r', encoding='utf-8') as f:
                self.artists = json.load(f)
            
            # Load tracks
            with open(self.data_path / "unified_tracks.json", 'r', encoding='utf-8') as f:
                self.tracks = json.load(f)
            
            # Load audio files
            with open(self.data_path / "unified_audio_files.json", 'r', encoding='utf-8') as f:
                self.audio_files = json.load(f)
            
            # Load cross-tradition mappings
            with open(self.data_path / "unified_cross_tradition_mappings.json", 'r', encoding='utf-8') as f:
                self.cross_tradition_mappings = json.load(f)
            
            # Load metadata
            with open(self.data_path / "unified_metadata.json", 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            print(f"✅ Loaded {len(self.ragas)} ragas, {len(self.artists)} artists, {len(self.tracks)} tracks")
            
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
                    'song_count': raga_data.get('song_count', 0)
                })
        
        return sorted(results, key=lambda x: x['song_count'], reverse=True)
    
    def filter_by_tradition(self, tradition: str) -> List[Dict[str, Any]]:
        """Filter ragas by tradition."""
        results = []
        
        for raga_id, raga_data in self.ragas.items():
            if raga_data.get('tradition', '').lower() == tradition.lower():
                results.append({
                    'raga_id': raga_id,
                    'name': raga_data.get('name', ''),
                    'sanskrit_name': raga_data.get('sanskrit_name', ''),
                    'tradition': raga_data.get('tradition', 'Unknown'),
                    'song_count': raga_data.get('song_count', 0)
                })
        
        return sorted(results, key=lambda x: x['song_count'], reverse=True)
    
    def get_top_ragas(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get top N ragas by song count."""
        results = []
        
        for raga_id, raga_data in self.ragas.items():
            results.append({
                'raga_id': raga_id,
                'name': raga_data.get('name', ''),
                'sanskrit_name': raga_data.get('sanskrit_name', ''),
                'tradition': raga_data.get('tradition', 'Unknown'),
                'song_count': raga_data.get('song_count', 0)
            })
        
        return sorted(results, key=lambda x: x['song_count'], reverse=True)[:n]
    
    def get_raga_details(self, raga_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific raga."""
        # Search for raga by name
        for raga_id, raga_data in self.ragas.items():
            if raga_data.get('name', '').lower() == raga_name.lower():
                return raga_data
        
        return None
    
    def get_cross_tradition_mappings(self) -> List[Dict[str, Any]]:
        """Get all cross-tradition mappings."""
        results = []
        
        for mapping_id, mapping_data in self.cross_tradition_mappings.items():
            results.append({
                'mapping_id': mapping_id,
                'raga_name': mapping_data.get('raga_name', ''),
                'tradition': mapping_data.get('tradition', ''),
                'mapped_to': mapping_data.get('mapped_to', ''),
                'equivalence_type': mapping_data.get('equivalence_type', ''),
                'confidence': mapping_data.get('confidence', ''),
                'score': mapping_data.get('score', 0)
            })
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        return {
            'total_ragas': len(self.ragas),
            'total_artists': len(self.artists),
            'total_tracks': len(self.tracks),
            'total_audio_files': len(self.audio_files),
            'total_cross_tradition_mappings': len(self.cross_tradition_mappings),
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
            print(f"{i:2d}. {raga['name']} ({raga['tradition']})")
            print(f"    Sanskrit: {raga['sanskrit_name']}")
            print(f"    Songs: {raga['song_count']:,}")
            print()
    
    def print_raga_details(self, raga_data: Dict[str, Any]):
        """Print detailed raga information."""
        print(f"\n🎵 Raga Details: {raga_data.get('name', 'Unknown')}")
        print("=" * 60)
        
        for key, value in raga_data.items():
            if key == 'metadata' and isinstance(value, dict):
                print(f"{key}:")
                for meta_key, meta_value in value.items():
                    print(f"  {meta_key}: {meta_value}")
            elif key == 'cross_tradition_mapping' and isinstance(value, dict):
                print(f"{key}:")
                for map_key, map_value in value.items():
                    print(f"  {map_key}: {map_value}")
            else:
                print(f"{key}: {value}")
    
    def print_cross_tradition_mappings(self, mappings: List[Dict[str, Any]]):
        """Print cross-tradition mappings."""
        print(f"\n🔗 Cross-Tradition Mappings ({len(mappings)} total)")
        print("=" * 60)
        
        for mapping in mappings:
            print(f"• {mapping['raga_name']} ({mapping['tradition']})")
            print(f"  → {mapping['mapped_to']}")
            print(f"  Type: {mapping['equivalence_type']}, Confidence: {mapping['confidence']}")
            print(f"  Score: {mapping['score']}")
            print()
    
    def print_database_stats(self, stats: Dict[str, Any]):
        """Print database statistics."""
        print(f"\n📊 Database Statistics")
        print("=" * 60)
        
        print(f"Total Ragas: {stats['total_ragas']:,}")
        print(f"Total Artists: {stats['total_artists']:,}")
        print(f"Total Tracks: {stats['total_tracks']:,}")
        print(f"Total Audio Files: {stats['total_audio_files']:,}")
        print(f"Cross-Tradition Mappings: {stats['total_cross_tradition_mappings']:,}")
        print()
        
        print("Tradition Distribution:")
        for tradition, count in stats['tradition_distribution'].get('ragas', {}).items():
            print(f"  {tradition}: {count:,} ragas")
        print()
        
        print("Source Distribution:")
        for source, count in stats['source_distribution'].get('ragas', {}).items():
            print(f"  {source}: {count:,} ragas")
    
    def interactive_mode(self):
        """Start interactive exploration mode."""
        print("\n🎵 Welcome to RagaSense-Data Interactive Explorer!")
        print("=" * 60)
        print("Commands:")
        print("  search <query>     - Search ragas by name")
        print("  filter <tradition> - Filter by tradition")
        print("  top <n>            - Show top N ragas")
        print("  raga <name>        - Show raga details")
        print("  cross-tradition    - Show cross-tradition mappings")
        print("  stats              - Show database statistics")
        print("  help               - Show this help")
        print("  quit               - Exit")
        print()
        
        while True:
            try:
                command = input("RagaSense> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    print("Goodbye! 🎵")
                    break
                elif cmd == 'help':
                    print("Available commands: search, filter, top, raga, cross-tradition, stats, help, quit")
                elif cmd == 'search' and len(command) > 1:
                    query = ' '.join(command[1:])
                    results = self.search_ragas(query)
                    self.print_search_results(results, f"Search Results for '{query}'")
                elif cmd == 'filter' and len(command) > 1:
                    tradition = command[1]
                    results = self.filter_by_tradition(tradition)
                    self.print_search_results(results, f"Ragas in {tradition} Tradition")
                elif cmd == 'top':
                    n = int(command[1]) if len(command) > 1 else 20
                    results = self.get_top_ragas(n)
                    self.print_search_results(results, f"Top {n} Ragas by Song Count")
                elif cmd == 'raga' and len(command) > 1:
                    raga_name = ' '.join(command[1:])
                    raga_data = self.get_raga_details(raga_name)
                    if raga_data:
                        self.print_raga_details(raga_data)
                    else:
                        print(f"Raga '{raga_name}' not found.")
                elif cmd == 'cross-tradition':
                    mappings = self.get_cross_tradition_mappings()
                    self.print_cross_tradition_mappings(mappings)
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
    parser = argparse.ArgumentParser(description='RagaSense-Data Exploration Tool')
    parser.add_argument('command', nargs='?', help='Command to execute')
    parser.add_argument('args', nargs='*', help='Command arguments')
    
    args = parser.parse_args()
    
    explorer = RagaSenseExplorer()
    
    if not args.command:
        explorer.interactive_mode()
        return
    
    command = args.command.lower()
    
    if command == 'search' and args.args:
        query = ' '.join(args.args)
        results = explorer.search_ragas(query)
        explorer.print_search_results(results, f"Search Results for '{query}'")
    
    elif command == 'filter' and args.args:
        tradition = args.args[0]
        results = explorer.filter_by_tradition(tradition)
        explorer.print_search_results(results, f"Ragas in {tradition} Tradition")
    
    elif command == 'top':
        n = int(args.args[0]) if args.args else 20
        results = explorer.get_top_ragas(n)
        explorer.print_search_results(results, f"Top {n} Ragas by Song Count")
    
    elif command == 'raga' and args.args:
        raga_name = ' '.join(args.args)
        raga_data = explorer.get_raga_details(raga_name)
        if raga_data:
            explorer.print_raga_details(raga_data)
        else:
            print(f"Raga '{raga_name}' not found.")
    
    elif command == 'cross-tradition':
        mappings = explorer.get_cross_tradition_mappings()
        explorer.print_cross_tradition_mappings(mappings)
    
    elif command == 'stats':
        stats = explorer.get_database_stats()
        explorer.print_database_stats(stats)
    
    elif command == 'interactive':
        explorer.interactive_mode()
    
    else:
        print("Invalid command. Use 'python3 explore_ragasense_data.py' for interactive mode.")
        print("Available commands: search, filter, top, raga, cross-tradition, stats, interactive")

if __name__ == "__main__":
    main()
