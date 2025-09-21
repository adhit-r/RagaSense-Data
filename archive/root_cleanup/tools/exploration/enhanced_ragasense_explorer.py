#!/usr/bin/env python3
"""
Enhanced RagaSense-Data Explorer
================================

This enhanced explorer uses the updated raga database with multiple sources
and provides comprehensive exploration capabilities.

Features:
- Multi-source raga information
- Cross-tradition mappings
- Saraga track integration
- Enhanced search and filtering
- Source attribution tracking
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

class EnhancedRagaSenseExplorer:
    """
    Enhanced exploration tool for RagaSense-Data database with multiple sources.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / "data"
        
        # Try to load updated database first, fallback to original
        updated_path = self.data_path / "updated_raga_sources" / "updated_unified_ragas.json"
        original_path = self.data_path / "unified_ragasense_final" / "unified_ragas.json"
        
        if updated_path.exists():
            self.ragas_path = updated_path
            self.database_version = "updated"
        else:
            self.ragas_path = original_path
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
            # Load ragas
            with open(self.ragas_path, 'r', encoding='utf-8') as f:
                self.ragas = json.load(f)
            
            # Load other components from original location
            base_path = self.data_path / "unified_ragasense_final"
            
            with open(base_path / "unified_artists.json", 'r', encoding='utf-8') as f:
                self.artists = json.load(f)
            
            with open(base_path / "unified_tracks.json", 'r', encoding='utf-8') as f:
                self.tracks = json.load(f)
            
            with open(base_path / "unified_audio_files.json", 'r', encoding='utf-8') as f:
                self.audio_files = json.load(f)
            
            with open(base_path / "unified_cross_tradition_mappings.json", 'r', encoding='utf-8') as f:
                self.cross_tradition_mappings = json.load(f)
            
            with open(base_path / "unified_metadata.json", 'r', encoding='utf-8') as f:
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
                    'song_count': raga_data.get('song_count', 0),
                    'sources': raga_data.get('sources', []),
                    'saraga_track_count': raga_data.get('saraga_track_count', 0)
                })
        
        return sorted(results, key=lambda x: x['song_count'], reverse=True)
    
    def get_multi_source_ragas(self) -> List[Dict[str, Any]]:
        """Get ragas with multiple sources."""
        results = []
        
        for raga_id, raga_data in self.ragas.items():
            sources = raga_data.get('sources', [])
            if len(sources) > 1:
                results.append({
                    'raga_id': raga_id,
                    'name': raga_data.get('name', ''),
                    'sanskrit_name': raga_data.get('sanskrit_name', ''),
                    'tradition': raga_data.get('tradition', 'Unknown'),
                    'song_count': raga_data.get('song_count', 0),
                    'sources': sources,
                    'saraga_track_count': raga_data.get('saraga_track_count', 0),
                    'saraga_metadata': raga_data.get('saraga_metadata', {})
                })
        
        return sorted(results, key=lambda x: x['saraga_track_count'], reverse=True)
    
    def get_top_ragas(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get top N ragas by song count."""
        results = []
        
        for raga_id, raga_data in self.ragas.items():
            results.append({
                'raga_id': raga_id,
                'name': raga_data.get('name', ''),
                'sanskrit_name': raga_data.get('sanskrit_name', ''),
                'tradition': raga_data.get('tradition', 'Unknown'),
                'song_count': raga_data.get('song_count', 0),
                'sources': raga_data.get('sources', []),
                'saraga_track_count': raga_data.get('saraga_track_count', 0)
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
        multi_source_count = len([r for r in self.ragas.values() if len(r.get('sources', [])) > 1])
        saraga_ragas = len([r for r in self.ragas.values() if 'saraga' in r.get('sources', [])])
        total_saraga_tracks = sum(r.get('saraga_track_count', 0) for r in self.ragas.values())
        
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
        
        # Metadata
        metadata = raga_data.get('metadata', {})
        if metadata:
            print(f"\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
    
    def print_multi_source_ragas(self, ragas: List[Dict[str, Any]]):
        """Print multi-source ragas."""
        print(f"\n🔗 Multi-Source Ragas ({len(ragas)} total)")
        print("=" * 60)
        
        for raga in ragas:
            sources_str = ", ".join(raga['sources'])
            saraga_info = f" (+{raga['saraga_track_count']} Saraga tracks)" if raga['saraga_track_count'] > 0 else ""
            
            print(f"• {raga['name']} ({raga['tradition']})")
            print(f"  Sources: {sources_str}")
            print(f"  Songs: {raga['song_count']:,}{saraga_info}")
            
            # Show Saraga metadata if available
            saraga_meta = raga.get('saraga_metadata', {})
            if saraga_meta:
                datasets = saraga_meta.get('datasets', [])
                artists = saraga_meta.get('artists', [])
                print(f"  Saraga Datasets: {', '.join(datasets)}")
                print(f"  Saraga Artists: {', '.join(artists[:3])}{'...' if len(artists) > 3 else ''}")
            print()
    
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
        
        print("Tradition Distribution:")
        for tradition, count in stats['tradition_distribution'].get('ragas', {}).items():
            print(f"  {tradition}: {count:,} ragas")
        print()
        
        print("Source Distribution:")
        for source, count in stats['source_distribution'].get('ragas', {}).items():
            print(f"  {source}: {count:,} ragas")
    
    def interactive_mode(self):
        """Start interactive exploration mode."""
        print("\n🎵 Welcome to Enhanced RagaSense-Data Explorer!")
        print("=" * 60)
        print("Commands:")
        print("  search <query>     - Search ragas by name")
        print("  filter <tradition> - Filter by tradition")
        print("  top <n>            - Show top N ragas")
        print("  multi-source       - Show ragas with multiple sources")
        print("  raga <name>        - Show raga details")
        print("  cross-tradition    - Show cross-tradition mappings")
        print("  stats              - Show database statistics")
        print("  help               - Show this help")
        print("  quit               - Exit")
        print()
        
        while True:
            try:
                command = input("Enhanced RagaSense> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    print("Goodbye! 🎵")
                    break
                elif cmd == 'help':
                    print("Available commands: search, filter, top, multi-source, raga, cross-tradition, stats, help, quit")
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
                elif cmd == 'multi-source':
                    results = self.get_multi_source_ragas()
                    self.print_multi_source_ragas(results)
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
    parser = argparse.ArgumentParser(description='Enhanced RagaSense-Data Exploration Tool')
    parser.add_argument('command', nargs='?', help='Command to execute')
    parser.add_argument('args', nargs='*', help='Command arguments')
    
    args = parser.parse_args()
    
    explorer = EnhancedRagaSenseExplorer()
    
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
    
    elif command == 'multi-source':
        results = explorer.get_multi_source_ragas()
        explorer.print_multi_source_ragas(results)
    
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
        print("Invalid command. Use 'python3 enhanced_ragasense_explorer.py' for interactive mode.")
        print("Available commands: search, filter, top, multi-source, raga, cross-tradition, stats, interactive")

if __name__ == "__main__":
    main()

