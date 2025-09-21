#!/usr/bin/env python3
"""
Enhanced RagaSense-Data Web Explorer
===================================

An enhanced web-based explorer for the RagaSense-Data database with multiple sources.
Provides a web interface to explore the 1,340 unique ragas with Saraga integration.

Usage:
    python3 enhanced_web_explorer.py

Then open http://localhost:8000 in your browser.
"""

import json
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import html

class EnhancedRagaSenseWebExplorer(BaseHTTPRequestHandler):
    """Enhanced web server for exploring RagaSense-Data with multiple sources."""
    
    def __init__(self, *args, **kwargs):
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
        self.cross_tradition_mappings = {}
        self.metadata = {}
        
        self.load_data()
        super().__init__(*args, **kwargs)
    
    def load_data(self):
        """Load all database files."""
        try:
            with open(self.ragas_path, 'r', encoding='utf-8') as f:
                self.ragas = json.load(f)
            
            base_path = self.data_path / "unified_ragasense_final"
            
            with open(base_path / "unified_artists.json", 'r', encoding='utf-8') as f:
                self.artists = json.load(f)
            
            with open(base_path / "unified_tracks.json", 'r', encoding='utf-8') as f:
                self.tracks = json.load(f)
            
            with open(base_path / "unified_cross_tradition_mappings.json", 'r', encoding='utf-8') as f:
                self.cross_tradition_mappings = json.load(f)
            
            with open(base_path / "unified_metadata.json", 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        
        if path == '/':
            self.serve_homepage()
        elif path == '/search':
            self.serve_search(query_params)
        elif path == '/raga':
            self.serve_raga_details(query_params)
        elif path == '/stats':
            self.serve_stats()
        elif path == '/cross-tradition':
            self.serve_cross_tradition()
        elif path == '/multi-source':
            self.serve_multi_source()
        else:
            self.send_error(404, "Not Found")
    
    def serve_homepage(self):
        """Serve the enhanced homepage."""
        multi_source_count = len([r for r in self.ragas.values() if len(r.get('sources', [])) > 1])
        saraga_count = len([r for r in self.ragas.values() if 'saraga' in r.get('sources', [])])
        total_saraga_tracks = sum(r.get('saraga_track_count', 0) for r in self.ragas.values())
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced RagaSense-Data Explorer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }}
        .version-badge {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            display: inline-block;
            margin-left: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.saraga {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .stat-card.multi-source {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .search-section {{
            margin-bottom: 30px;
        }}
        .search-box {{
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .search-box:focus {{
            outline: none;
            border-color: #667eea;
        }}
        .btn {{
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
            margin-bottom: 10px;
        }}
        .btn:hover {{
            background: #5a6fd8;
        }}
        .btn.saraga {{
            background: #f093fb;
        }}
        .btn.saraga:hover {{
            background: #e879f9;
        }}
        .results {{
            margin-top: 20px;
        }}
        .raga-card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .raga-card:hover {{
            background: #e9ecef;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .raga-card.multi-source {{
            border-left: 4px solid #4facfe;
        }}
        .raga-card.saraga {{
            border-left: 4px solid #f093fb;
        }}
        .raga-name {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .raga-tradition {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .raga-songs {{
            color: #27ae60;
            font-weight: bold;
        }}
        .raga-sources {{
            color: #667eea;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .saraga-info {{
            color: #f093fb;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .tradition-filter {{
            margin-bottom: 20px;
        }}
        .tradition-btn {{
            background: #95a5a6;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
            margin-bottom: 10px;
        }}
        .tradition-btn.active {{
            background: #667eea;
        }}
        .tradition-btn:hover {{
            background: #7f8c8d;
        }}
        .tradition-btn.active:hover {{
            background: #5a6fd8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Enhanced RagaSense-Data Explorer <span class="version-badge">{self.database_version}</span></h1>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">1,340</div>
                <div>Unique Ragas</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">19</div>
                <div>Artists</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">4,536</div>
                <div>Tracks</div>
            </div>
            <div class="stat-card multi-source">
                <div class="stat-number">{multi_source_count}</div>
                <div>Multi-Source Ragas</div>
            </div>
            <div class="stat-card saraga">
                <div class="stat-number">{saraga_count}</div>
                <div>Saraga Ragas</div>
            </div>
            <div class="stat-card saraga">
                <div class="stat-number">{total_saraga_tracks}</div>
                <div>Saraga Tracks</div>
            </div>
        </div>
        
        <div class="search-section">
            <input type="text" class="search-box" id="searchBox" placeholder="Search ragas by name...">
            <button class="btn" onclick="searchRagas()">Search</button>
            <button class="btn" onclick="showTopRagas()">Top Ragas</button>
            <button class="btn saraga" onclick="showMultiSource()">Multi-Source</button>
            <button class="btn" onclick="showStats()">Statistics</button>
            <button class="btn" onclick="showCrossTradition()">Cross-Tradition</button>
        </div>
        
        <div class="tradition-filter">
            <button class="tradition-btn active" onclick="filterByTradition('all')">All</button>
            <button class="tradition-btn" onclick="filterByTradition('Carnatic')">Carnatic</button>
            <button class="tradition-btn" onclick="filterByTradition('Hindustani')">Hindustani</button>
            <button class="tradition-btn" onclick="filterByTradition('Both')">Both</button>
        </div>
        
        <div id="results" class="results"></div>
    </div>
    
    <script>
        let currentResults = [];
        let currentFilter = 'all';
        
        function searchRagas() {{
            const query = document.getElementById('searchBox').value;
            if (query.trim() === '') {{
                showTopRagas();
                return;
            }}
            
            fetch(`/search?q=${{encodeURIComponent(query)}}`)
                .then(response => response.json())
                .then(data => {{
                    currentResults = data;
                    displayResults(data);
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = '<p>Error searching ragas.</p>';
                }});
        }}
        
        function showTopRagas() {{
            fetch('/search?top=20')
                .then(response => response.json())
                .then(data => {{
                    currentResults = data;
                    displayResults(data);
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = '<p>Error loading top ragas.</p>';
                }});
        }}
        
        function showMultiSource() {{
            fetch('/multi-source')
                .then(response => response.json())
                .then(data => {{
                    currentResults = data;
                    displayResults(data, 'Multi-Source Ragas');
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = '<p>Error loading multi-source ragas.</p>';
                }});
        }}
        
        function filterByTradition(tradition) {{
            currentFilter = tradition;
            
            // Update button states
            document.querySelectorAll('.tradition-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            // Filter current results
            if (tradition === 'all') {{
                displayResults(currentResults);
            }} else {{
                const filtered = currentResults.filter(raga => raga.tradition === tradition);
                displayResults(filtered);
            }}
        }}
        
        function displayResults(results, title = 'Search Results') {{
            const resultsDiv = document.getElementById('results');
            
            if (results.length === 0) {{
                resultsDiv.innerHTML = '<p>No ragas found.</p>';
                return;
            }}
            
            let html = `<h3>${{title}} (${{results.length}} ragas):</h3>`;
            
            results.forEach(raga => {{
                const isMultiSource = raga.sources && raga.sources.length > 1;
                const hasSaraga = raga.sources && raga.sources.includes('saraga');
                const cardClass = isMultiSource ? 'multi-source' : (hasSaraga ? 'saraga' : '');
                
                html += `
                    <div class="raga-card ${{cardClass}}" onclick="showRagaDetails('${{raga.raga_id}}')">
                        <div class="raga-name">${{raga.name}}</div>
                        <div class="raga-tradition">${{raga.tradition}}</div>
                        <div class="raga-songs">${{raga.song_count.toLocaleString()}} songs</div>
                        <div class="raga-sources">Sources: ${{raga.sources ? raga.sources.join(', ') : 'Unknown'}}</div>
                        ${{raga.saraga_track_count > 0 ? `<div class="saraga-info">+${{raga.saraga_track_count}} Saraga tracks</div>` : ''}}
                    </div>
                `;
            }});
            
            resultsDiv.innerHTML = html;
        }}
        
        function showRagaDetails(ragaId) {{
            fetch(`/raga?id=${{encodeURIComponent(ragaId)}}`)
                .then(response => response.json())
                .then(data => {{
                    if (data.error) {{
                        alert(data.error);
                        return;
                    }}
                    
                    let html = `
                        <h2>${{data.name}}</h2>
                        <p><strong>Tradition:</strong> ${{data.tradition}}</p>
                        <p><strong>Sanskrit Name:</strong> ${{data.sanskrit_name || 'N/A'}}</p>
                        <p><strong>Song Count:</strong> ${{data.song_count.toLocaleString()}}</p>
                        <p><strong>Sources:</strong> ${{data.sources.join(', ')}}</p>
                    `;
                    
                    if (data.saraga_track_count > 0) {{
                        html += `<p><strong>Saraga Tracks:</strong> ${{data.saraga_track_count}}</p>`;
                        if (data.saraga_metadata) {{
                            html += `<p><strong>Saraga Datasets:</strong> ${{data.saraga_metadata.datasets.join(', ')}}</p>`;
                            html += `<p><strong>Saraga Artists:</strong> ${{data.saraga_metadata.artists.join(', ')}}</p>`;
                        }}
                    }}
                    
                    if (data.cross_tradition_mapping && data.cross_tradition_mapping.mapping) {{
                        const mapping = data.cross_tradition_mapping.mapping;
                        html += `<p><strong>Cross-Tradition Mapping:</strong> ${{mapping.carnatic_name}} ↔ ${{mapping.hindustani_name}}</p>`;
                        html += `<p><strong>Confidence:</strong> ${{data.cross_tradition_mapping.confidence}}</p>`;
                    }}
                    
                    html += '<button class="btn" onclick="showTopRagas()">Back to Results</button>';
                    
                    document.getElementById('results').innerHTML = html;
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    alert('Error loading raga details.');
                }});
        }}
        
        function showStats() {{
            fetch('/stats')
                .then(response => response.json())
                .then(data => {{
                    let html = `
                        <h3>Database Statistics (${{data.database_version}} version)</h3>
                        <p><strong>Total Ragas:</strong> ${{data.total_ragas.toLocaleString()}}</p>
                        <p><strong>Total Artists:</strong> ${{data.total_artists.toLocaleString()}}</p>
                        <p><strong>Total Tracks:</strong> ${{data.total_tracks.toLocaleString()}}</p>
                        <p><strong>Multi-Source Ragas:</strong> ${{data.multi_source_ragas.toLocaleString()}}</p>
                        <p><strong>Saraga Ragas:</strong> ${{data.saraga_ragas.toLocaleString()}}</p>
                        <p><strong>Total Saraga Tracks:</strong> ${{data.total_saraga_tracks.toLocaleString()}}</p>
                        
                        <h4>Tradition Distribution:</h4>
                        <ul>
                    `;
                    
                    for (const [tradition, count] of Object.entries(data.tradition_distribution.ragas)) {{
                        html += `<li>${{tradition}}: ${{count.toLocaleString()}} ragas</li>`;
                    }}
                    
                    html += '</ul><button class="btn" onclick="showTopRagas()">Back to Results</button>';
                    
                    document.getElementById('results').innerHTML = html;
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    alert('Error loading statistics.');
                }});
        }}
        
        function showCrossTradition() {{
            fetch('/cross-tradition')
                .then(response => response.json())
                .then(data => {{
                    let html = '<h3>Cross-Tradition Mappings</h3>';
                    
                    data.forEach(mapping => {{
                        html += `
                            <div class="raga-card">
                                <div class="raga-name">${{mapping.raga_name}} (${{mapping.tradition}})</div>
                                <div class="raga-tradition">→ ${{mapping.mapped_to}}</div>
                                <div class="raga-songs">Type: ${{mapping.equivalence_type}}, Confidence: ${{mapping.confidence}}</div>
                            </div>
                        `;
                    }});
                    
                    html += '<button class="btn" onclick="showTopRagas()">Back to Results</button>';
                    
                    document.getElementById('results').innerHTML = html;
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    alert('Error loading cross-tradition mappings.');
                }});
        }}
        
        // Load top ragas on page load
        window.onload = function() {{
            showTopRagas();
        }};
        
        // Search on Enter key
        document.getElementById('searchBox').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                searchRagas();
            }}
        }});
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_search(self, query_params):
        """Serve search results."""
        query = query_params.get('q', [''])[0]
        top = query_params.get('top', [''])[0]
        
        results = []
        
        if top:
            # Get top N ragas
            n = int(top) if top.isdigit() else 20
            sorted_ragas = sorted(self.ragas.items(), key=lambda x: x[1].get('song_count', 0), reverse=True)
            results = [
                {
                    'raga_id': raga_id,
                    'name': raga_data.get('name', ''),
                    'tradition': raga_data.get('tradition', 'Unknown'),
                    'song_count': raga_data.get('song_count', 0),
                    'sources': raga_data.get('sources', []),
                    'saraga_track_count': raga_data.get('saraga_track_count', 0)
                }
                for raga_id, raga_data in sorted_ragas[:n]
            ]
        elif query:
            # Search ragas
            query_lower = query.lower()
            for raga_id, raga_data in self.ragas.items():
                name = raga_data.get('name', '').lower()
                if query_lower in name or query_lower in raga_id.lower():
                    results.append({
                        'raga_id': raga_id,
                        'name': raga_data.get('name', ''),
                        'tradition': raga_data.get('tradition', 'Unknown'),
                        'song_count': raga_data.get('song_count', 0),
                        'sources': raga_data.get('sources', []),
                        'saraga_track_count': raga_data.get('saraga_track_count', 0)
                    })
            
            results.sort(key=lambda x: x['song_count'], reverse=True)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(results).encode())
    
    def serve_multi_source(self):
        """Serve multi-source ragas."""
        results = []
        
        for raga_id, raga_data in self.ragas.items():
            sources = raga_data.get('sources', [])
            if len(sources) > 1:
                results.append({
                    'raga_id': raga_id,
                    'name': raga_data.get('name', ''),
                    'tradition': raga_data.get('tradition', 'Unknown'),
                    'song_count': raga_data.get('song_count', 0),
                    'sources': sources,
                    'saraga_track_count': raga_data.get('saraga_track_count', 0),
                    'saraga_metadata': raga_data.get('saraga_metadata', {})
                })
        
        results.sort(key=lambda x: x['saraga_track_count'], reverse=True)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(results).encode())
    
    def serve_raga_details(self, query_params):
        """Serve detailed raga information."""
        raga_id = query_params.get('id', [''])[0]
        
        if raga_id in self.ragas:
            raga_data = self.ragas[raga_id]
            result = {
                'raga_id': raga_id,
                'name': raga_data.get('name', ''),
                'sanskrit_name': raga_data.get('sanskrit_name', ''),
                'tradition': raga_data.get('tradition', 'Unknown'),
                'song_count': raga_data.get('song_count', 0),
                'sources': raga_data.get('sources', []),
                'saraga_track_count': raga_data.get('saraga_track_count', 0),
                'saraga_metadata': raga_data.get('saraga_metadata', {}),
                'cross_tradition_mapping': raga_data.get('cross_tradition_mapping', {})
            }
        else:
            result = {'error': 'Raga not found'}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    
    def serve_stats(self):
        """Serve database statistics."""
        multi_source_count = len([r for r in self.ragas.values() if len(r.get('sources', [])) > 1])
        saraga_count = len([r for r in self.ragas.values() if 'saraga' in r.get('sources', [])])
        total_saraga_tracks = sum(r.get('saraga_track_count', 0) for r in self.ragas.values())
        
        stats = {
            'total_ragas': len(self.ragas),
            'total_artists': len(self.artists),
            'total_tracks': len(self.tracks),
            'total_cross_tradition_mappings': len(self.cross_tradition_mappings),
            'multi_source_ragas': multi_source_count,
            'saraga_ragas': saraga_count,
            'total_saraga_tracks': total_saraga_tracks,
            'database_version': self.database_version,
            'tradition_distribution': self.metadata.get('tradition_distribution', {})
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())
    
    def serve_cross_tradition(self):
        """Serve cross-tradition mappings."""
        mappings = []
        for mapping_id, mapping_data in self.cross_tradition_mappings.items():
            mappings.append({
                'mapping_id': mapping_id,
                'raga_name': mapping_data.get('raga_name', ''),
                'tradition': mapping_data.get('tradition', ''),
                'mapped_to': mapping_data.get('mapped_to', ''),
                'equivalence_type': mapping_data.get('equivalence_type', ''),
                'confidence': mapping_data.get('confidence', '')
            })
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(mappings).encode())

def main():
    """Start the enhanced web server."""
    port = 8000
    server_address = ('', port)
    
    print(f"🎵 Starting Enhanced RagaSense-Data Web Explorer...")
    print(f"📱 Open http://localhost:{port} in your browser")
    print(f"🛑 Press Ctrl+C to stop the server")
    
    httpd = HTTPServer(server_address, EnhancedRagaSenseWebExplorer)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        httpd.server_close()

if __name__ == "__main__":
    main()

