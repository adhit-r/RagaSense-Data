#!/usr/bin/env python3
"""
RagaSense - Perfect Information Architecture
User-centered design with flawless UX
"""

from flask import Flask, render_template_string, jsonify
import json
import sys
import os

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant

app = Flask(__name__)

def create_perfect_dataset():
    """Create dataset with perfect information architecture"""
    print("🎯 Creating perfect user-focused dataset...")
    
    assistant = DataUnificationAssistant()
    ragas = []
    
    # Get all unique ragas
    processed = set()
    
    for key, entity in assistant.saraga_data.items():
        name = entity.get('original_name', key)
        if name not in processed:
            result = assistant.create_enhanced_unified_entity(name, entity.get('tradition'))
            if result['status'] == 'unified':
                raga = create_user_raga(name, result)
                ragas.append(raga)
                processed.add(name)
    
    for key, entity in assistant.ramanarunachalam_data.items():
        name = entity.get('original_name', key)
        if name not in processed:
            result = assistant.create_enhanced_unified_entity(name, entity.get('tradition'))
            if result['status'] == 'unified':
                raga = create_user_raga(name, result)
                ragas.append(raga)
                processed.add(name)
    
    # Sort by importance
    ragas.sort(key=lambda r: r['score'], reverse=True)
    
    print(f"✅ Created {len(ragas)} perfectly structured ragas")
    return ragas

def create_user_raga(name, result):
    """Create perfect user-focused raga structure"""
    data = result.get('merged_data', {})
    
    # What matters to users
    has_audio = bool(data.get('recording_info'))
    has_videos = len(data.get('youtube_videos', [])) > 0
    has_theory = bool(data.get('arohana') and data.get('avarohana'))
    
    # Priority score for sorting
    score = 0
    if has_audio: score += 100
    if has_videos: score += len(data.get('youtube_videos', [])) * 0.5
    if has_theory: score += 50
    
    return {
        'name': name,
        'tradition': result.get('tradition', 'unknown'),
        
        # Essential info users need first
        'summary': {
            'tradition': result.get('tradition', 'unknown').title(),
            'has_audio': has_audio,
            'has_videos': has_videos,
            'has_theory': has_theory,
            'video_count': len(data.get('youtube_videos', [])),
        },
        
        # Musical knowledge
        'music': {
            'arohana': data.get('arohana', ''),
            'avarohana': data.get('avarohana', ''),
            'melakartha': data.get('melakartha', ''),
        },
        
        # Audio experience
        'audio': {
            'available': has_audio,
            'artist': data.get('artist', ''),
            'form': data.get('form', ''),
        },
        
        # Video experience
        'videos': data.get('youtube_videos', [])[:5],  # Top 5
        
        # Trust indicators
        'quality': {
            'confidence': int(result.get('confidence', 0) * 100),
            'sources': get_sources(result),
        },
        
        'score': score
    }

def get_sources(result):
    """Get source info"""
    sources = []
    if result.get('saraga_found'):
        sources.append('Saraga Research')
    if result.get('ramanarunachalam_found'):
        sources.append('Classical Archives')
    return sources

# Store data
ragas_data = []

@app.route('/')
def index():
    """Perfect UI with flawless information architecture"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎵 RagaSense - Indian Classical Ragas</title>
    <style>
        * { 
            margin: 0; padding: 0; box-sizing: border-box; 
        }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc; color: #1e293b; line-height: 1.5;
        }
        
        /* Header - Clean, focused */
        .header {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white; padding: 4rem 2rem; text-align: center;
        }
        
        .header h1 { 
            font-size: 3rem; font-weight: 800; margin-bottom: 1rem; 
        }
        
        .header p { 
            font-size: 1.25rem; opacity: 0.9; max-width: 600px; margin: 0 auto;
        }
        
        /* Search - Primary action */
        .search {
            background: white; padding: 2rem; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            position: sticky; top: 0; z-index: 50;
        }
        
        .search-container {
            max-width: 1200px; margin: 0 auto;
            display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
        }
        
        .search-input {
            flex: 1; min-width: 300px; padding: 1rem 1.5rem;
            border: 2px solid #e2e8f0; border-radius: 50px;
            font-size: 1rem; transition: all 0.3s;
        }
        
        .search-input:focus {
            outline: none; border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        .filters {
            display: flex; gap: 0.5rem; flex-wrap: wrap;
        }
        
        .filter {
            padding: 0.5rem 1rem; background: #f1f5f9;
            border: 1px solid #e2e8f0; border-radius: 1.5rem;
            cursor: pointer; transition: all 0.2s; font-size: 0.875rem;
        }
        
        .filter:hover, .filter.active {
            background: #3b82f6; color: white; border-color: #3b82f6;
        }
        
        /* Stats */
        .stats {
            background: #f8fafc; padding: 1.5rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .stats-grid {
            max-width: 1200px; margin: 0 auto;
            display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap;
        }
        
        .stat {
            text-align: center;
        }
        
        .stat-number {
            font-size: 2rem; font-weight: 700; color: #3b82f6;
        }
        
        .stat-label {
            font-size: 0.875rem; color: #64748b; margin-top: 0.25rem;
        }
        
        /* Main content */
        .main {
            max-width: 1400px; margin: 0 auto; padding: 2rem;
        }
        
        /* Perfect raga cards */
        .grid {
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 1.5rem;
        }
        
        .card {
            background: white; border-radius: 1rem; padding: 1.5rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            transition: all 0.3s; border: 1px solid #f1f5f9;
            position: relative;
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }
        
        .card.carnatic::before {
            content: ''; position: absolute; top: 0; left: 0;
            width: 4px; height: 100%; background: #ef4444; border-radius: 1rem 0 0 1rem;
        }
        
        .card.hindustani::before {
            content: ''; position: absolute; top: 0; left: 0;
            width: 4px; height: 100%; background: #3b82f6; border-radius: 1rem 0 0 1rem;
        }
        
        /* Card content hierarchy */
        .card-header {
            margin-bottom: 1rem;
        }
        
        .raga-name {
            font-size: 1.5rem; font-weight: 700; color: #1e293b;
            margin-bottom: 0.5rem;
        }
        
        .raga-info {
            display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
        }
        
        .tradition {
            padding: 0.25rem 0.75rem; border-radius: 0.75rem;
            font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        }
        
        .tradition.carnatic { background: #fef2f2; color: #dc2626; }
        .tradition.hindustani { background: #eff6ff; color: #2563eb; }
        
        /* Theory section */
        .theory {
            background: linear-gradient(135deg, #f8faff, #fff8f8);
            padding: 1rem; border-radius: 0.75rem; margin: 1rem 0;
        }
        
        .theory-title {
            font-weight: 600; color: #374151; margin-bottom: 0.5rem;
            font-size: 0.875rem;
        }
        
        .scale {
            font-family: 'SF Mono', monospace; font-size: 0.875rem;
            color: #1f2937; margin: 0.25rem 0;
        }
        
        /* Experience tags */
        .experiences {
            display: flex; gap: 0.5rem; margin: 1rem 0; flex-wrap: wrap;
        }
        
        .exp {
            display: flex; align-items: center; gap: 0.375rem;
            padding: 0.375rem 0.75rem; border-radius: 1rem;
            font-size: 0.75rem; font-weight: 500;
        }
        
        .exp-audio { background: #f0fdf4; color: #16a34a; }
        .exp-video { background: #fef3c7; color: #d97706; }
        .exp-theory { background: #eff6ff; color: #2563eb; }
        
        /* Trust footer */
        .trust {
            display: flex; justify-content: space-between; align-items: center;
            margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #f1f5f9;
        }
        
        .confidence {
            display: flex; align-items: center; gap: 0.5rem;
        }
        
        .confidence-dot {
            width: 0.75rem; height: 0.75rem; border-radius: 50%;
        }
        
        .confidence-high { background: #16a34a; }
        .confidence-medium { background: #d97706; }
        .confidence-low { background: #dc2626; }
        
        .sources {
            font-size: 0.75rem; color: #64748b;
        }
        
        /* Empty state */
        .empty {
            text-align: center; padding: 4rem 2rem; color: #64748b;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .header h1 { font-size: 2rem; }
            .search-container { flex-direction: column; }
            .search-input { min-width: auto; }
            .grid { grid-template-columns: 1fr; }
            .stats-grid { gap: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 RagaSense</h1>
        <p>Discover the beauty of Indian classical ragas with authentic recordings and complete musical theory</p>
    </div>

    <div class="search">
        <div class="search-container">
            <input type="text" class="search-input" id="searchInput" 
                   placeholder="Search ragas..." oninput="search()">
            
            <div class="filters">
                <div class="filter active" onclick="filter('all')">All</div>
                <div class="filter" onclick="filter('carnatic')">Carnatic</div>
                <div class="filter" onclick="filter('hindustani')">Hindustani</div>
                <div class="filter" onclick="filter('audio')">Audio</div>
                <div class="filter" onclick="filter('video')">Videos</div>
                <div class="filter" onclick="filter('theory')">Theory</div>
            </div>
        </div>
    </div>

    <div class="stats">
        <div class="stats-grid" id="stats">
        </div>
    </div>

    <div class="main">
        <div class="grid" id="grid">
        </div>
        
        <div class="empty" id="empty" style="display: none;">
            <h3>No ragas found</h3>
            <p>Try a different search or filter</p>
        </div>
    </div>

    <script>
        let ragas = [];
        let filtered = [];
        let currentFilter = 'all';
        let searchTerm = '';

        async function loadData() {
            const response = await fetch('/api/ragas');
            ragas = await response.json();
            filtered = [...ragas];
            updateStats();
            render();
        }

        function updateStats() {
            const stats = {
                total: ragas.length,
                carnatic: ragas.filter(r => r.tradition === 'carnatic').length,
                hindustani: ragas.filter(r => r.tradition === 'hindustani').length,
                audio: ragas.filter(r => r.summary.has_audio).length,
                videos: ragas.filter(r => r.summary.has_videos).length
            };

            document.getElementById('stats').innerHTML = `
                <div class="stat">
                    <div class="stat-number">${stats.total}</div>
                    <div class="stat-label">Total Ragas</div>
                </div>
                <div class="stat">
                    <div class="stat-number">${stats.carnatic}</div>
                    <div class="stat-label">Carnatic</div>
                </div>
                <div class="stat">
                    <div class="stat-number">${stats.hindustani}</div>
                    <div class="stat-label">Hindustani</div>
                </div>
                <div class="stat">
                    <div class="stat-number">${stats.audio}</div>
                    <div class="stat-label">Audio</div>
                </div>
                <div class="stat">
                    <div class="stat-number">${stats.videos}</div>
                    <div class="stat-label">Videos</div>
                </div>
            `;
        }

        function render() {
            const grid = document.getElementById('grid');
            const empty = document.getElementById('empty');
            
            if (filtered.length === 0) {
                grid.style.display = 'none';
                empty.style.display = 'block';
                return;
            }
            
            grid.style.display = 'grid';
            empty.style.display = 'none';
            
            grid.innerHTML = filtered.map(raga => `
                <div class="card ${raga.tradition}">
                    <div class="card-header">
                        <div class="raga-name">${raga.name}</div>
                        <div class="raga-info">
                            <div class="tradition ${raga.tradition}">${raga.summary.tradition}</div>
                        </div>
                    </div>
                    
                    ${raga.music.arohana || raga.music.avarohana ? `
                        <div class="theory">
                            <div class="theory-title">🎼 Musical Structure</div>
                            ${raga.music.arohana ? `<div class="scale">↗ ${raga.music.arohana}</div>` : ''}
                            ${raga.music.avarohana ? `<div class="scale">↘ ${raga.music.avarohana}</div>` : ''}
                            ${raga.music.melakartha ? `<div style="margin-top: 0.5rem; font-size: 0.75rem; color: #6b7280;">Melakartha ${raga.music.melakartha}</div>` : ''}
                        </div>
                    ` : ''}
                    
                    <div class="experiences">
                        ${raga.summary.has_audio ? `<div class="exp exp-audio">🎵 Audio</div>` : ''}
                        ${raga.summary.has_videos ? `<div class="exp exp-video">📺 ${raga.summary.video_count} Videos</div>` : ''}
                        ${raga.summary.has_theory ? `<div class="exp exp-theory">📖 Theory</div>` : ''}
                    </div>
                    
                    ${raga.audio.available ? `
                        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 0.5rem; margin: 0.75rem 0;">
                            <div style="font-size: 0.875rem; color: #374151;">
                                <strong>Recording:</strong> ${raga.audio.artist} • ${raga.audio.form}
                            </div>
                        </div>
                    ` : ''}
                    
                    <div class="trust">
                        <div class="confidence">
                            <div class="confidence-dot confidence-${getConfidenceLevel(raga.quality.confidence)}"></div>
                            <span style="font-size: 0.875rem; font-weight: 500;">${raga.quality.confidence}% verified</span>
                        </div>
                        <div class="sources">${raga.quality.sources.join(' • ')}</div>
                    </div>
                </div>
            `).join('');
        }

        function getConfidenceLevel(conf) {
            if (conf >= 80) return 'high';
            if (conf >= 60) return 'medium';
            return 'low';
        }

        function applyFilters() {
            filtered = ragas.filter(raga => {
                if (currentFilter === 'carnatic' && raga.tradition !== 'carnatic') return false;
                if (currentFilter === 'hindustani' && raga.tradition !== 'hindustani') return false;
                if (currentFilter === 'audio' && !raga.summary.has_audio) return false;
                if (currentFilter === 'video' && !raga.summary.has_videos) return false;
                if (currentFilter === 'theory' && !raga.summary.has_theory) return false;
                if (searchTerm && !raga.name.toLowerCase().includes(searchTerm)) return false;
                return true;
            });
            render();
        }

        function filter(type) {
            currentFilter = type;
            document.querySelectorAll('.filter').forEach(f => f.classList.remove('active'));
            event.target.classList.add('active');
            applyFilters();
        }

        function search() {
            searchTerm = document.getElementById('searchInput').value.toLowerCase();
            applyFilters();
        }

        loadData();
    </script>
</body>
</html>
    ''')

@app.route('/api/ragas')
def api_ragas():
    """API endpoint for raga data"""
    return jsonify(ragas_data)

def main():
    """Start the perfect RagaSense explorer"""
    global ragas_data
    
    print("🎵 RagaSense - Perfect Information Architecture")
    print("=" * 60)
    
    ragas_data = create_perfect_dataset()
    
    # Stats
    total = len(ragas_data)
    carnatic = len([r for r in ragas_data if r['tradition'] == 'carnatic'])
    hindustani = len([r for r in ragas_data if r['tradition'] == 'hindustani'])
    with_audio = len([r for r in ragas_data if r['summary']['has_audio']])
    with_videos = len([r for r in ragas_data if r['summary']['has_videos']])
    
    print(f"✅ Perfect dataset ready:")
    print(f"   📊 Total: {total} ragas")
    print(f"   🎼 Carnatic: {carnatic}")
    print(f"   🎵 Hindustani: {hindustani}")
    print(f"   🎶 Audio: {with_audio}")
    print(f"   📺 Videos: {with_videos}")
    
    print(f"\n🌐 Perfect UX running at: http://localhost:5006")
    print("\n🎯 Features:")
    print("   ✅ Flawless information architecture")
    print("   ✅ User-centered design")
    print("   ✅ Perfect visual hierarchy")
    print("   ✅ Instant search and filtering")
    print("   ✅ Trust indicators")
    print("   ✅ Mobile-responsive")
    
    app.run(host='0.0.0.0', port=5006, debug=False)

if __name__ == "__main__":
    main()