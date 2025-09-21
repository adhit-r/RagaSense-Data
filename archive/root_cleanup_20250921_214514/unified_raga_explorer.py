#!/usr/bin/env python3
"""
Unified Raga Explorer - Modern UI/UX
One raga = One card with complete source attribution
"""

from flask import Flask, render_template_string, jsonify, request
import json
import sys
import os

# Add the scripts/utilities directory to Python path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts', 'utilities')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from data_unification_assistant import DataUnificationAssistant

app = Flask(__name__)

def create_unified_raga_dataset():
    """Create unified dataset with one entry per raga"""
    print("🔄 Creating unified raga dataset...")
    
    assistant = DataUnificationAssistant()
    unified_ragas = {}
    
    # Process all ragas and unify them
    processed_ragas = set()
    
    # First pass: Collect all unique raga names
    all_raga_names = set()
    for key, entity in assistant.saraga_data.items():
        all_raga_names.add(entity.get('original_name', key))
    for key, entity in assistant.ramanarunachalam_data.items():
        all_raga_names.add(entity.get('original_name', key))
    
    # Second pass: Create unified entries
    for raga_name in all_raga_names:
        if raga_name in processed_ragas:
            continue
            
        # Find in both datasets
        saraga_match = None
        rama_match = None
        tradition = 'unknown'
        
        # Find in Saraga
        for key, entity in assistant.saraga_data.items():
            if entity.get('original_name', key) == raga_name or assistant.normalize_name(entity.get('original_name', key)) == assistant.normalize_name(raga_name):
                saraga_match = entity
                tradition = entity.get('tradition', 'unknown')
                break
        
        # Find in Ramanarunachalam  
        for key, entity in assistant.ramanarunachalam_data.items():
            if entity.get('original_name', key) == raga_name or assistant.normalize_name(entity.get('original_name', key)) == assistant.normalize_name(raga_name):
                rama_match = entity
                if tradition == 'unknown':
                    tradition = entity.get('tradition', 'unknown')
                break
        
        # Use enhanced unification
        if saraga_match or rama_match:
            result = assistant.create_enhanced_unified_entity(raga_name, tradition)
            
            if result['status'] == 'unified':
                unified_raga = create_unified_raga_entry(raga_name, result, saraga_match, rama_match)
                unified_ragas[raga_name] = unified_raga
                processed_ragas.add(raga_name)
    
    print(f"✅ Created {len(unified_ragas)} unified raga entries")
    return unified_ragas

def create_unified_raga_entry(raga_name, unification_result, saraga_data, rama_data):
    """Create a unified raga entry with complete source attribution"""
    merged_data = unification_result.get('merged_data', {})
    
    # Extract data from each source
    saraga_contribution = {}
    rama_contribution = {}
    
    if saraga_data:
        saraga_raw = saraga_data.get('data', {})
        saraga_contribution = {
            'available': True,
            'audio_file': saraga_raw.get('audio_file', ''),
            'metadata_file': saraga_raw.get('metadata_file', ''),
            'artist': saraga_raw.get('artist', ''),
            'form': saraga_raw.get('form', ''),
            'taala': saraga_raw.get('taala', ''),
            'duration': saraga_raw.get('duration', ''),
            'recording_info': saraga_raw.get('recording_info', {}),
            'album': saraga_raw.get('album', ''),
            'tonic': saraga_raw.get('tonic', '')
        }
    else:
        saraga_contribution = {'available': False}
    
    if rama_data:
        rama_raw = rama_data.get('data', {})
        videos = merged_data.get('youtube_videos', [])
        rama_contribution = {
            'available': True,
            'arohana': rama_raw.get('arohana', ''),
            'avarohana': rama_raw.get('avarohana', ''),
            'melakartha': rama_raw.get('melakartha', ''),
            'songs_count': len(rama_raw.get('songs', [])),
            'videos_count': len(videos),
            'youtube_videos': videos,
            'stats': rama_raw.get('stats', [])
        }
    else:
        rama_contribution = {'available': False}
    
    # Create unified entry
    unified_entry = {
        'name': raga_name,
        'normalized_name': unification_result.get('normalized_name', ''),
        'tradition': unification_result.get('tradition', 'unknown'),
        'confidence': unification_result.get('confidence', 0.0),
        'confidence_reason': unification_result.get('confidence_reason', ''),
        
        # Source availability
        'sources': {
            'saraga': saraga_contribution,
            'ramanarunachalam': rama_contribution
        },
        
        # Unified musical theory (primarily from Ramanarunachalam)
        'musical_theory': {
            'arohana': merged_data.get('arohana', ''),
            'avarohana': merged_data.get('avarohana', ''),
            'melakartha': merged_data.get('melakartha', ''),
            'available': bool(merged_data.get('arohana') or merged_data.get('avarohana') or merged_data.get('melakartha'))
        },
        
        # Audio and multimedia
        'multimedia': {
            'has_audio': bool(saraga_contribution.get('audio_file') or saraga_contribution.get('metadata_file')),
            'has_videos': bool(rama_contribution.get('videos_count', 0) > 0),
            'video_count': rama_contribution.get('videos_count', 0),
            'sample_videos': (rama_contribution.get('youtube_videos') or [])[:3]  # First 3 videos
        },
        
        # Performance details (from Saraga)
        'performance': {
            'artist': saraga_contribution.get('artist', ''),
            'form': saraga_contribution.get('form', ''),
            'taala': saraga_contribution.get('taala', ''),
            'duration': saraga_contribution.get('duration', ''),
            'album': saraga_contribution.get('album', '')
        },
        
        # Quality indicators
        'data_quality': {
            'completeness': calculate_completeness_score(saraga_contribution, rama_contribution),
            'has_theory': rama_contribution.get('available', False),
            'has_audio': saraga_contribution.get('available', False),
            'has_videos': rama_contribution.get('videos_count', 0) > 0
        }
    }
    
    return unified_entry

def calculate_completeness_score(saraga_data, rama_data):
    """Calculate how complete the raga data is (0-100)"""
    score = 0
    
    # Saraga contributions (40 points max)
    if saraga_data.get('available'):
        score += 20  # Base availability
        if saraga_data.get('audio_file'): score += 5
        if saraga_data.get('artist'): score += 5  
        if saraga_data.get('form'): score += 5
        if saraga_data.get('recording_info'): score += 5
    
    # Ramanarunachalam contributions (60 points max) 
    if rama_data.get('available'):
        score += 20  # Base availability
        if rama_data.get('arohana'): score += 10
        if rama_data.get('avarohana'): score += 10
        if rama_data.get('melakartha'): score += 10
        if rama_data.get('videos_count', 0) > 0: score += 10
    
    return min(score, 100)

# Global data store
unified_data = []

@app.route('/')
def index():
    """Main page with modern UI"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎵 Unified Raga Explorer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);
            border-radius: 16px; padding: 30px; margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.5em; margin-bottom: 10px;
        }
        
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px; margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.9); padding: 20px;
            border-radius: 12px; text-align: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        
        .stat-number { font-size: 2em; font-weight: 700; color: #667eea; }
        .stat-label { color: #666; font-size: 0.9em; margin-top: 5px; }
        
        .controls {
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);
            padding: 20px; border-radius: 12px; margin-bottom: 20px;
        }
        
        .search-input {
            width: 100%; padding: 12px 20px; border: 2px solid #e1e5e9;
            border-radius: 25px; font-size: 16px; margin-bottom: 15px;
        }
        
        .filter-buttons { display: flex; gap: 10px; flex-wrap: wrap; }
        
        .filter-btn {
            padding: 8px 16px; border: 2px solid #e1e5e9; background: white;
            border-radius: 20px; cursor: pointer; transition: all 0.3s ease;
        }
        
        .filter-btn.active { background: #667eea; color: white; }
        
        .raga-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 20px;
        }
        
        .raga-card {
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);
            border-radius: 16px; padding: 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: all 0.3s ease; border-left: 4px solid #667eea;
        }
        
        .raga-card:hover { transform: translateY(-4px); }
        .raga-card.carnatic { border-left-color: #e74c3c; }
        .raga-card.hindustani { border-left-color: #3498db; }
        
        .raga-header {
            display: flex; justify-content: space-between;
            align-items: flex-start; margin-bottom: 15px;
        }
        
        .raga-title { font-size: 1.4em; font-weight: 700; color: #2c3e50; }
        .raga-tradition { font-size: 0.9em; color: #7f8c8d; text-transform: uppercase; }
        
        .confidence-badge {
            padding: 4px 12px; border-radius: 12px; font-size: 0.8em; font-weight: 600;
        }
        
        .confidence-high { background: #d4edda; color: #155724; }
        .confidence-medium { background: #fff3cd; color: #856404; }
        .confidence-low { background: #f8d7da; color: #721c24; }
        
        .source-section { margin: 15px 0; }
        
        .source-title {
            font-weight: 600; color: #495057; margin-bottom: 8px;
            display: flex; align-items: center; gap: 8px;
        }
        
        .source-available { color: #28a745; }
        .source-unavailable { color: #dc3545; }
        
        .source-details {
            background: #f8f9fa; padding: 12px; border-radius: 8px; font-size: 0.9em;
        }
        
        .source-item {
            display: flex; justify-content: space-between; margin: 4px 0;
        }
        
        .musical-theory {
            background: linear-gradient(135deg, #667eea20, #764ba220);
            padding: 15px; border-radius: 12px; margin: 15px 0;
        }
        
        .theory-item {
            display: flex; justify-content: space-between; margin: 6px 0; font-size: 0.9em;
        }
        
        .theory-label { font-weight: 500; color: #666; }
        .theory-value { color: #2c3e50; font-family: monospace; }
        
        .multimedia-section { display: flex; gap: 10px; margin: 15px 0; }
        
        .media-badge {
            padding: 6px 12px; border-radius: 16px; font-size: 0.8em; font-weight: 500;
        }
        
        .media-audio { background: #e8f5e8; color: #2d5a2d; }
        .media-video { background: #ffe8e8; color: #5a2d2d; }
        
        .completeness-bar {
            width: 100%; height: 6px; background: #e9ecef;
            border-radius: 3px; margin: 10px 0; overflow: hidden;
        }
        
        .completeness-fill {
            height: 100%; background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Unified Raga Explorer</h1>
        <p>One raga, complete story - Audio, Theory, Videos, all sources unified</p>
        
        <div class="stats-grid" id="statsGrid">
            <!-- Stats populated by JavaScript -->
        </div>
    </div>

    <div class="controls">
        <input type="text" id="searchInput" class="search-input" placeholder="Search ragas..." oninput="filterRagas()">
        
        <div class="filter-buttons">
            <button class="filter-btn active" onclick="filterByTradition('all')">All</button>
            <button class="filter-btn" onclick="filterByTradition('carnatic')">🎼 Carnatic</button>
            <button class="filter-btn" onclick="filterByTradition('hindustani')">🎵 Hindustani</button>
            <button class="filter-btn" onclick="filterByData('audio')">Has Audio</button>
            <button class="filter-btn" onclick="filterByData('video')">Has Videos</button>
            <button class="filter-btn" onclick="filterByData('theory')">Has Theory</button>
        </div>
    </div>

    <div id="ragaGrid" class="raga-grid"></div>

    <script>
        let allRagas = [];
        let filteredRagas = [];
        let currentFilters = { tradition: 'all', data: null, search: '' };

        async function loadData() {
            const response = await fetch('/api/unified-ragas');
            allRagas = await response.json();
            filteredRagas = [...allRagas];
            updateStats();
            renderRagas();
        }

        function updateStats() {
            const stats = {
                total: allRagas.length,
                carnatic: allRagas.filter(r => r.tradition === 'carnatic').length,
                hindustani: allRagas.filter(r => r.tradition === 'hindustani').length,
                withAudio: allRagas.filter(r => r.multimedia.has_audio).length,
                withVideos: allRagas.filter(r => r.multimedia.has_videos).length,
                withTheory: allRagas.filter(r => r.musical_theory.available).length
            };

            document.getElementById('statsGrid').innerHTML = `
                <div class="stat-card"><div class="stat-number">${stats.total}</div><div class="stat-label">Total Ragas</div></div>
                <div class="stat-card"><div class="stat-number">${stats.carnatic}</div><div class="stat-label">🎼 Carnatic</div></div>
                <div class="stat-card"><div class="stat-number">${stats.hindustani}</div><div class="stat-label">🎵 Hindustani</div></div>
                <div class="stat-card"><div class="stat-number">${stats.withAudio}</div><div class="stat-label">With Audio</div></div>
                <div class="stat-card"><div class="stat-number">${stats.withVideos}</div><div class="stat-label">With Videos</div></div>
                <div class="stat-card"><div class="stat-number">${stats.withTheory}</div><div class="stat-label">With Theory</div></div>
            `;
        }

        function renderRagas() {
            const grid = document.getElementById('ragaGrid');
            
            grid.innerHTML = filteredRagas.map(raga => `
                <div class="raga-card ${raga.tradition}">
                    <div class="raga-header">
                        <div>
                            <div class="raga-title">${raga.name}</div>
                            <div class="raga-tradition">${raga.tradition}</div>
                        </div>
                        <div class="confidence-badge ${getConfidenceClass(raga.confidence)}">
                            ${(raga.confidence * 100).toFixed(0)}%
                        </div>
                    </div>
                    
                    ${raga.musical_theory.available ? `
                        <div class="musical-theory">
                            <div style="font-weight: 600; margin-bottom: 10px;">🎼 Musical Theory</div>
                            ${raga.musical_theory.arohana ? `<div class="theory-item"><span class="theory-label">Arohana:</span> <span class="theory-value">${raga.musical_theory.arohana}</span></div>` : ''}
                            ${raga.musical_theory.avarohana ? `<div class="theory-item"><span class="theory-label">Avarohana:</span> <span class="theory-value">${raga.musical_theory.avarohana}</span></div>` : ''}
                            ${raga.musical_theory.melakartha ? `<div class="theory-item"><span class="theory-label">Melakartha:</span> <span class="theory-value">${raga.musical_theory.melakartha}</span></div>` : ''}
                        </div>
                    ` : ''}
                    
                    <div class="source-section">
                        <div class="source-title ${raga.sources.saraga.available ? 'source-available' : 'source-unavailable'}">
                            📻 Saraga ${raga.sources.saraga.available ? '✓' : '✗'}
                        </div>
                        ${raga.sources.saraga.available ? `
                            <div class="source-details">
                                ${raga.sources.saraga.artist ? `<div class="source-item"><span>Artist:</span> <span>${raga.sources.saraga.artist}</span></div>` : ''}
                                ${raga.sources.saraga.form ? `<div class="source-item"><span>Form:</span> <span>${raga.sources.saraga.form}</span></div>` : ''}
                                ${raga.sources.saraga.audio_file ? `<div class="source-item"><span>Audio:</span> <span>Available</span></div>` : ''}
                            </div>
                        ` : '<div class="source-details">No data available</div>'}
                    </div>
                    
                    <div class="source-section">
                        <div class="source-title ${raga.sources.ramanarunachalam.available ? 'source-available' : 'source-unavailable'}">
                            🏛️ Ramanarunachalam ${raga.sources.ramanarunachalam.available ? '✓' : '✗'}
                        </div>
                        ${raga.sources.ramanarunachalam.available ? `
                            <div class="source-details">
                                ${raga.sources.ramanarunachalam.videos_count ? `<div class="source-item"><span>Videos:</span> <span>${raga.sources.ramanarunachalam.videos_count}</span></div>` : ''}
                                ${raga.sources.ramanarunachalam.songs_count ? `<div class="source-item"><span>Songs:</span> <span>${raga.sources.ramanarunachalam.songs_count}</span></div>` : ''}
                            </div>
                        ` : '<div class="source-details">No data available</div>'}
                    </div>
                    
                    <div class="multimedia-section">
                        ${raga.multimedia.has_audio ? '<div class="media-badge media-audio">🎵 Audio</div>' : ''}
                        ${raga.multimedia.has_videos ? `<div class="media-badge media-video">📺 ${raga.multimedia.video_count} Videos</div>` : ''}
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-size: 0.9em; color: #666;">Completeness</span>
                            <span style="font-size: 0.9em; font-weight: 600;">${raga.data_quality.completeness}%</span>
                        </div>
                        <div class="completeness-bar">
                            <div class="completeness-fill" style="width: ${raga.data_quality.completeness}%"></div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function getConfidenceClass(confidence) {
            if (confidence >= 0.9) return 'confidence-high';
            if (confidence >= 0.7) return 'confidence-medium';
            return 'confidence-low';
        }

        function applyFilters() {
            filteredRagas = allRagas.filter(raga => {
                if (currentFilters.tradition !== 'all' && raga.tradition !== currentFilters.tradition) return false;
                if (currentFilters.data === 'audio' && !raga.multimedia.has_audio) return false;
                if (currentFilters.data === 'video' && !raga.multimedia.has_videos) return false;
                if (currentFilters.data === 'theory' && !raga.musical_theory.available) return false;
                if (currentFilters.search && !raga.name.toLowerCase().includes(currentFilters.search)) return false;
                return true;
            });
            renderRagas();
        }

        function filterByTradition(tradition) {
            currentFilters.tradition = tradition;
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            applyFilters();
        }

        function filterByData(type) {
            currentFilters.data = currentFilters.data === type ? null : type;
            event.target.classList.toggle('active');
            applyFilters();
        }

        function filterRagas() {
            currentFilters.search = document.getElementById('searchInput').value.toLowerCase();
            applyFilters();
        }

        loadData();
    </script>
</body>
</html>
    """)

@app.route('/api/unified-ragas')
def api_unified_ragas():
    """API endpoint for unified raga data"""
    return jsonify(unified_data)

def main():
    """Start the unified raga explorer"""
    global unified_data
    
    print("🎵 Unified Raga Explorer")
    print("=" * 50)
    
    # Create unified dataset
    unified_ragas = create_unified_raga_dataset()
    unified_data = list(unified_ragas.values())
    
    # Statistics
    total_ragas = len(unified_data)
    carnatic_count = len([r for r in unified_data if r['tradition'] == 'carnatic'])
    hindustani_count = len([r for r in unified_data if r['tradition'] == 'hindustani'])
    with_audio = len([r for r in unified_data if r['multimedia']['has_audio']])
    with_videos = len([r for r in unified_data if r['multimedia']['has_videos']])
    with_theory = len([r for r in unified_data if r['musical_theory']['available']])
    
    print(f"✅ Unified Dataset Created:")
    print(f"   📊 Total Ragas: {total_ragas}")
    print(f"   🎼 Carnatic: {carnatic_count}")
    print(f"   🎵 Hindustani: {hindustani_count}")
    print(f"   🎶 With Audio: {with_audio}")
    print(f"   📺 With Videos: {with_videos}")
    print(f"   📖 With Theory: {with_theory}")
    
    print("\n🌐 Starting unified explorer...")
    print("📊 Access at: http://localhost:5005")
    print("\n🎯 Features:")
    print("   ✅ One raga = One card")
    print("   ✅ Complete source attribution")
    print("   ✅ Audio file integration from Saraga")
    print("   ✅ Musical theory from Ramanarunachalam")
    print("   ✅ YouTube videos with counts")
    print("   ✅ Data completeness scoring")
    
    app.run(host='0.0.0.0', port=5005, debug=False)

if __name__ == "__main__":
    main()