#!/usr/bin/env python3
"""
RagaSense Dataset Explorer
=========================

Interactive web interface to explore the complete RagaSense dataset
Includes search, filtering, statistics, and detailed raga information
"""

import json
import os
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

class DatasetExplorer:
    def __init__(self):
        self.dataset_info = {}
        self.saraga_data = {}
        self.ramanarunachalam_data = {}
        self.integration_results = {}
        self.ml_datasets = {}
        self.search_index = {}
        
        self.load_all_data()
        self.build_search_index()
    
    def load_all_data(self):
        """Load all available dataset files"""
        print("🔄 Loading dataset...")
        
        # 1. Load dataset manifest
        try:
            with open('data/DATASET_MANIFEST.json', 'r') as f:
                self.dataset_info = json.load(f)
        except:
            self.dataset_info = {"version": "2.0.0", "statistics": {}}
        
        # 2. Load Saraga metadata
        try:
            # Try both possible file locations
            saraga_files = [
                'data/02_raw/extracted_saraga_metadata/carnatic_metadata_extracted.json',
                'data/02_raw/extracted_saraga_metadata/hindustani_metadata_extracted.json',
                'data/02_raw/extracted_saraga_metadata/combined_metadata_extracted.json'
            ]
            
            for file_path in saraga_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        tradition = 'carnatic' if 'carnatic' in file_path else 'hindustani' if 'hindustani' in file_path else 'combined'
                        self.saraga_data[tradition] = data
        except Exception as e:
            print(f"⚠️ Could not load Saraga data: {e}")
        
        # 3. Load Ramanarunachalam data samples
        try:
            ramanarunachalam_dir = 'data/01_source/ramanarunachalam'
            for tradition in ['Carnatic', 'Hindustani']:
                raga_dir = os.path.join(ramanarunachalam_dir, tradition, 'raga')
                if os.path.exists(raga_dir):
                    tradition_key = tradition.lower()
                    self.ramanarunachalam_data[tradition_key] = {}
                    
                    # Load first 20 ragas as samples
                    files = [f for f in os.listdir(raga_dir) if f.endswith('.json')][:20]
                    for filename in files:
                        raga_name = filename[:-5]
                        try:
                            with open(os.path.join(raga_dir, filename), 'r') as f:
                                self.ramanarunachalam_data[tradition_key][raga_name] = json.load(f)
                        except:
                            pass
        except Exception as e:
            print(f"⚠️ Could not load Ramanarunachalam data: {e}")
        
        # 4. Load integration results
        try:
            integration_files = [
                'data/03_processed/metadata/integration_summary_20250921_112230.json'
            ]
            
            for file_path in integration_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        self.integration_results = json.load(f)
                    break
        except Exception as e:
            print(f"⚠️ Could not load integration results: {e}")
        
        # 5. Load ML datasets
        try:
            ml_files = [
                'data/04_ml_datasets/final_ml_ready_dataset_20250914_115657.json',
                'data/04_ml_datasets/final_ml_dataset_summary_20250914_115657.json'
            ]
            
            for file_path in ml_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        filename = os.path.basename(file_path)
                        self.ml_datasets[filename] = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load ML datasets: {e}")
        
        print("✅ Dataset loaded successfully!")
    
    def build_search_index(self):
        """Build search index for fast lookups"""
        self.search_index = {
            'ragas': [],
            'traditions': set(),
            'composers': set(),
            'regions': set()
        }
        
        # Index Saraga data
        for tradition, data in self.saraga_data.items():
            if isinstance(data, dict):
                for raga_name, details in data.items():
                    self.search_index['ragas'].append({
                        'name': raga_name,
                        'tradition': tradition,
                        'source': 'Saraga',
                        'details': details
                    })
                    self.search_index['traditions'].add(tradition)
        
        # Index Ramanarunachalam data
        for tradition, ragas in self.ramanarunachalam_data.items():
            for raga_name, details in ragas.items():
                self.search_index['ragas'].append({
                    'name': raga_name,
                    'tradition': tradition,
                    'source': 'Ramanarunachalam',
                    'details': details
                })
                self.search_index['traditions'].add(tradition)
        
        # Convert sets to lists for JSON serialization
        self.search_index['traditions'] = list(self.search_index['traditions'])
        self.search_index['composers'] = list(self.search_index['composers'])
        self.search_index['regions'] = list(self.search_index['regions'])
    
    def search_ragas(self, query="", tradition="", source="", limit=50):
        """Search ragas with filters"""
        results = []
        query_lower = query.lower() if query else ""
        
        for raga in self.search_index['ragas']:
            # Apply filters
            if tradition and raga['tradition'] != tradition:
                continue
            if source and raga['source'] != source:
                continue
            if query and query_lower not in raga['name'].lower():
                continue
            
            results.append(raga)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_raga_details(self, raga_name, source):
        """Get detailed information about a specific raga"""
        for raga in self.search_index['ragas']:
            if raga['name'] == raga_name and raga['source'] == source:
                return raga
        return None
    
    def get_dataset_statistics(self):
        """Get comprehensive dataset statistics"""
        stats = {
            'overview': self.dataset_info.get('statistics', {}),
            'ragas': {
                'total': len(self.search_index['ragas']),
                'by_tradition': {},
                'by_source': {}
            },
            'traditions': list(self.search_index['traditions']),
            'integration': self.integration_results.get('summary', {})
        }
        
        # Count by tradition and source
        for raga in self.search_index['ragas']:
            tradition = raga['tradition']
            source = raga['source']
            
            if tradition not in stats['ragas']['by_tradition']:
                stats['ragas']['by_tradition'][tradition] = 0
            stats['ragas']['by_tradition'][tradition] += 1
            
            if source not in stats['ragas']['by_source']:
                stats['ragas']['by_source'][source] = 0
            stats['ragas']['by_source'][source] += 1
        
        return stats

# Initialize explorer
explorer = DatasetExplorer()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RagaSense Dataset Explorer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; }
        .navbar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card { border: none; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .stat-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
        .raga-card { transition: transform 0.2s; cursor: pointer; }
        .raga-card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,0,0,0.15); }
        .badge-tradition { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
        .badge-source { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
        .search-box { border-radius: 25px; border: 2px solid #e9ecef; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; }
        .tradition-carnatic { border-left: 4px solid #ff6b6b; }
        .tradition-hindustani { border-left: 4px solid #4ecdc4; }
        .tradition-combined { border-left: 4px solid #45b7d1; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-music me-2"></i>RagaSense Dataset Explorer
            </span>
            <span class="text-light">
                <i class="fas fa-database me-1"></i>Version 2.0.0
            </span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Statistics Dashboard -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="card-body text-center">
                        <i class="fas fa-music fa-2x mb-2"></i>
                        <h3 id="total-ragas">0</h3>
                        <p class="mb-0">Total Ragas</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="card-body text-center">
                        <i class="fas fa-layer-group fa-2x mb-2"></i>
                        <h3 id="total-traditions">0</h3>
                        <p class="mb-0">Traditions</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="card-body text-center">
                        <i class="fas fa-database fa-2x mb-2"></i>
                        <h3 id="total-sources">0</h3>
                        <p class="mb-0">Data Sources</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="card-body text-center">
                        <i class="fas fa-chart-line fa-2x mb-2"></i>
                        <h3 id="integration-rate">0%</h3>
                        <p class="mb-0">Integration Rate</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Search and Filter Section -->
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title"><i class="fas fa-search me-2"></i>Search & Filter</h5>
                        <div class="row">
                            <div class="col-md-4">
                                <input type="text" id="search-query" class="form-control search-box" placeholder="Search ragas...">
                            </div>
                            <div class="col-md-3">
                                <select id="tradition-filter" class="form-select">
                                    <option value="">All Traditions</option>
                                    <option value="carnatic">Carnatic</option>
                                    <option value="hindustani">Hindustani</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <select id="source-filter" class="form-select">
                                    <option value="">All Sources</option>
                                    <option value="Saraga">Saraga Dataset</option>
                                    <option value="Ramanarunachalam">Ramanarunachalam Dataset</option>
                                </select>
                            </div>
                            <div class="col-md-2">
                                <button class="btn btn-primary w-100" onclick="searchRagas()">
                                    <i class="fas fa-search me-1"></i>Search
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">
                            <i class="fas fa-list me-2"></i>Ragas 
                            <span class="badge bg-secondary" id="results-count">0</span>
                        </h5>
                        <div id="ragas-container" class="row">
                            <!-- Ragas will be loaded here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Raga Detail Modal -->
    <div class="modal fade" id="ragaModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="ragaModalTitle">Raga Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="ragaModalBody">
                    <!-- Raga details will be loaded here -->
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let allRagas = [];
        let currentResults = [];

        // Load initial data
        document.addEventListener('DOMContentLoaded', function() {
            loadStatistics();
            searchRagas();
        });

        function loadStatistics() {
            fetch('/api/statistics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-ragas').textContent = data.ragas.total;
                    document.getElementById('total-traditions').textContent = data.traditions.length;
                    document.getElementById('total-sources').textContent = Object.keys(data.ragas.by_source).length;
                    
                    const integrationRate = data.integration.match_rate || 0;
                    document.getElementById('integration-rate').textContent = (integrationRate * 100).toFixed(1) + '%';
                });
        }

        function searchRagas() {
            const query = document.getElementById('search-query').value;
            const tradition = document.getElementById('tradition-filter').value;
            const source = document.getElementById('source-filter').value;

            const params = new URLSearchParams({
                query: query,
                tradition: tradition,
                source: source,
                limit: 100
            });

            fetch(`/api/search?${params}`)
                .then(response => response.json())
                .then(data => {
                    currentResults = data;
                    displayRagas(data);
                });
        }

        function displayRagas(ragas) {
            const container = document.getElementById('ragas-container');
            const count = document.getElementById('results-count');
            
            count.textContent = ragas.length;
            
            if (ragas.length === 0) {
                container.innerHTML = '<div class="col-12"><div class="alert alert-info">No ragas found. Try adjusting your search criteria.</div></div>';
                return;
            }

            container.innerHTML = ragas.map(raga => `
                <div class="col-md-6 col-lg-4 mb-3">
                    <div class="card raga-card tradition-${raga.tradition}" onclick="showRagaDetails('${raga.name}', '${raga.source}')">
                        <div class="card-body">
                            <h6 class="card-title">${raga.name}</h6>
                            <div class="mb-2">
                                <span class="badge badge-tradition me-1">${raga.tradition}</span>
                                <span class="badge badge-source">${raga.source}</span>
                            </div>
                            <small class="text-muted">
                                <i class="fas fa-info-circle me-1"></i>Click for details
                            </small>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function showRagaDetails(ragaName, source) {
            fetch(`/api/raga/${encodeURIComponent(ragaName)}/${encodeURIComponent(source)}`)
                .then(response => response.json())
                .then(raga => {
                    document.getElementById('ragaModalTitle').textContent = raga.name;
                    
                    const details = raga.details || {};
                    let content = `
                        <div class="row">
                            <div class="col-md-6">
                                <h6><i class="fas fa-tag me-2"></i>Basic Information</h6>
                                <p><strong>Name:</strong> ${raga.name}</p>
                                <p><strong>Tradition:</strong> ${raga.tradition}</p>
                                <p><strong>Source:</strong> ${raga.source}</p>
                            </div>
                            <div class="col-md-6">
                                <h6><i class="fas fa-music me-2"></i>Musical Details</h6>
                    `;
                    
                    if (details.arohana) content += `<p><strong>Arohana:</strong> ${details.arohana}</p>`;
                    if (details.avarohana) content += `<p><strong>Avarohana:</strong> ${details.avarohana}</p>`;
                    if (details.melakartha) content += `<p><strong>Melakartha:</strong> ${details.melakartha}</p>`;
                    
                    content += `
                            </div>
                        </div>
                    `;
                    
                    if (details.audio_files && details.audio_files.length > 0) {
                        content += `
                            <hr>
                            <h6><i class="fas fa-headphones me-2"></i>Audio Files</h6>
                            <p>Available: ${details.audio_files.length} files</p>
                        `;
                    }
                    
                    document.getElementById('ragaModalBody').innerHTML = content;
                    new bootstrap.Modal(document.getElementById('ragaModal')).show();
                });
        }

        // Search on Enter key
        document.getElementById('search-query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchRagas();
            }
        });
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/statistics')
def get_statistics():
    return jsonify(explorer.get_dataset_statistics())

@app.route('/api/search')
def search_ragas():
    query = request.args.get('query', '')
    tradition = request.args.get('tradition', '')
    source = request.args.get('source', '')
    limit = int(request.args.get('limit', 50))
    
    results = explorer.search_ragas(query, tradition, source, limit)
    return jsonify(results)

@app.route('/api/raga/<raga_name>/<source>')
def get_raga_details(raga_name, source):
    raga = explorer.get_raga_details(raga_name, source)
    if raga:
        return jsonify(raga)
    else:
        return jsonify({'error': 'Raga not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting RagaSense Dataset Explorer...")
    print("📊 Dataset loaded and indexed")
    print("🌐 Open your browser to: http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002)