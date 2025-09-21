#!/usr/bin/env python3
"""
Improved Raga Mapper with Better UX and Existing Mappings Display
"""

import json
import os
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

class ImprovedRagaMapper:
    """Improved raga mapper with better UX"""
    
    def __init__(self):
        self.load_existing_mappings()
        self.load_raga_data()
    
    def load_existing_mappings(self):
        """Load all existing cross-tradition mappings"""
        self.existing_mappings = {
            'exact_matches': [],
            'similar_matches': [],
            'semantic_matches': []
        }
        
        # Load from multiple mapping files
        mapping_files = [
            'data/04_ml_datasets/unified/cross_tradition_mappings_20250914_155215.json',
            'data/04_ml_datasets/unified/semantic_cross_tradition_mappings_20250914_175606.json'
        ]
        
        for file_path in mapping_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract exact matches
                    if 'exact_matches' in data:
                        for raga in data['exact_matches']:
                            self.existing_mappings['exact_matches'].append({
                                'saraga_raga': raga,
                                'ramanarunachalam_raga': raga,
                                'confidence': 1.0,
                                'match_type': 'exact_match',
                                'source': 'existing'
                            })
                    
                    # Extract similar matches
                    if 'similar_matches' in data:
                        for match in data['similar_matches']:
                            if len(match) == 2:
                                self.existing_mappings['similar_matches'].append({
                                    'saraga_raga': match[0],
                                    'ramanarunachalam_raga': match[1],
                                    'confidence': 0.8,
                                    'match_type': 'similar_match',
                                    'source': 'existing'
                                })
                    
                    # Extract semantic matches
                    if 'semantic_matches' in data:
                        for match in data['semantic_matches']:
                            self.existing_mappings['semantic_matches'].append({
                                'saraga_raga': match['saraga_raga'],
                                'ramanarunachalam_raga': match['ramanarunachalam_raga'],
                                'confidence': match.get('confidence', 0.8),
                                'match_type': match.get('match_type', 'semantic_match'),
                                'reasoning': match.get('reasoning', ''),
                                'source': 'existing'
                            })
                
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        # Combine all mappings
        self.all_mappings = (
            self.existing_mappings['exact_matches'] + 
            self.existing_mappings['similar_matches'] + 
            self.existing_mappings['semantic_matches']
        )
        
        print(f"✅ Loaded {len(self.all_mappings)} existing mappings")
    
    def load_raga_data(self):
        """Load raga data for search and filtering"""
        self.saraga_ragas = set()
        self.ramanarunachalam_ragas = set()
        
        # Load from unified dataset
        dataset_files = []
        for file in os.listdir('data/04_ml_datasets/unified/'):
            if file.startswith('unified_raga_dataset_') and file.endswith('.json'):
                dataset_files.append(file)
        
        if dataset_files:
            latest_file = sorted(dataset_files)[-1]
            try:
                with open(f'data/04_ml_datasets/unified/{latest_file}', 'r') as f:
                    data = json.load(f)
                
                if 'ragas' in data:
                    for raga_name, raga_info in data['ragas'].items():
                        sources = raga_info.get('sources', [])
                        if 'saraga' in sources:
                            self.saraga_ragas.add(raga_name)
                        if 'ramanarunachalam' in sources:
                            self.ramanarunachalam_ragas.add(raga_name)
                
                print(f"✅ Loaded {len(self.saraga_ragas)} Saraga ragas and {len(self.ramanarunachalam_ragas)} Ramanarunachalam ragas")
            
            except Exception as e:
                print(f"Error loading dataset: {e}")
    
    def get_unmapped_saraga_ragas(self):
        """Get Saraga ragas that don't have mappings yet"""
        mapped_saraga = set()
        for mapping in self.all_mappings:
            mapped_saraga.add(mapping['saraga_raga'].lower())
        
        unmapped = []
        for raga in self.saraga_ragas:
            if raga.lower() not in mapped_saraga:
                unmapped.append(raga)
        
        return sorted(unmapped)
    
    def search_ramanarunachalam(self, query):
        """Search Ramanarunachalam ragas"""
        if not query:
            return []
        
        query = query.lower()
        matches = []
        for raga in self.ramanarunachalam_ragas:
            if query in raga.lower():
                matches.append(raga)
        
        return sorted(matches)[:10]  # Limit to 10 results

# Initialize mapper
mapper = ImprovedRagaMapper()

@app.route('/')
def index():
    """Main interface with improved UX"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RagaSense-Data | Cross-Tradition Raga Mapping</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .hero-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 0; }
        .mapping-card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
        .exact-match { border-left: 4px solid #28a745; background-color: #f8fff8; }
        .similar-match { border-left: 4px solid #ffc107; background-color: #fffdf5; }
        .semantic-match { border-left: 4px solid #17a2b8; background-color: #f0f9ff; }
        .stats-card { background: #f8f9fa; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
        .search-results { max-height: 200px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; }
        .search-item { padding: 0.5rem; cursor: pointer; border-bottom: 1px solid #eee; }
        .search-item:hover { background-color: #f8f9fa; }
        .unmapped-item { padding: 0.5rem; margin: 0.25rem 0; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="hero-section">
        <div class="container text-center">
            <h1 class="display-4">🎵 RagaSense-Data</h1>
            <p class="lead">Cross-Tradition Raga Mapping Interface</p>
            <p>Map Saraga ragas to Ramanarunachalam ragas with confidence scoring</p>
        </div>
    </div>
    
    <div class="container mt-4">
        <!-- Statistics -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="stats-card text-center">
                    <h4>{{ mapper.all_mappings|length }}</h4>
                    <p>Total Mappings</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stats-card text-center">
                    <h4>{{ mapper.existing_mappings.exact_matches|length }}</h4>
                    <p>Exact Matches</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stats-card text-center">
                    <h4>{{ mapper.existing_mappings.similar_matches|length }}</h4>
                    <p>Similar Matches</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stats-card text-center">
                    <h4>{{ mapper.get_unmapped_saraga_ragas()|length }}</h4>
                    <p>Unmapped Saraga</p>
                </div>
            </div>
        </div>
        
        <!-- Navigation Tabs -->
        <ul class="nav nav-tabs" id="main-tabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="existing-tab" data-bs-toggle="tab" data-bs-target="#existing" type="button">📋 Existing Mappings</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="mapping-tab" data-bs-toggle="tab" data-bs-target="#mapping" type="button">➕ Create New Mapping</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="unmapped-tab" data-bs-toggle="tab" data-bs-target="#unmapped" type="button">🔍 Unmapped Ragas</button>
            </li>
        </ul>
        
        <!-- Tab Content -->
        <div class="tab-content" id="main-tabs-content">
            <!-- Existing Mappings Tab -->
            <div class="tab-pane fade show active" id="existing" role="tabpanel">
                <div class="mt-4">
                    <h3>📋 Existing Cross-Tradition Mappings</h3>
                    <p class="text-muted">These are the mappings we already have in our database</p>
                    
                    <!-- Filter buttons -->
                    <div class="mb-3">
                        <button class="btn btn-outline-success btn-sm" onclick="filterMappings('exact')">Exact Matches ({{ mapper.existing_mappings.exact_matches|length }})</button>
                        <button class="btn btn-outline-warning btn-sm" onclick="filterMappings('similar')">Similar Matches ({{ mapper.existing_mappings.similar_matches|length }})</button>
                        <button class="btn btn-outline-info btn-sm" onclick="filterMappings('semantic')">Semantic Matches ({{ mapper.existing_mappings.semantic_matches|length }})</button>
                        <button class="btn btn-outline-primary btn-sm" onclick="filterMappings('all')">Show All</button>
                    </div>
                    
                    <div id="mappings-container">
                        <!-- Mappings will be loaded here -->
                    </div>
                </div>
            </div>
            
            <!-- Create New Mapping Tab -->
            <div class="tab-pane fade" id="mapping" role="tabpanel">
                <div class="mt-4">
                    <h3>➕ Create New Cross-Tradition Mapping</h3>
                    <p class="text-muted">Map a Saraga raga to a Ramanarunachalam raga</p>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>📚 Saraga Raga (Source)</h5>
                                </div>
                                <div class="card-body">
                                    <input type="text" class="form-control" id="saraga-input" placeholder="Type Saraga raga name...">
                                    <div id="saraga-suggestions" class="search-results mt-2" style="display: none;"></div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>🎼 Ramanarunachalam Raga (Target)</h5>
                                </div>
                                <div class="card-body">
                                    <input type="text" class="form-control" id="ramanarunachalam-input" placeholder="Type Ramanarunachalam raga name...">
                                    <div id="ramanarunachalam-suggestions" class="search-results mt-2" style="display: none;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-3">
                        <div class="col-md-6">
                            <label class="form-label">Confidence Level</label>
                            <select class="form-select" id="confidence-select">
                                <option value="1.0">Very High (100%) - Exact match</option>
                                <option value="0.9">High (90%) - Very similar</option>
                                <option value="0.8" selected>Medium-High (80%) - Similar</option>
                                <option value="0.7">Medium (70%) - Somewhat similar</option>
                                <option value="0.6">Low-Medium (60%) - Possibly related</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Match Type</label>
                            <select class="form-select" id="match-type-select">
                                <option value="exact_match">Exact Match</option>
                                <option value="similar_match" selected>Similar Match</option>
                                <option value="semantic_match">Semantic Match</option>
                                <option value="cultural_equivalent">Cultural Equivalent</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="mt-3">
                        <label class="form-label">Notes (Optional)</label>
                        <textarea class="form-control" id="mapping-notes" rows="3" placeholder="Any additional context or reasoning for this mapping..."></textarea>
                    </div>
                    
                    <div class="mt-3">
                        <button class="btn btn-primary btn-lg" onclick="createMapping()">Create Mapping</button>
                        <button class="btn btn-secondary btn-lg" onclick="clearForm()">Clear Form</button>
                    </div>
                </div>
            </div>
            
            <!-- Unmapped Ragas Tab -->
            <div class="tab-pane fade" id="unmapped" role="tabpanel">
                <div class="mt-4">
                    <h3>🔍 Unmapped Saraga Ragas</h3>
                    <p class="text-muted">These Saraga ragas don't have cross-tradition mappings yet</p>
                    
                    <div class="mb-3">
                        <input type="text" class="form-control" id="unmapped-search" placeholder="Search unmapped ragas...">
                    </div>
                    
                    <div id="unmapped-container">
                        <!-- Unmapped ragas will be loaded here -->
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Load existing mappings on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadExistingMappings();
            loadUnmappedRagas();
        });
        
        // Search functionality
        document.getElementById('saraga-input').addEventListener('input', function() {
            const query = this.value;
            if (query.length > 1) {
                searchRagas('saraga', query);
            } else {
                document.getElementById('saraga-suggestions').style.display = 'none';
            }
        });
        
        document.getElementById('ramanarunachalam-input').addEventListener('input', function() {
            const query = this.value;
            if (query.length > 1) {
                searchRagas('ramanarunachalam', query);
            } else {
                document.getElementById('ramanarunachalam-suggestions').style.display = 'none';
            }
        });
        
        function searchRagas(type, query) {
            fetch(`/api/search-${type}?q=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById(`${type}-suggestions`);
                    container.innerHTML = '';
                    
                    if (data.length > 0) {
                        data.forEach(raga => {
                            const item = document.createElement('div');
                            item.className = 'search-item';
                            item.textContent = raga;
                            item.onclick = () => {
                                document.getElementById(`${type}-input`).value = raga;
                                container.style.display = 'none';
                            };
                            container.appendChild(item);
                        });
                        container.style.display = 'block';
                    } else {
                        container.style.display = 'none';
                    }
                });
        }
        
        function loadExistingMappings() {
            fetch('/api/existing-mappings')
                .then(response => response.json())
                .then(data => {
                    renderMappings(data);
                });
        }
        
        function renderMappings(mappings) {
            const container = document.getElementById('mappings-container');
            container.innerHTML = '';
            
            if (mappings.length === 0) {
                container.innerHTML = '<p class="text-muted">No mappings found.</p>';
                return;
            }
            
            mappings.forEach(mapping => {
                const card = document.createElement('div');
                card.className = `mapping-card ${mapping.match_type.replace('_', '-')}`;
                
                const confidencePercent = Math.round(mapping.confidence * 100);
                const matchTypeLabel = mapping.match_type.replace('_', ' ').toUpperCase();
                
                card.innerHTML = `
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <h5>${mapping.saraga_raga} → ${mapping.ramanarunachalam_raga}</h5>
                            <div class="row">
                                <div class="col-md-3">
                                    <small class="text-muted">Type: <strong>${matchTypeLabel}</strong></small>
                                </div>
                                <div class="col-md-3">
                                    <small class="text-muted">Confidence: <strong>${confidencePercent}%</strong></small>
                                </div>
                                <div class="col-md-3">
                                    <small class="text-muted">Source: <strong>${mapping.source}</strong></small>
                                </div>
                            </div>
                            ${mapping.reasoning ? `<p class="mt-2 mb-0"><small class="text-muted">${mapping.reasoning}</small></p>` : ''}
                        </div>
                        <div class="ms-3">
                            <span class="badge bg-${mapping.match_type === 'exact_match' ? 'success' : mapping.match_type === 'similar_match' ? 'warning' : 'info'}">${matchTypeLabel}</span>
                        </div>
                    </div>
                `;
                
                container.appendChild(card);
            });
        }
        
        function filterMappings(type) {
            fetch('/api/existing-mappings')
                .then(response => response.json())
                .then(data => {
                    let filtered = data;
                    if (type !== 'all') {
                        filtered = data.filter(mapping => mapping.match_type === type);
                    }
                    renderMappings(filtered);
                });
        }
        
        function loadUnmappedRagas() {
            fetch('/api/unmapped-ragas')
                .then(response => response.json())
                .then(data => {
                    renderUnmappedRagas(data);
                });
        }
        
        function renderUnmappedRagas(ragas) {
            const container = document.getElementById('unmapped-container');
            container.innerHTML = '';
            
            if (ragas.length === 0) {
                container.innerHTML = '<p class="text-muted">All Saraga ragas are mapped!</p>';
                return;
            }
            
            ragas.forEach(raga => {
                const item = document.createElement('div');
                item.className = 'unmapped-item';
                item.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <span><strong>${raga}</strong></span>
                        <button class="btn btn-sm btn-outline-primary" onclick="mapRaga('${raga}')">Map This Raga</button>
                    </div>
                `;
                container.appendChild(item);
            });
        }
        
        function mapRaga(raga) {
            // Switch to mapping tab and pre-fill Saraga raga
            document.getElementById('mapping-tab').click();
            document.getElementById('saraga-input').value = raga;
        }
        
        function createMapping() {
            const saragaRaga = document.getElementById('saraga-input').value.trim();
            const ramanarunachalamRaga = document.getElementById('ramanarunachalam-input').value.trim();
            const confidence = parseFloat(document.getElementById('confidence-select').value);
            const matchType = document.getElementById('match-type-select').value;
            const notes = document.getElementById('mapping-notes').value.trim();
            
            if (!saragaRaga || !ramanarunachalamRaga) {
                alert('Please fill in both raga names');
                return;
            }
            
            const mapping = {
                saraga_raga: saragaRaga,
                ramanarunachalam_raga: ramanarunachalamRaga,
                confidence: confidence,
                match_type: matchType,
                notes: notes
            };
            
            fetch('/api/create-mapping', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(mapping)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Mapping created successfully!');
                    clearForm();
                    loadExistingMappings();
                    loadUnmappedRagas();
                } else {
                    alert('Error: ' + data.error);
                }
            });
        }
        
        function clearForm() {
            document.getElementById('saraga-input').value = '';
            document.getElementById('ramanarunachalam-input').value = '';
            document.getElementById('confidence-select').value = '0.8';
            document.getElementById('match-type-select').value = 'similar_match';
            document.getElementById('mapping-notes').value = '';
            document.getElementById('saraga-suggestions').style.display = 'none';
            document.getElementById('ramanarunachalam-suggestions').style.display = 'none';
        }
    </script>
</body>
</html>
    """, mapper=mapper)

# API Routes
@app.route('/api/existing-mappings')
def api_existing_mappings():
    """Get all existing mappings"""
    return jsonify(mapper.all_mappings)

@app.route('/api/unmapped-ragas')
def api_unmapped_ragas():
    """Get unmapped Saraga ragas"""
    return jsonify(mapper.get_unmapped_saraga_ragas())

@app.route('/api/search-saraga')
def api_search_saraga():
    """Search Saraga ragas"""
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    query = query.lower()
    matches = []
    for raga in mapper.saraga_ragas:
        if query in raga.lower():
            matches.append(raga)
    
    return jsonify(sorted(matches)[:10])

@app.route('/api/search-ramanarunachalam')
def api_search_ramanarunachalam():
    """Search Ramanarunachalam ragas"""
    return jsonify(mapper.search_ramanarunachalam(request.args.get('q', '')))

@app.route('/api/create-mapping', methods=['POST'])
def api_create_mapping():
    """Create a new mapping"""
    data = request.get_json()
    
    # Add to existing mappings
    new_mapping = {
        'saraga_raga': data['saraga_raga'],
        'ramanarunachalam_raga': data['ramanarunachalam_raga'],
        'confidence': data['confidence'],
        'match_type': data['match_type'],
        'reasoning': data.get('notes', ''),
        'source': 'user_created',
        'created_at': datetime.now().isoformat()
    }
    
    mapper.all_mappings.append(new_mapping)
    
    # Save to file (you can implement this)
    return jsonify({'success': True, 'mapping': new_mapping})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🎵 Starting Improved Raga Mapper...")
    print(f"📊 Loaded {len(mapper.all_mappings)} existing mappings")
    print(f"🔍 {len(mapper.get_unmapped_saraga_ragas())} unmapped Saraga ragas")
    print(f"🌐 Interface: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

