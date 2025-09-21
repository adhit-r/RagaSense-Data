#!/usr/bin/env python3
"""
Raga Data Unification Web Interface
==================================

Web-based interface for the Data Unification Assistant
Provides interactive raga comparison and analysis
"""

import json
import os
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add the scripts/utilities directory to the path
scripts_path = Path(__file__).parent / 'scripts' / 'utilities'
sys.path.insert(0, str(scripts_path))

try:
    from data_unification_assistant import DataUnificationAssistant
except ImportError:
    print("⚠️ Could not import DataUnificationAssistant")
    print("Make sure scripts/utilities/data_unification_assistant.py exists")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# Initialize the unification assistant
assistant = DataUnificationAssistant()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RagaSense Data Unification Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { padding-top: 2rem; }
        .card { 
            border: none; 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            background: rgba(255,255,255,0.95);
        }
        .card-header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px 15px 0 0 !important;
        }
        .confidence-high { background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%); }
        .confidence-medium { background: linear-gradient(135deg, #FFD93D 0%, #FF6B6B 100%); }
        .confidence-low { background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%); }
        .field-analysis { background: #f8f9fa; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
        .dataset-badge { font-size: 0.8em; padding: 0.3rem 0.6rem; }
        .search-box { border-radius: 25px; border: 2px solid #e9ecef; padding: 0.8rem 1.2rem; }
        .btn-primary { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border: none;
            border-radius: 25px;
            padding: 0.8rem 2rem;
        }
        .result-card { margin-bottom: 2rem; transition: transform 0.2s; }
        .result-card:hover { transform: translateY(-2px); }
        .json-display { 
            background: #f8f9fa; 
            border-radius: 10px; 
            padding: 1rem; 
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
            max-height: 400px;
            overflow-y: auto;
        }
        .navbar { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); }
        .navbar-brand { color: white !important; font-weight: bold; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg">
        <div class="container-fluid">
            <span class="navbar-brand">
                <i class="fas fa-music me-2"></i>RagaSense Data Unification Assistant
            </span>
            <span class="text-white">
                <i class="fas fa-database me-1"></i>Saraga + Ramanarunachalam Integration
            </span>
        </div>
    </nav>

    <div class="container">
        <!-- Input Section -->
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-search me-2"></i>Raga Unification</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-8">
                                <input type="text" id="ragaInput" class="form-control search-box" 
                                       placeholder="Enter raga name (e.g., Bhairavi, Yaman, Kambhoji)">
                            </div>
                            <div class="col-md-4">
                                <button class="btn btn-primary w-100" onclick="unifyRaga()">
                                    <i class="fas fa-magic me-1"></i>Unify Data
                                </button>
                            </div>
                        </div>
                        <div class="mt-3">
                            <small class="text-muted">
                                <i class="fas fa-info-circle me-1"></i>
                                Enter a raga name to compare and merge data from Saraga and Ramanarunachalam datasets
                            </small>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div id="resultsSection" style="display: none;">
            <div class="row">
                <div class="col-md-12">
                    <div class="card result-card">
                        <div class="card-header">
                            <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>Unification Results</h5>
                        </div>
                        <div class="card-body" id="resultsContent">
                            <!-- Results will be populated here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Loading Section -->
        <div id="loadingSection" style="display: none;">
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Analyzing and unifying raga data...</p>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function unifyRaga() {
            const ragaName = document.getElementById('ragaInput').value.trim();
            
            if (!ragaName) {
                alert('Please enter a raga name');
                return;
            }

            showLoading();
            
            fetch('/api/unify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ entity_name: ragaName })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayResults(data);
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                alert('Error occurred during unification');
            });
        }

        function showLoading() {
            document.getElementById('loadingSection').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
        }

        function hideLoading() {
            document.getElementById('loadingSection').style.display = 'none';
        }

        function displayResults(data) {
            const resultsContent = document.getElementById('resultsContent');
            
            if (data.status === 'not_found') {
                resultsContent.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        <strong>Not Found:</strong> "${data.entity}" was not found in either dataset.
                    </div>
                `;
            } else {
                resultsContent.innerHTML = generateResultsHTML(data);
            }
            
            document.getElementById('resultsSection').style.display = 'block';
        }

        function generateResultsHTML(data) {
            const confidenceClass = getConfidenceClass(data.confidence);
            const confidencePercent = (data.confidence * 100).toFixed(1);
            
            return `
                <!-- Overview -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card text-center ${confidenceClass} text-white">
                            <div class="card-body">
                                <h4>${confidencePercent}%</h4>
                                <p class="mb-0">Match Confidence</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center ${data.saraga_found ? 'bg-success' : 'bg-secondary'} text-white">
                            <div class="card-body">
                                <h4><i class="fas ${data.saraga_found ? 'fa-check' : 'fa-times'}"></i></h4>
                                <p class="mb-0">Saraga Dataset</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center ${data.ramanarunachalam_found ? 'bg-success' : 'bg-secondary'} text-white">
                            <div class="card-body">
                                <h4><i class="fas ${data.ramanarunachalam_found ? 'fa-check' : 'fa-times'}"></i></h4>
                                <p class="mb-0">Ramanarunachalam</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center bg-info text-white">
                            <div class="card-body">
                                <h4>${data.field_analysis.overlap_percentage.toFixed(1)}%</h4>
                                <p class="mb-0">Field Overlap</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Field Analysis -->
                <div class="field-analysis">
                    <h6><i class="fas fa-chart-bar me-2"></i>Field Analysis</h6>
                    <div class="row">
                        <div class="col-md-4">
                            <strong>Common Fields (${data.field_analysis.common_fields.length}):</strong>
                            <ul class="list-unstyled mt-2">
                                ${data.field_analysis.common_fields.map(field => `<li><span class="badge bg-success me-1">${field}</span></li>`).join('')}
                            </ul>
                        </div>
                        <div class="col-md-4">
                            <strong>Missing from Saraga (${data.field_analysis.missing_from_a.length}):</strong>
                            <ul class="list-unstyled mt-2">
                                ${data.field_analysis.missing_from_a.map(field => `<li><span class="badge bg-warning me-1">${field}</span></li>`).join('')}
                            </ul>
                        </div>
                        <div class="col-md-4">
                            <strong>Missing from Ramanarunachalam (${data.field_analysis.missing_from_b.length}):</strong>
                            <ul class="list-unstyled mt-2">
                                ${data.field_analysis.missing_from_b.map(field => `<li><span class="badge bg-info me-1">${field}</span></li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>

                <!-- Confidence Reason -->
                <div class="alert alert-info">
                    <strong>Match Reason:</strong> ${data.confidence_reason}
                </div>

                <!-- Merged Data Preview -->
                <div class="mt-4">
                    <h6><i class="fas fa-code me-2"></i>Merged Data Structure</h6>
                    <div class="json-display">
                        <pre>${JSON.stringify(data.merged_data, null, 2)}</pre>
                    </div>
                </div>

                <!-- Raw Results -->
                <div class="mt-4">
                    <h6><i class="fas fa-database me-2"></i>Complete Analysis Results</h6>
                    <div class="json-display">
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>
                </div>
            `;
        }

        function getConfidenceClass(confidence) {
            if (confidence >= 0.8) return 'confidence-high';
            if (confidence >= 0.6) return 'confidence-medium';
            return 'confidence-low';
        }

        // Allow Enter key to trigger search
        document.getElementById('ragaInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                unifyRaga();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/unify', methods=['POST'])
def unify_entity():
    try:
        data = request.get_json()
        entity_name = data.get('entity_name', '')
        
        if not entity_name:
            return jsonify({'error': 'Entity name is required'}), 400
        
        result = assistant.unify_entity(entity_name)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_unify', methods=['POST'])
def batch_unify_entities():
    try:
        data = request.get_json()
        entity_names = data.get('entity_names', [])
        
        if not entity_names:
            return jsonify({'error': 'Entity names list is required'}), 400
        
        results = assistant.batch_unify_entities(entity_names)
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/available_ragas')
def get_available_ragas():
    """Get list of available ragas from both datasets"""
    try:
        saraga_ragas = list(assistant.saraga_data.keys())
        ramanarunachalam_ragas = list(assistant.ramanarunachalam_data.keys())
        
        return jsonify({
            'saraga_ragas': saraga_ragas[:20],  # First 20 for preview
            'ramanarunachalam_ragas': ramanarunachalam_ragas[:20],
            'total_saraga': len(saraga_ragas),
            'total_ramanarunachalam': len(ramanarunachalam_ragas)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting RagaSense Data Unification Web Interface...")
    print("🔗 Connecting to datasets...")
    print(f"📊 Loaded {len(assistant.saraga_data)} Saraga ragas")
    print(f"📚 Loaded {len(assistant.ramanarunachalam_data)} Ramanarunachalam ragas")
    print("🌐 Open your browser to: http://localhost:5003")
    app.run(debug=True, host='0.0.0.0', port=5003)