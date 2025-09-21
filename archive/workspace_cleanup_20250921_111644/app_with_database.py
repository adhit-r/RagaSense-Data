#!/usr/bin/env python3
"""
RagaSense-Data Flask Application with Vercel Postgres Database
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

class DatabaseManager:
    """Database connection and query manager"""
    
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Connect to Vercel Postgres database"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST'),
                database=os.getenv('POSTGRES_DATABASE'),
                user=os.getenv('POSTGRES_USER'),
                password=os.getenv('POSTGRES_PASSWORD'),
                port=os.getenv('POSTGRES_PORT', 5432)
            )
            print("✅ Connected to Vercel Postgres database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.connection = None
    
    def get_cursor(self):
        """Get database cursor"""
        if not self.connection:
            self.connect()
        return self.connection.cursor(cursor_factory=RealDictCursor)
    
    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        try:
            cursor = self.get_cursor()
            cursor.execute(query, params)
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                self.connection.commit()
                return cursor.rowcount
        except Exception as e:
            print(f"Database error: {e}")
            if self.connection:
                self.connection.rollback()
            return None

# Initialize database manager
db = DatabaseManager()

class CollaborativeRagaMapper:
    """Collaborative raga mapping system with database backend"""
    
    def __init__(self):
        self.db = db
    
    def get_pending_mappings(self):
        """Get pending mapping proposals"""
        query = """
            SELECT mp.*, 
                   COUNT(mv.id) as vote_count,
                   COUNT(CASE WHEN mv.vote_type = 'approve' THEN 1 END) as approve_count,
                   COUNT(CASE WHEN mv.vote_type = 'reject' THEN 1 END) as reject_count
            FROM mapping_proposals mp
            LEFT JOIN mapping_votes mv ON mp.id = mv.proposal_id
            WHERE mp.status = 'pending'
            GROUP BY mp.id
            ORDER BY mp.created_at DESC
        """
        return self.db.execute_query(query)
    
    def get_approved_mappings(self):
        """Get approved mapping proposals"""
        query = """
            SELECT mp.*, 
                   COUNT(mv.id) as vote_count,
                   COUNT(CASE WHEN mv.vote_type = 'approve' THEN 1 END) as approve_count,
                   COUNT(CASE WHEN mv.vote_type = 'reject' THEN 1 END) as reject_count
            FROM mapping_proposals mp
            LEFT JOIN mapping_votes mv ON mp.id = mv.proposal_id
            WHERE mp.status = 'approved'
            GROUP BY mp.id
            ORDER BY mp.created_at DESC
        """
        return self.db.execute_query(query)
    
    def get_rejected_mappings(self):
        """Get rejected mapping proposals"""
        query = """
            SELECT mp.*, 
                   COUNT(mv.id) as vote_count,
                   COUNT(CASE WHEN mv.vote_type = 'approve' THEN 1 END) as approve_count,
                   COUNT(CASE WHEN mv.vote_type = 'reject' THEN 1 END) as reject_count
            FROM mapping_proposals mp
            LEFT JOIN mapping_votes mv ON mp.id = mv.proposal_id
            WHERE mp.status = 'rejected'
            GROUP BY mp.id
            ORDER BY mp.created_at DESC
        """
        return self.db.execute_query(query)
    
    def propose_mapping(self, saraga_raga, ramanarunachalam_raga, proposed_by, confidence, notes):
        """Propose a new mapping"""
        query = """
            INSERT INTO mapping_proposals 
            (saraga_raga, ramanarunachalam_raga, proposed_by, confidence, notes)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """
        result = self.db.execute_query(query, (saraga_raga, ramanarunachalam_raga, proposed_by, confidence, notes))
        return result[0]['id'] if result else None
    
    def vote_on_mapping(self, proposal_id, user_id, vote_type, confidence, notes):
        """Vote on a mapping proposal"""
        # Check if user already voted
        check_query = "SELECT id FROM mapping_votes WHERE proposal_id = %s AND user_id = %s"
        existing = self.db.execute_query(check_query, (proposal_id, user_id))
        
        if existing:
            # Update existing vote
            query = """
                UPDATE mapping_votes 
                SET vote_type = %s, confidence = %s, notes = %s
                WHERE proposal_id = %s AND user_id = %s
            """
            self.db.execute_query(query, (vote_type, confidence, notes, proposal_id, user_id))
        else:
            # Insert new vote
            query = """
                INSERT INTO mapping_votes 
                (proposal_id, user_id, vote_type, confidence, notes)
                VALUES (%s, %s, %s, %s, %s)
            """
            self.db.execute_query(query, (proposal_id, user_id, vote_type, confidence, notes))
        
        # Check if proposal should be auto-approved/rejected
        self.check_proposal_status(proposal_id)
    
    def check_proposal_status(self, proposal_id):
        """Check if proposal should be auto-approved or rejected based on votes"""
        query = """
            SELECT 
                COUNT(*) as total_votes,
                COUNT(CASE WHEN vote_type = 'approve' THEN 1 END) as approve_votes,
                COUNT(CASE WHEN vote_type = 'reject' THEN 1 END) as reject_votes
            FROM mapping_votes 
            WHERE proposal_id = %s
        """
        result = self.db.execute_query(query, (proposal_id,))
        
        if result:
            stats = result[0]
            total_votes = stats['total_votes']
            approve_votes = stats['approve_votes']
            reject_votes = stats['reject_votes']
            
            if total_votes >= 3:
                approval_rate = approve_votes / total_votes
                rejection_rate = reject_votes / total_votes
                
                if approval_rate >= 0.67:  # 67% approval
                    self.update_proposal_status(proposal_id, 'approved')
                elif rejection_rate >= 0.33:  # 33% rejection
                    self.update_proposal_status(proposal_id, 'rejected')
    
    def update_proposal_status(self, proposal_id, status):
        """Update proposal status"""
        query = "UPDATE mapping_proposals SET status = %s WHERE id = %s"
        self.db.execute_query(query, (status, proposal_id))
    
    def get_user_stats(self, user_id):
        """Get user contribution statistics"""
        query = """
            SELECT 
                COUNT(CASE WHEN mp.proposed_by = %s THEN 1 END) as proposals_made,
                COUNT(CASE WHEN mv.user_id = %s THEN 1 END) as votes_cast,
                COUNT(CASE WHEN mp.proposed_by = %s AND mp.status = 'approved' THEN 1 END) as proposals_approved,
                COUNT(CASE WHEN mp.proposed_by = %s AND mp.status = 'pending' THEN 1 END) as proposals_pending
            FROM mapping_proposals mp
            FULL OUTER JOIN mapping_votes mv ON mv.user_id = %s
        """
        result = self.db.execute_query(query, (user_id, user_id, user_id, user_id, user_id))
        return result[0] if result else {}
    
    def get_ragas_by_tradition(self, tradition=None):
        """Get ragas filtered by tradition"""
        if tradition:
            query = "SELECT * FROM ragas WHERE tradition = %s ORDER BY name"
            return self.db.execute_query(query, (tradition,))
        else:
            query = "SELECT * FROM ragas ORDER BY name"
            return self.db.execute_query(query)
    
    def search_ragas(self, search_term):
        """Search ragas by name"""
        query = "SELECT * FROM ragas WHERE name ILIKE %s ORDER BY name"
        return self.db.execute_query(query, (f'%{search_term}%',))

# Initialize collaborative mapper
collaborative_mapper = CollaborativeRagaMapper()

@app.route('/')
def index():
    """Main landing page"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RagaSense-Data | Unified Indian Classical Music Dataset</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .hero-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 0; }
        .stat-card { background: white; border-radius: 8px; padding: 1.5rem; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .feature-card { border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; margin: 0.5rem 0; }
    </style>
</head>
<body>
    <div class="hero-section">
        <div class="container text-center">
            <h1 class="display-4">🎵 RagaSense-Data</h1>
            <p class="lead">Unified Indian Classical Music Dataset with Collaborative Mapping</p>
            <p>Comprehensive collection of Carnatic and Hindustani music data with research-grade annotations</p>
        </div>
    </div>
    
    <div class="container mt-5">
        <div class="row">
            <div class="col-md-4">
                <div class="stat-card text-center">
                    <h3>🎼 Dataset Overview</h3>
                    <p>Explore our unified dataset combining Ramanarunachalam and Saraga sources</p>
                    <a href="/dataset" class="btn btn-primary">View Dataset</a>
                </div>
            </div>
            <div class="col-md-4">
                <div class="stat-card text-center">
                    <h3>🔍 Raga Explorer</h3>
                    <p>Search and explore ragas from both Carnatic and Hindustani traditions</p>
                    <a href="/explorer" class="btn btn-success">Explore Ragas</a>
                </div>
            </div>
            <div class="col-md-4">
                <div class="stat-card text-center">
                    <h3>👥 Collaborative Mapping</h3>
                    <p>Propose and vote on raga mappings with the community</p>
                    <a href="/collaborative" class="btn btn-warning">Join Community</a>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """)

@app.route('/collaborative')
def collaborative_mapping():
    """Collaborative mapping interface"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Collaborative Raga Mapping - RagaSense-Data</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .mapping-card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
        .status-approved { border-left: 4px solid #28a745; }
        .status-pending { border-left: 4px solid #ffc107; }
        .status-rejected { border-left: 4px solid #dc3545; }
        .vote-buttons { margin-top: 1rem; }
        .stats-card { background: #f8f9fa; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
        .hero-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 0; }
    </style>
</head>
<body>
    <div class="hero-section">
        <div class="container text-center">
            <h1 class="display-4">🎵 Collaborative Raga Mapping</h1>
            <p class="lead">Help build consensus on raga mappings between Saraga and Ramanarunachalam datasets</p>
            <a href="/" class="btn btn-light btn-lg">← Back to Main Interface</a>
        </div>
    </div>
    
    <div class="container mt-4">
        <!-- User ID Input -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5>👤 Your Identity</h5>
                        <input type="text" class="form-control" id="user-id" placeholder="Enter your name or ID">
                        <small class="text-muted">This helps track contributions and prevent duplicate votes</small>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5>📊 Your Stats</h5>
                        <div id="user-stats">Enter your ID to see stats</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- New Mapping Proposal -->
        <div class="card mb-4">
            <div class="card-header">
                <h5>➕ Propose New Mapping</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <label class="form-label">Saraga Raga</label>
                        <input type="text" class="form-control" id="saraga-raga" placeholder="e.g., sankarabharanam">
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Ramanarunachalam Raga</label>
                        <input type="text" class="form-control" id="ramanarunachalam-raga" placeholder="e.g., Sankarabharanam">
                    </div>
                    <div class="col-md-2">
                        <label class="form-label">Confidence</label>
                        <select class="form-select" id="confidence">
                            <option value="0.9">Very High (90%)</option>
                            <option value="0.8">High (80%)</option>
                            <option value="0.7">Medium-High (70%)</option>
                            <option value="0.6">Medium (60%)</option>
                            <option value="0.5" selected>Low-Medium (50%)</option>
                        </select>
                    </div>
                    <div class="col-md-2">
                        <label class="form-label">&nbsp;</label>
                        <button class="btn btn-primary d-block w-100" onclick="proposeMapping()">Propose</button>
                    </div>
                </div>
                <div class="row mt-2">
                    <div class="col-12">
                        <label class="form-label">Notes (optional)</label>
                        <textarea class="form-control" id="notes" rows="2" placeholder="Any additional context or reasoning..."></textarea>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Navigation Tabs -->
        <ul class="nav nav-tabs" id="mapping-tabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="pending-tab" data-bs-toggle="tab" data-bs-target="#pending" type="button">⏳ Pending Review</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="approved-tab" data-bs-toggle="tab" data-bs-target="#approved" type="button">✅ Approved</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="rejected-tab" data-bs-toggle="tab" data-bs-target="#rejected" type="button">❌ Rejected</button>
            </li>
        </ul>
        
        <!-- Tab Content -->
        <div class="tab-content" id="mapping-tabs-content">
            <div class="tab-pane fade show active" id="pending" role="tabpanel">
                <div id="pending-mappings">Loading...</div>
            </div>
            <div class="tab-pane fade" id="approved" role="tabpanel">
                <div id="approved-mappings">Loading...</div>
            </div>
            <div class="tab-pane fade" id="rejected" role="tabpanel">
                <div id="rejected-mappings">Loading...</div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentUserId = '';
        document.addEventListener('DOMContentLoaded', function() { loadMappings(); });
        document.getElementById('user-id').addEventListener('input', function() { 
            currentUserId = this.value.trim();
            if (currentUserId) loadUserStats();
        });
        
        function loadMappings() { 
            loadPendingMappings();
            loadApprovedMappings();
            loadRejectedMappings();
        }
        
        function loadPendingMappings() {
            fetch('/api/collaborative/pending')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('pending-mappings').innerHTML = renderMappings(data, 'pending');
                });
        }
        
        function loadApprovedMappings() {
            fetch('/api/collaborative/approved')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('approved-mappings').innerHTML = renderMappings(data, 'approved');
                });
        }
        
        function loadRejectedMappings() {
            fetch('/api/collaborative/rejected')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('rejected-mappings').innerHTML = renderMappings(data, 'rejected');
                });
        }
        
        function renderMappings(mappings, status) {
            if (mappings.length === 0) {
                return '<div class="text-center text-gray-500 py-8">No ' + status + ' mappings found.</div>';
            }
            
            return mappings.map(mapping => `
                <div class="mapping-card ${status === 'approved' ? 'status-approved' : status === 'rejected' ? 'status-rejected' : 'status-pending'}">
                    <div class="d-flex justify-content-between">
                        <div class="flex-grow-1">
                            <h5>${mapping.saraga_raga} → ${mapping.ramanarunachalam_raga}</h5>
                            <p><strong>Proposed by:</strong> ${mapping.proposed_by}</p>
                            <p><strong>Confidence:</strong> ${Math.round(mapping.confidence * 100)}%</p>
                            ${mapping.notes ? `<p><strong>Notes:</strong> ${mapping.notes}</p>` : ''}
                        </div>
                        <div class="ms-3">
                            <div class="stats-card">
                                <div><strong>Votes:</strong> ${mapping.vote_count || 0}</div>
                                <div><strong>Approve:</strong> ${mapping.approve_count || 0}</div>
                                <div><strong>Reject:</strong> ${mapping.reject_count || 0}</div>
                            </div>
                            ${status === 'pending' && currentUserId ? `
                                <div class="vote-buttons">
                                    <button class="btn btn-success btn-sm" onclick="voteOnMapping('${mapping.id}', 'approve')">Approve</button>
                                    <button class="btn btn-danger btn-sm" onclick="voteOnMapping('${mapping.id}', 'reject')">Reject</button>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        function proposeMapping() {
            if (!currentUserId) {
                alert('Please enter your user ID first');
                return;
            }
            
            const saragaRaga = document.getElementById('saraga-raga').value.trim();
            const ramanarunachalamRaga = document.getElementById('ramanarunachalam-raga').value.trim();
            const confidence = parseFloat(document.getElementById('confidence').value);
            const notes = document.getElementById('notes').value.trim();
            
            if (!saragaRaga || !ramanarunachalamRaga) {
                alert('Please fill in both raga names');
                return;
            }
            
            fetch('/api/collaborative/propose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    saraga_raga: saragaRaga,
                    ramanarunachalam_raga: ramanarunachalamRaga,
                    proposed_by: currentUserId,
                    confidence: confidence,
                    notes: notes
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Mapping proposed successfully!');
                    document.getElementById('saraga-raga').value = '';
                    document.getElementById('ramanarunachalam-raga').value = '';
                    document.getElementById('notes').value = '';
                    loadMappings();
                } else {
                    alert('Error: ' + data.error);
                }
            });
        }
        
        function voteOnMapping(proposalId, voteType) {
            if (!currentUserId) {
                alert('Please enter your user ID first');
                return;
            }
            
            const confidence = parseFloat(prompt('Rate your confidence in this vote (0.5-1.0):', '0.8'));
            if (isNaN(confidence) || confidence < 0.5 || confidence > 1.0) {
                alert('Please enter a valid confidence between 0.5 and 1.0');
                return;
            }
            
            const notes = prompt('Optional notes about your vote:') || '';
            
            fetch('/api/collaborative/vote', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    proposal_id: proposalId,
                    user_id: currentUserId,
                    vote_type: voteType,
                    confidence: confidence,
                    notes: notes
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Vote recorded successfully!');
                    loadMappings();
                } else {
                    alert('Error: ' + data.error);
                }
            });
        }
        
        function loadUserStats() {
            if (!currentUserId) return;
            
            fetch(`/api/collaborative/user-stats/${currentUserId}`)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('user-stats').innerHTML = `
                        <div>📝 Proposals: ${data.proposals_made || 0}</div>
                        <div>🗳️ Votes: ${data.votes_cast || 0}</div>
                        <div>✅ Approved: ${data.proposals_approved || 0}</div>
                        <div>⏳ Pending: ${data.proposals_pending || 0}</div>
                    `;
                });
        }
    </script>
</body>
</html>
    """)

# API Routes
@app.route('/api/collaborative/pending')
def api_pending_mappings():
    """Get pending mappings"""
    mappings = collaborative_mapper.get_pending_mappings()
    return jsonify(mappings if mappings else [])

@app.route('/api/collaborative/approved')
def api_approved_mappings():
    """Get approved mappings"""
    mappings = collaborative_mapper.get_approved_mappings()
    return jsonify(mappings if mappings else [])

@app.route('/api/collaborative/rejected')
def api_rejected_mappings():
    """Get rejected mappings"""
    mappings = collaborative_mapper.get_rejected_mappings()
    return jsonify(mappings if mappings else [])

@app.route('/api/collaborative/propose', methods=['POST'])
def api_propose_mapping():
    """Propose a new mapping"""
    data = request.get_json()
    
    proposal_id = collaborative_mapper.propose_mapping(
        data['saraga_raga'],
        data['ramanarunachalam_raga'],
        data['proposed_by'],
        data['confidence'],
        data.get('notes', '')
    )
    
    if proposal_id:
        return jsonify({'success': True, 'proposal_id': proposal_id})
    else:
        return jsonify({'success': False, 'error': 'Failed to create proposal'})

@app.route('/api/collaborative/vote', methods=['POST'])
def api_vote_on_mapping():
    """Vote on a mapping proposal"""
    data = request.get_json()
    
    collaborative_mapper.vote_on_mapping(
        data['proposal_id'],
        data['user_id'],
        data['vote_type'],
        data['confidence'],
        data.get('notes', '')
    )
    
    return jsonify({'success': True})

@app.route('/api/collaborative/user-stats/<user_id>')
def api_user_stats(user_id):
    """Get user statistics"""
    stats = collaborative_mapper.get_user_stats(user_id)
    return jsonify(stats)

@app.route('/api/ragas')
def api_ragas():
    """Get ragas with optional tradition filter"""
    tradition = request.args.get('tradition')
    search = request.args.get('search')
    
    if search:
        ragas = collaborative_mapper.search_ragas(search)
    elif tradition:
        ragas = collaborative_mapper.get_ragas_by_tradition(tradition)
    else:
        ragas = collaborative_mapper.get_ragas_by_tradition()
    
    return jsonify(ragas if ragas else [])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

