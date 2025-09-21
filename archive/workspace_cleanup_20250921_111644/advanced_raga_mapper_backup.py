#!/usr/bin/env python3
"""
Advanced 3-Panel Raga Mapper with Professional UX
"""

import json
import os
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from datetime import datetime
import difflib

app = Flask(__name__)
CORS(app)

class AdvancedRagaMapper:
    """Advanced raga mapper with 3-panel layout and professional UX"""
    
    def __init__(self):
        self.load_existing_mappings()
        self.load_raga_data()
        self.session_mappings = []  # Current session mappings
    
    def load_existing_mappings(self):
        """Load all existing cross-tradition mappings"""
        self.existing_mappings = []
        
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
                            self.existing_mappings.append({
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
                                self.existing_mappings.append({
                                    'saraga_raga': match[0],
                                    'ramanarunachalam_raga': match[1],
                                    'confidence': 0.8,
                                    'match_type': 'similar_match',
                                    'source': 'existing'
                                })
                    
                    # Extract semantic matches
                    if 'semantic_matches' in data:
                        for match in data['semantic_matches']:
                            self.existing_mappings.append({
                                'saraga_raga': match['saraga_raga'],
                                'ramanarunachalam_raga': match['ramanarunachalam_raga'],
                                'confidence': match.get('confidence', 0.8),
                                'match_type': match.get('match_type', 'semantic_match'),
                                'reasoning': match.get('reasoning', ''),
                                'source': 'existing'
                            })
                
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"✅ Loaded {len(self.existing_mappings)} existing mappings")
    
    def load_raga_data(self):
        """Load raga data from separate Saraga and Ramanarunachalam datasets"""
        self.saraga_ragas = {
            'carnatic': [],
            'hindustani': [],
            'all': []
        }
        self.ramanarunachalam_ragas = []
        self.raga_details = {}  # Store detailed raga information
        self.ramanarunachalam_details = {}  # Store Ramanarunachalam raga details
        
        # Load actual Saraga dataset
        self.load_saraga_data()
        
        # Load actual Ramanarunachalam dataset
        self.load_ramanarunachalam_data()
        
        # Load cross-tradition mappings for Saraga-only ragas
        self.load_saraga_only_ragas()
        
        # Combine all Saraga ragas
        self.saraga_ragas['all'] = self.saraga_ragas['carnatic'] + self.saraga_ragas['hindustani']
        
        print(f"✅ Loaded {len(self.saraga_ragas['carnatic'])} Carnatic and {len(self.saraga_ragas['hindustani'])} Hindustani Saraga ragas")
        print(f"✅ Loaded {len(self.ramanarunachalam_ragas)} Ramanarunachalam ragas")
    
    def load_saraga_data(self):
        """Load actual Saraga dataset"""
        print("📊 Loading Saraga dataset...")
        
        try:
            with open('data/02_raw/extracted_saraga_metadata/combined_metadata_extracted.json', 'r') as f:
                saraga_data = json.load(f)
            
            for recording in saraga_data:
                raga_name = recording.get('raga_name', '').lower()
                tradition = recording.get('tradition', '').lower()
                
                if raga_name and tradition:
                    if tradition == 'carnatic' and raga_name not in self.saraga_ragas['carnatic']:
                        self.saraga_ragas['carnatic'].append(raga_name)
                        self.raga_details[raga_name] = {
                            'name': recording.get('raga_name', ''),
                            'tradition': 'Carnatic',
                            'melakartha': '',  # Saraga doesn't have melakartha info
                            'arohana': '',     # Saraga doesn't have arohana info
                            'avarohana': '',   # Saraga doesn't have avarohana info
                            'data_source': 'Saraga Dataset',
                            'counts': {
                                'composers': 0,
                                'songs': 0,
                                'types': 0,
                                'videos': 0
                            },
                            'id': recording.get('raga_name', '')
                        }
                    elif tradition == 'hindustani' and raga_name not in self.saraga_ragas['hindustani']:
                        self.saraga_ragas['hindustani'].append(raga_name)
                        self.raga_details[raga_name] = {
                            'name': recording.get('raga_name', ''),
                            'tradition': 'Hindustani',
                            'melakartha': '',  # Saraga doesn't have melakartha info
                            'arohana': '',     # Saraga doesn't have arohana info
                            'avarohana': '',   # Saraga doesn't have avarohana info
                            'data_source': 'Saraga Dataset',
                            'counts': {
                                'composers': 0,
                                'songs': 0,
                                'types': 0,
                                'videos': 0
                            },
                            'id': recording.get('raga_name', '')
                        }
            
        except Exception as e:
            print(f"❌ Error loading Saraga data: {e}")
            # Fallback to minimal data
            self.saraga_ragas['carnatic'] = ['bhairavi', 'kalyani', 'sankarabharanam']
            self.saraga_ragas['hindustani'] = ['yaman', 'kafi', 'khamaj']
    
    def load_ramanarunachalam_data(self):
        """Load actual Ramanarunachalam dataset"""
        print("📊 Loading Ramanarunachalam dataset...")
        
        try:
            ramanarunachalam_path = 'data/01_source/ramanarunachalam'
            
            for tradition in ['Carnatic', 'Hindustani']:
                path = os.path.join(ramanarunachalam_path, tradition, 'raga')
                if os.path.exists(path):
                    for filename in os.listdir(path):
                        if filename.endswith('.json'):
                            raga_name = filename[:-5]  # Remove .json extension
                            raga_lower = raga_name.lower()
                            
                            # Add to Ramanarunachalam ragas list
                            if raga_name not in self.ramanarunachalam_ragas:
                                self.ramanarunachalam_ragas.append(raga_name)
                            
                            # Load detailed raga information
                            try:
                                with open(os.path.join(path, filename), 'r') as f:
                                    raga_data = json.load(f)
                                
                                self.ramanarunachalam_details[raga_lower] = {
                                    'name': raga_data.get('name', raga_name),
                                    'tradition': raga_data.get('tradition', tradition),
                                    'melakartha': f"{raga_data.get('melakarta', '')} {raga_data.get('name', '')}" if raga_data.get('melakarta') else '',
                                    'arohana': raga_data.get('arohana', ''),
                                    'avarohana': raga_data.get('avarohana', ''),
                                    'data_source': 'Ramanarunachalam Dataset',
                                    'counts': {
                                        'composers': len(raga_data.get('composers', [])),
                                        'songs': len(raga_data.get('songs', [])),
                                        'types': 1,
                                        'videos': 0
                                    },
                                    'id': raga_data.get('name', raga_name)
                                }
                            except Exception as e:
                                print(f"⚠️ Error loading {filename}: {e}")
            
        except Exception as e:
            print(f"❌ Error loading Ramanarunachalam data: {e}")
            # Fallback to minimal data
            self.ramanarunachalam_ragas = ['Bhairavi', 'Kalyani', 'Sankarabharanam']
    
    def load_saraga_only_ragas(self):
        """Load Saraga-only ragas from cross-tradition mappings"""
        print("📊 Loading Saraga-only ragas...")
        
        try:
            with open('data/04_ml_datasets/unified/cross_tradition_mappings_final_20250914_163006.json', 'r') as f:
                mappings_data = json.load(f)
            
            # Add Saraga-only ragas (these are the actual Saraga ragas)
            if 'saraga_only' in mappings_data:
                for raga in mappings_data['saraga_only']:
                    raga_lower = raga.lower()
                    # Classify Saraga ragas by tradition (most are Carnatic from Saraga dataset)
                    if any(keyword in raga_lower for keyword in ['bhairavi', 'kambhoji', 'mohanam', 'kalyani', 'sankarabharanam', 'janaranjani', 'lalit', 'sindhubhairavi', 'karaharapriya', 'ranjani', 'rasikapriya']):
                        if raga_lower not in self.saraga_ragas['carnatic']:
                            self.saraga_ragas['carnatic'].append(raga_lower)
                            # Add basic info for Saraga-only ragas
                            if raga_lower not in self.raga_details:
                                self.raga_details[raga_lower] = {
                                    'name': raga,
                                    'tradition': 'Carnatic',
                                    'melakartha': '',
                                    'arohana': '',
                                    'avarohana': '',
                                    'data_source': 'Saraga Dataset (Cross-tradition)',
                                    'counts': {
                                        'composers': 0,
                                        'songs': 0,
                                        'types': 0,
                                        'videos': 0
                                    },
                                    'id': raga
                                }
                    else:
                        if raga_lower not in self.saraga_ragas['hindustani']:
                            self.saraga_ragas['hindustani'].append(raga_lower)
                            # Add basic info for Saraga-only ragas
                            if raga_lower not in self.raga_details:
                                self.raga_details[raga_lower] = {
                                    'name': raga,
                                    'tradition': 'Hindustani',
                                    'melakartha': '',
                                    'arohana': '',
                                    'avarohana': '',
                                    'data_source': 'Saraga Dataset (Cross-tradition)',
                                    'counts': {
                                        'composers': 0,
                                        'songs': 0,
                                        'types': 0,
                                        'videos': 0
                                    },
                                    'id': raga
                                }
            
        except Exception as e:
            print(f"❌ Error loading Saraga-only ragas: {e}")
    
    def get_mapped_saraga_ragas(self):
                        self.saraga_ragas['carnatic'].append(raga_name)
                        self.ramanarunachalam_ragas.append(raga_info.get('raga_name', ''))
                        # Store detailed raga information
                        self.raga_details[raga_name] = {
                            'name': raga_info.get('raga_name', ''),
                            'tradition': 'Carnatic',
                            'melakartha': raga_info.get('melakartha', ''),
                            'arohana': raga_info.get('arohana', ''),
                            'avarohana': raga_info.get('avarohana', ''),
                            'data_source': 'Saraga Dataset',
                            'counts': raga_info.get('counts', {}),
                            'id': raga_info.get('id', '')
                        }
                        
                        # Also store in Ramanarunachalam details if it's from Ramanarunachalam
                        audio_files = raga_info.get('audio_files', [])
                        if audio_files and any('ramanarunachalam' in file.lower() for file in audio_files):
                            self.ramanarunachalam_details[raga_name] = {
                                'name': raga_info.get('raga_name', ''),
                                'tradition': 'Carnatic',
                                'melakartha': raga_info.get('melakartha', ''),
                                'arohana': raga_info.get('arohana', ''),
                                'avarohana': raga_info.get('avarohana', ''),
                                'data_source': 'Ramanarunachalam Dataset',
                                'counts': raga_info.get('counts', {}),
                                'id': raga_info.get('id', '')
                            }
            
            # Extract Hindustani ragas with details
            if 'hindustani' in tradition_data and 'ragas' in tradition_data['hindustani']:
                for raga_info in tradition_data['hindustani']['ragas']:
                    raga_name = raga_info.get('raga_name', '').lower()
                    if raga_name:
                        self.saraga_ragas['hindustani'].append(raga_name)
                        self.ramanarunachalam_ragas.append(raga_info.get('raga_name', ''))
                        # Store detailed raga information
                        self.raga_details[raga_name] = {
                            'name': raga_info.get('raga_name', ''),
                            'tradition': 'Hindustani',
                            'melakartha': raga_info.get('melakartha', ''),
                            'arohana': raga_info.get('arohana', ''),
                            'avarohana': raga_info.get('avarohana', ''),
                            'data_source': 'Saraga Dataset',
                            'counts': raga_info.get('counts', {}),
                            'id': raga_info.get('id', '')
                        }
                        
                        # Also store in Ramanarunachalam details if it's from Ramanarunachalam
                        audio_files = raga_info.get('audio_files', [])
                        if audio_files and any('ramanarunachalam' in file.lower() for file in audio_files):
                            self.ramanarunachalam_details[raga_name] = {
                                'name': raga_info.get('raga_name', ''),
                                'tradition': 'Hindustani',
                                'melakartha': raga_info.get('melakartha', ''),
                                'arohana': raga_info.get('arohana', ''),
                                'avarohana': raga_info.get('avarohana', ''),
                                'data_source': 'Ramanarunachalam Dataset',
                                'counts': raga_info.get('counts', {}),
                                'id': raga_info.get('id', '')
                            }
            
            # Load Saraga-specific ragas from cross-tradition mappings
            with open('data/04_ml_datasets/unified/cross_tradition_mappings_final_20250914_163006.json', 'r') as f:
                mappings_data = json.load(f)
            
            # Add Saraga-only ragas (these are the actual Saraga ragas)
            if 'saraga_only' in mappings_data:
                for raga in mappings_data['saraga_only']:
                    raga_lower = raga.lower()
                    # Classify Saraga ragas by tradition (most are Carnatic from Saraga dataset)
                    if any(keyword in raga_lower for keyword in ['bhairavi', 'kambhoji', 'mohanam', 'kalyani', 'sankarabharanam', 'janaranjani', 'lalit', 'sindhubhairavi', 'karaharapriya', 'ranjani', 'rasikapriya']):
                        if raga_lower not in self.saraga_ragas['carnatic']:
                            self.saraga_ragas['carnatic'].append(raga_lower)
                            # Add basic info for Saraga-only ragas
                            if raga_lower not in self.raga_details:
                                self.raga_details[raga_lower] = {
                                    'name': raga,
                                    'tradition': 'Carnatic',
                                    'melakartha': '',
                                    'arohana': '',
                                    'avarohana': '',
                                    'data_source': 'Saraga Dataset (Cross-tradition)',
                                    'counts': {},
                                    'id': raga
                                }
                    else:
                        if raga_lower not in self.saraga_ragas['hindustani']:
                            self.saraga_ragas['hindustani'].append(raga_lower)
                            # Add basic info for Saraga-only ragas
                            if raga_lower not in self.raga_details:
                                self.raga_details[raga_lower] = {
                                    'name': raga,
                                    'tradition': 'Hindustani',
                                    'melakartha': '',
                                    'arohana': '',
                                    'avarohana': '',
                                    'data_source': 'Saraga Dataset (Cross-tradition)',
                                    'counts': {},
                                    'id': raga
                                }
            
            # Add common ragas to both traditions
            if 'common_ragas' in mappings_data:
                for raga in mappings_data['common_ragas']:
                    raga_lower = raga.lower()
                    if raga_lower not in self.saraga_ragas['carnatic']:
                        self.saraga_ragas['carnatic'].append(raga_lower)
                    if raga_lower not in self.saraga_ragas['hindustani']:
                        self.saraga_ragas['hindustani'].append(raga_lower)
                    # Add basic info for common ragas
                    if raga_lower not in self.raga_details:
                        self.raga_details[raga_lower] = {
                            'name': raga,
                            'tradition': 'Common',
                            'melakartha': '',
                            'arohana': '',
                            'avarohana': '',
                            'data_source': 'Cross-tradition Common',
                            'counts': {},
                            'id': raga
                        }
            
            # Combine all ragas
            self.saraga_ragas['all'] = list(set(self.saraga_ragas['carnatic'] + self.saraga_ragas['hindustani']))
            
            # Sort all lists
            for key in self.saraga_ragas:
                self.saraga_ragas[key].sort()
            self.ramanarunachalam_ragas.sort()
            
            print(f"✅ Loaded {len(self.saraga_ragas['all'])} Saraga ragas ({len(self.saraga_ragas['carnatic'])} Carnatic, {len(self.saraga_ragas['hindustani'])} Hindustani)")
            print(f"✅ Loaded {len(self.ramanarunachalam_ragas)} Ramanarunachalam ragas")
            print(f"✅ Loaded {len(self.raga_details)} raga details")
            
        except Exception as e:
            print(f"Error loading real data: {e}")
            # Fallback to a minimal correct dataset
            self.saraga_ragas = {
                'carnatic': ['bhairavi', 'kambhoji', 'mohanam', 'kalyani', 'sankarabharanam', 'janaranjani', 'lalit', 'sindhubhairavi', 'karaharapriya', 'ranjani', 'rasikapriya'],
                'hindustani': ['yaman', 'bhairav', 'khamaj', 'kafi', 'asavari', 'bageshri', 'darbari', 'desh', 'hamsadhwani', 'jaunpuri', 'kedar', 'malkauns', 'marwa', 'miyan ki malhar', 'puriya', 'sarang', 'shuddha kalyan', 'todi', 'yaman kalyan'],
                'all': []
            }
            self.saraga_ragas['all'] = self.saraga_ragas['carnatic'] + self.saraga_ragas['hindustani']
            self.ramanarunachalam_ragas = ['Bhairavi', 'Kambhoji', 'Mohanam', 'Kalyani', 'Sankarabharanam', 'Janaranjani', 'Lalit', 'Sindhubhairavi', 'Karaharapriya', 'Ranjani', 'Rasikapriya', 'Bhairav', 'Yaman', 'Khamaj', 'Kafi', 'Asavari', 'Bageshri', 'Darbari', 'Desh', 'Hamsadhwani', 'Jaunpuri', 'Kedar', 'Malkauns', 'Marwa', 'Miyan ki malhar', 'Puriya', 'Sarang', 'Shuddha kalyan', 'Todi', 'Yaman kalyan']
            print(f"⚠️ Using fallback data: {len(self.saraga_ragas['all'])} Saraga ragas ({len(self.saraga_ragas['carnatic'])} Carnatic, {len(self.saraga_ragas['hindustani'])} Hindustani)")
            print(f"⚠️ Using fallback data: {len(self.ramanarunachalam_ragas)} Ramanarunachalam ragas")
    
    def get_mapped_saraga_ragas(self):
        """Get Saraga ragas that have mappings"""
        mapped = set()
        for mapping in self.existing_mappings + self.session_mappings:
            mapped.add(mapping['saraga_raga'].lower())
        return mapped
    
    def get_unmapped_saraga_ragas(self, tradition='all'):
        """Get unmapped Saraga ragas for a tradition"""
        mapped = self.get_mapped_saraga_ragas()
        unmapped = []
        for raga in self.saraga_ragas[tradition]:
            if raga.lower() not in mapped:
                unmapped.append(raga)
        return unmapped
    
    def find_candidate_matches(self, saraga_raga):
        """Find candidate matches for a Saraga raga with enhanced matching algorithms"""
        candidates = []
        
        # Get Saraga raga details for musical structure matching
        saraga_details = self.raga_details.get(saraga_raga.lower(), {})
        saraga_arohana = saraga_details.get('arohana', '')
        saraga_avarohana = saraga_details.get('avarohana', '')
        saraga_melakartha = saraga_details.get('melakartha', '')
        
        # Check existing mappings first
        for mapping in self.existing_mappings:
            if mapping['saraga_raga'].lower() == saraga_raga.lower():
                candidates.append({
                    'raga': mapping['ramanarunachalam_raga'],
                    'confidence': mapping['confidence'],
                    'match_type': mapping['match_type'],
                    'source': 'existing',
                    'data_source': 'Cross-tradition mapping',
                    'reasoning': mapping.get('reasoning', '')
                })
        
        # If no existing mapping, find matches using enhanced algorithms
        if not candidates:
            # 1. EXACT MUSICAL STRUCTURE MATCH (100% confidence)
            if saraga_arohana and saraga_avarohana and saraga_melakartha:
                for ramanarunachalam_raga in self.ramanarunachalam_ragas:
                    ramanarunachalam_lower = ramanarunachalam_raga.lower()
                    ramanarunachalam_details = self.ramanarunachalam_details.get(ramanarunachalam_lower, {})
                    
                    if (saraga_arohana == ramanarunachalam_details.get('arohana', '') and 
                        saraga_avarohana == ramanarunachalam_details.get('avarohana', '') and 
                        saraga_melakartha == ramanarunachalam_details.get('melakartha', '') and
                        saraga_arohana and saraga_avarohana and saraga_melakartha):
                        candidates.append({
                            'raga': ramanarunachalam_raga,
                            'confidence': 1.0,
                            'match_type': 'musical_structure_match',
                            'source': 'algorithm',
                            'data_source': 'Ramanarunachalam Dataset',
                            'reasoning': f'Exact musical structure match: Arohana={saraga_arohana}, Avarohana={saraga_avarohana}, Melakartha={saraga_melakartha}'
                        })
            
            # 2. EXACT NAME MATCH (95% confidence)
            if not candidates:
                for ramanarunachalam_raga in self.ramanarunachalam_ragas:
                    if saraga_raga.lower() == ramanarunachalam_raga.lower():
                        candidates.append({
                            'raga': ramanarunachalam_raga,
                            'confidence': 0.95,
                            'match_type': 'exact_name_match',
                            'source': 'algorithm',
                            'data_source': 'Ramanarunachalam Dataset',
                            'reasoning': 'Exact name match'
                        })
            
            # 3. MELAKARTHA MATCH (90% confidence) - Same parent scale
            if not candidates and saraga_melakartha:
                for ramanarunachalam_raga in self.ramanarunachalam_ragas:
                    ramanarunachalam_lower = ramanarunachalam_raga.lower()
                    ramanarunachalam_details = self.ramanarunachalam_details.get(ramanarunachalam_lower, {})
                    ramanarunachalam_melakartha = ramanarunachalam_details.get('melakartha', '')
                    
                    if (saraga_melakartha == ramanarunachalam_melakartha and 
                        saraga_melakartha and ramanarunachalam_melakartha):
                        candidates.append({
                            'raga': ramanarunachalam_raga,
                            'confidence': 0.90,
                            'match_type': 'melakartha_match',
                            'source': 'algorithm',
                            'data_source': 'Ramanarunachalam Dataset',
                            'reasoning': f'Same melakartha (parent scale): {saraga_melakartha}'
                        })
            
            # 4. AROHANA/AVAROHANA MATCH (85% confidence) - Same musical structure
            if not candidates and saraga_arohana and saraga_avarohana:
                for ramanarunachalam_raga in self.ramanarunachalam_ragas:
                    ramanarunachalam_lower = ramanarunachalam_raga.lower()
                    ramanarunachalam_details = self.ramanarunachalam_details.get(ramanarunachalam_lower, {})
                    ramanarunachalam_arohana = ramanarunachalam_details.get('arohana', '')
                    ramanarunachalam_avarohana = ramanarunachalam_details.get('avarohana', '')
                    
                    if (saraga_arohana == ramanarunachalam_arohana and 
                        saraga_avarohana == ramanarunachalam_avarohana and
                        saraga_arohana and saraga_avarohana):
                        candidates.append({
                            'raga': ramanarunachalam_raga,
                            'confidence': 0.85,
                            'match_type': 'arohana_avarohana_match',
                            'source': 'algorithm',
                            'data_source': 'Ramanarunachalam Dataset',
                            'reasoning': f'Same arohana/avarohana: Arohana={saraga_arohana}, Avarohana={saraga_avarohana}'
                        })
            
            # 5. HIGH SIMILARITY NAME MATCH (80% confidence)
            if not candidates:
                for ramanarunachalam_raga in self.ramanarunachalam_ragas:
                    similarity = difflib.SequenceMatcher(None, saraga_raga.lower(), ramanarunachalam_raga.lower()).ratio()
                    if similarity >= 0.8:  # 80% similarity threshold
                        candidates.append({
                            'raga': ramanarunachalam_raga,
                            'confidence': 0.80,
                            'match_type': 'high_similarity_match',
                            'source': 'algorithm',
                            'data_source': 'Ramanarunachalam Dataset',
                            'reasoning': f'High name similarity: {similarity:.2f}'
                        })
            
            # 6. FUZZY MATCH (lower confidence)
            if not candidates:
                matches = difflib.get_close_matches(
                    saraga_raga, 
                    self.ramanarunachalam_ragas, 
                    n=5, 
                    cutoff=0.3
                )
                
                for match in matches:
                    similarity = difflib.SequenceMatcher(None, saraga_raga.lower(), match.lower()).ratio()
                    confidence = min(similarity * 1.2, 0.75)  # Cap at 75% for fuzzy matches
                    
                    candidates.append({
                        'raga': match,
                        'confidence': confidence,
                        'match_type': 'fuzzy_match',
                        'source': 'algorithm',
                        'data_source': 'Ramanarunachalam Dataset',
                        'reasoning': f'Fuzzy match (similarity: {similarity:.2f})'
                    })
        
        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates
    
    def add_session_mapping(self, saraga_raga, ramanarunachalam_raga, confidence, match_type, notes=''):
        """Add a mapping to the current session"""
        mapping = {
            'saraga_raga': saraga_raga,
            'ramanarunachalam_raga': ramanarunachalam_raga,
            'confidence': confidence,
            'match_type': match_type,
            'notes': notes,
            'source': 'session',
            'created_at': datetime.now().isoformat()
        }
        self.session_mappings.append(mapping)
        return mapping
    
    def remove_session_mapping(self, index):
        """Remove a mapping from the current session"""
        if 0 <= index < len(self.session_mappings):
            return self.session_mappings.pop(index)
        return None

# Initialize mapper
mapper = AdvancedRagaMapper()

@app.route('/')
def index():
    """Main 3-panel interface"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RagaSense-Data | Advanced Raga Mapper</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .main-container { height: 100vh; display: flex; flex-direction: column; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 0; }
        .panel-container { flex: 1; display: flex; gap: 1rem; padding: 1rem; }
        .panel { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .left-panel { flex: 1; min-width: 250px; }
        .premapped-panel { flex: 1; min-width: 300px; }
        .middle-panel { flex: 1.5; min-width: 400px; }
        .right-panel { flex: 1; min-width: 300px; }
        
        .panel-header { background: #f8f9fa; border-bottom: 1px solid #dee2e6; padding: 1rem; border-radius: 8px 8px 0 0; }
        .panel-content { padding: 1rem; height: calc(100% - 80px); overflow-y: auto; }
        
        .tradition-tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .tradition-tab { padding: 0.5rem 1rem; border: 1px solid #dee2e6; background: white; cursor: pointer; border-radius: 4px; }
        .tradition-tab.active { background: #007bff; color: white; }
        
        .filter-tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .filter-tab { padding: 0.25rem 0.75rem; border: 1px solid #dee2e6; background: white; cursor: pointer; border-radius: 4px; font-size: 0.9rem; }
        .filter-tab.active { background: #28a745; color: white; }
        
        .raga-item { padding: 0.75rem; border: 1px solid #dee2e6; margin: 0.25rem 0; border-radius: 4px; cursor: pointer; transition: all 0.2s; }
        .raga-item:hover { background: #f8f9fa; border-color: #007bff; }
        .raga-item.selected { background: #e3f2fd; border-color: #2196f3; }
        .raga-item.mapped { border-left: 4px solid #28a745; }
        .raga-item.unmapped { border-left: 4px solid #ffc107; }
        
        .candidate-card { padding: 1rem; border: 1px solid #dee2e6; margin: 0.5rem 0; border-radius: 8px; }
        .candidate-card.high-confidence { border-left: 4px solid #28a745; background: #f8fff8; }
        .candidate-card.medium-confidence { border-left: 4px solid #ffc107; background: #fffdf5; }
        .candidate-card.low-confidence { border-left: 4px solid #6c757d; background: #f8f9fa; }
        .candidate-card.musical-match { border-left: 4px solid #dc3545; background: #fff5f5; box-shadow: 0 2px 8px rgba(220, 53, 69, 0.2); }
        .candidate-card.exact-name-match { border-left: 4px solid #17a2b8; background: #f0f8ff; box-shadow: 0 2px 8px rgba(23, 162, 184, 0.2); }
        .candidate-card.melakartha-match { border-left: 4px solid #6f42c1; background: #f8f5ff; box-shadow: 0 2px 8px rgba(111, 66, 193, 0.2); }
        .candidate-card.arohana-avarohana-match { border-left: 4px solid #fd7e14; background: #fff8f0; box-shadow: 0 2px 8px rgba(253, 126, 20, 0.2); }
        .candidate-card.high-similarity-match { border-left: 4px solid #20c997; background: #f0fff8; box-shadow: 0 2px 8px rgba(32, 201, 151, 0.2); }
        
        .confidence-badge { padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold; }
        .confidence-high { background: #d4edda; color: #155724; }
        .confidence-medium { background: #fff3cd; color: #856404; }
        .confidence-low { background: #d1ecf1; color: #0c5460; }
        .confidence-musical { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
        .confidence-exact-name { background: #d1ecf1; color: #0c5460; border: 2px solid #17a2b8; }
        .confidence-melakartha { background: #e2d9f3; color: #4a2c7a; border: 2px solid #6f42c1; }
        .confidence-arohana-avarohana { background: #ffeaa7; color: #d63031; border: 2px solid #fd7e14; }
        .confidence-high-similarity { background: #d1f2eb; color: #00b894; border: 2px solid #20c997; }
        
        .mapping-item { padding: 0.75rem; border: 1px solid #dee2e6; margin: 0.25rem 0; border-radius: 4px; background: #f8f9fa; }
        .mapping-item .mapping-text { font-weight: 500; color: #495057; }
        .mapping-item .mapping-ragas { font-size: 1.1rem; margin: 0.5rem 0; }
        .mapping-item .source-raga { color: #007bff; font-weight: 600; }
        .mapping-item .dest-raga { color: #28a745; font-weight: 600; }
        
        .premapped-item { padding: 0.75rem; border: 1px solid #dee2e6; margin: 0.25rem 0; border-radius: 4px; background: #f0f8ff; border-left: 4px solid #007bff; }
        .premapped-item .premapped-text { font-weight: 500; color: #495057; }
        .premapped-item .premapped-ragas { font-size: 1.1rem; margin: 0.5rem 0; }
        .premapped-item .premapped-source { color: #007bff; font-weight: 600; }
        .premapped-item .premapped-dest { color: #28a745; font-weight: 600; }
        .mapping-item .mapping-details { font-size: 0.8rem; color: #6c757d; }
        
        .progress-bar-container { background: #e9ecef; border-radius: 10px; height: 20px; margin: 1rem 0; }
        .progress-bar { background: linear-gradient(90deg, #28a745, #20c997); height: 100%; border-radius: 10px; transition: width 0.3s; }
        
        .search-box { width: 100%; padding: 0.5rem; border: 1px solid #dee2e6; border-radius: 4px; margin-bottom: 1rem; }
        
        .action-buttons { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
        .btn-sm { padding: 0.25rem 0.5rem; font-size: 0.8rem; }
        
        .empty-state { text-align: center; color: #6c757d; padding: 2rem; }
        .empty-state i { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }
        
        .keyboard-hint { font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem; }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Header -->
        <div class="header">
            <div class="container-fluid">
                <div class="row align-items-center">
                    <div class="col">
                        <h4 class="mb-0"><i class="fas fa-music"></i> RagaSense-Data | Advanced Raga Mapper</h4>
                    </div>
                    <div class="col-auto">
                        <div class="d-flex gap-3 align-items-center">
                            <span id="progress-text">Mapped 0/{{ mapper.saraga_ragas.all|length }} Saraga ragas</span>
                            <button class="btn btn-light btn-sm" onclick="exportMappings()">
                                <i class="fas fa-download"></i> Export JSON
                            </button>
                            <button class="btn btn-light btn-sm" onclick="undoLast()">
                                <i class="fas fa-undo"></i> Undo Last
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Progress Bar -->
        <div class="container-fluid px-3">
            <div class="progress-bar-container">
                <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
            </div>
        </div>
        
        <!-- 3-Panel Layout -->
        <div class="panel-container">
            <!-- Left Panel: Saraga Ragas -->
            <div class="panel left-panel">
                <div class="panel-header">
                    <h5><i class="fas fa-list"></i> SARAGA RAGAS</h5>
                </div>
                <div class="panel-content">
                    <!-- Tradition Tabs -->
                    <div class="tradition-tabs">
                        <div class="tradition-tab active" data-tradition="all" onclick="switchTradition('all')">
                            All ({{ mapper.saraga_ragas.all|length }})
                        </div>
                        <div class="tradition-tab" data-tradition="carnatic" onclick="switchTradition('carnatic')">
                            Carnatic ({{ mapper.saraga_ragas.carnatic|length }})
                        </div>
                        <div class="tradition-tab" data-tradition="hindustani" onclick="switchTradition('hindustani')">
                            Hindustani ({{ mapper.saraga_ragas.hindustani|length }})
                        </div>
                    </div>
                    
                    <!-- Filter Tabs -->
                    <div class="filter-tabs">
                        <div class="filter-tab active" data-filter="all" onclick="switchFilter('all')">All</div>
                        <div class="filter-tab" data-filter="mapped" onclick="switchFilter('mapped')">Mapped</div>
                        <div class="filter-tab" data-filter="unmapped" onclick="switchFilter('unmapped')">Unmapped</div>
                    </div>
                    
                    <!-- Search -->
                    <input type="text" class="search-box" id="saraga-search" placeholder="🔍 Search Saraga ragas..." onkeyup="searchRagas()">
                    
                    <!-- Raga List -->
                    <div id="raga-list">
                        <!-- Ragas will be loaded here -->
                    </div>
                </div>
            </div>
            
            <!-- Pre-mapped Matches Panel -->
            <div class="panel premapped-panel">
                <div class="panel-header">
                    <h5><i class="fas fa-star"></i> PRE-MAPPED MATCHES</h5>
                    <div class="mt-2">
                        <small class="text-muted">100% confirmed matches from existing analysis</small>
                    </div>
                </div>
                <div class="panel-content" id="premapped-content">
                    <div class="text-center text-muted">
                        <i class="fas fa-spinner fa-spin"></i> Loading pre-mapped matches...
                    </div>
                </div>
            </div>
            
            <!-- Middle Panel: Candidate Matches -->
            <div class="panel middle-panel">
                <div class="panel-header">
                    <h5><i class="fas fa-search"></i> MATCH CANDIDATES</h5>
                    <div class="mt-2">
                        <small class="text-muted">
                            <span class="badge bg-info me-1">Cross-tradition mapping</span> = Existing mapping from analysis
                            <span class="badge bg-primary ms-2 me-1">Ramanarunachalam Dataset</span> = Algorithm-suggested match
                        </small>
                    </div>
                </div>
                <div class="panel-content">
                    <div id="candidate-content">
                        <div class="empty-state">
                            <i class="fas fa-mouse-pointer"></i>
                            <p>Select a Saraga raga from the left panel to see candidate matches</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Right Panel: Confirmed Mappings -->
            <div class="panel right-panel">
                <div class="panel-header">
                    <h5><i class="fas fa-check-circle"></i> CONFIRMED MAPPINGS</h5>
                </div>
                <div class="panel-content">
                    <div id="mappings-list">
                        <div class="empty-state">
                            <i class="fas fa-shopping-cart"></i>
                            <p>No mappings confirmed yet</p>
                            <small>Confirmed mappings will appear here</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentTradition = 'all';
        let currentFilter = 'all';
        let selectedRaga = null;
        let currentRagas = [];
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadRagas();
            updateProgress();
            
            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                    e.preventDefault();
                    navigateRagas(e.key === 'ArrowUp' ? -1 : 1);
                } else if (e.key === 'Enter' && selectedRaga) {
                    e.preventDefault();
                    confirmFirstCandidate();
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    clearSelection();
                }
            });
        });
        
        function switchTradition(tradition) {
            currentTradition = tradition;
            document.querySelectorAll('.tradition-tab').forEach(tab => tab.classList.remove('active'));
            document.querySelector(`[data-tradition="${tradition}"]`).classList.add('active');
            loadRagas();
        }
        
        function switchFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.filter-tab').forEach(tab => tab.classList.remove('active'));
            document.querySelector(`[data-filter="${filter}"]`).classList.add('active');
            loadRagas();
        }
        
        function loadRagas() {
            fetch(`/api/ragas?tradition=${currentTradition}&filter=${currentFilter}`)
                .then(response => response.json())
                .then(data => {
                    currentRagas = data;
                    renderRagas(data);
                });
        }
        
        function loadPremappedMatches() {
            fetch('/api/premapped-matches')
                .then(response => response.json())
                .then(data => {
                    renderPremappedMatches(data);
                });
        }
        
        function renderPremappedMatches(premapped) {
            const container = document.getElementById('premapped-content');
            
            if (premapped.length === 0) {
                container.innerHTML = '<div class="empty-state"><i class="fas fa-star"></i><p>No pre-mapped matches found</p></div>';
                return;
            }
            
            let html = '';
            premapped.forEach((match, index) => {
                html += `
                    <div class="premapped-item">
                        <div class="premapped-text">
                            <strong>Saraga Dataset</strong> → <strong>Ramanarunachalam Dataset</strong>
                        </div>
                        <div class="premapped-ragas">
                            <span class="premapped-source">${match.saraga_raga}</span> → <span class="premapped-dest">${match.ramanarunachalam_raga}</span>
                        </div>
                        <div class="premapped-details">
                            ${Math.round(match.confidence * 100)}% confidence | ${match.match_type.replace('_', ' ')}
                        </div>
                        <div class="action-buttons">
                            <button class="btn btn-outline-info btn-sm" onclick="showPremappedDetails('${match.saraga_raga}', '${match.ramanarunachalam_raga}')">
                                <i class="fas fa-info-circle"></i> Details
                            </button>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        function renderRagas(ragas) {
            const container = document.getElementById('raga-list');
            container.innerHTML = '';
            
            if (ragas.length === 0) {
                container.innerHTML = '<div class="empty-state"><i class="fas fa-search"></i><p>No ragas found</p></div>';
                return;
            }
            
            ragas.forEach((raga, index) => {
                const item = document.createElement('div');
                item.className = `raga-item ${raga.mapped ? 'mapped' : 'unmapped'}`;
                item.onclick = () => selectRaga(raga, index);
                
                const statusIcon = raga.mapped ? '✅' : '⏳';
                const traditionTag = raga.tradition === 'carnatic' ? '🎵' : '🎶';
                
                item.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <div class="flex-grow-1">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <strong>${statusIcon} ${raga.name}</strong>
                                    <div class="text-muted small">
                                        ${traditionTag} ${raga.tradition.charAt(0).toUpperCase() + raga.tradition.slice(1)}
                                    </div>
                                </div>
                                <button class="btn btn-sm btn-outline-info" onclick="event.stopPropagation(); showRagaInfo('${raga.name}')" title="Show raga details">
                                    <i class="fas fa-info-circle"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                `;
                
                container.appendChild(item);
            });
        }
        
        function selectRaga(raga, index) {
            selectedRaga = raga;
            
            // Update UI
            document.querySelectorAll('.raga-item').forEach(item => item.classList.remove('selected'));
            document.querySelectorAll('.raga-item')[index].classList.add('selected');
            
            // Load candidates
            loadCandidates(raga.name);
        }
        
        function loadCandidates(saragaRaga) {
            fetch(`/api/candidates?raga=${encodeURIComponent(saragaRaga)}`)
                .then(response => response.json())
                .then(data => {
                    renderCandidates(saragaRaga, data);
                });
        }
        
        function renderCandidates(saragaRaga, candidates) {
            const container = document.getElementById('candidate-content');
            
            if (candidates.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-search"></i>
                        <p>No candidates found for <strong>${saragaRaga}</strong></p>
                        <button class="btn btn-primary" onclick="searchManually('${saragaRaga}')">
                            <i class="fas fa-search"></i> Search Manually
                        </button>
                    </div>
                `;
                return;
            }
            
            let html = `
                <div class="mb-3">
                    <h6>Selected: <strong>${saragaRaga}</strong></h6>
                    <small class="text-muted">${candidates.length} candidate(s) found</small>
                </div>
            `;
            
            candidates.forEach((candidate, index) => {
                // Determine styling based on match type
                let confidenceClass, confidenceBadgeClass, badgeText;
                
                switch(candidate.match_type) {
                    case 'musical_structure_match':
                        confidenceClass = 'musical-match';
                        confidenceBadgeClass = 'confidence-musical';
                        badgeText = '100% MUSICAL MATCH';
                        break;
                    case 'exact_name_match':
                        confidenceClass = 'exact-name-match';
                        confidenceBadgeClass = 'confidence-exact-name';
                        badgeText = '95% EXACT NAME';
                        break;
                    case 'melakartha_match':
                        confidenceClass = 'melakartha-match';
                        confidenceBadgeClass = 'confidence-melakartha';
                        badgeText = '90% MELAKARTHA';
                        break;
                    case 'arohana_avarohana_match':
                        confidenceClass = 'arohana-avarohana-match';
                        confidenceBadgeClass = 'confidence-arohana-avarohana';
                        badgeText = '85% MUSICAL STRUCTURE';
                        break;
                    case 'high_similarity_match':
                        confidenceClass = 'high-similarity-match';
                        confidenceBadgeClass = 'confidence-high-similarity';
                        badgeText = '80% HIGH SIMILARITY';
                        break;
                    default:
                        confidenceClass = candidate.confidence >= 0.8 ? 'high-confidence' : 
                                        candidate.confidence >= 0.5 ? 'medium-confidence' : 'low-confidence';
                        confidenceBadgeClass = candidate.confidence >= 0.8 ? 'confidence-high' : 
                                             candidate.confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';
                        badgeText = Math.round(candidate.confidence * 100) + '% match';
                }
                
                html += `
                    <div class="candidate-card ${confidenceClass}">
                        <div class="d-flex justify-content-between align-items-start">
                            <div class="flex-grow-1">
                                <h6 class="mb-1">${candidate.raga}</h6>
                                <div class="mb-2">
                                    <span class="confidence-badge ${confidenceBadgeClass}">
                                        ${badgeText}
                                    </span>
                                    <span class="badge bg-secondary ms-2">${candidate.match_type.replace('_', ' ')}</span>
                                    <span class="badge bg-info ms-2">${candidate.data_source || 'Unknown Source'}</span>
                                </div>
                                ${candidate.reasoning ? `<small class="text-muted">${candidate.reasoning}</small>` : ''}
                            </div>
                        </div>
                        <div class="action-buttons">
                            <button class="btn btn-success btn-sm" onclick="confirmMapping('${saragaRaga}', '${candidate.raga}', ${candidate.confidence}, '${candidate.match_type}')">
                                <i class="fas fa-check"></i> Confirm
                            </button>
                            <button class="btn btn-outline-secondary btn-sm" onclick="rejectCandidate(${index})">
                                <i class="fas fa-times"></i> Reject
                            </button>
                            <button class="btn btn-outline-info btn-sm" onclick="showMoreInfo('${candidate.raga}')">
                                <i class="fas fa-info-circle"></i> More Info
                            </button>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        function confirmMapping(saragaRaga, ramanarunachalamRaga, confidence, matchType) {
            fetch('/api/confirm-mapping', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    saraga_raga: saragaRaga,
                    ramanarunachalam_raga: ramanarunachalamRaga,
                    confidence: confidence,
                    match_type: matchType
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadMappings();
                    updateProgress();
                    loadRagas(); // Refresh raga list
                    
                    // Show success message
                    showToast('Mapping confirmed!', 'success');
                }
            });
        }
        
        function loadMappings() {
            fetch('/api/session-mappings')
                .then(response => response.json())
                .then(data => {
                    renderMappings(data);
                });
        }
        
        function renderMappings(mappings) {
            const container = document.getElementById('mappings-list');
            
            if (mappings.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-shopping-cart"></i>
                        <p>No mappings confirmed yet</p>
                        <small>Confirmed mappings will appear here</small>
                    </div>
                `;
                return;
            }
            
            let html = '';
            mappings.forEach((mapping, index) => {
                html += `
                    <div class="mapping-item">
                        <div class="mapping-text">
                            <strong>Saraga Dataset</strong> → <strong>Ramanarunachalam Dataset</strong>
                        </div>
                        <div class="mapping-ragas">
                            <span class="source-raga">${mapping.saraga_raga}</span> → <span class="dest-raga">${mapping.ramanarunachalam_raga}</span>
                        </div>
                        <div class="mapping-details">
                            ${Math.round(mapping.confidence * 100)}% confidence | ${mapping.match_type.replace('_', ' ')}
                        </div>
                        <div class="action-buttons">
                            <button class="btn btn-outline-info btn-sm" onclick="showMappingDetails('${mapping.saraga_raga}', '${mapping.ramanarunachalam_raga}')">
                                <i class="fas fa-info-circle"></i> Details
                            </button>
                            <button class="btn btn-outline-danger btn-sm" onclick="removeMapping(${index})">
                                <i class="fas fa-trash"></i> Remove
                            </button>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        function updateProgress() {
            fetch('/api/progress')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('progress-text').textContent = 
                        `Mapped ${data.mapped}/${data.total} Saraga ragas`;
                    document.getElementById('progress-bar').style.width = `${data.percentage}%`;
                });
        }
        
        function searchRagas() {
            const query = document.getElementById('saraga-search').value.toLowerCase();
            const filtered = currentRagas.filter(raga => 
                raga.name.toLowerCase().includes(query)
            );
            renderRagas(filtered);
        }
        
        function navigateRagas(direction) {
            const items = document.querySelectorAll('.raga-item');
            const currentIndex = Array.from(items).findIndex(item => item.classList.contains('selected'));
            
            let newIndex = currentIndex + direction;
            if (newIndex < 0) newIndex = items.length - 1;
            if (newIndex >= items.length) newIndex = 0;
            
            if (items[newIndex]) {
                items[newIndex].click();
            }
        }
        
        function confirmFirstCandidate() {
            const confirmBtn = document.querySelector('.candidate-card .btn-success');
            if (confirmBtn) {
                confirmBtn.click();
            }
        }
        
        function clearSelection() {
            selectedRaga = null;
            document.querySelectorAll('.raga-item').forEach(item => item.classList.remove('selected'));
            document.getElementById('candidate-content').innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-mouse-pointer"></i>
                    <p>Select a Saraga raga from the left panel to see candidate matches</p>
                </div>
            `;
        }
        
        function removeMapping(index) {
            fetch('/api/remove-mapping', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: index })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadMappings();
                    updateProgress();
                    loadRagas();
                    showToast('Mapping removed', 'info');
                }
            });
        }
        
        function exportMappings() {
            fetch('/api/export-mappings')
                .then(response => response.json())
                .then(data => {
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `raga_mappings_${new Date().toISOString().split('T')[0]}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                });
        }
        
        function undoLast() {
            fetch('/api/undo-last', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        loadMappings();
                        updateProgress();
                        loadRagas();
                        showToast('Last mapping undone', 'info');
                    } else {
                        showToast('No mappings to undo', 'warning');
                    }
                });
        }
        
        function showToast(message, type) {
            // Simple toast notification
            const toast = document.createElement('div');
            toast.className = `alert alert-${type} position-fixed`;
            toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
            toast.textContent = message;
            document.body.appendChild(toast);
            
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 3000);
        }
        
        function showRagaInfo(ragaName) {
            fetch(`/api/raga-details?raga=${encodeURIComponent(ragaName)}`)
                .then(response => response.json())
                .then(data => {
                    if (Object.keys(data).length === 0) {
                        showToast('No detailed information available for this raga', 'warning');
                        return;
                    }
                    
                    // Create modal content
                    const modalContent = `
                        <div class="modal fade" id="ragaInfoModal" tabindex="-1">
                            <div class="modal-dialog modal-lg">
                                <div class="modal-content">
                                    <div class="modal-header">
                                        <h5 class="modal-title">
                                            <i class="fas fa-music"></i> ${data.name || ragaName}
                                        </h5>
                                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                    </div>
                                    <div class="modal-body">
                                        <div class="row">
                                            <div class="col-md-6">
                                                <h6>Basic Information</h6>
                                                <table class="table table-sm">
                                                    <tr><td><strong>Tradition:</strong></td><td>${data.tradition || 'Unknown'}</td></tr>
                                                    <tr><td><strong>Data Source:</strong></td><td><span class="badge bg-info">${data.data_source || 'Unknown'}</span></td></tr>
                                                    <tr><td><strong>ID:</strong></td><td>${data.id || 'N/A'}</td></tr>
                                                </table>
                                            </div>
                                            <div class="col-md-6">
                                                <h6>Musical Structure</h6>
                                                <table class="table table-sm">
                                                    <tr><td><strong>Melakartha:</strong></td><td>${data.melakartha || 'N/A'}</td></tr>
                                                    <tr><td><strong>Arohana:</strong></td><td><code>${data.arohana || 'N/A'}</code></td></tr>
                                                    <tr><td><strong>Avarohana:</strong></td><td><code>${data.avarohana || 'N/A'}</code></td></tr>
                                                </table>
                                            </div>
                                        </div>
                                        ${data.counts && Object.keys(data.counts).length > 0 ? `
                                            <div class="mt-3">
                                                <h6>Dataset Statistics</h6>
                                                <div class="row">
                                                    ${Object.entries(data.counts).map(([key, value]) => `
                                                        <div class="col-md-3">
                                                            <div class="text-center">
                                                                <div class="h5 text-primary">${value}</div>
                                                                <small class="text-muted">${key.charAt(0).toUpperCase() + key.slice(1)}</small>
                                                            </div>
                                                        </div>
                                                    `).join('')}
                                                </div>
                                            </div>
                                        ` : ''}
                                    </div>
                                    <div class="modal-footer">
                                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // Remove existing modal if any
                    const existingModal = document.getElementById('ragaInfoModal');
                    if (existingModal) {
                        existingModal.remove();
                    }
                    
                    // Add modal to body
                    document.body.insertAdjacentHTML('beforeend', modalContent);
                    
                    // Show modal
                    const modal = new bootstrap.Modal(document.getElementById('ragaInfoModal'));
                    modal.show();
                    
                    // Clean up modal when hidden
                    document.getElementById('ragaInfoModal').addEventListener('hidden.bs.modal', function() {
                        this.remove();
                    });
                })
                .catch(error => {
                    console.error('Error loading raga details:', error);
                    showToast('Error loading raga details', 'danger');
                });
        }
        
        function showMappingDetails(saragaRaga, ramanarunachalamRaga) {
            // Fetch details for both ragas
            Promise.all([
                fetch(`/api/raga-details?raga=${encodeURIComponent(saragaRaga)}`).then(r => r.json()),
                fetch(`/api/ramanarunachalam-details?raga=${encodeURIComponent(ramanarunachalamRaga)}`).then(r => r.json())
            ])
            .then(([saragaData, ramanarunachalamData]) => {
                // Create modal content
                const modalContent = `
                    <div class="modal fade" id="mappingDetailsModal" tabindex="-1">
                        <div class="modal-dialog modal-xl">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">
                                        <i class="fas fa-exchange-alt"></i> Mapping Details: ${saragaRaga} → ${ramanarunachalamRaga}
                                    </h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="row">
                                        <!-- Saraga Dataset -->
                                        <div class="col-md-6">
                                            <div class="card">
                                                <div class="card-header bg-primary text-white">
                                                    <h6 class="mb-0"><i class="fas fa-database"></i> Saraga Dataset</h6>
                                                </div>
                                                <div class="card-body">
                                                    <h5 class="text-primary">${saragaData.name || saragaRaga}</h5>
                                                    <table class="table table-sm">
                                                        <tr><td><strong>Tradition:</strong></td><td>${saragaData.tradition || 'Unknown'}</td></tr>
                                                        <tr><td><strong>Data Source:</strong></td><td><span class="badge bg-info">${saragaData.data_source || 'Unknown'}</span></td></tr>
                                                        <tr><td><strong>ID:</strong></td><td>${saragaData.id || 'N/A'}</td></tr>
                                                        <tr><td><strong>Melakartha:</strong></td><td>${saragaData.melakartha || 'N/A'}</td></tr>
                                                        <tr><td><strong>Arohana:</strong></td><td><code>${saragaData.arohana || 'N/A'}</code></td></tr>
                                                        <tr><td><strong>Avarohana:</strong></td><td><code>${saragaData.avarohana || 'N/A'}</code></td></tr>
                                                    </table>
                                                    ${saragaData.counts && Object.keys(saragaData.counts).length > 0 ? `
                                                        <h6>Dataset Statistics</h6>
                                                        <div class="row">
                                                            ${Object.entries(saragaData.counts).map(([key, value]) => `
                                                                <div class="col-6">
                                                                    <div class="text-center">
                                                                        <div class="h6 text-primary">${value}</div>
                                                                        <small class="text-muted">${key.charAt(0).toUpperCase() + key.slice(1)}</small>
                                                                    </div>
                                                                </div>
                                                            `).join('')}
                                                        </div>
                                                    ` : ''}
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <!-- Ramanarunachalam Dataset -->
                                        <div class="col-md-6">
                                            <div class="card">
                                                <div class="card-header bg-success text-white">
                                                    <h6 class="mb-0"><i class="fas fa-database"></i> Ramanarunachalam Dataset</h6>
                                                </div>
                                                <div class="card-body">
                                                    <h5 class="text-success">${ramanarunachalamData.name || ramanarunachalamRaga}</h5>
                                                    <table class="table table-sm">
                                                        <tr><td><strong>Tradition:</strong></td><td>${ramanarunachalamData.tradition || 'Unknown'}</td></tr>
                                                        <tr><td><strong>Data Source:</strong></td><td><span class="badge bg-success">${ramanarunachalamData.data_source || 'Ramanarunachalam Dataset'}</span></td></tr>
                                                        <tr><td><strong>ID:</strong></td><td>${ramanarunachalamData.id || 'N/A'}</td></tr>
                                                        <tr><td><strong>Melakartha:</strong></td><td>${ramanarunachalamData.melakartha || 'N/A'}</td></tr>
                                                        <tr><td><strong>Arohana:</strong></td><td><code>${ramanarunachalamData.arohana || 'N/A'}</code></td></tr>
                                                        <tr><td><strong>Avarohana:</strong></td><td><code>${ramanarunachalamData.avarohana || 'N/A'}</code></td></tr>
                                                    </table>
                                                    ${ramanarunachalamData.counts && Object.keys(ramanarunachalamData.counts).length > 0 ? `
                                                        <h6>Dataset Statistics</h6>
                                                        <div class="row">
                                                            ${Object.entries(ramanarunachalamData.counts).map(([key, value]) => `
                                                                <div class="col-6">
                                                                    <div class="text-center">
                                                                        <div class="h6 text-success">${value}</div>
                                                                        <small class="text-muted">${key.charAt(0).toUpperCase() + key.slice(1)}</small>
                                                                    </div>
                                                                </div>
                                                            `).join('')}
                                                        </div>
                                                    ` : ''}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                // Remove existing modal if any
                const existingModal = document.getElementById('mappingDetailsModal');
                if (existingModal) {
                    existingModal.remove();
                }
                
                // Add modal to body
                document.body.insertAdjacentHTML('beforeend', modalContent);
                
                // Show modal
                const modal = new bootstrap.Modal(document.getElementById('mappingDetailsModal'));
                modal.show();
                
                // Clean up modal when hidden
                document.getElementById('mappingDetailsModal').addEventListener('hidden.bs.modal', function() {
                    this.remove();
                });
            })
            .catch(error => {
                console.error('Error loading mapping details:', error);
                showToast('Error loading mapping details', 'danger');
            });
        }
        
        function showPremappedDetails(saragaRaga, ramanarunachalamRaga) {
            showMappingDetails(saragaRaga, ramanarunachalamRaga);
        }
        
        // Load initial data
        loadMappings();
        loadPremappedMatches();
    </script>
</body>
</html>
    """, mapper=mapper)

# API Routes
@app.route('/api/ragas')
def api_ragas():
    """Get ragas based on tradition and filter"""
    tradition = request.args.get('tradition', 'all')
    filter_type = request.args.get('filter', 'all')
    
    ragas = mapper.saraga_ragas[tradition]
    mapped_ragas = mapper.get_mapped_saraga_ragas()
    
    result = []
    for raga in ragas:
        is_mapped = raga.lower() in mapped_ragas
        
        # Apply filter
        if filter_type == 'mapped' and not is_mapped:
            continue
        if filter_type == 'unmapped' and is_mapped:
            continue
        
        result.append({
            'name': raga,
            'tradition': 'carnatic' if raga in mapper.saraga_ragas['carnatic'] else 'hindustani',
            'mapped': is_mapped
        })
    
    return jsonify(result)

@app.route('/api/candidates')
def api_candidates():
    """Get candidate matches for a raga"""
    raga = request.args.get('raga', '')
    if not raga:
        return jsonify([])
    
    candidates = mapper.find_candidate_matches(raga)
    return jsonify(candidates)

@app.route('/api/raga-details')
def get_raga_details():
    """Get detailed information for a Saraga raga"""
    raga = request.args.get('raga', '').lower()
    if not raga:
        return jsonify({})
    
    details = mapper.raga_details.get(raga, {})
    return jsonify(details)

@app.route('/api/ramanarunachalam-details')
def get_ramanarunachalam_details():
    """Get detailed information for a Ramanarunachalam raga"""
    raga = request.args.get('raga', '').lower()
    if not raga:
        return jsonify({})
    
    details = mapper.ramanarunachalam_details.get(raga, {})
    return jsonify(details)

@app.route('/api/premapped-matches')
def get_premapped_matches():
    """Get all pre-mapped high-confidence matches"""
    premapped = []
    
    # Add existing mappings with high confidence
    for mapping in mapper.existing_mappings:
        if mapping['confidence'] >= 0.80:  # 80%+ confidence considered pre-mapped
            premapped.append({
                'saraga_raga': mapping['saraga_raga'],
                'ramanarunachalam_raga': mapping['ramanarunachalam_raga'],
                'confidence': mapping['confidence'],
                'match_type': mapping['match_type'],
                'reasoning': mapping.get('reasoning', ''),
                'source': 'existing'
            })
    
    # Add algorithm-discovered high-confidence matches
    for saraga_raga in mapper.saraga_ragas['all']:
        candidates = mapper.find_candidate_matches(saraga_raga)
        for candidate in candidates:
            if (candidate['confidence'] >= 0.80 and 
                candidate['source'] == 'algorithm' and
                not any(p['saraga_raga'].lower() == saraga_raga.lower() and 
                       p['ramanarunachalam_raga'].lower() == candidate['raga'].lower() 
                       for p in premapped)):
                premapped.append({
                    'saraga_raga': saraga_raga,
                    'ramanarunachalam_raga': candidate['raga'],
                    'confidence': candidate['confidence'],
                    'match_type': candidate['match_type'],
                    'reasoning': candidate['reasoning'],
                    'source': 'algorithm'
                })
    
    # Sort by confidence (highest first)
    premapped.sort(key=lambda x: x['confidence'], reverse=True)
    return jsonify(premapped)

@app.route('/api/confirm-mapping', methods=['POST'])
def api_confirm_mapping():
    """Confirm a mapping"""
    data = request.get_json()
    mapping = mapper.add_session_mapping(
        data['saraga_raga'],
        data['ramanarunachalam_raga'],
        data['confidence'],
        data['match_type']
    )
    return jsonify({'success': True, 'mapping': mapping})

@app.route('/api/session-mappings')
def api_session_mappings():
    """Get current session mappings"""
    return jsonify(mapper.session_mappings)

@app.route('/api/remove-mapping', methods=['POST'])
def api_remove_mapping():
    """Remove a mapping from session"""
    data = request.get_json()
    removed = mapper.remove_session_mapping(data['index'])
    return jsonify({'success': removed is not None, 'removed': removed})

@app.route('/api/progress')
def api_progress():
    """Get mapping progress"""
    total = len(mapper.saraga_ragas['all'])
    mapped = len(mapper.get_mapped_saraga_ragas())
    percentage = (mapped / total * 100) if total > 0 else 0
    
    return jsonify({
        'total': total,
        'mapped': mapped,
        'percentage': percentage
    })

@app.route('/api/export-mappings')
def api_export_mappings():
    """Export all mappings"""
    all_mappings = mapper.existing_mappings + mapper.session_mappings
    return jsonify({
        'existing_mappings': mapper.existing_mappings,
        'session_mappings': mapper.session_mappings,
        'total_count': len(all_mappings),
        'exported_at': datetime.now().isoformat()
    })

@app.route('/api/undo-last', methods=['POST'])
def api_undo_last():
    """Undo last mapping"""
    if mapper.session_mappings:
        removed = mapper.session_mappings.pop()
        return jsonify({'success': True, 'removed': removed})
    return jsonify({'success': False})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5009))
    print("🎵 Starting Advanced 3-Panel Raga Mapper...")
    print(f"📊 Loaded {len(mapper.existing_mappings)} existing mappings")
    print(f"🎵 {len(mapper.saraga_ragas['all'])} Saraga ragas ({len(mapper.saraga_ragas['carnatic'])} Carnatic, {len(mapper.saraga_ragas['hindustani'])} Hindustani)")
    print(f"🎼 {len(mapper.ramanarunachalam_ragas)} Ramanarunachalam ragas")
    print(f"🌐 Interface: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
