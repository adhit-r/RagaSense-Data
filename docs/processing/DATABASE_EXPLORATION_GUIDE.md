# RagaSense-Data Database Exploration Guide

## 🎯 **What Kind of Database Do We Have?**

### **Database Type: JSON-Based Unified Database**
- **Format**: JSON files with structured metadata
- **Size**: 5.7 MB total unified database
- **Structure**: Multi-component database with cross-references
- **Quality**: High-quality, validated data with source attribution

### **Database Components:**
1. **`unified_ragas.json`** (0.9 MB) - 1,340 unique ragas
2. **`unified_artists.json`** (0.0 MB) - 19 professional vocalists
3. **`unified_tracks.json`** (2.5 MB) - 4,536 compositions
4. **`unified_audio_files.json`** (2.0 MB) - 4,536 audio file references
5. **`unified_cross_tradition_mappings.json`** (0.0 MB) - 12 validated equivalences
6. **`unified_metadata.json`** (0.0 MB) - Database metadata and statistics

## 🔍 **Database Structure for Each Raga**

### **Raga Record Fields:**
```json
{
  "raga_id": "Kalyani",
  "name": "Kalyani",
  "sanskrit_name": "kalyANi",
  "tradition": "Both",
  "song_count": 6244,
  "sample_duration": "8:44",
  "file_path": "Carnatic/raga/Kalyani.json",
  "cross_tradition_mapping": {
    "type": "similar",
    "mapping": {
      "raga_name": "Kalyani",
      "carnatic_name": "Kalyani", 
      "hindustani_name": "Yaman",
      "confidence": "high",
      "evidence": "expert_knowledge"
    },
    "confidence": "high"
  },
  "metadata": {
    "source": "ramanarunachalam",
    "last_updated": "2025-09-08T10:48:36.989764",
    "quality_score": 2.8,
    "merged_from": ["kalyani"],
    "duplicate_count": 2
  },
  "sources": ["ramanarunachalam"],
  "source_priority": "primary",
  "last_updated": "2025-09-08T22:03:54.571532"
}
```

### **Key Fields Explained:**
- **`raga_id`**: Unique identifier
- **`name`**: Common raga name
- **`sanskrit_name`**: Sanskrit transliteration
- **`tradition`**: Carnatic, Hindustani, or Both
- **`song_count`**: Number of compositions in this raga
- **`cross_tradition_mapping`**: Equivalence to other tradition
- **`sources`**: Data source attribution
- **`metadata`**: Quality scores and processing info

## 🛠️ **How to Explore the Data**

### **1. Command-Line Explorer (Recommended)**

#### **Installation:**
```bash
# The explorer is already created in your project
python3 explore_ragasense_data.py
```

#### **Available Commands:**
```bash
# Interactive mode (recommended)
python3 explore_ragasense_data.py

# Search for ragas
python3 explore_ragasense_data.py search "Bhairavi"

# Filter by tradition
python3 explore_ragasense_data.py filter "Carnatic"

# Show top ragas
python3 explore_ragasense_data.py top 20

# Get raga details
python3 explore_ragasense_data.py raga "Kalyani"

# Show cross-tradition mappings
python3 explore_ragasense_data.py cross-tradition

# Show database statistics
python3 explore_ragasense_data.py stats
```

#### **Interactive Mode Features:**
- **Search**: Find ragas by name
- **Filter**: Filter by tradition (Carnatic/Hindustani/Both)
- **Browse**: View top ragas by song count
- **Details**: Get comprehensive raga information
- **Cross-tradition**: Explore raga equivalences
- **Statistics**: View database metrics

### **2. Web-Based Explorer**

#### **Start Web Server:**
```bash
python3 web_explorer.py
```

#### **Access Interface:**
- **URL**: http://localhost:8000
- **Features**: 
  - Visual search interface
  - Clickable raga cards
  - Tradition filtering
  - Statistics dashboard
  - Cross-tradition mappings

### **3. Direct JSON Access**

#### **Load Data in Python:**
```python
import json

# Load raga database
with open('data/unified_ragasense_final/unified_ragas.json', 'r') as f:
    ragas = json.load(f)

# Search for specific raga
raga_name = "Kalyani"
for raga_id, raga_data in ragas.items():
    if raga_data.get('name') == raga_name:
        print(f"Found: {raga_data}")
        break
```

#### **Load in Other Languages:**
```javascript
// JavaScript/Node.js
const fs = require('fs');
const ragas = JSON.parse(fs.readFileSync('data/unified_ragasense_final/unified_ragas.json', 'utf8'));
```

## 📊 **Database Statistics**

### **Overall Statistics:**
- **Total Ragas**: 1,340 (all unique)
- **Total Artists**: 19 professional vocalists
- **Total Tracks**: 4,536 compositions
- **Total Audio Files**: 4,536 audio file references
- **Cross-Tradition Mappings**: 12 validated equivalences

### **Tradition Distribution:**
- **Carnatic**: 1,143 ragas (85.3%)
- **Hindustani**: 132 ragas (9.9%)
- **Both**: 65 ragas (4.9%)

### **Top Ragas by Song Count:**
1. **Ragamalika**: 9,810 songs
2. **Thodi**: 6,321 songs
3. **Kalyani**: 6,244 songs
4. **Sankarabharanam**: 5,192 songs
5. **Bhairavi**: 4,519 songs

## 🔍 **Exploration Examples**

### **Search Examples:**
```bash
# Search for ragas containing "bhairavi"
python3 explore_ragasense_data.py search "bhairavi"

# Find all Carnatic ragas
python3 explore_ragasense_data.py filter "Carnatic"

# Get top 10 ragas by popularity
python3 explore_ragasense_data.py top 10

# Get detailed info for Kalyani
python3 explore_ragasense_data.py raga "Kalyani"
```

### **Web Interface Examples:**
1. **Open**: http://localhost:8000
2. **Search**: Type "Bhairavi" in search box
3. **Filter**: Click "Carnatic" to see only Carnatic ragas
4. **Browse**: Click "Top Ragas" to see most popular ragas
5. **Details**: Click any raga card for detailed information

## 🎯 **Research Applications**

### **For Machine Learning:**
- **Raga Classification**: 1,340 unique ragas for training
- **Audio Analysis**: 4,536 audio file references
- **Cross-Tradition Learning**: 65 ragas common to both traditions
- **Feature Extraction**: High-quality metadata for ML features

### **For Musicological Research:**
- **Cross-Tradition Analysis**: 12 validated equivalences
- **Raga Relationships**: Comprehensive raga database
- **Performance Analysis**: Track-level data with artist attribution
- **Quality Assessment**: Source attribution and quality scores

### **For Data Science:**
- **Unified Schema**: Consistent data model
- **Source Attribution**: Complete data provenance
- **Quality Metrics**: Comprehensive data quality indicators
- **Scalable Structure**: Ready for additional data sources

## 📁 **File Locations**

### **Database Files:**
- **Main Database**: `data/unified_ragasense_final/unified_ragasense_database.json`
- **Ragas**: `data/unified_ragasense_final/unified_ragas.json`
- **Artists**: `data/unified_ragasense_final/unified_artists.json`
- **Tracks**: `data/unified_ragasense_final/unified_tracks.json`
- **Audio Files**: `data/unified_ragasense_final/unified_audio_files.json`
- **Cross-Tradition**: `data/unified_ragasense_final/unified_cross_tradition_mappings.json`

### **Exploration Tools:**
- **CLI Explorer**: `explore_ragasense_data.py`
- **Web Explorer**: `web_explorer.py`
- **Documentation**: `DATABASE_EXPLORATION_GUIDE.md`

## 🚀 **Quick Start**

### **1. Command Line (Recommended):**
```bash
# Start interactive exploration
python3 explore_ragasense_data.py

# Then use commands like:
# search "Bhairavi"
# filter "Carnatic"
# top 20
# raga "Kalyani"
# stats
# quit
```

### **2. Web Interface:**
```bash
# Start web server
python3 web_explorer.py

# Open browser to http://localhost:8000
```

### **3. Direct Access:**
```python
import json
with open('data/unified_ragasense_final/unified_ragas.json', 'r') as f:
    ragas = json.load(f)
print(f"Total ragas: {len(ragas)}")
```

## 🎵 **Conclusion**

The RagaSense-Data database provides a comprehensive, high-quality resource for exploring Indian classical music. With 1,340 unique ragas, 4,536 tracks, and 12 cross-tradition mappings, it offers rich opportunities for research, analysis, and machine learning applications.

The database is designed for easy exploration through both command-line and web interfaces, making it accessible to researchers, musicians, and data scientists interested in Indian classical music.
