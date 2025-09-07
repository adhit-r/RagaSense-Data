# RagaSense-Data: Project Summary

## 🎯 Project Overview

RagaSense-Data is a comprehensive, unified dataset repository for Indian Classical Music research that addresses the fragmented data landscape in computational musicology. The project creates a single, standardized platform for both Carnatic and Hindustani traditions with advanced relationship mapping and quality assurance.

## 🏗️ Architecture

### Hybrid Database Approach
- **Neo4j Graph Database**: For complex raga relationship mapping
- **Vector Database**: For audio similarity search and ML embeddings  
- **Structured JSON Files**: For metadata with standardized schemas
- **Weights & Biases**: For experiment tracking and model monitoring

### Core Components
1. **Data Ingestion Pipeline**: Automated processing from 10+ sources
2. **Cross-Tradition Mapping**: Expert-validated raga relationships
3. **Quality Assurance**: Multi-tier validation system
4. **Research Tools**: Validation, analysis, and processing utilities

## 📊 Data Sources Integration

### Carnatic Music (Priority 1)
- **Saraga Carnatic Music Dataset**: Time-aligned annotations
- **Carnatic Music Repository (ramanarunachalam)**: Comprehensive metadata
- **Google AudioSet Carnatic Music**: Audio clips with labels

### Hindustani Music (Priority 1)  
- **SANGEET XML Dataset**: Metadata and notations
- **Hindustani Music Repository (ramanarunachalam)**: Structured metadata
- **Thaat and Raga Forest (TRF)**: Classification datasets

### Multi-Style & Metadata
- **Indian Music Instruments**: Instrument-specific audio
- **Cross-tradition repositories**: Relationship data

## 🔗 Cross-Tradition Mapping System

### Relationship Types
- **SAME**: Identical ragas (e.g., Kalyani ↔ Yaman)
- **SIMILAR**: Very similar with minor differences (e.g., Kharaharapriya ↔ Kafi)
- **RELATED**: Shared characteristics
- **DERIVED**: One evolved from another
- **UNIQUE**: Exists only in one tradition

### Expert Validation
- Multi-expert review process
- Confidence scoring (0.0-1.0)
- Supporting evidence and references
- Disagreement resolution protocols

## 🛠️ Technical Implementation

### Data Processing Pipeline
```bash
# Full ingestion
python tools/ingestion/datasource_ingestion.py

# Validation
python tools/validation/data_validator.py

# Neo4j setup
python tools/graph-db/setup-neo4j.py
```

### Quality Metrics
- **Schema Compliance**: >95% target
- **Quality Score**: 0.0-1.0 scale
- **Expert Validation**: Required for core ragas
- **Cross-Reference Accuracy**: >95% for mapped ragas

### Weights & Biases Integration
- **Experiment Tracking**: Data ingestion, validation, mapping
- **Model Monitoring**: Audio analysis, classification
- **Dashboard**: Real-time metrics and alerts
- **Artifacts**: Dataset versions, model checkpoints

## 📁 Repository Structure

```
RagaSense-Data/
├── data/                    # Core dataset files
│   ├── carnatic/           # Carnatic tradition data
│   ├── hindustani/         # Hindustani tradition data
│   └── unified/            # Cross-tradition mappings
├── schemas/                # Data schemas and validation
├── tools/                  # Processing and analysis tools
│   ├── ingestion/          # Data ingestion pipeline
│   ├── validation/         # Quality assurance
│   └── graph-db/           # Neo4j management
├── docs/                   # Comprehensive documentation
├── config/                 # Project configuration
├── examples/               # Usage examples
└── governance/             # Community guidelines
```

## 🎵 Key Features

### 1. Unified Data Schema
- Single JSON schema for both traditions
- Standardized swara notation (S, R1, R2, G2, G3, M1, M2, P, D1, D2, N2, N3)
- Comprehensive metadata fields
- Quality scoring system

### 2. Advanced Relationship Mapping
- Graph-based raga relationships
- Cypher queries for complex analysis
- Pathfinding and neighborhood analysis
- Pattern matching across traditions

### 3. Quality Assurance
- Multi-tier validation system
- Expert review workflow
- Automated quality scoring
- Continuous monitoring

### 4. Research-Ready Data
- Clean, validated datasets
- Standardized formats
- Rich metadata
- ML-optimized structure

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/ragasense/ragasense-data.git
cd ragasense-data
pip install -r requirements.txt
cp env.example .env  # Configure environment
```

### Quick Start
```bash
# Process high-priority sources
python tools/ingestion/datasource_ingestion.py --priority 1

# Validate data quality
python tools/validation/data_validator.py

# Set up relationship mapping
python tools/graph-db/setup-neo4j.py
```

## 📈 Success Metrics

### Data Quality
- **Target**: >95% schema compliance
- **Target**: >90% expert validation agreement
- **Target**: >1000 hours of audio data
- **Target**: 200+ ragas mapped and verified

### Research Impact
- **Target**: 50+ academic citations per year
- **Target**: 100+ educational institutions using dataset
- **Target**: 10+ novel applications built
- **Target**: 100+ active community contributors

### Community Health
- **Target**: >4.0/5 contributor satisfaction
- **Target**: <48h response time for reviews
- **Target**: All major regions represented
- **Target**: 50+ active experts engaged

## 🔒 Legal & Ethical Framework

### Licensing
- **Metadata & Annotations**: CC-BY-SA 4.0
- **Audio Content**: Individual licensing per source
- **Research Use**: Academic use encouraged
- **Commercial Use**: Separate licensing required

### Cultural Sensitivity
- Respect for traditional knowledge
- Clear consent and attribution
- Regional representation
- Cultural accuracy validation

## 🌟 Innovation Highlights

### 1. First Unified Dataset
- Single platform for both traditions
- Standardized cross-tradition mapping
- Expert-validated relationships

### 2. Advanced Technology Stack
- Graph database for relationships
- Vector database for similarity
- ML experiment tracking
- Automated quality assurance

### 3. Community-Driven Approach
- Open contribution model
- Expert governance
- Transparent validation
- Cultural sensitivity

## 🎯 Future Roadmap

### Phase 1 (Current)
- Core dataset creation
- Basic relationship mapping
- Quality assurance system

### Phase 2 (Next 6 months)
- Advanced audio analysis
- ML model development
- API development
- Community expansion

### Phase 3 (Next 12 months)
- Real-time processing
- Advanced analytics
- Educational tools
- Commercial applications

## 📞 Support & Community

- **Documentation**: Comprehensive guides in `docs/`
- **Issues**: GitHub issue tracker
- **Discussions**: Community forums
- **Email**: Direct contact with maintainers

## 📄 Citation

```bibtex
@dataset{ragasense2024,
  title={RagaSense-Data: A Unified Dataset for Indian Classical Music Research},
  author={RagaSense Community},
  year={2024},
  url={https://github.com/ragasense/ragasense-data},
  license={CC-BY-SA-4.0}
}
```

---

**RagaSense-Data**: *Empowering the next generation of Indian Classical Music research through unified, high-quality data and advanced relationship mapping.*

*Built with ❤️ for the global music research community*
