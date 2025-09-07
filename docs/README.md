# RagaSense-Data Documentation

Welcome to the comprehensive documentation for RagaSense-Data, the unified dataset repository for Indian Classical Music research.

## Table of Contents

1. [Getting Started](getting-started.md)
2. [Data Schema](data-schema.md)
3. [Data Ingestion](data-ingestion.md)
4. [Cross-Tradition Mapping](cross-tradition-mapping.md)
5. [Quality Assurance](quality-assurance.md)
6. [API Reference](api-reference.md)
7. [Contributing](contributing.md)
8. [Troubleshooting](troubleshooting.md)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ragasense/ragasense-data.git
cd ragasense-data

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Basic Usage

```bash
# Run data ingestion
python tools/ingestion/datasource_ingestion.py --priority 1

# Validate data quality
python tools/validation/data_validator.py

# Set up Neo4j for relationship mapping
python tools/graph-db/setup-neo4j.py
```

## Project Overview

RagaSense-Data is designed to solve the fragmented data landscape in Indian Classical Music research by providing:

- **Unified Schema**: Single standardized format for both Carnatic and Hindustani traditions
- **Cross-Tradition Mapping**: Explicit relationships between ragas across traditions
- **Quality Assurance**: Multi-tier validation with expert review
- **Research-Ready**: Clean, validated data optimized for ML workflows
- **Community-Driven**: Open contribution model with expert governance

## Architecture

The project uses a hybrid database approach:

- **Neo4j**: Graph database for raga relationship mapping
- **Vector Database**: For audio similarity search and ML embeddings
- **Structured Files**: JSON metadata with standardized schemas
- **Weights & Biases**: Experiment tracking and model monitoring

## Data Sources

The dataset integrates data from multiple sources:

### Carnatic Music
- Saraga Carnatic Music Dataset
- Google AudioSet Carnatic Music
- Carnatic Music Repository (ramanarunachalam)
- Sanidha Multi-Modal Dataset

### Hindustani Music
- SANGEET XML Dataset
- Hindustani Music Repository (ramanarunachalam)
- Thaat and Raga Forest (TRF)

### Multi-Style
- Indian Music Instruments Dataset
- Cross-tradition metadata repositories

## Key Features

### 1. Unified Data Schema
- Single JSON schema for both traditions
- Standardized swara notation
- Comprehensive metadata fields
- Quality scoring system

### 2. Cross-Tradition Mapping
- Identical raga relationships (e.g., Kalyani ↔ Yaman)
- Similar raga mappings (e.g., Kharaharapriya ↔ Kafi)
- Expert-validated confidence scores
- Supporting evidence and references

### 3. Quality Assurance
- Schema validation
- Expert review process
- Automated quality scoring
- Continuous monitoring

### 4. Research Tools
- Data ingestion pipeline
- Validation framework
- Relationship analysis
- Audio processing utilities

## Getting Help

- **Documentation**: Check the specific guides in this directory
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions
- **Email**: Contact the maintainers

## License

- **Metadata & Annotations**: CC-BY-SA 4.0
- **Audio Content**: Individual licensing per source
- **Research Use**: Academic use encouraged with attribution

## Citation

If you use this dataset in your research, please cite:

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

*Empowering the next generation of Indian Classical Music research through unified, high-quality data.*
