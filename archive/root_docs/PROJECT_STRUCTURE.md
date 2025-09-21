# RagaSense-Data Project Structure

## 📁 Directory Organization

```
RagaSense-Data/
├── 📄 README.md                    # Main project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 pyproject.toml              # Project configuration
├── 📄 .gitignore                  # Git ignore rules
├── 📄 PROJECT_STRUCTURE.md        # This file
│
├── 📁 data/                       # Main dataset directory
│   ├── 📁 raw/                   # Original, unprocessed data
│   │   ├── ramanarunachalam/     # Ramanarunachalam repository
│   │   ├── saraga_datasets/      # Saraga 1.5 datasets
│   │   ├── saraga_carnatic_melody_synth/ # Melody synth data
│   │   └── carnatic_varnam/      # Varnam dataset
│   ├── 📁 processed/             # Cleaned and processed data
│   │   ├── unified_ragas.json    # Main raga database
│   │   ├── unified_artists.json  # Artist database
│   │   ├── unified_tracks.json   # Track database
│   │   └── cross_tradition_mappings.json
│   ├── 📁 ml_ready/              # ML training datasets
│   └── 📁 exports/               # Community export formats
│
├── 📁 scripts/                    # Processing and analysis scripts
│   ├── 📁 data_processing/       # Data processing pipelines
│   │   ├── process_saraga_datasets.py
│   │   ├── simple_saraga_processor.py
│   │   ├── comprehensive_data_processor.py
│   │   └── extract_audio_features.py
│   ├── 📁 analysis/              # Data analysis tools
│   │   ├── analyze_real_data.py
│   │   ├── analyze_saraga_datasets.py
│   │   └── comprehensive_data_analysis.py
│   ├── 📁 exploration/           # Data exploration tools
│   │   ├── explore_ragasense_data.py
│   │   └── web_explorer.py
│   ├── 📁 integration/           # Dataset integration
│   │   ├── integrate_all_datasets.py
│   │   ├── create_unified_dataset.py
│   │   └── unified_dataset_integration.py
│   └── 📁 utilities/             # Utility scripts
│       └── organize_workspace.py
│
├── 📁 ml_models/                  # Machine learning models
│   ├── 📁 training/              # Training scripts
│   │   └── gpu_optimized_trainer.py
│   ├── 📁 models/                # Trained models
│   ├── 📁 inference/             # Inference scripts
│   ├── 📁 experiments/           # ML experiments
│   ├── raga_detection_system.py  # Core ML system
│   ├── raga_detection_api.py     # API server
│   ├── train_raga_model.py       # Training script
│   ├── demo_raga_detection.py    # Demo script
│   ├── requirements.txt          # ML dependencies
│   └── README.md                 # ML documentation
│
├── 📁 tools/                      # Development tools
│   ├── 📁 analysis/              # Analysis tools
│   ├── 📁 audio/                 # Audio processing tools
│   ├── 📁 conversion/            # Format conversion tools
│   ├── 📁 data_processing/       # Data processing tools
│   ├── 📁 exploration/           # Exploration tools
│   ├── 📁 export/                # Export tools
│   ├── 📁 graph-db/              # Graph database tools
│   ├── 📁 ingestion/             # Data ingestion tools
│   ├── 📁 ml/                    # ML tools
│   ├── 📁 utils/                 # Utility tools
│   ├── 📁 validation/            # Validation tools
│   └── 📁 web/                   # Web tools
│
├── 📁 docs/                       # Documentation
│   ├── 📁 overview/              # Project overview
│   ├── 📁 processing/            # Processing documentation
│   ├── 📁 api-reference/         # API documentation
│   ├── 📁 contribution-guide/    # Contribution guidelines
│   ├── 📁 dataset-guide/         # Dataset usage guide
│   ├── 📁 data-sources/          # Data source documentation
│   ├── 📁 ml_models/             # ML model documentation
│   └── 📁 website/               # Website documentation
│
├── 📁 schemas/                    # Database schemas
│   ├── graph-schema.cypher       # Neo4j graph schema
│   ├── neo4j_schema.cypher       # Neo4j schema
│   ├── mapping-schema.json       # Mapping schema
│   └── metadata-schema.json      # Metadata schema
│
├── 📁 examples/                   # Usage examples
│   ├── basic-queries/            # Basic query examples
│   ├── research-examples/        # Research examples
│   └── sample_data/              # Sample data files
│
├── 📁 tests/                      # Test files
│
├── 📁 config/                     # Configuration files
│   ├── project_config.yaml       # Project configuration
│   └── wandb_config.yaml         # Weights & Biases config
│
├── 📁 governance/                 # Data governance
│   ├── data_policy.md            # Data policy
│   ├── quality_standards.md      # Quality standards
│   ├── access_control.md         # Access control
│   └── data_lineage.json         # Data lineage
│
├── 📁 website/                    # Web interface
│   ├── 📁 src/                   # Source code
│   ├── 📁 dist/                  # Built website
│   ├── package.json              # Node dependencies
│   ├── astro.config.mjs          # Astro configuration
│   └── vercel.json               # Vercel deployment config
│
├── 📁 logs/                       # Log files
│   ├── 📁 analysis/              # Analysis logs
│   ├── 📁 data_processing/       # Processing logs
│   ├── 📁 exploration/           # Exploration logs
│   └── 📁 integration/           # Integration logs
│
└── 📁 archive/                    # Archived files
    ├── 📁 root_files/            # Archived root files
    ├── 📁 old_scripts/           # Archived scripts
    ├── 📁 old_docs/              # Archived documentation
    ├── 📁 data_versions/         # Archived data versions
    ├── 📁 analysis_results/      # Archived analysis results
    ├── 📁 downloads/             # Archived downloads
    ├── 📁 duplicates/            # Duplicate files
    └── 📁 config/                # Archived configuration
```

## 🎯 Best Practices Applied

### 1. **Clear Separation of Concerns**
- **Data**: Raw, processed, ML-ready, and export formats
- **Scripts**: Organized by function (processing, analysis, exploration, integration)
- **Tools**: Development and utility tools
- **Documentation**: Comprehensive and well-organized

### 2. **Scalable Structure**
- **Modular design**: Easy to add new components
- **Clear naming**: Descriptive directory and file names
- **Logical grouping**: Related files grouped together

### 3. **Development Best Practices**
- **Configuration management**: Centralized config files
- **Testing**: Dedicated tests directory
- **Documentation**: Comprehensive docs structure
- **Version control**: Proper .gitignore and project config

### 4. **Data Management**
- **Raw data preservation**: Original data never modified
- **Processing pipeline**: Clear data flow from raw to ML-ready
- **Export formats**: Multiple formats for different use cases
- **Archive system**: Old versions preserved but organized

### 5. **Community Ready**
- **Clear documentation**: README, guides, and examples
- **Easy installation**: requirements.txt and pyproject.toml
- **Web interface**: Live website for data exploration
- **API access**: RESTful API for programmatic access

## 🚀 Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Explore data
python3 scripts/exploration/explore_ragasense_data.py

# Process data
python3 scripts/data_processing/simple_saraga_processor.py

# Train ML model
python3 ml_models/training/gpu_optimized_trainer.py
```

### Web Interface
Visit: https://ragasense-data-j26pv45x8-radhi1991s-projects.vercel.app

This structure follows industry best practices for data science and machine learning projects, making it easy to navigate, maintain, and contribute to.

