# RagaSense Workspace Organization Plan

## 🎯 Current State Analysis

### Data Sources Available
1. **Ramanarunachalam_Music_Repository** ✅ (Complete - 868 ragas, 105,339 songs)
2. **Saraga 1.5 Carnatic** ✅ (Downloaded - needs processing)
3. **Saraga 1.5 Hindustani** ✅ (Downloaded - needs processing)  
4. **Saraga Carnatic Melody Synth** ✅ (Processed - 339 audio files)
5. **Carnatic Varnam** ✅ (Downloaded - needs processing)

### Current Issues
- **Data folder is messy** - 20+ subdirectories with overlapping content
- **Multiple versions** of same datasets (unified_ragasense_dataset, unified_ragasense_final, etc.)
- **Old/unused files** scattered throughout
- **No clear governance** structure
- **Tools folder underutilized**
- **No GPU utilization** for ML training
- **No Weights & Biases** integration

## 🏗️ Proposed Organization Structure

```
RagaSense-Data/
├── 📁 data/                          # Clean, organized data
│   ├── 📁 raw/                       # Original, unprocessed data
│   │   ├── ramanarunachalam/         # Ramanarunachalam repository
│   │   ├── saraga/                   # All Saraga datasets
│   │   └── carnatic_varnam/          # Varnam dataset
│   ├── 📁 processed/                 # Cleaned, processed data
│   │   ├── unified_ragas.json        # Main raga database
│   │   ├── unified_artists.json      # Artist database
│   │   ├── unified_tracks.json       # Track database
│   │   └── cross_tradition_mappings.json
│   ├── 📁 ml_ready/                  # ML training data
│   │   ├── audio_features/           # Extracted audio features
│   │   ├── embeddings/               # Vector embeddings
│   │   └── training_data/            # Training datasets
│   └── 📁 exports/                   # Export formats
│       ├── csv/                      # CSV exports
│       ├── parquet/                  # Parquet files
│       └── sqlite/                   # SQLite database
├── 📁 ml_models/                     # ML models and training
│   ├── models/                       # Trained models
│   ├── training/                     # Training scripts
│   ├── inference/                    # Inference scripts
│   └── experiments/                  # ML experiments
├── 📁 tools/                         # Utility tools
│   ├── data_processing/              # Data processing tools
│   ├── analysis/                     # Analysis tools
│   ├── validation/                   # Data validation
│   └── export/                       # Export tools
├── 📁 scripts/                       # Main processing scripts
├── 📁 docs/                          # Documentation
├── 📁 governance/                    # Data governance
├── 📁 archive/                       # Archived files
└── 📁 logs/                          # Log files
```

## 📋 Organization Tasks

### Phase 1: Archive Old Files
- [ ] Move old data versions to archive/
- [ ] Archive unused scripts and docs
- [ ] Archive old analysis results
- [ ] Archive duplicate datasets

### Phase 2: Reorganize Data Folder
- [ ] Create new data structure
- [ ] Move raw data to data/raw/
- [ ] Consolidate processed data
- [ ] Create ML-ready datasets

### Phase 3: Optimize Tools
- [ ] Reorganize tools folder
- [ ] Create data processing pipeline
- [ ] Add GPU support for ML
- [ ] Integrate Weights & Biases

### Phase 4: Governance & Documentation
- [ ] Create data governance structure
- [ ] Update documentation
- [ ] Create data lineage tracking
- [ ] Add quality metrics

## 🚀 Implementation Plan

### Immediate Actions
1. **Archive old files** - Move to archive/ without deletion
2. **Consolidate data** - Keep only latest, clean versions
3. **Reorganize structure** - Create clear hierarchy
4. **Add GPU support** - Enable Mac GPU for ML training
5. **Integrate W&B** - Add experiment tracking

### Data Consolidation Strategy
- **Keep**: `unified_ragasense_final/` (latest, cleanest)
- **Archive**: All other unified_* versions
- **Process**: Saraga datasets for ML training
- **Organize**: Raw data by source

### ML Optimization
- **GPU Support**: Enable Mac GPU for training
- **W&B Integration**: Track experiments and models
- **Data Pipeline**: Automated processing pipeline
- **Model Versioning**: Proper model management

## 📊 Expected Outcomes

### Benefits
- **Clean workspace** - Easy navigation
- **Clear data lineage** - Know what's what
- **Better performance** - GPU acceleration
- **Experiment tracking** - W&B integration
- **Scalable structure** - Ready for growth

### Metrics
- **Data reduction**: ~50% fewer files in main directories
- **Processing speed**: 3-5x faster with GPU
- **Experiment tracking**: 100% of ML runs tracked
- **Data quality**: Automated validation pipeline
