# RagaSense-Data

A comprehensive, unified dataset for Indian Classical Music research, combining data from multiple sources including Ramanarunachalam, Saraga, and Carnatic Varnam datasets.

## 🎵 Overview

RagaSense-Data is the largest unified dataset for Indian Classical Music, containing:
- **1,340+ unique ragas** from both Carnatic and Hindustani traditions
- **100,000+ songs** with metadata and cross-tradition mappings
- **Audio features** extracted for machine learning applications
- **Cross-tradition mappings** validated by musicological experts

## 📊 Dataset Statistics

- **Ramanarunachalam**: 868 ragas, 105,339 songs
- **Saraga 1.5 Carnatic**: 1,982 audio files, 498 metadata files
- **Saraga 1.5 Hindustani**: Processing in progress
- **Saraga Melody Synth**: 339 audio files
- **Carnatic Varnam**: Processing in progress

## 🚀 Quick Start

### Data Access
```bash
# Explore the dataset
python3 scripts/exploration/explore_ragasense_data.py

# Web interface
python3 scripts/exploration/web_explorer.py
```

### ML Training
```bash
# Extract audio features
python3 scripts/data_processing/extract_audio_features.py

# Train raga detection model
python3 ml_models/training/gpu_optimized_trainer.py
```

## 📁 Project Structure

```
RagaSense-Data/
├── data/                    # Main dataset
│   ├── raw/                # Original data sources
│   ├── processed/          # Cleaned and processed data
│   ├── ml_ready/          # ML training datasets
│   └── exports/           # Community export formats
├── scripts/               # Processing and analysis scripts
│   ├── data_processing/   # Data processing pipelines
│   ├── analysis/          # Data analysis tools
│   ├── exploration/       # Data exploration tools
│   ├── integration/       # Dataset integration
│   └── utilities/         # Utility scripts
├── ml_models/            # Machine learning models
├── tools/                # Development tools
├── docs/                 # Documentation
├── schemas/              # Database schemas
└── website/              # Web interface
```

## 🌐 Web Interface

Visit our live website: [RagaSense-Data](https://ragasense-data-j26pv45x8-radhi1991s-projects.vercel.app)

## 📚 Documentation

- [Dataset Guide](docs/dataset-guide/)
- [API Reference](docs/api-reference/)
- [Contribution Guide](docs/contribution-guide/)

## 🤝 Contributing

We welcome contributions! Please see our [Contribution Guide](docs/contribution-guide/) for details.

## 📄 License

This dataset is released under [License Type] for research and educational purposes.

## 📞 Contact

For questions or collaboration, please contact [Contact Information].

## 🙏 Acknowledgments

- Ramanarunachalam Music Repository
- Saraga Dataset Team
- Carnatic Varnam Dataset Contributors
- Musicological experts who validated cross-tradition mappings
