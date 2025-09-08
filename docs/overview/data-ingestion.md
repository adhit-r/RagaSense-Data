# Data Ingestion Guide

This guide explains how to use the RagaSense-Data ingestion pipeline to process data from various sources.

## Overview

The data ingestion system automatically downloads, processes, and validates data from multiple sources listed in `datasources.md`. It converts all data to a unified format and ensures quality standards.

## Quick Start

### Basic Ingestion

```bash
# Run full ingestion pipeline
python tools/ingestion/datasource_ingestion.py

# Process only high-priority sources
python tools/ingestion/datasource_ingestion.py --priority 1

# Process specific source
python tools/ingestion/datasource_ingestion.py --source "Saraga Carnatic Music Dataset"
```

### With Weights & Biases

```bash
# Enable W&B tracking
python tools/ingestion/datasource_ingestion.py --wandb
```

## Data Sources

The ingestion system processes data from these sources:

### Priority 1 (Critical)
- **Saraga Carnatic Music Dataset**: Time-aligned annotations
- **Carnatic Music Repository (ramanarunachalam)**: Comprehensive metadata
- **SANGEET XML Dataset**: Hindustani metadata and notations
- **Hindustani Music Repository (ramanarunachalam)**: Structured metadata

### Priority 2 (High)
- **Google AudioSet Carnatic Music**: Audio clips with labels
- **Sanidha Multi-Modal Dataset**: Studio recordings with video
- **Carnatic Music Website (ramanarunachalam)**: Web-scraped metadata

### Priority 3+ (Medium/Low)
- **Indian Music Instruments**: Instrument-specific audio
- **Thaat and Raga Forest (TRF)**: Classification datasets

## Ingestion Process

### 1. Download Phase
- Downloads data from various sources
- Supports multiple methods: direct, git, kaggle, API, scraping
- Handles different formats: JSON, XML, CSV, audio, video

### 2. Processing Phase
- Converts source data to unified schema
- Standardizes swara notation across traditions
- Extracts and validates metadata
- Generates unique identifiers

### 3. Validation Phase
- Schema validation against JSON schema
- Quality scoring based on completeness
- Expert validation for critical data
- Cross-reference checking

### 4. Storage Phase
- Saves unified metadata to appropriate directories
- Organizes by tradition (carnatic/hindustani)
- Creates cross-tradition mapping files
- Generates ingestion reports

## Configuration

### Environment Variables

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Weights & Biases
WANDB_PROJECT=ragasense-data
WANDB_ENTITY=your_entity
WANDB_API_KEY=your_api_key

# Kaggle API
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

### Project Configuration

Edit `config/project_config.yaml` to customize:

```yaml
processing:
  audio:
    target_sample_rate: 44100
    target_bit_depth: 16
    
  batch_processing:
    batch_size: 100
    max_workers: 4
```

## Output Structure

After ingestion, data is organized as:

```
data/
├── carnatic/
│   ├── metadata/          # Unified metadata files
│   ├── audio/            # Audio files
│   └── annotations/      # Expert annotations
├── hindustani/
│   ├── metadata/
│   ├── audio/
│   └── annotations/
└── unified/
    ├── mappings/         # Cross-tradition mappings
    └── relationships/    # Relationship graphs
```

## Quality Metrics

The ingestion system tracks:

- **Schema Compliance Rate**: Percentage of files passing schema validation
- **Quality Score**: Average quality score (0.0-1.0)
- **Processing Time**: Time taken for each source
- **Error Rates**: Number and types of errors encountered

## Monitoring

### Weights & Biases Dashboard

When enabled, W&B tracks:

- Ingestion progress and success rates
- Data quality metrics over time
- Processing performance
- Error patterns and trends

### Log Files

Detailed logs are saved to `logs/`:

- `ingestion_report_YYYYMMDD_HHMMSS.json`: Comprehensive report
- Processing logs with timestamps
- Error logs with stack traces

## Troubleshooting

### Common Issues

1. **Download Failures**
   ```bash
   # Check network connectivity
   # Verify source URLs
   # Check API credentials (Kaggle, etc.)
   ```

2. **Schema Validation Errors**
   ```bash
   # Run validation separately
   python tools/validation/data_validator.py --file path/to/file.json
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size in config
   # Process sources individually
   # Use --priority flag to limit scope
   ```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python tools/ingestion/datasource_ingestion.py
```

## Advanced Usage

### Custom Source Processing

To add a new data source:

1. Add source configuration to `datasource_ingestion.py`
2. Implement download method
3. Create processing logic
4. Add validation rules

### Batch Processing

```bash
# Process sources in parallel
python tools/ingestion/datasource_ingestion.py --parallel

# Resume interrupted ingestion
python tools/ingestion/datasource_ingestion.py --resume
```

### Quality Thresholds

```bash
# Set custom quality thresholds
python tools/ingestion/datasource_ingestion.py --quality-threshold 0.8
```

## Best Practices

1. **Start Small**: Begin with priority 1 sources
2. **Monitor Progress**: Use W&B dashboard for tracking
3. **Validate Early**: Run validation after each source
4. **Backup Data**: Keep original downloads in `downloads/`
5. **Document Issues**: Report problems with detailed logs

## Support

For issues with data ingestion:

1. Check the troubleshooting section
2. Review log files in `logs/`
3. Open an issue on GitHub with:
   - Error messages
   - Log files
   - Configuration details
   - Steps to reproduce
