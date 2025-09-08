# RagaSense-Data Architecture Summary

## 🎵 Overview

RagaSense-Data is a comprehensive, unified dataset for Indian Classical Music that integrates data from multiple sources (Carnatic and Hindustani traditions) with advanced data structures for research and analysis.

## 📊 Current Dataset Statistics

### **Fixed Dataset (After Ragamalika Classification)**
- **Total Individual Ragas**: 5,393
- **Carnatic Ragas**: 880
- **Hindustani Ragas**: 4,513
- **Composition Forms**: 10 (Ragamalika, Talamalika, etc.)
- **Ragamalika Compositions Mapped**: 6
- **Individual Ragas Extracted from Ragamalika**: 20

### **Top Ragas by Song Count**
1. **Unknownraga**: 84,645 songs (needs investigation)
2. **Kalyani**: 2,824 songs
3. **Bhairavi**: 1,998 songs
4. **Thodi**: 1,864 songs
5. **Kaanada**: 1,848 songs

## 🏗️ Architecture Components

### 1. **Core Data Layer**
```
📁 data/
├── unified_ragasense_dataset/          # Original unified dataset
├── ragamalika_classification_fixed/    # Fixed dataset with proper classifications
├── ragamalika_mapping/                 # Ragamalika composition mappings
└── cleaned_ragasense_dataset/          # Cleaned raga data
```

### 2. **Database Schemas**

#### **JSON Database Structure**
- **`unified_ragas_database_fixed.json`**: Individual ragas only (5,393 ragas)
- **`composition_forms_database.json`**: Composition forms (Ragamalika, Talamalika)
- **`ragamalika_compositions_database.json`**: Specific ragamalika compositions
- **`unified_artists_database.json`**: Artist information
- **`unified_composers_database.json`**: Composer information
- **`unified_songs_database.json`**: Song information

#### **Neo4j Graph Database**
- **Nodes**: Raga, Artist, Composer, Song, RagamalikaComposition, CrossTraditionMapping
- **Relationships**: SIMILAR_TO, PERFORMS, COMPOSED, IN_RAGA, CONTAINS, MAPS
- **Schema**: `neo4j_schema.cypher`

#### **Vector Database**
- **Collections**: ragas, songs, artists, composers
- **Vector Types**: Audio Features, Melodic Patterns, Rhythmic Patterns, Text Embeddings
- **Schema**: `vector_database_schema.py`

## 🎼 Ragamalika Composition Mappings

### **Mapped Compositions**
1. **Valachi Vachi** (Patnam Subramania Iyer)
   - **Type**: Navaragamalika Varnam
   - **Ragas**: 9 (Kedaram, Shankarabharanam, Kalyani, Begada, Kambhoji, Yadukulakamboji, Bilahari, Mohanam, Shree)

2. **Bhavayami Raghuramam** (Swathi Thirunal)
   - **Type**: Ragamalika Kriti
   - **Ragas**: 6 (Saveri, Kalyani, Bhairavi, Kambhoji, Yadukulakamboji, Bilahari)

3. **Sri Viswanatham Bhajeham** (Muthuswami Dikshitar)
   - **Type**: Ragamalika Kriti
   - **Ragas**: 14 (Shankarabharanam, Kalyani, Bhairavi, Kambhoji, Yadukulakamboji, Bilahari, Mohanam, Shree, Kedaram, Begada, Hamsadhwani, Madhyamavathi, Sindhubhairavi, Kapi)

4. **Kurai Onrum Illai** (C. Rajagopalachari)
   - **Type**: Ragamalika Devotional
   - **Ragas**: 3 (Sivaranjani, Kapi, Sindhu Bhairavi)

5. **Manasa Verutarula** (Ramaswami Dikshitar)
   - **Type**: Ragamalika Kriti
   - **Ragas**: 48 total, 17 unique

6. **Sivamohanasakti Nannu** (Ramaswami Dikshitar)
   - **Type**: Ragamalika Kriti
   - **Ragas**: 44 total, 17 unique

## 🔗 Cross-Tradition Mappings

### **Identified Mappings**
- **Kalyani** (Carnatic) ↔ **Yaman** (Hindustani) - Identical
- **Bhairavi** (Carnatic) ↔ **Bhairavi** (Hindustani) - Identical
- **Thodi** (Carnatic) ↔ **Miyan ki Todi** (Hindustani) - Similar
- **Kambhoji** (Carnatic) ↔ **Khamaj** (Hindustani) - Similar
- **Mohanam** (Carnatic) ↔ **Bhoop** (Hindustani) - Similar
- **Shankarabharanam** (Carnatic) ↔ **Bilaval** (Hindustani) - Similar

## 🎯 Key Features

### **1. Proper Ragamalika Handling**
- ✅ **Reclassified** Ragamalika as composition form, not individual raga
- ✅ **Extracted** individual ragas from ragamalika compositions
- ✅ **Created** mapping database for ragamalika → constituent ragas
- ✅ **Updated** statistics to count individual ragas separately

### **2. Data Quality Improvements**
- ✅ **Fixed** ragamalika classification issues
- ✅ **Identified** 20 unique ragas from ragamalika compositions
- ✅ **Created** separate databases for composition forms
- ✅ **Preserved** ragamalika relationships while tracking constituent ragas

### **3. Advanced Data Structures**
- ✅ **Neo4j Graph Database** for relationship mapping
- ✅ **Vector Database** for similarity search and content-based analysis
- ✅ **Comprehensive Metadata** with quality scores and validation
- ✅ **Cross-Tradition Mappings** with similarity scores

## 🚀 Future Enhancements

### **Immediate Tasks**
1. **Investigate "Unknownraga"** entries (84,645 songs)
2. **Split combined raga names** (e.g., "Bilaval, Hameer")
3. **Validate raga classifications** against authoritative sources
4. **Integrate Saraga 1.5 datasets** (metadata processing currently failing)

### **Advanced Features**
1. **Audio Feature Extraction** from YouTube links
2. **Automated Raga Classification** using ML models
3. **Real-time Similarity Search** using vector database
4. **Graph-based Relationship Discovery** using Neo4j
5. **Quality Validation System** with automated scoring

## 📁 File Structure

```
RagaSense-Data/
├── data/
│   ├── unified_ragasense_dataset/          # Original dataset
│   ├── ragamalika_classification_fixed/    # Fixed dataset
│   ├── ragamalika_mapping/                 # Ragamalika mappings
│   └── cleaned_ragasense_dataset/          # Cleaned data
├── downloads/                              # Raw data sources
├── schemas/                                # Database schemas
├── create_ragamalika_mapping.py           # Ragamalika mapping script
├── fix_ragamalika_classification.py       # Classification fix script
├── neo4j_schema.cypher                    # Neo4j schema
├── vector_database_schema.py              # Vector database schema
├── data_architecture_visualizations.md    # Architecture diagrams
└── RagaSense-Data_Architecture_Summary.md # This file
```

## 🎵 Impact

The RagaSense-Data dataset now provides:

1. **Accurate Raga Classification**: Individual ragas properly separated from composition forms
2. **Ragamalika Analysis**: Complete mapping of ragamalika compositions to constituent ragas
3. **Cross-Tradition Research**: Structured mappings between Carnatic and Hindustani ragas
4. **Advanced Querying**: Graph and vector-based search capabilities
5. **Research-Ready Data**: Comprehensive metadata with quality validation

This makes RagaSense-Data a powerful resource for musicological research, machine learning applications, and cultural preservation of Indian Classical Music traditions.

## 🔧 Technical Implementation

### **Data Processing Pipeline**
1. **Data Ingestion** from multiple sources
2. **Data Validation** and quality checks
3. **Ragamalika Classification** and individual raga extraction
4. **Cross-Tradition Mapping** with similarity scoring
5. **Metadata Enrichment** and quality validation
6. **Multi-Database Storage** (JSON, Neo4j, Vector)

### **Query Capabilities**
- **Similarity Search**: Find similar ragas, songs, artists
- **Relationship Discovery**: Graph-based relationship analysis
- **Cross-Tradition Analysis**: Compare Carnatic and Hindustani traditions
- **Ragamalika Decomposition**: Extract individual ragas from compositions
- **Quality Assessment**: Automated data quality scoring

The architecture is designed to be scalable, maintainable, and research-friendly, providing a solid foundation for advanced musicological analysis and machine learning applications.
