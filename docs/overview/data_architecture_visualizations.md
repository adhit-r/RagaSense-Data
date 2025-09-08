# RagaSense-Data Architecture Visualizations

## 1. Core Entity Relationship Diagram (ERD)

```mermaid
erDiagram
    INDIVIDUAL_RAGA {
        string raga_id PK
        string name
        string tradition
        array sources
        int song_count
        object metadata
        array youtube_links
        int melakarta_number
        string parent_scale
        array composition_forms
    }
    
    COMPOSITION_FORM {
        string form_id PK
        string name
        string type
        string description
        string tradition
        array sources
        int song_count
        object metadata
        array youtube_links
        string original_raga_id
    }
    
    RAGAMALIKA_COMPOSITION {
        string composition_id PK
        string name
        string composer
        string type
        string tradition
        array constituent_ragas
        int total_ragas
        array unique_ragas
        int unique_raga_count
        array songs
        object metadata
        array youtube_links
    }
    
    ARTIST {
        string artist_id PK
        string name
        string tradition
        array sources
        int song_count
        object metadata
        array youtube_links
        array specializations
        string period
    }
    
    COMPOSER {
        string composer_id PK
        string name
        string tradition
        array sources
        int song_count
        object metadata
        array youtube_links
        string period
        array compositions
    }
    
    SONG {
        string song_id PK
        string title
        string tradition
        array raga_ids
        string artist_id FK
        string composer_id FK
        string composition_form_id FK
        object metadata
        array youtube_links
        string duration
        string language
    }
    
    CROSS_TRADITION_MAPPING {
        string mapping_id PK
        string carnatic_raga_id FK
        string hindustani_raga_id FK
        string relationship_type
        float similarity_score
        string notes
    }
    
    INDIVIDUAL_RAGA ||--o{ SONG : "contains"
    COMPOSITION_FORM ||--o{ SONG : "contains"
    RAGAMALIKA_COMPOSITION ||--o{ INDIVIDUAL_RAGA : "comprises"
    ARTIST ||--o{ SONG : "performs"
    COMPOSER ||--o{ SONG : "composes"
    INDIVIDUAL_RAGA ||--o{ CROSS_TRADITION_MAPPING : "maps_to"
```

## 2. Ragamalika Composition Structure

```mermaid
graph TD
    A[Ragamalika Composition] --> B[Valachi Vachi]
    A --> C[Bhavayami Raghuramam]
    A --> D[Sri Viswanatham Bhajeham]
    A --> E[Kurai Onrum Illai]
    A --> F[Manasa Verutarula]
    A --> G[Sivamohanasakti Nannu]
    
    B --> B1[Kedaram]
    B --> B2[Shankarabharanam]
    B --> B3[Kalyani]
    B --> B4[Begada]
    B --> B5[Kambhoji]
    B --> B6[Yadukulakamboji]
    B --> B7[Bilahari]
    B --> B8[Mohanam]
    B --> B9[Shree]
    
    C --> C1[Saveri]
    C --> C2[Kalyani]
    C --> C3[Bhairavi]
    C --> C4[Kambhoji]
    C --> C5[Yadukulakamboji]
    C --> C6[Bilahari]
    
    D --> D1[Shankarabharanam]
    D --> D2[Kalyani]
    D --> D3[Bhairavi]
    D --> D4[Kambhoji]
    D --> D5[Yadukulakamboji]
    D --> D6[Bilahari]
    D --> D7[Mohanam]
    D --> D8[Shree]
    D --> D9[Kedaram]
    D --> D10[Begada]
    D --> D11[Hamsadhwani]
    D --> D12[Madhyamavathi]
    D --> D13[Sindhubhairavi]
    D --> D14[Kapi]
    
    E --> E1[Sivaranjani]
    E --> E2[Kapi]
    E --> E3[Sindhu Bhairavi]
    
    F --> F1[17 Unique Ragas]
    G --> G1[17 Unique Ragas]
```

## 3. Vector Database Architecture

```mermaid
graph TB
    subgraph "Vector Database (ChromaDB/Pinecone)"
        V1[Audio Feature Vectors]
        V2[Melodic Pattern Vectors]
        V3[Rhythmic Pattern Vectors]
        V4[Text Embeddings]
        V5[Metadata Vectors]
    end
    
    subgraph "Vector Types"
        V1 --> V1A[MFCC Features]
        V1 --> V1B[Spectral Features]
        V1 --> V1C[Chroma Features]
        V1 --> V1D[Tonnetz Features]
        
        V2 --> V2A[Raga Scale Vectors]
        V2 --> V2B[Melodic Contour]
        V2 --> V2C[Phrase Patterns]
        
        V3 --> V3A[Tala Pattern Vectors]
        V3 --> V3B[Rhythmic Density]
        V3 --> V3C[Accent Patterns]
        
        V4 --> V4A[Raga Name Embeddings]
        V4 --> V4B[Composer Embeddings]
        V4 --> V4C[Lyrics Embeddings]
        
        V5 --> V5A[Metadata Embeddings]
        V5 --> V5B[Context Embeddings]
    end
    
    subgraph "Vector Operations"
        VO1[Similarity Search]
        VO2[Clustering]
        VO3[Classification]
        VO4[Recommendation]
        VO5[Anomaly Detection]
    end
    
    V1 --> VO1
    V2 --> VO1
    V3 --> VO1
    V4 --> VO1
    V5 --> VO1
    
    V1 --> VO2
    V2 --> VO2
    V3 --> VO2
    
    V1 --> VO3
    V2 --> VO3
    V4 --> VO3
    
    V1 --> VO4
    V2 --> VO4
    V4 --> VO4
    
    V1 --> VO5
    V2 --> VO5
```

## 4. Neo4j Graph Database Schema

```mermaid
graph LR
    subgraph "Neo4j Graph Database"
        R[Raga] --> A[Artist]
        R --> C[Composer]
        R --> S[Song]
        R --> RT[RagaType]
        R --> RS[RagaScale]
        R --> RM[RagaMapping]
        
        A --> S
        C --> S
        S --> Y[YouTube]
        S --> M[Metadata]
        
        R --> R2[Raga]
        RM --> R2
        
        subgraph "Raga Relationships"
            R --> |"similar_to"| R2
            R --> |"derived_from"| R2
            R --> |"parent_of"| R2
            R --> |"child_of"| R2
        end
        
        subgraph "Composition Relationships"
            RC[RagamalikaComposition] --> R
            RC --> |"contains"| R
            RC --> C
        end
        
        subgraph "Cross-Tradition Mappings"
            CTM[CrossTraditionMapping] --> R
            CTM --> R2
            CTM --> |"identical"| R2
            CTM --> |"similar"| R2
            CTM --> |"derived"| R2
        end
    end
```

## 5. Data Flow Architecture

```mermaid
flowchart TD
    subgraph "Data Sources"
        DS1[Ramanarunachalam]
        DS2[Saraga 1.5 Carnatic]
        DS3[Saraga 1.5 Hindustani]
        DS4[Saraga Carnatic Melody Synth]
        DS5[YouTube Links]
    end
    
    subgraph "Data Processing Pipeline"
        DP1[Data Ingestion]
        DP2[Data Validation]
        DP3[Data Cleaning]
        DP4[Ragamalika Classification]
        DP5[Cross-Tradition Mapping]
        DP6[Metadata Enrichment]
    end
    
    subgraph "Storage Layer"
        SL1[Unified JSON Database]
        SL2[Vector Database]
        SL3[Neo4j Graph Database]
        SL4[Composition Forms DB]
        SL5[Ragamalika Mapping DB]
    end
    
    subgraph "API Layer"
        API1[REST API]
        API2[GraphQL API]
        API3[Vector Search API]
        API4[Recommendation API]
    end
    
    subgraph "Applications"
        APP1[Web Interface]
        APP2[Research Tools]
        APP3[ML Models]
        APP4[Analytics Dashboard]
    end
    
    DS1 --> DP1
    DS2 --> DP1
    DS3 --> DP1
    DS4 --> DP1
    DS5 --> DP1
    
    DP1 --> DP2
    DP2 --> DP3
    DP3 --> DP4
    DP4 --> DP5
    DP5 --> DP6
    
    DP6 --> SL1
    DP6 --> SL2
    DP6 --> SL3
    DP6 --> SL4
    DP6 --> SL5
    
    SL1 --> API1
    SL2 --> API3
    SL3 --> API2
    SL4 --> API1
    SL5 --> API1
    
    API1 --> APP1
    API2 --> APP2
    API3 --> APP3
    API4 --> APP4
```

## 6. Metadata Structure Visualization

```mermaid
graph TD
    subgraph "Raga Metadata"
        RM[Raga Metadata] --> RM1[Basic Info]
        RM --> RM2[Musical Properties]
        RM --> RM3[Relationships]
        RM --> RM4[Sources]
        
        RM1 --> RM1A[name]
        RM1 --> RM1B[tradition]
        RM1 --> RM1C[raga_id]
        RM1 --> RM1D[song_count]
        
        RM2 --> RM2A[melakarta_number]
        RM2 --> RM2B[parent_scale]
        RM2 --> RM2C[arohana]
        RM2 --> RM2D[avarohana]
        RM2 --> RM2E[swaras]
        
        RM3 --> RM3A[cross_tradition_mappings]
        RM3 --> RM3B[similar_ragas]
        RM3 --> RM3C[derived_ragas]
        RM3 --> RM3D[composition_forms]
        
        RM4 --> RM4A[sources]
        RM4 --> RM4B[youtube_links]
        RM4 --> RM4C[metadata_quality]
    end
    
    subgraph "Song Metadata"
        SM[Song Metadata] --> SM1[Basic Info]
        SM --> SM2[Musical Info]
        SM --> SM3[Performance Info]
        SM --> SM4[Media Info]
        
        SM1 --> SM1A[title]
        SM1 --> SM1B[language]
        SM1 --> SM1C[duration]
        SM1 --> SM1D[tradition]
        
        SM2 --> SM2A[raga_ids]
        SM2 --> SM2B[tala]
        SM2 --> SM2C[composition_form]
        SM2 --> SM2D[genre]
        
        SM3 --> SM3A[artist_id]
        SM3 --> SM3B[composer_id]
        SM3 --> SM3C[performance_date]
        SM3 --> SM3D[venue]
        
        SM4 --> SM4A[youtube_links]
        SM4 --> SM4B[audio_quality]
        SM4 --> SM4C[transcription]
        SM4 --> SM4D[annotations]
    end
```

## 7. Cross-Tradition Mapping Visualization

```mermaid
graph LR
    subgraph "Carnatic Ragas"
        C1[Kalyani]
        C2[Bhairavi]
        C3[Thodi]
        C4[Kambhoji]
        C5[Mohanam]
        C6[Shankarabharanam]
    end
    
    subgraph "Hindustani Ragas"
        H1[Yaman]
        H2[Bhairavi]
        H3[Miyan ki Todi]
        H4[Khamaj]
        H5[Bhoop]
        H6[Bilaval]
    end
    
    subgraph "Mapping Relationships"
        M1[Identical]
        M2[Similar]
        M3[Derived]
        M4[Unique]
    end
    
    C1 --> |"Identical"| H1
    C2 --> |"Identical"| H2
    C3 --> |"Similar"| H3
    C4 --> |"Similar"| H4
    C5 --> |"Similar"| H5
    C6 --> |"Similar"| H6
    
    C1 -.-> |"0.95 similarity"| H1
    C2 -.-> |"0.98 similarity"| H2
    C3 -.-> |"0.87 similarity"| H3
    C4 -.-> |"0.82 similarity"| H4
    C5 -.-> |"0.89 similarity"| H5
    C6 -.-> |"0.91 similarity"| H6
```

## 8. Vector Search and Similarity Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        I1[Audio File]
        I2[Text Query]
        I3[Raga Name]
        I4[Metadata Query]
    end
    
    subgraph "Feature Extraction"
        FE1[Audio Features]
        FE2[Text Embeddings]
        FE3[Metadata Vectors]
        FE4[Hybrid Features]
    end
    
    subgraph "Vector Database"
        VD1[Audio Vectors]
        VD2[Text Vectors]
        VD3[Metadata Vectors]
        VD4[Composite Vectors]
    end
    
    subgraph "Similarity Search"
        SS1[Cosine Similarity]
        SS2[Euclidean Distance]
        SS3[Manhattan Distance]
        SS4[Custom Distance]
    end
    
    subgraph "Results"
        R1[Similar Ragas]
        R2[Similar Songs]
        R3[Similar Artists]
        R4[Recommendations]
    end
    
    I1 --> FE1
    I2 --> FE2
    I3 --> FE3
    I4 --> FE4
    
    FE1 --> VD1
    FE2 --> VD2
    FE3 --> VD3
    FE4 --> VD4
    
    VD1 --> SS1
    VD2 --> SS2
    VD3 --> SS3
    VD4 --> SS4
    
    SS1 --> R1
    SS2 --> R2
    SS3 --> R3
    SS4 --> R4
```

## 9. Neo4j Cypher Query Examples

```cypher
// Find all ragas similar to Kalyani
MATCH (r1:Raga {name: "Kalyani"})-[:SIMILAR_TO]->(r2:Raga)
RETURN r2.name, r2.tradition

// Find ragamalika compositions containing Kalyani
MATCH (rc:RagamalikaComposition)-[:CONTAINS]->(r:Raga {name: "Kalyani"})
RETURN rc.name, rc.composer, rc.type

// Find cross-tradition mappings for Bhairavi
MATCH (ctm:CrossTraditionMapping)-[:MAPS]->(r:Raga {name: "Bhairavi"})
RETURN ctm.relationship_type, ctm.similarity_score

// Find artists who perform in multiple ragas
MATCH (a:Artist)-[:PERFORMS]->(s:Song)-[:IN_RAGA]->(r:Raga)
WITH a, collect(DISTINCT r.name) as ragas
WHERE size(ragas) > 5
RETURN a.name, ragas

// Find ragamalika compositions with most ragas
MATCH (rc:RagamalikaComposition)
RETURN rc.name, rc.total_ragas
ORDER BY rc.total_ragas DESC
LIMIT 10
```

## 10. Data Quality and Validation Flow

```mermaid
flowchart TD
    subgraph "Data Quality Checks"
        DQ1[Schema Validation]
        DQ2[Data Completeness]
        DQ3[Data Consistency]
        DQ4[Cross-Reference Validation]
        DQ5[Business Rule Validation]
    end
    
    subgraph "Quality Metrics"
        QM1[Completeness Score]
        QM2[Accuracy Score]
        QM3[Consistency Score]
        QM4[Validity Score]
        QM5[Overall Quality Score]
    end
    
    subgraph "Quality Issues"
        QI1[Missing Data]
        QI2[Invalid Data]
        QI3[Inconsistent Data]
        QI4[Duplicate Data]
        QI5[Outdated Data]
    end
    
    subgraph "Quality Actions"
        QA1[Data Cleaning]
        QA2[Data Enrichment]
        QA3[Data Standardization]
        QA4[Data Validation]
        QA5[Quality Monitoring]
    end
    
    DQ1 --> QM1
    DQ2 --> QM2
    DQ3 --> QM3
    DQ4 --> QM4
    DQ5 --> QM5
    
    QM1 --> QI1
    QM2 --> QI2
    QM3 --> QI3
    QM4 --> QI4
    QM5 --> QI5
    
    QI1 --> QA1
    QI2 --> QA2
    QI3 --> QA3
    QI4 --> QA4
    QI5 --> QA5
```

This comprehensive visualization shows the complete architecture of our RagaSense-Data system, including:

1. **Entity Relationship Diagrams** for all data structures
2. **Ragamalika composition mappings** with constituent ragas
3. **Vector database architecture** for similarity search
4. **Neo4j graph database schema** for relationship mapping
5. **Data flow architecture** from sources to applications
6. **Metadata structure** for ragas and songs
7. **Cross-tradition mapping** relationships
8. **Vector search and similarity** architecture
9. **Neo4j Cypher query examples** for common operations
10. **Data quality and validation** flow

The system now properly handles ragamalika compositions by extracting individual ragas while preserving the composition relationships, making it much more accurate and research-ready!
