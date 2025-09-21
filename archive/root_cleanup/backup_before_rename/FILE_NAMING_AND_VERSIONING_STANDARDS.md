# File Naming and Versioning Standards

## 🎯 **CURRENT PROBLEMS IDENTIFIED**

### **❌ Chaotic Naming Issues**
- **Inconsistent naming**: `test_model_direct.py` vs `test_model_simple.py` vs `test_model_neural_network.py`
- **No versioning**: `raga_classifier_api.py` vs `raga_classifier_api_working.py`
- **Unclear purpose**: `extract_saraga_audio_features.py` vs `train_simple_raga_classifier.py`
- **Mixed conventions**: `phase1_enhancement.py` vs `phase2_simplified_ml.py`
- **No timestamps**: Files overwritten without version history
- **No status indicators**: `working`, `old`, `new` suffixes everywhere

---

## 📋 **PROPOSED NAMING STANDARDS**

### **1. File Naming Convention**
```
[category]_[purpose]_[version]_[status].[extension]
```

### **2. Categories**
- **`ml`** - Machine Learning models and training
- **`data`** - Data processing and analysis
- **`api`** - API endpoints and services
- **`util`** - Utility scripts and helpers
- **`test`** - Testing scripts and validation
- **`doc`** - Documentation files
- **`config`** - Configuration files
- **`script`** - General processing scripts

### **3. Purposes**
- **`train`** - Training scripts
- **`predict`** - Prediction/inference scripts
- **`extract`** - Feature extraction
- **`process`** - Data processing
- **`validate`** - Validation scripts
- **`analyze`** - Analysis scripts
- **`integrate`** - Integration scripts
- **`classify`** - Classification scripts

### **4. Versioning**
- **`v1.0`** - Major version (breaking changes)
- **`v1.1`** - Minor version (new features)
- **`v1.1.1`** - Patch version (bug fixes)
- **`v1.1.1-alpha`** - Alpha release
- **`v1.1.1-beta`** - Beta release
- **`v1.1.1-rc1`** - Release candidate

### **5. Status Indicators**
- **`stable`** - Production ready
- **`dev`** - Development version
- **`experimental`** - Experimental features
- **`deprecated`** - Deprecated, use newer version
- **`archived`** - Archived, not maintained

---

## 🔧 **IMPLEMENTATION PLAN**

### **Phase 1: Rename Core ML Files**
```
OLD → NEW
├── raga_classifier_api.py → api_raga_classifier_v1.0_stable.py
├── raga_classifier_api_working.py → api_raga_classifier_v1.1_stable.py
├── train_simple_raga_classifier.py → ml_train_classifier_v1.0_stable.py
├── extract_saraga_audio_features.py → data_extract_features_v1.0_stable.py
├── test_model_direct.py → test_model_direct_v1.0_dev.py
├── test_model_simple.py → test_model_simple_v1.0_dev.py
└── test_model_neural_network.py → test_model_neural_v1.0_dev.py
```

### **Phase 2: Rename Processing Scripts**
```
OLD → NEW
├── phase1_enhancement.py → script_phase1_enhancement_v1.0_archived.py
├── phase2_simplified_ml.py → script_phase2_ml_v1.0_archived.py
├── phase3_composer_relationship_fix.py → data_fix_composer_relations_v1.0_stable.py
├── phase3_youtube_validation.py → data_validate_youtube_v1.0_stable.py
└── phase4_youtube_dataset_creation.py → data_create_youtube_dataset_v1.0_dev.py
```

### **Phase 3: Rename Analysis Files**
```
OLD → NEW
├── yue_model_analysis_simplified.py → ml_analyze_yue_model_v1.0_stable.py
├── yue_model_comprehensive_analysis.py → ml_analyze_yue_comprehensive_v1.0_stable.py
├── yue_rhythmic_analysis.py → ml_analyze_yue_rhythmic_v1.0_stable.py
└── real_raga_classifier.py → ml_train_real_classifier_v1.0_dev.py
```

### **Phase 4: Rename Documentation**
```
OLD → NEW
├── CORRECTED_RAGA_STATISTICS.md → doc_raga_statistics_corrected_v1.0_stable.md
├── REAL_DATA_ANALYSIS_RESULTS.md → doc_real_data_analysis_v1.0_stable.md
├── YUE_RHYTHMIC_ANALYSIS_SUMMARY.md → doc_yue_rhythmic_analysis_v1.0_stable.md
└── ML_TRAINING_SUCCESS_SUMMARY.md → doc_ml_training_success_v1.0_stable.md
```

---

## 📁 **DIRECTORY STRUCTURE WITH VERSIONING**

### **ML Models Directory**
```
ml_models/
├── core_models/
│   ├── ml_raga_detection_system_v1.0_stable.py
│   ├── ml_raga_detection_system_v1.1_dev.py
│   └── ml_demo_raga_detection_v1.0_stable.py
├── api_models/
│   ├── api_raga_classifier_v1.0_stable.py
│   ├── api_raga_classifier_v1.1_stable.py
│   └── api_raga_classifier_v2.0_dev.py
├── training_scripts/
│   ├── ml_train_classifier_v1.0_stable.py
│   ├── ml_train_classifier_v1.1_dev.py
│   └── ml_train_gpu_optimized_v1.0_stable.py
└── experiments/
    ├── ml_experiment_yue_model_v1.0_experimental.py
    └── ml_experiment_rhythmic_analysis_v1.0_experimental.py
```

### **Data Processing Directory**
```
data/
├── organized_processed/
│   ├── data_extract_saraga_features_v1.0_stable.py
│   ├── data_process_unified_dataset_v1.0_stable.py
│   └── data_validate_youtube_links_v1.0_stable.py
└── organized_raw/
    └── data_download_saraga_v1.0_stable.py
```

### **Archive Directory**
```
archive/
├── organized_by_type/
│   ├── ml_models_v1.0_archived/
│   ├── data_processing_v1.0_archived/
│   └── analysis_v1.0_archived/
└── organized_by_date/
    ├── 2025-09-09/
    ├── 2025-09-08/
    └── 2025-09-07/
```

---

## 🔄 **VERSIONING WORKFLOW**

### **1. Development Workflow**
```
1. Create new version: ml_train_classifier_v1.1_dev.py
2. Test and validate
3. Mark as stable: ml_train_classifier_v1.1_stable.py
4. Archive old version: ml_train_classifier_v1.0_archived.py
5. Update documentation
```

### **2. Release Workflow**
```
1. Alpha: ml_train_classifier_v1.1.0-alpha.py
2. Beta: ml_train_classifier_v1.1.0-beta.py
3. RC: ml_train_classifier_v1.1.0-rc1.py
4. Stable: ml_train_classifier_v1.1.0_stable.py
```

### **3. Maintenance Workflow**
```
1. Bug fix: ml_train_classifier_v1.1.1_stable.py
2. Feature add: ml_train_classifier_v1.2.0_stable.py
3. Breaking change: ml_train_classifier_v2.0.0_stable.py
```

---

## 📊 **BENEFITS OF NEW SYSTEM**

### **1. Clear Versioning**
- **Easy tracking** of file versions
- **Rollback capability** to previous versions
- **Change history** visible in filenames
- **Release management** with alpha/beta/stable

### **2. Consistent Naming**
- **Predictable structure** for all files
- **Easy searching** by category and purpose
- **Clear status** indication
- **Professional appearance**

### **3. Better Organization**
- **Logical grouping** by function
- **Easy maintenance** and updates
- **Team collaboration** with clear conventions
- **Scalable structure** for future growth

---

## 🚀 **IMPLEMENTATION STEPS**

### **Step 1: Create Versioning Script**
- Automated renaming script
- Version validation
- Archive old versions

### **Step 2: Rename Existing Files**
- Apply new naming convention
- Update all references
- Test functionality

### **Step 3: Update Documentation**
- Update all file references
- Create version history
- Update README files

### **Step 4: Establish Workflow**
- Team guidelines
- Automated checks
- CI/CD integration

---

## ⚠️ **IMPORTANT NOTES**

- **Backup before renaming** - Preserve all existing files
- **Update all references** - Fix imports and documentation
- **Test after renaming** - Ensure functionality preserved
- **Document changes** - Keep version history

**This systematic approach will create a professional, maintainable, and scalable file organization system!**
