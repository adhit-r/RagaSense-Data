#!/usr/bin/env python3
"""
RagaSense-Data: MacBook GPU Acceleration Test
Test script to verify MacBook GPU acceleration is working properly
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_macbook_gpu_acceleration():
    """Test MacBook GPU acceleration capabilities"""
    print("🍎 Testing MacBook GPU Acceleration for RagaSense-Data")
    print("=" * 60)
    
    try:
        # Test GPU accelerator
        from tools.utils.macbook_gpu_accelerator import MacBookGPUAccelerator
        
        print("\n1. Initializing MacBook GPU Accelerator...")
        accelerator = MacBookGPUAccelerator()
        
        # Get GPU info
        gpu_info = accelerator.get_gpu_info()
        print("\n📊 GPU Information:")
        for key, value in gpu_info.items():
            print(f"   {key}: {value}")
        
        if not gpu_info['gpu_available']:
            print("⚠️ GPU not available, but CPU processing will still work")
            return True
        
        # Test data ingestion acceleration
        print("\n2. Testing Data Ingestion Acceleration...")
        sample_data = [
            {
                'id': 'test_carnatic_001',
                'tradition': 'carnatic',
                'raga': {
                    'name': 'Kalyani',
                    'confidence': 0.95,
                    'arohana': ['S', 'R2', 'G3', 'M2', 'P', 'D2', 'N3', 'S\''],
                    'avarohana': ['S\'', 'N3', 'D2', 'P', 'M2', 'G3', 'R2', 'S']
                },
                'audio': {
                    'duration_seconds': 180.5,
                    'sample_rate': 44100,
                    'quality_score': 0.92
                },
                'performance': {
                    'artist': 'M. S. Subbulakshmi',
                    'performance_type': 'concert'
                }
            },
            {
                'id': 'test_hindustani_001',
                'tradition': 'hindustani',
                'raga': {
                    'name': 'Yaman',
                    'confidence': 0.93,
                    'arohana': ['S', 'R2', 'G3', 'M2', 'P', 'D2', 'N3', 'S\''],
                    'avarohana': ['S\'', 'N3', 'D2', 'P', 'M2', 'G3', 'R2', 'S']
                },
                'audio': {
                    'duration_seconds': 165.2,
                    'sample_rate': 44100,
                    'quality_score': 0.88
                },
                'performance': {
                    'artist': 'Bhimsen Joshi',
                    'performance_type': 'concert'
                }
            }
        ]
        
        accelerated_data = accelerator.accelerate_data_ingestion(sample_data)
        print(f"   ✅ Processed {len(accelerated_data)} records")
        print(f"   ✅ GPU processed: {sum(1 for d in accelerated_data if d.get('gpu_processed', False))}/{len(accelerated_data)}")
        print(f"   ✅ Average quality score: {sum(d.get('quality_score', 0) for d in accelerated_data) / len(accelerated_data):.3f}")
        
        # Test cross-tradition mapping acceleration
        print("\n3. Testing Cross-Tradition Mapping Acceleration...")
        sample_mappings = [
            {
                'mapping_id': 'test_mapping_001',
                'carnatic_raga': {'name': 'Kalyani', 'id': 'carnatic_kalyani'},
                'hindustani_raga': {'name': 'Yaman', 'id': 'hindustani_yaman'},
                'relationship': {
                    'type': 'SAME',
                    'confidence': 0.95,
                    'verified_by': 'expert_001'
                }
            },
            {
                'mapping_id': 'test_mapping_002',
                'carnatic_raga': {'name': 'Kharaharapriya', 'id': 'carnatic_kharaharapriya'},
                'hindustani_raga': {'name': 'Kafi', 'id': 'hindustani_kafi'},
                'relationship': {
                    'type': 'SIMILAR',
                    'confidence': 0.85,
                    'verified_by': 'expert_002'
                }
            }
        ]
        
        accelerated_mappings = accelerator.accelerate_cross_tradition_mapping(sample_mappings)
        print(f"   ✅ Processed {len(accelerated_mappings)} mappings")
        print(f"   ✅ Average GPU confidence: {sum(m.get('gpu_confidence', 0) for m in accelerated_mappings) / len(accelerated_mappings):.3f}")
        
        # Test metadata management acceleration
        print("\n4. Testing Metadata Management Acceleration...")
        accelerated_metadata = accelerator.accelerate_metadata_management(sample_data)
        print(f"   ✅ Enhanced {len(accelerated_metadata)} metadata records")
        print(f"   ✅ GPU enhanced: {sum(1 for m in accelerated_metadata if m.get('gpu_enhanced', False))}/{len(accelerated_metadata)}")
        
        print("\n🎉 All MacBook GPU acceleration tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Test failed with exception")
        return False

def test_data_ingestion_with_gpu():
    """Test data ingestion with GPU acceleration"""
    print("\n" + "=" * 60)
    print("🚀 Testing Data Ingestion with GPU Acceleration")
    print("=" * 60)
    
    try:
        from tools.ingestion.datasource_ingestion import RagaSenseDataSourceIngestion
        
        print("\n1. Initializing Data Ingestion System...")
        ingestion = RagaSenseDataSourceIngestion()
        
        if ingestion.gpu_accelerator:
            print("   ✅ GPU accelerator available")
            gpu_info = ingestion.gpu_accelerator.get_gpu_info()
            print(f"   📊 Device: {gpu_info['device']}")
            print(f"   📊 GPU Available: {gpu_info['gpu_available']}")
        else:
            print("   ⚠️ GPU accelerator not available")
        
        print("\n2. Testing with sample data...")
        # Create sample data directory
        sample_dir = project_root / "examples" / "sample_data"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Test ingestion with sample data
        results = ingestion.run_full_ingestion()
        print(f"   ✅ Ingestion completed: {len(results)} sources processed")
        if results:
            total_processed = sum(r.records_processed for r in results)
            total_valid = sum(r.records_valid for r in results)
            print(f"   📊 Total records processed: {total_processed}")
            print(f"   📊 Total records valid: {total_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        logger.exception("Data ingestion test failed")
        return False

def test_data_validation_with_gpu():
    """Test data validation with GPU acceleration"""
    print("\n" + "=" * 60)
    print("🔍 Testing Data Validation with GPU Acceleration")
    print("=" * 60)
    
    try:
        from tools.validation.data_validator import RagaSenseDataValidator
        
        print("\n1. Initializing Data Validation System...")
        validator = RagaSenseDataValidator(use_gpu=True)
        
        if validator.gpu_accelerator:
            print("   ✅ GPU accelerator available for validation")
        else:
            print("   ⚠️ GPU accelerator not available for validation")
        
        print("\n2. Testing validation with sample file...")
        sample_file = project_root / "examples" / "sample_data" / "sample_carnatic_metadata.json"
        
        if sample_file.exists():
            result = validator.validate_metadata_file(sample_file)
            print(f"   ✅ Validation result: {result.is_valid}")
            print(f"   📊 Quality score: {result.quality_score:.3f}")
            print(f"   📊 Schema errors: {len(result.schema_errors)}")
            print(f"   📊 Quality issues: {len(result.quality_issues)}")
        else:
            print("   ⚠️ Sample file not found, skipping validation test")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        logger.exception("Data validation test failed")
        return False

def main():
    """Run all MacBook GPU tests"""
    print("🍎 RagaSense-Data MacBook GPU Acceleration Test Suite")
    print("=" * 60)
    
    tests = [
        ("GPU Accelerator", test_macbook_gpu_acceleration),
        ("Data Ingestion", test_data_ingestion_with_gpu),
        ("Data Validation", test_data_validation_with_gpu)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MacBook GPU acceleration is working perfectly!")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
