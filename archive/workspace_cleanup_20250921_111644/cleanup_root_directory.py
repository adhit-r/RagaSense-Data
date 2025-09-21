#!/usr/bin/env python3
"""
Clean up root directory by moving files to appropriate locations
"""

import os
import shutil
from pathlib import Path

def cleanup_root_directory():
    """Clean up the root directory by organizing files"""
    
    print("🧹 Cleaning up root directory...")
    
    # Files to move to archive
    files_to_archive = [
        # Log files
        "comprehensive_saraga_coverage_analysis.log",
        "create_final_ml_ready_dataset.log", 
        "create_unified_raga_dataset_final.log",
        "extract_and_process_remaining_saraga_audio.log",
        "process_remaining_unmatched_saraga_files.log",
        
        # PNG files (analysis images)
        "coverage_analysis.png",
        "cross_tradition_mapping_overview.png", 
        "detailed_raga_mappings.png",
        "problematic_ragas_analysis.png",
        
        # Markdown reports
        "ai_semantic_matching_report.md",
        "CROSS_TRADITION_MAPPING_SUCCESS_REPORT.md",
        "cross_tradition_mapping_visual_report_20250914_163510.md",
        
        # Python scripts (move to scripts directory)
        "cleanup_data_directories.py",
        "cleanup_processed_metadata.py", 
        "cleanup_scripts_directory.py",
        "deploy.py",
        "load_existing_mappings.py",
        "create_vercel_data.py",
        
        # Flask apps (move to archive)
        "collaborative_raga_mapper.py",
        "simple_matcher_fixed.py",
        "simple_raga_matcher.py",
        
        # Templates (move to archive)
        "templates/",
        
        # Other files
        "todo.md",
        "pyproject.toml",
        "Procfile",
        "runtime.txt"
    ]
    
    # Create directories if they don't exist
    os.makedirs("archive/root_cleanup/junk_files", exist_ok=True)
    os.makedirs("archive/root_cleanup/logs", exist_ok=True)
    os.makedirs("archive/root_cleanup/images", exist_ok=True)
    os.makedirs("archive/root_cleanup/reports", exist_ok=True)
    os.makedirs("scripts/cleanup", exist_ok=True)
    os.makedirs("scripts/flask_apps", exist_ok=True)
    
    moved_count = 0
    
    for file_path in files_to_archive:
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.log'):
                    dest = f"archive/root_cleanup/logs/{file_path}"
                elif file_path.endswith('.png'):
                    dest = f"archive/root_cleanup/images/{file_path}"
                elif file_path.endswith('.md'):
                    dest = f"archive/root_cleanup/reports/{file_path}"
                elif file_path in ["cleanup_data_directories.py", "cleanup_processed_metadata.py", "cleanup_scripts_directory.py", "deploy.py", "load_existing_mappings.py", "create_vercel_data.py"]:
                    dest = f"scripts/cleanup/{file_path}"
                elif file_path in ["collaborative_raga_mapper.py", "simple_matcher_fixed.py", "simple_raga_matcher.py"]:
                    dest = f"scripts/flask_apps/{file_path}"
                elif file_path == "templates/":
                    dest = f"archive/root_cleanup/{file_path}"
                else:
                    dest = f"archive/root_cleanup/junk_files/{file_path}"
                
                if os.path.isdir(file_path):
                    shutil.move(file_path, dest)
                else:
                    shutil.move(file_path, dest)
                
                print(f"  ✅ Moved {file_path} → {dest}")
                moved_count += 1
                
            except Exception as e:
                print(f"  ❌ Failed to move {file_path}: {e}")
    
    # Clean up __pycache__ directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(pycache_path)
                    print(f"  🗑️  Removed {pycache_path}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ❌ Failed to remove {pycache_path}: {e}")
    
    print(f"\n✅ Cleanup complete! Moved/removed {moved_count} files/directories")
    print("📁 Files organized into:")
    print("  • archive/root_cleanup/logs/ - Log files")
    print("  • archive/root_cleanup/images/ - Analysis images") 
    print("  • archive/root_cleanup/reports/ - Markdown reports")
    print("  • scripts/cleanup/ - Cleanup scripts")
    print("  • scripts/flask_apps/ - Flask applications")
    print("  • archive/root_cleanup/junk_files/ - Other files")

if __name__ == "__main__":
    cleanup_root_directory()

