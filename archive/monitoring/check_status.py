#!/usr/bin/env python3
"""
Quick status check for nightly processing
"""

import subprocess
import os
from datetime import datetime
from pathlib import Path

def check_status():
    print("🌙 RagaSense-Data Nightly Processing Status")
    print("=" * 50)
    print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if process is running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        process_found = False
        for line in lines:
            if 'nightly_data_processing.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    print(f"✅ Process running (PID: {pid}, CPU: {cpu}%, MEM: {mem}%)")
                    process_found = True
                    break
        
        if not process_found:
            print("❌ Process not running")
    
    except Exception as e:
        print(f"Error checking process: {e}")
    
    # Check log files
    log_files = ['nightly_processing.log', 'nightly_output.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            print(f"📝 {log_file}: {size} bytes")
            
            if size > 0:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"   Last line: {lines[-1].strip()}")
                except:
                    pass
        else:
            print(f"📝 {log_file}: Not found")
    
    # Check W&B runs
    wandb_dir = Path("wandb")
    if wandb_dir.exists():
        runs = list(wandb_dir.glob("run-*"))
        print(f"📊 W&B runs: {len(runs)}")
    
    print("=" * 50)

if __name__ == "__main__":
    check_status()
