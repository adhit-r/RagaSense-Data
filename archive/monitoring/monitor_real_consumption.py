#!/usr/bin/env python3
"""
Monitor Real Data Consumption
"""

import time
import subprocess
import os
from datetime import datetime
from pathlib import Path

def check_real_consumption_status():
    print("🌙 REAL DATA CONSUMPTION MONITOR")
    print("=" * 50)
    print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if process is running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        process_found = False
        for line in lines:
            if 'consume_all_data_overnight.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    print(f"✅ REAL DATA CONSUMPTION RUNNING (PID: {pid}, CPU: {cpu}%, MEM: {mem}%)")
                    process_found = True
                    break
        
        if not process_found:
            print("❌ Real data consumption not running")
            return False
    
    except Exception as e:
        print(f"Error checking process: {e}")
        return False
    
    # Check log files
    log_files = ['consume_all_data.log', 'consume_all_data_output.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            print(f"📝 {log_file}: {size} bytes")
            
            if size > 0:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Show last few lines
                            for line in lines[-3:]:
                                print(f"   {line.strip()}")
                except:
                    pass
        else:
            print(f"📝 {log_file}: Not found")
    
    # Check downloads directory
    downloads_dir = Path("downloads")
    if downloads_dir.exists():
        download_items = list(downloads_dir.iterdir())
        print(f"📁 Downloads: {len(download_items)} items")
        
        # Show recent downloads
        recent_items = sorted(download_items, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
        for item in recent_items:
            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            print(f"   📂 {item.name}: {size // (1024*1024)}MB")
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        data_items = list(data_dir.iterdir())
        print(f"📊 Processed data: {len(data_items)} directories")
    
    # Check W&B runs
    wandb_dir = Path("wandb")
    if wandb_dir.exists():
        runs = list(wandb_dir.glob("run-*"))
        print(f"📊 W&B runs: {len(runs)}")
    
    print("=" * 50)
    return True

if __name__ == "__main__":
    check_real_consumption_status()
