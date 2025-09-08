#!/usr/bin/env python3
"""
Monitor All Night Processing Pipeline
"""

import time
import subprocess
import os
from datetime import datetime
from pathlib import Path

def check_all_night_status():
    print("🌙 ALL NIGHT PROCESSING MONITOR")
    print("=" * 50)
    print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if process is running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        process_found = False
        for line in lines:
            if 'all_night_processing.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    print(f"✅ ALL NIGHT PROCESS RUNNING (PID: {pid}, CPU: {cpu}%, MEM: {mem}%)")
                    process_found = True
                    break
        
        if not process_found:
            print("❌ All night process not running")
            return False
    
    except Exception as e:
        print(f"Error checking process: {e}")
        return False
    
    # Check log files
    log_files = ['all_night_processing.log', 'all_night_output.log']
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
    
    # Check cycle reports
    logs_dir = Path("logs")
    if logs_dir.exists():
        cycle_reports = list(logs_dir.glob("cycle_*_report_*.json"))
        print(f"📊 Cycle reports: {len(cycle_reports)}")
        
        if cycle_reports:
            # Show latest cycle info
            latest_report = max(cycle_reports, key=lambda x: x.stat().st_mtime)
            try:
                import json
                with open(latest_report, 'r') as f:
                    data = json.load(f)
                    print(f"   Latest cycle: {data.get('cycle', 'Unknown')}")
                    print(f"   Total runtime: {data.get('total_processing_time', 'Unknown')}")
            except:
                pass
    
    print("=" * 50)
    return True

def monitor_continuously():
    """Monitor continuously"""
    print("🔄 Starting continuous monitoring...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            is_running = check_all_night_status()
            
            if not is_running:
                print("❌ Process stopped - monitoring ended")
                break
            
            print("⏳ Waiting 60 seconds for next check...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        monitor_continuously()
    else:
        check_all_night_status()
