#!/usr/bin/env python3
"""
RagaSense-Data: Nightly Processing Monitor
Monitor the progress of the long-running data processing pipeline
"""

import time
import os
import subprocess
from pathlib import Path
from datetime import datetime

def check_process_status():
    """Check if the nightly processing is still running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if 'nightly_data_processing.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    return True, pid
        return False, None
    except Exception as e:
        print(f"Error checking process: {e}")
        return False, None

def monitor_logs():
    """Monitor the log files for progress"""
    log_files = [
        'nightly_processing.log',
        'nightly_output.log'
    ]
    
    print("📊 Monitoring Nightly Processing Pipeline")
    print("=" * 50)
    
    while True:
        is_running, pid = check_process_status()
        
        if is_running:
            print(f"✅ Process running (PID: {pid}) - {datetime.now().strftime('%H:%M:%S')}")
        else:
            print(f"❌ Process not running - {datetime.now().strftime('%H:%M:%S')}")
            break
        
        # Check log files
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line:
                                print(f"📝 Latest: {last_line}")
                except Exception as e:
                    print(f"Error reading {log_file}: {e}")
        
        print("-" * 50)
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_logs()
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
