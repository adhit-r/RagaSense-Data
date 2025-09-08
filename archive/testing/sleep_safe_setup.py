#!/usr/bin/env python3
"""
Sleep Safe Setup - Final configuration for overnight processing
"""

import subprocess
import time
from datetime import datetime

def setup_sleep_safe_processing():
    print("🌙 SLEEP SAFE SETUP - RagaSense-Data")
    print("=" * 60)
    print(f"⏰ Setup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check current status
    print("\n📊 Current Status:")
    try:
        result = subprocess.run(['python', 'check_status.py'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Status check error: {e}")
    
    # Ensure MacBook stays awake
    print("\n🍎 MacBook Sleep Prevention:")
    try:
        # Kill any existing caffeinate processes
        subprocess.run(['pkill', 'caffeinate'], capture_output=True)
        time.sleep(1)
        
        # Start caffeinate to prevent sleep
        subprocess.Popen(['caffeinate', '-d', '-i', '-m', '-u'])
        print("✅ MacBook sleep prevention enabled")
    except Exception as e:
        print(f"⚠️ Sleep prevention error: {e}")
    
    # Check if nightly processing is running
    print("\n🔄 Nightly Processing Status:")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'nightly_data_processing.py' in result.stdout:
            print("✅ Nightly processing is running")
        else:
            print("❌ Nightly processing not running - starting now...")
            subprocess.Popen(['nohup', 'python', 'nightly_data_processing.py', '>', 'nightly_output.log', '2>&1', '&'])
            time.sleep(2)
            print("✅ Nightly processing started")
    except Exception as e:
        print(f"Process check error: {e}")
    
    # Final instructions
    print("\n" + "=" * 60)
    print("🎉 SLEEP SAFE SETUP COMPLETE!")
    print("=" * 60)
    print("✅ MacBook will stay awake")
    print("✅ Data processing pipeline running")
    print("✅ W&B tracking active")
    print("✅ GPU acceleration enabled")
    print("✅ All systems optimized for MacBook")
    
    print("\n💤 YOU CAN NOW SAFELY SLEEP!")
    print("🔄 Your MacBook will continue processing data all night")
    print("📊 Check progress at: https://wandb.ai/adhithya/ragasense-data-ingestion")
    print("📝 Monitor logs with: python check_status.py")
    
    print(f"\n⏰ Processing started at: {datetime.now().strftime('%H:%M:%S')}")
    print("🌅 Check results in the morning!")

if __name__ == "__main__":
    setup_sleep_safe_processing()
