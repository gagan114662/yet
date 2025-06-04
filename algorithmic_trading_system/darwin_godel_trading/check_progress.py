#!/usr/bin/env python3
"""
Quick progress checker for breakthrough evolution
"""

import subprocess
import time
import sys
from datetime import datetime

def check_progress():
    """Check if breakthrough system is making progress"""
    print("🔍 Checking breakthrough evolution progress...")
    
    # Check if process is running
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        
        if "breakthrough_dgm.py" in result.stdout:
            print("✅ Breakthrough system is running")
            
            # Check recent QuantConnect projects
            try:
                qc_result = subprocess.run(
                    ["/home/vandan/.local/bin/lean", "cloud", "pull"],
                    cwd="/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace",
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if "Breakthrough_" in qc_result.stdout:
                    breakthrough_projects = [line for line in qc_result.stdout.split('\n') if 'Breakthrough_' in line]
                    print(f"📊 Found {len(breakthrough_projects)} breakthrough projects on QuantConnect")
                    
                    # Show latest few
                    for project in breakthrough_projects[-5:]:
                        print(f"   📈 {project.strip()}")
                else:
                    print("❌ No breakthrough projects found yet")
                    
            except Exception as e:
                print(f"⚠️ Could not check QuantConnect: {e}")
                
        else:
            print("❌ Breakthrough system not running")
            
    except Exception as e:
        print(f"❌ Error checking process: {e}")

if __name__ == "__main__":
    check_progress()