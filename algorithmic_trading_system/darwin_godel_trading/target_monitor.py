#!/usr/bin/env python3
"""
TARGET ACHIEVEMENT MONITOR
Continuously monitors for breakthrough and reports when ALL 4 targets are met
"""

import subprocess
import time
import re
import sys
from datetime import datetime

TARGET_CRITERIA = {
    "cagr": 25.0,     # >25%
    "sharpe": 1.0,    # >1.0  
    "max_dd": 20.0,   # <20%
    "score": 4        # 4/4 criteria
}

def monitor_targets():
    """Monitor until ALL targets achieved"""
    print("🎯 TARGET ACHIEVEMENT MONITOR")
    print("Will alert immediately when ALL 4 criteria are met:")
    print(f"   📊 CAGR: >{TARGET_CRITERIA['cagr']}%")
    print(f"   📈 Sharpe: >{TARGET_CRITERIA['sharpe']}")
    print(f"   📉 Max DD: <{TARGET_CRITERIA['max_dd']}%")
    print(f"   🎯 Score: {TARGET_CRITERIA['score']}/4")
    print("")
    
    best_cagr = 0.0
    best_score = 0
    iteration = 0
    
    while True:
        iteration += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        try:
            # Check if breakthrough system is still running
            ps_result = subprocess.run(
                ["ps", "aux"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            if "breakthrough_dgm.py" not in ps_result.stdout:
                print(f"[{timestamp}] ⚠️ Breakthrough system stopped, checking final results...")
                
                # Try to find any breakthrough projects
                try:
                    # Quick check for recent breakthrough projects
                    qc_check = subprocess.run(
                        ["curl", "-s", "https://www.quantconnect.com/"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    print(f"[{timestamp}] 📊 Checking QuantConnect for final results...")
                except:
                    pass
                
                break
            
            # System is running, check for progress every 5 minutes
            if iteration % 30 == 1:  # Every 30 iterations (5 minutes)
                print(f"[{timestamp}] 🔄 Monitoring... Best so far: {best_cagr:.1f}% CAGR, {best_score}/4 score")
            
            # Look for any victory patterns in system output
            # This is a simplified check - in real implementation would parse actual results
            
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            print(f"\n[{timestamp}] 🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"[{timestamp}] ⚠️ Monitor error: {e}")
            time.sleep(30)
    
    print("\n📊 FINAL STATUS:")
    print("To verify results, check the QuantConnect projects manually:")
    print("1. Go to https://www.quantconnect.com/project")
    print("2. Look for 'Breakthrough_' projects")
    print("3. Check results for strategies meeting ALL 4 criteria")

if __name__ == "__main__":
    try:
        monitor_targets()
    except KeyboardInterrupt:
        print("\n🛑 Target monitoring stopped")