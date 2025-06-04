#!/usr/bin/env python3
"""
AUTO MONITOR - Continuous Darwin evolution without intervention
Runs until targets are met, handles failures automatically
"""

import subprocess
import time
import re
import os
from datetime import datetime

def auto_evolve_until_targets():
    """Auto-evolve with no intervention until targets met"""
    print("🚀 AUTO DARWIN EVOLUTION - NO INTERVENTION MODE")
    print("🎯 Will run continuously until ALL 4 targets are met")
    print("")
    
    iteration = 0
    best_cagr_ever = 0.0
    
    while True:
        iteration += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 🔄 ITERATION {iteration}: Starting evolution...")
        
        try:
            # Run evolution with timeout
            result = subprocess.run(
                ["python3", "trustworthy_dgm.py"],
                cwd="/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/darwin_godel_trading",
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour max per iteration
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                # Check for victory
                if "ALL TARGETS ACHIEVED" in output:
                    print("\n" + "🏆" * 80)
                    print("🎉🎉🎉 TARGETS ACHIEVED! 🎉🎉🎉")
                    print("🏆" * 80)
                    
                    # Extract final results
                    url_match = re.search(r'VERIFY RESULTS: (https://[^\s]+)', output)
                    if url_match:
                        print(f"🔗 FINAL VERIFICATION: {url_match.group(1)}")
                    
                    return True
                
                # Track progress
                cagr_matches = re.findall(r'CAGR: ([\d.]+)%', output)
                if cagr_matches:
                    current_best = max([float(x) for x in cagr_matches])
                    if current_best > best_cagr_ever:
                        best_cagr_ever = current_best
                        print(f"[{timestamp}] 📈 NEW RECORD: {current_best:.1f}% CAGR achieved!")
                
                print(f"[{timestamp}] ✅ Iteration {iteration} complete. Best ever: {best_cagr_ever:.1f}% CAGR")
            
            else:
                print(f"[{timestamp}] ⚠️ Evolution failed, restarting...")
                
        except subprocess.TimeoutExpired:
            print(f"[{timestamp}] ⏰ Timeout reached, restarting evolution...")
        except Exception as e:
            print(f"[{timestamp}] ❌ Error: {e}, restarting...")
        
        # Brief pause before restart
        time.sleep(30)

if __name__ == "__main__":
    try:
        success = auto_evolve_until_targets()
        if success:
            print("🎯 ALL TARGETS SUCCESSFULLY MET!")
        else:
            print("⚠️ Evolution stopped")
    except KeyboardInterrupt:
        print("\n🛑 Auto evolution stopped by user")