#!/usr/bin/env python3
"""
PROGRESS MONITOR
Continuously monitors Darwin evolution and reports when targets are met
"""

import subprocess
import time
import re
import os
from datetime import datetime

TARGET_CRITERIA = {
    "cagr": 0.25,           # >25%
    "sharpe": 1.0,          # >1.0  
    "max_drawdown": 0.20,   # <20%
    "avg_profit": 0.0075    # >0.75%
}

def monitor_evolution():
    """Monitor the trustworthy DGM evolution"""
    print("🔍 MONITORING DARWIN EVOLUTION FOR TARGET ACHIEVEMENT")
    print("🎯 Will alert when ALL 4 criteria are met:")
    print(f"   📊 CAGR: >{TARGET_CRITERIA['cagr']*100:.0f}%")
    print(f"   📈 Sharpe: >{TARGET_CRITERIA['sharpe']:.1f}")
    print(f"   📉 Max DD: <{TARGET_CRITERIA['max_drawdown']*100:.0f}%")
    print(f"   💰 Avg Profit: >{TARGET_CRITERIA['avg_profit']*100:.2f}%")
    print("")
    
    generation = 0
    best_cagr = 0.0
    best_score = 0
    
    while True:
        try:
            # Run one iteration of evolution
            result = subprocess.run(
                ["python3", "trustworthy_dgm.py"],
                cwd="/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/darwin_godel_trading",
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes max
            )
            
            if result.returncode == 0:
                # Parse results for progress
                output = result.stdout
                
                # Look for victory celebration
                if "ALL TARGETS ACHIEVED" in output:
                    print("\n" + "🏆" * 80)
                    print("🎉🎉🎉 TARGET ACHIEVEMENT CONFIRMED! 🎉🎉🎉")
                    print("🏆" * 80)
                    
                    # Extract final results
                    cagr_match = re.search(r'FINAL CAGR: ([\d.]+)%', output)
                    sharpe_match = re.search(r'FINAL SHARPE: ([\d.]+)', output)
                    dd_match = re.search(r'FINAL MAX DD: ([\d.]+)%', output)
                    url_match = re.search(r'VERIFY RESULTS: (https://[^\s]+)', output)
                    
                    if all([cagr_match, sharpe_match, dd_match, url_match]):
                        print(f"✅ FINAL CAGR: {cagr_match.group(1)}%")
                        print(f"✅ FINAL SHARPE: {sharpe_match.group(1)}")
                        print(f"✅ FINAL MAX DD: {dd_match.group(1)}%")
                        print(f"🔗 VERIFICATION: {url_match.group(1)}")
                        print("\n🎯 ALL TARGETS SUCCESSFULLY MET!")
                        return True
                
                # Track progress
                cagr_matches = re.findall(r'CAGR: ([\d.]+)%', output)
                score_matches = re.findall(r'SCORE: ([\d.]+)/4', output)
                gen_matches = re.findall(r'GENERATION (\d+)', output)
                
                if cagr_matches and score_matches:
                    current_cagr = max([float(x) for x in cagr_matches])
                    current_score = max([float(x) for x in score_matches])
                    
                    if gen_matches:
                        generation = max([int(x) for x in gen_matches])
                    
                    # Report significant progress
                    if current_cagr > best_cagr + 1.0 or current_score > best_score:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] 📈 PROGRESS: Gen {generation} | CAGR: {current_cagr:.1f}% | Score: {current_score:.0f}/4")
                        
                        if current_cagr >= 10.0:
                            print(f"             🚀 Significant progress: {current_cagr:.1f}% CAGR!")
                        if current_cagr >= 20.0:
                            print(f"             ⚡ Approaching target: {current_cagr:.1f}% CAGR!")
                        
                        best_cagr = current_cagr
                        best_score = current_score
            
            else:
                # Handle errors
                print(f"⚠️ Evolution iteration failed: {result.stderr[:100]}")
            
            # Brief pause before next check
            time.sleep(30)
            
        except subprocess.TimeoutExpired:
            print("⚠️ Evolution timeout, restarting...")
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            return False
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    success = monitor_evolution()
    if success:
        print("✅ TARGET MONITORING COMPLETE - ALL CRITERIA MET!")
    else:
        print("⚠️ Monitoring stopped")