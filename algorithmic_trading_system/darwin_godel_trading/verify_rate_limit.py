#!/usr/bin/env python3
"""
Simple verification script - YOU can run this yourself
"""
import subprocess
import time

print("Testing QuantConnect rate limit...")

# Try to run 2 backtests quickly
for i in range(2):
    print(f"\nAttempt {i+1}:")
    start = time.time()
    
    result = subprocess.run(
        ["/home/vandan/.local/bin/lean", "cloud", "backtest", "simple_test_strategy", "--name", f"verify_{i}"],
        cwd="/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace",
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start
    
    if "Too many backtest requests" in result.stderr:
        print(f"❌ RATE LIMITED after {elapsed:.1f}s")
    elif result.returncode == 0:
        print(f"✅ SUCCESS after {elapsed:.1f}s")
    else:
        print(f"⚠️ Other error after {elapsed:.1f}s")
    
    if i == 0:
        print("Waiting 3 seconds before next attempt...")
        time.sleep(3)