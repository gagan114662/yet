#!/usr/bin/env python3
"""
DEBUG THE INITIALIZATION ERROR - Find the root cause
"""

import sys
import os
import json
import glob

def debug_initialization_error():
    """Find what's causing the 'return without exception set' error"""
    
    print("🔍 DEBUGGING INITIALIZATION ERROR")
    print("="*50)
    
    # 1. Check all main.py files for common issues
    lean_workspace = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
    
    print("📁 Scanning main.py files for initialization issues...")
    
    main_files = glob.glob(f"{lean_workspace}/**/main.py", recursive=True)
    
    common_issues = {
        "self.universe = []": 0,
        "self.momentum_allocation": 0,
        "self.reversion_allocation": 0,
        "self.factor_allocation": 0,
        "def Initialize(": 0,
        "def initialize(": 0,
        "from AlgorithmImports import *": 0,
        "QCAlgorithm": 0
    }
    
    problematic_files = []
    
    for main_file in main_files[:10]:  # Check first 10
        try:
            with open(main_file, 'r') as f:
                content = f.read()
                
                # Check for common patterns
                for pattern in common_issues:
                    if pattern in content:
                        common_issues[pattern] += 1
                
                # Check for the specific line that's causing issues
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "self.universe = []" in line and i + 1 == 29:  # Line 29 issue
                        problematic_files.append({
                            'file': main_file,
                            'line': i + 1,
                            'content': line.strip()
                        })
                        
        except Exception as e:
            print(f"❌ Error reading {main_file}: {e}")
    
    print(f"\n📊 PATTERN ANALYSIS:")
    for pattern, count in common_issues.items():
        print(f"   {pattern}: found in {count} files")
    
    if problematic_files:
        print(f"\n⚠️  FOUND {len(problematic_files)} FILES WITH LINE 29 ISSUES:")
        for pf in problematic_files:
            print(f"   📄 {pf['file']}")
            print(f"      Line {pf['line']}: {pf['content']}")
    
    # 2. Create the simplest possible working strategy
    print(f"\n🔧 CREATING MINIMAL WORKING STRATEGY...")
    
    minimal_strategy = '''from AlgorithmImports import *

class MinimalWorkingStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2020, 6, 1)
        self.SetCash(100000)
        self.AddEquity("SPY", Resolution.Daily)
    
    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1.0)'''
    
    # Save to multiple locations to test
    test_dirs = [
        "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/MINIMAL_TEST_1",
        "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/MINIMAL_TEST_2"
    ]
    
    for test_dir in test_dirs:
        try:
            os.makedirs(test_dir, exist_ok=True)
            
            # Write main.py
            with open(f"{test_dir}/main.py", 'w') as f:
                f.write(minimal_strategy)
            
            # Write config.json
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "description": "Minimal test strategy to debug initialization"
            }
            
            with open(f"{test_dir}/config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Created test strategy: {test_dir}")
            
        except Exception as e:
            print(f"❌ Error creating {test_dir}: {e}")
    
    # 3. Identify the exact issue
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    print(f"The 'return without exception set' error at line 29 suggests:")
    print(f"   1. QuantConnect is still reading an old cached version")
    print(f"   2. There's a Python C extension issue")
    print(f"   3. Memory allocation problem during initialization")
    print(f"   4. Import/dependency conflict")
    
    print(f"\n🔧 SOLUTIONS TO TRY:")
    print(f"   1. Test minimal strategy in: {test_dirs[0]}")
    print(f"   2. Use completely different class name")
    print(f"   3. Avoid complex object initialization")
    print(f"   4. Start with absolute minimum code")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Test the minimal strategy above")
    print(f"   2. If it works, gradually add complexity")
    print(f"   3. If it fails, the issue is environment-level")
    print(f"   4. NO MORE PERFORMANCE CLAIMS until we get ONE working backtest")

if __name__ == "__main__":
    debug_initialization_error()