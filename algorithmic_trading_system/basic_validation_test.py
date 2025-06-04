#!/usr/bin/env python3
"""
BASIC VALIDATION TEST - Start with simple setup verification
Step 1: Verify basic components work before complex evolution
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime

# Paths
LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
LEAN_CLI = "/home/vandan/.local/bin/lean"

class BasicValidator:
    """Simple validator to test core functionality"""
    
    def __init__(self):
        self.test_results = {}
        
    def test_lean_cli(self):
        """Test 1: Verify LEAN CLI works"""
        print("🔧 TEST 1: LEAN CLI Functionality")
        print("-" * 40)
        
        try:
            result = subprocess.run([LEAN_CLI, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ LEAN CLI working: {version}")
                self.test_results["lean_cli"] = {"status": "PASS", "version": version}
                return True
            else:
                print(f"❌ LEAN CLI failed: {result.stderr}")
                self.test_results["lean_cli"] = {"status": "FAIL", "error": result.stderr}
                return False
                
        except Exception as e:
            print(f"❌ LEAN CLI error: {e}")
            self.test_results["lean_cli"] = {"status": "ERROR", "error": str(e)}
            return False
    
    def test_simple_strategy_creation(self):
        """Test 2: Create a simple strategy project"""
        print("\n🏗️ TEST 2: Simple Strategy Creation")
        print("-" * 40)
        
        project_name = f"validation_test_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        try:
            # Create project directory
            os.makedirs(project_path, exist_ok=True)
            
            # Create simple strategy
            simple_strategy = '''from AlgorithmImports import *

class ValidationStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 6, 30)
        self.SetCash(100000)
        
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Simple indicators
        self.sma_short = self.SMA("SPY", 10, Resolution.Daily)
        self.sma_long = self.SMA("SPY", 20, Resolution.Daily)
        
        # Weekly rebalancing
        self.Schedule.On(self.DateRules.WeekStart("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.Rebalance)
        
    def Rebalance(self):
        if not self.sma_short.IsReady or not self.sma_long.IsReady:
            return
            
        # Simple momentum with realistic leverage
        if self.sma_short.Current.Value > self.sma_long.Current.Value:
            self.SetHoldings(self.spy, 1.5)  # 1.5x leverage - realistic
        else:
            self.SetHoldings(self.spy, 0.5)  # Reduced position in downtrend
    
    def OnData(self, data):
        pass'''
            
            # Write strategy file
            with open(os.path.join(project_path, "main.py"), 'w') as f:
                f.write(simple_strategy)
            
            # Write config
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "local-id": abs(hash(project_name)) % 1000000
            }
            with open(os.path.join(project_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Simple strategy created: {project_name}")
            print(f"   Path: {project_path}")
            print(f"   Strategy: SPY momentum with 1.5x leverage")
            print(f"   Period: Jan-Jun 2022 (6 months)")
            
            self.test_results["strategy_creation"] = {
                "status": "PASS", 
                "project": project_name,
                "path": project_path
            }
            
            return project_name, project_path
            
        except Exception as e:
            print(f"❌ Strategy creation failed: {e}")
            self.test_results["strategy_creation"] = {"status": "FAIL", "error": str(e)}
            return None, None
    
    def test_local_backtest(self, project_name):
        """Test 3: Run local backtest"""
        print("\n📊 TEST 3: Local Backtest Execution")
        print("-" * 40)
        
        if not project_name:
            print("❌ Cannot run backtest - no project created")
            self.test_results["local_backtest"] = {"status": "SKIP", "reason": "No project"}
            return False
        
        try:
            print(f"🚀 Running backtest for: {project_name}")
            print("   This may take 30-60 seconds...")
            
            # Run backtest with timeout
            result = subprocess.run(
                [LEAN_CLI, "backtest", project_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Backtest completed successfully!")
                
                # Parse basic results
                output = result.stdout
                cagr = self._extract_metric(output, "Compounding Annual Return")
                sharpe = self._extract_metric(output, "Sharpe Ratio")
                drawdown = self._extract_metric(output, "Drawdown")
                
                print(f"📈 Results Summary:")
                print(f"   CAGR: {cagr}%")
                print(f"   Sharpe: {sharpe}")
                print(f"   Max Drawdown: {drawdown}%")
                
                self.test_results["local_backtest"] = {
                    "status": "PASS",
                    "cagr": cagr,
                    "sharpe": sharpe,
                    "drawdown": drawdown,
                    "output_length": len(output)
                }
                
                return True
                
            else:
                print(f"❌ Backtest failed:")
                print(f"   Error: {result.stderr[:200] if result.stderr else 'Unknown'}")
                print(f"   Return code: {result.returncode}")
                
                self.test_results["local_backtest"] = {
                    "status": "FAIL",
                    "error": result.stderr,
                    "return_code": result.returncode
                }
                
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Backtest timed out (>2 minutes)")
            self.test_results["local_backtest"] = {"status": "TIMEOUT", "timeout": 120}
            return False
        except Exception as e:
            print(f"❌ Backtest error: {e}")
            self.test_results["local_backtest"] = {"status": "ERROR", "error": str(e)}
            return False
    
    def _extract_metric(self, output, metric_name):
        """Extract metric from LEAN output"""
        try:
            for line in output.split('\n'):
                if metric_name in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        value = parts[-1].replace('%', '').replace('$', '').replace(',', '')
                        return float(value)
            return "N/A"
        except:
            return "N/A"
    
    def test_quantconnect_api(self):
        """Test 4: Basic QuantConnect API connectivity"""
        print("\n☁️ TEST 4: QuantConnect API Connectivity")
        print("-" * 40)
        
        try:
            sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')
            from working_qc_api import QuantConnectCloudAPI
            
            # Initialize API
            api = QuantConnectCloudAPI(
                "357130", 
                "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
            )
            
            # Test basic API call
            test_project_id = api.create_project("ValidationAPITest")
            
            if test_project_id:
                print(f"✅ QuantConnect API working!")
                print(f"   Created test project: {test_project_id}")
                
                self.test_results["qc_api"] = {
                    "status": "PASS",
                    "test_project": test_project_id
                }
                return True
            else:
                print("❌ QuantConnect API failed to create project")
                self.test_results["qc_api"] = {"status": "FAIL", "error": "Project creation failed"}
                return False
                
        except Exception as e:
            print(f"❌ QuantConnect API error: {e}")
            self.test_results["qc_api"] = {"status": "ERROR", "error": str(e)}
            return False
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print("🚀 STARTING BASIC VALIDATION SUITE")
        print("=" * 60)
        print(f"Timestamp: {datetime.now()}")
        print(f"LEAN Workspace: {LEAN_WORKSPACE}")
        print(f"LEAN CLI: {LEAN_CLI}")
        
        # Test sequence
        tests_passed = 0
        total_tests = 4
        
        # Test 1: LEAN CLI
        if self.test_lean_cli():
            tests_passed += 1
        
        # Test 2: Strategy Creation
        project_name, project_path = self.test_simple_strategy_creation()
        if project_name:
            tests_passed += 1
        
        # Test 3: Local Backtest
        if self.test_local_backtest(project_name):
            tests_passed += 1
        
        # Test 4: QuantConnect API
        if self.test_quantconnect_api():
            tests_passed += 1
        
        # Final Report
        print("\n" + "=" * 60)
        print("🏁 VALIDATION SUMMARY")
        print("=" * 60)
        
        success_rate = (tests_passed / total_tests) * 100
        print(f"Tests Passed: {tests_passed}/{total_tests} ({success_rate:.0f}%)")
        
        for test_name, result in self.test_results.items():
            status = result.get("status", "UNKNOWN")
            if status == "PASS":
                print(f"✅ {test_name}: {status}")
            elif status == "FAIL":
                print(f"❌ {test_name}: {status} - {result.get('error', 'Unknown error')}")
            elif status == "ERROR":
                print(f"💥 {test_name}: {status} - {result.get('error', 'Unknown error')}")
            else:
                print(f"⚠️ {test_name}: {status}")
        
        if tests_passed == total_tests:
            print("\n🎉 ALL TESTS PASSED - Ready for next phase!")
            print("✅ LEAN CLI works")
            print("✅ Strategy creation works") 
            print("✅ Local backtesting works")
            print("✅ QuantConnect API works")
            print("\n📋 NEXT STEPS:")
            print("1. Test simple strategy with 12% CAGR target")
            print("2. Run basic evolution with realistic targets")
            print("3. Gradually increase complexity")
        elif tests_passed >= 2:
            print("\n⚠️ PARTIAL SUCCESS - Some components need fixing")
            print("💡 Focus on fixing failed tests before proceeding")
        else:
            print("\n❌ MAJOR ISSUES - Core setup needs work")
            print("🔧 Fix basic setup before attempting evolution")
        
        # Save results
        timestamp = int(time.time())
        results_file = f"/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/validation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'tests_passed': tests_passed,
                'total_tests': total_tests,
                'success_rate': success_rate,
                'results': self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved: validation_results_{timestamp}.json")
        
        return tests_passed == total_tests

def main():
    validator = BasicValidator()
    success = validator.run_full_validation()
    
    if success:
        print("\n🎯 READY FOR PHASE 2: Simple Strategy Testing")
    else:
        print("\n🔧 FIX ISSUES BEFORE PROCEEDING")
    
    return success

if __name__ == "__main__":
    main()