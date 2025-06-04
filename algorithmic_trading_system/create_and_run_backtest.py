#!/usr/bin/env python3
"""
CREATE AND RUN BACKTEST - Use the working API to actually run a strategy and get results
"""

import sys
import time
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI
import json

def create_and_run_backtest():
    """Create a new backtest, run it, and get the results"""
    
    print("🚀 CREATING AND RUNNING BACKTEST")
    print("="*50)
    
    # Initialize API
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    # Define our minimal working strategy
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
    
    print("📊 Strategy: Minimal SPY Buy-and-Hold")
    print("📅 Period: Jan 2020 - Jun 2020 (6 months)")
    print("💰 Starting Capital: $100,000")
    
    # Deploy the complete strategy using the working API
    print(f"\n🔧 DEPLOYING STRATEGY...")
    
    try:
        result = api.deploy_strategy("MinimalTest", minimal_strategy)
        
        if result.get('success'):
            project_id = result['project_id']
            backtest_id = result['backtest_id']
            
            print(f"✅ DEPLOYMENT SUCCESSFUL!")
            print(f"📊 Project ID: {project_id}")
            print(f"🎯 Backtest ID: {backtest_id}")
            print(f"🔗 URL: https://www.quantconnect.com/terminal/{project_id}#open/{backtest_id}")
            
            # Wait for backtest to complete
            print(f"\n⏳ WAITING FOR BACKTEST TO COMPLETE...")
            max_wait = 120  # 2 minutes max
            wait_time = 0
            
            while wait_time < max_wait:
                print(f"   Checking results... ({wait_time}s elapsed)")
                
                try:
                    results = api.read_backtest_results(project_id, backtest_id)
                    
                    if results:
                        print(f"\n🎉 BACKTEST COMPLETED!")
                        print(f"📊 ACTUAL RESULTS:")
                        print(f"="*40)
                        
                        # Display the key metrics
                        key_metrics = ['cagr', 'sharpe', 'total_orders', 'win_rate', 'drawdown', 'net_profit']
                        
                        for metric in key_metrics:
                            if metric in results:
                                value = results[metric]
                                print(f"   {metric.upper()}: {value}")
                        
                        print(f"\n📋 ALL AVAILABLE METRICS:")
                        print(f"-"*40)
                        for key, value in results.items():
                            print(f"   {key}: {value}")
                        
                        # Save to file
                        timestamp = int(time.time())
                        filename = f"/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/ACTUAL_RESULTS_{timestamp}.json"
                        
                        with open(filename, 'w') as f:
                            json.dump({
                                'project_id': project_id,
                                'backtest_id': backtest_id,
                                'strategy': 'MinimalTest',
                                'results': results,
                                'timestamp': timestamp
                            }, f, indent=2)
                        
                        print(f"\n💾 Results saved to: ACTUAL_RESULTS_{timestamp}.json")
                        print(f"🎯 THIS IS REAL DATA - NO MORE THEORETICAL CLAIMS!")
                        
                        return results
                        
                except Exception as e:
                    print(f"   ❌ Error reading results: {e}")
                
                # Wait 10 seconds before checking again
                time.sleep(10)
                wait_time += 10
            
            print(f"\n⏰ TIMEOUT: Backtest took longer than {max_wait} seconds")
            print(f"🔗 Check manually: https://www.quantconnect.com/terminal/{project_id}#open/{backtest_id}")
            
        else:
            print(f"❌ DEPLOYMENT FAILED: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    create_and_run_backtest()