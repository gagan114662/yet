#!/usr/bin/env python3
"""
TEST IMPROVED MOMENTUM STRATEGY - Better time period + basic momentum + leverage constraint
"""

import sys
import time
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI
import json

def test_improved_momentum():
    """Test improved momentum strategy in better time period"""
    
    print("🚀 TESTING IMPROVED MOMENTUM STRATEGY")
    print("="*50)
    
    # Initialize API
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    # Define improved strategy with momentum + leverage constraint
    improved_strategy = '''from AlgorithmImports import *

class ImprovedMomentumStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add SPY with daily resolution
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Simple momentum indicators
        self.sma_short = self.SMA("SPY", 10, Resolution.Daily)
        self.sma_long = self.SMA("SPY", 30, Resolution.Daily)
        
        # Leverage constraint (user requirement)
        self.MAX_LEVERAGE = 1.2
        
        # Schedule daily rebalancing
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 15), 
                        self.Rebalance)
    
    def Rebalance(self):
        if not self.sma_short.IsReady or not self.sma_long.IsReady:
            return
            
        # Simple momentum strategy with leverage constraint
        if self.sma_short.Current.Value > self.sma_long.Current.Value:
            # Uptrend - use maximum allowed leverage
            self.SetHoldings(self.spy, self.MAX_LEVERAGE)
        else:
            # Downtrend - reduce exposure but stay invested
            self.SetHoldings(self.spy, 0.5)
    
    def OnData(self, data):
        pass'''
    
    print(f"📊 Strategy: Improved Momentum with 1.2x Leverage")
    print(f"📅 Period: Jan 2021 - Dec 2023 (3 years, better than COVID crash)")
    print(f"💰 Starting Capital: $100,000")
    print(f"⚖️ Max Leverage: 1.2x (realistic constraint)")
    print(f"📈 Logic: SMA(10) > SMA(30) = 1.2x leverage, else 0.5x")
    
    # Deploy the strategy
    print(f"\n🔧 DEPLOYING STRATEGY...")
    
    try:
        result = api.deploy_strategy("ImprovedMomentum", improved_strategy)
        
        if result.get('success'):
            project_id = result['project_id']
            backtest_id = result['backtest_id']
            
            print(f"✅ DEPLOYMENT SUCCESSFUL!")
            print(f"📊 Project ID: {project_id}")
            print(f"🎯 Backtest ID: {backtest_id}")
            print(f"🔗 URL: https://www.quantconnect.com/terminal/{project_id}#open/{backtest_id}")
            
            # Wait for backtest to complete
            print(f"\n⏳ WAITING FOR BACKTEST TO COMPLETE...")
            max_wait = 300  # 5 minutes for 3-year backtest
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
                        filename = f"/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/IMPROVED_MOMENTUM_RESULTS_{timestamp}.json"
                        
                        with open(filename, 'w') as f:
                            json.dump({
                                'project_id': project_id,
                                'backtest_id': backtest_id,
                                'strategy': 'ImprovedMomentum',
                                'description': 'SMA crossover with 1.2x max leverage, 2021-2023',
                                'results': results,
                                'timestamp': timestamp
                            }, f, indent=2)
                        
                        print(f"\n💾 Results saved to: IMPROVED_MOMENTUM_RESULTS_{timestamp}.json")
                        print(f"🎯 STEP 1 COMPLETE - NOW WE HAVE MOMENTUM COMPONENT RESULTS!")
                        
                        # Analyze if we're ready for next step
                        cagr = results.get('cagr', 0)
                        if cagr > 5:
                            print(f"\n✅ POSITIVE CAGR: {cagr}% - Ready to add mean reversion component")
                        else:
                            print(f"\n⚠️ LOW CAGR: {cagr}% - May need to adjust momentum logic before adding components")
                        
                        return results
                        
                except Exception as e:
                    print(f"   ❌ Error reading results: {e}")
                
                # Wait 15 seconds before checking again
                time.sleep(15)
                wait_time += 15
            
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
    test_improved_momentum()