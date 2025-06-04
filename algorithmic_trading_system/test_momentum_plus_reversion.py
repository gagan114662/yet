#!/usr/bin/env python3
"""
TEST MOMENTUM + MEAN REVERSION STRATEGY - 50% momentum + 30% mean reversion
"""

import sys
import time
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI
import json

def test_momentum_plus_reversion():
    """Test momentum + mean reversion strategy"""
    
    print("🚀 TESTING MOMENTUM + MEAN REVERSION STRATEGY")
    print("="*55)
    
    # Initialize API
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    # Define multi-component strategy
    multi_strategy = '''from AlgorithmImports import *

class MomentumReversionStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add primary SPY
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Add mean reversion asset (VTI for diversification)
        self.vti = self.AddEquity("VTI", Resolution.Daily).Symbol
        
        # Momentum indicators (for SPY)
        self.sma_short = self.SMA("SPY", 10, Resolution.Daily)
        self.sma_long = self.SMA("SPY", 30, Resolution.Daily)
        
        # Mean reversion indicators (for VTI)
        self.vti_sma = self.SMA("VTI", 20, Resolution.Daily)
        self.vti_bb = self.BB("VTI", 20, 2, Resolution.Daily)
        
        # Component allocations (user specified)
        self.MOMENTUM_ALLOCATION = 0.5   # 50%
        self.REVERSION_ALLOCATION = 0.3  # 30%
        self.CASH_BUFFER = 0.2          # 20% cash buffer
        
        # Leverage constraint (user requirement)
        self.MAX_LEVERAGE = 1.2
        
        # Schedule daily rebalancing
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 15), 
                        self.Rebalance)
    
    def Rebalance(self):
        if not self.sma_short.IsReady or not self.vti_bb.IsReady:
            return
        
        # MOMENTUM COMPONENT (50% allocation)
        momentum_signal = 0
        if self.sma_short.Current.Value > self.sma_long.Current.Value:
            momentum_signal = self.MOMENTUM_ALLOCATION * self.MAX_LEVERAGE
        else:
            momentum_signal = self.MOMENTUM_ALLOCATION * 0.5  # Reduced exposure in downtrend
        
        # MEAN REVERSION COMPONENT (30% allocation)
        reversion_signal = 0
        vti_price = self.Securities[self.vti].Price
        if vti_price < self.vti_bb.LowerBand.Current.Value:
            # Oversold - buy
            reversion_signal = self.REVERSION_ALLOCATION * 1.0
        elif vti_price > self.vti_bb.UpperBand.Current.Value:
            # Overbought - reduce position
            reversion_signal = self.REVERSION_ALLOCATION * 0.2
        else:
            # Neutral
            reversion_signal = self.REVERSION_ALLOCATION * 0.6
        
        # Apply allocations
        self.SetHoldings(self.spy, momentum_signal)
        self.SetHoldings(self.vti, reversion_signal)
        
        # Debug logging
        total_allocation = momentum_signal + reversion_signal
        self.Debug(f"Momentum: {momentum_signal:.2f}, Reversion: {reversion_signal:.2f}, Total: {total_allocation:.2f}")
    
    def OnData(self, data):
        pass'''
    
    print(f"📊 Strategy: 50% Momentum + 30% Mean Reversion")
    print(f"📅 Period: Jan 2021 - Dec 2023 (3 years)")
    print(f"💰 Starting Capital: $100,000")
    print(f"⚖️ Max Leverage: 1.2x per component")
    print(f"📈 Momentum: SPY with SMA crossover")
    print(f"📉 Reversion: VTI with Bollinger Bands")
    print(f"🎯 Target: Improve on 6.31% CAGR baseline")
    
    # Deploy the strategy
    print(f"\n🔧 DEPLOYING MULTI-COMPONENT STRATEGY...")
    
    try:
        result = api.deploy_strategy("MomentumReversion", multi_strategy)
        
        if result.get('success'):
            project_id = result['project_id']
            backtest_id = result['backtest_id']
            
            print(f"✅ DEPLOYMENT SUCCESSFUL!")
            print(f"📊 Project ID: {project_id}")
            print(f"🎯 Backtest ID: {backtest_id}")
            print(f"🔗 URL: https://www.quantconnect.com/terminal/{project_id}#open/{backtest_id}")
            
            # Wait for backtest to complete
            print(f"\n⏳ WAITING FOR BACKTEST TO COMPLETE...")
            max_wait = 300  # 5 minutes
            wait_time = 0
            
            while wait_time < max_wait:
                print(f"   Checking results... ({wait_time}s elapsed)")
                
                try:
                    results = api.read_backtest_results(project_id, backtest_id)
                    
                    if results:
                        print(f"\n🎉 BACKTEST COMPLETED!")
                        print(f"📊 ACTUAL RESULTS:")
                        print(f"="*40)
                        
                        # Display key metrics
                        key_metrics = ['cagr', 'sharpe', 'total_orders', 'win_rate', 'drawdown', 'net_profit']
                        
                        for metric in key_metrics:
                            if metric in results:
                                value = results[metric]
                                print(f"   {metric.upper()}: {value}")
                        
                        # Save to file
                        timestamp = int(time.time())
                        filename = f"/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/MOMENTUM_REVERSION_RESULTS_{timestamp}.json"
                        
                        with open(filename, 'w') as f:
                            json.dump({
                                'project_id': project_id,
                                'backtest_id': backtest_id,
                                'strategy': 'MomentumReversion',
                                'description': '50% momentum + 30% mean reversion, 1.2x max leverage',
                                'components': {
                                    'momentum': '50% - SPY SMA crossover',
                                    'reversion': '30% - VTI Bollinger Bands',
                                    'cash_buffer': '20%'
                                },
                                'results': results,
                                'timestamp': timestamp
                            }, f, indent=2)
                        
                        print(f"\n💾 Results saved to: MOMENTUM_REVERSION_RESULTS_{timestamp}.json")
                        
                        # Analyze results vs previous
                        cagr = results.get('cagr', 0)
                        baseline_cagr = 6.31  # From previous momentum-only test
                        
                        if cagr > baseline_cagr:
                            improvement = cagr - baseline_cagr
                            print(f"\n✅ IMPROVEMENT: {cagr:.2f}% vs {baseline_cagr:.2f}% (+{improvement:.2f}%)")
                            if cagr > 10:
                                print(f"🎯 STRONG PERFORMANCE - Ready to add factor component for 12% target")
                            else:
                                print(f"🔧 MODERATE IMPROVEMENT - Factor component may push us to 12% target")
                        else:
                            decline = baseline_cagr - cagr
                            print(f"\n⚠️ DECLINE: {cagr:.2f}% vs {baseline_cagr:.2f}% (-{decline:.2f}%)")
                            print(f"💡 Mean reversion may be hurting performance in trending market")
                        
                        print(f"🏁 STEP 2 COMPLETE - Ready for factor component test")
                        
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
    test_momentum_plus_reversion()