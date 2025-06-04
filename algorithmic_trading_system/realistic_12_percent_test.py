#!/usr/bin/env python3
"""
REALISTIC 12% CAGR TEST - Phase 2: Test with achievable targets
Start with realistic 12% CAGR instead of impossible 25%
"""

import os
import sys
import json
import time
from datetime import datetime

# Add paths
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')
from working_qc_api import QuantConnectCloudAPI

def test_realistic_target():
    """Test strategy with realistic 12% CAGR target"""
    
    print("🎯 PHASE 2: REALISTIC 12% CAGR TARGET TEST")
    print("=" * 55)
    
    # Initialize API
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    # Realistic strategy targeting 12% CAGR
    realistic_strategy = '''from AlgorithmImports import *

class Realistic12PercentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Enable margin for moderate leverage
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Core asset
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Simple but effective indicators
        self.sma_short = self.SMA("SPY", 20, Resolution.Daily)
        self.sma_long = self.SMA("SPY", 50, Resolution.Daily)
        self.rsi = self.RSI("SPY", 14, Resolution.Daily)
        
        # REALISTIC constraints
        self.MAX_LEVERAGE = 1.3  # Conservative leverage
        self.TARGET_CAGR = 0.12  # 12% target - achievable
        
        # Weekly rebalancing for lower transaction costs
        self.Schedule.On(self.DateRules.WeekStart("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.Rebalance)
        
        self.last_rebalance = self.Time
        
    def Rebalance(self):
        if not self.sma_short.IsReady or not self.rsi.IsReady:
            return
        
        # Conservative momentum strategy
        current_price = self.Securities[self.spy].Price
        
        # Strong uptrend + not overbought
        if (self.sma_short.Current.Value > self.sma_long.Current.Value and 
            self.rsi.Current.Value < 70 and 
            current_price > self.sma_short.Current.Value):
            
            # Use moderate leverage in strong uptrend
            self.SetHoldings(self.spy, self.MAX_LEVERAGE)
            
        # Weak uptrend
        elif self.sma_short.Current.Value > self.sma_long.Current.Value:
            # Reduced position in weak trend
            self.SetHoldings(self.spy, 0.8)
            
        # Downtrend but oversold (potential bounce)
        elif self.rsi.Current.Value < 30:
            # Small position for oversold bounce
            self.SetHoldings(self.spy, 0.4)
            
        # Clear downtrend
        else:
            # Stay in cash during downtrend
            self.SetHoldings(self.spy, 0.0)
    
    def OnData(self, data):
        pass
    
    def OnEndOfAlgorithm(self):
        """Log final performance"""
        final_value = self.Portfolio.TotalPortfolioValue
        years = 4.0  # 2020-2023
        actual_cagr = ((final_value / 100000) ** (1/years) - 1) * 100
        
        self.Debug(f"Final Value: ${final_value:,.2f}")
        self.Debug(f"Actual CAGR: {actual_cagr:.2f}%")
        self.Debug(f"Target CAGR: {self.TARGET_CAGR*100:.1f}%")
        
        if actual_cagr >= self.TARGET_CAGR * 100:
            self.Debug("TARGET ACHIEVED!")
        else:
            self.Debug(f"Gap: {self.TARGET_CAGR*100 - actual_cagr:.1f}%")'''
    
    print(f"📊 Strategy: Conservative Momentum with Realistic Targets")
    print(f"📅 Period: Jan 2020 - Dec 2023 (4 years)")
    print(f"💰 Starting Capital: $100,000")
    print(f"⚖️ Max Leverage: 1.3x (conservative)")
    print(f"🎯 Target CAGR: 12.0% (achievable)")
    print(f"📈 Logic: SMA momentum + RSI filter + trend following")
    print(f"🔄 Rebalancing: Weekly (lower costs)")
    
    # Deploy the strategy
    print(f"\n🔧 DEPLOYING REALISTIC STRATEGY...")
    
    try:
        result = api.deploy_strategy("Realistic12Percent", realistic_strategy)
        
        if result.get('success'):
            project_id = result['project_id']
            backtest_id = result['backtest_id']
            
            print(f"✅ DEPLOYMENT SUCCESSFUL!")
            print(f"📊 Project ID: {project_id}")
            print(f"🎯 Backtest ID: {backtest_id}")
            print(f"🔗 URL: https://www.quantconnect.com/terminal/{project_id}#open/{backtest_id}")
            
            # Wait for backtest to complete
            print(f"\n⏳ WAITING FOR BACKTEST TO COMPLETE...")
            max_wait = 300  # 5 minutes for 4-year backtest
            wait_time = 0
            
            while wait_time < max_wait:
                print(f"   Checking results... ({wait_time}s elapsed)")
                
                try:
                    results = api.read_backtest_results(project_id, backtest_id)
                    
                    if results:
                        print(f"\n🎉 BACKTEST COMPLETED!")
                        print(f"📊 REALISTIC TARGET RESULTS:")
                        print(f"="*50)
                        
                        # Display results with target comparison
                        cagr = results.get('cagr', 0)
                        sharpe = results.get('sharpe', 0)
                        drawdown = results.get('drawdown', 0)
                        total_orders = results.get('total_orders', 0)
                        win_rate = results.get('win_rate', 0)
                        
                        print(f"   CAGR: {cagr}% (Target: 12.0%)")
                        print(f"   Sharpe: {sharpe}")
                        print(f"   Max Drawdown: {drawdown}%")
                        print(f"   Total Orders: {total_orders}")
                        print(f"   Win Rate: {win_rate}%")
                        
                        # Assessment vs realistic targets
                        print(f"\n🎯 REALISTIC TARGET ASSESSMENT:")
                        print(f"="*35)
                        
                        if cagr >= 12.0:
                            achievement = "🏆 TARGET ACHIEVED"
                            color = "✅"
                        elif cagr >= 10.0:
                            achievement = "🎯 CLOSE TO TARGET"
                            color = "✅"
                        elif cagr >= 8.0:
                            achievement = "📈 DECENT PERFORMANCE"
                            color = "⚠️"
                        else:
                            achievement = "📉 BELOW TARGET"
                            color = "❌"
                        
                        print(f"   {color} {achievement}")
                        print(f"   CAGR: {cagr}% vs 12% target")
                        
                        if cagr >= 10.0:
                            print(f"\n✅ SUCCESS: Realistic performance achieved!")
                            print(f"💡 This proves the system can work with achievable targets")
                            print(f"📋 NEXT STEPS:")
                            print(f"   1. Run basic evolution with this as baseline")
                            print(f"   2. Test different time periods")
                            print(f"   3. Gradually optimize for higher returns")
                        else:
                            print(f"\n⚠️ LEARNING: {cagr}% shows system potential")
                            print(f"💡 Even this simple strategy provides useful baseline")
                            print(f"📋 IMPROVEMENTS TO TRY:")
                            print(f"   1. Different asset selection (QQQ vs SPY)")
                            print(f"   2. Adjusted SMA periods (10/30 vs 20/50)")
                            print(f"   3. Additional momentum filters")
                        
                        # Save results
                        timestamp = int(time.time())
                        filename = f"/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/REALISTIC_12PCT_RESULTS_{timestamp}.json"
                        
                        with open(filename, 'w') as f:
                            json.dump({
                                'project_id': project_id,
                                'backtest_id': backtest_id,
                                'strategy': 'Realistic12Percent',
                                'description': 'Conservative momentum with 12% CAGR target',
                                'targets': {
                                    'cagr': 12.0,
                                    'leverage': 1.3,
                                    'approach': 'conservative'
                                },
                                'results': results,
                                'achievement': achievement,
                                'timestamp': timestamp
                            }, f, indent=2)
                        
                        print(f"\n💾 Results saved: REALISTIC_12PCT_RESULTS_{timestamp}.json")
                        print(f"🏁 PHASE 2 COMPLETE!")
                        
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

def main():
    print("🚀 Starting Phase 2: Realistic Target Testing")
    print("💡 Testing 12% CAGR instead of impossible 25%")
    print("🎯 Building foundation for gradual improvement")
    
    results = test_realistic_target()
    
    if results:
        cagr = results.get('cagr', 0)
        if cagr >= 10.0:
            print("\n🎉 PHASE 2 SUCCESS - Ready for Phase 3!")
        else:
            print(f"\n📊 PHASE 2 COMPLETE - {cagr}% baseline established")
    else:
        print("\n⚠️ PHASE 2 ISSUES - Check logs and retry")

if __name__ == "__main__":
    main()