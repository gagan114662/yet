#!/usr/bin/env python3
"""
FINAL 12% CAGR TARGET SYSTEM - Complete multi-component strategy with realistic leverage
50% Momentum + 30% Mean Reversion + 20% Factor Strategy
"""

import sys
import time
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI
import json

def deploy_final_system():
    """Deploy the complete 12% CAGR target system"""
    
    print("🚀 DEPLOYING FINAL 12% CAGR TARGET SYSTEM")
    print("="*55)
    
    # Initialize API
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    # Define complete multi-component strategy
    final_strategy = '''from AlgorithmImports import *

class Final12PercentTargetSystem(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Enable margin for leverage
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Assets for each component
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol   # Momentum
        self.vti = self.AddEquity("VTI", Resolution.Daily).Symbol   # Mean Reversion  
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol   # Factor (Tech)
        self.iwm = self.AddEquity("IWM", Resolution.Daily).Symbol   # Factor (Small Cap)
        
        # MOMENTUM INDICATORS (SPY)
        self.sma_short = self.SMA("SPY", 10, Resolution.Daily)
        self.sma_long = self.SMA("SPY", 30, Resolution.Daily)
        self.spy_rsi = self.RSI("SPY", 14, Resolution.Daily)
        
        # MEAN REVERSION INDICATORS (VTI)
        self.vti_bb = self.BB("VTI", 20, 2, Resolution.Daily)
        self.vti_rsi = self.RSI("VTI", 14, Resolution.Daily)
        
        # FACTOR INDICATORS (QQQ/IWM ratio and momentum)
        self.qqq_sma = self.SMA("QQQ", 20, Resolution.Daily)
        self.iwm_sma = self.SMA("IWM", 20, Resolution.Daily)
        
        # USER-SPECIFIED ALLOCATIONS
        self.MOMENTUM_ALLOCATION = 0.5   # 50%
        self.REVERSION_ALLOCATION = 0.3  # 30%
        self.FACTOR_ALLOCATION = 0.2     # 20%
        
        # USER-SPECIFIED LEVERAGE CONSTRAINT
        self.MAX_LEVERAGE = 1.2
        
        # TARGET METRICS (user requirements)
        self.TARGET_CAGR = 12.0
        self.TARGET_SHARPE = 1.0
        self.MAX_DRAWDOWN = 20.0
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.Rebalance)
        
        self.Debug("Final 12% CAGR Target System initialized")
    
    def Rebalance(self):
        if not self.sma_short.IsReady or not self.vti_bb.IsReady or not self.qqq_sma.IsReady:
            return
        
        # COMPONENT 1: MOMENTUM (50% allocation)
        momentum_weight = self.CalculateMomentumWeight()
        momentum_allocation = self.MOMENTUM_ALLOCATION * momentum_weight * self.MAX_LEVERAGE
        
        # COMPONENT 2: MEAN REVERSION (30% allocation) 
        reversion_weight = self.CalculateReversionWeight()
        reversion_allocation = self.REVERSION_ALLOCATION * reversion_weight
        
        # COMPONENT 3: FACTOR (20% allocation)
        factor_allocation_qqq, factor_allocation_iwm = self.CalculateFactorAllocation()
        
        # Apply allocations with leverage constraints
        self.SetHoldings(self.spy, momentum_allocation)
        self.SetHoldings(self.vti, reversion_allocation)
        self.SetHoldings(self.qqq, factor_allocation_qqq)
        self.SetHoldings(self.iwm, factor_allocation_iwm)
        
        # Debug total allocation
        total_allocation = abs(momentum_allocation) + abs(reversion_allocation) + abs(factor_allocation_qqq) + abs(factor_allocation_iwm)
        self.Debug(f"Total allocation: {total_allocation:.2f} (Max: {self.MAX_LEVERAGE})")
    
    def CalculateMomentumWeight(self):
        """Calculate momentum component weight"""
        # Strong uptrend with good momentum
        if (self.sma_short.Current.Value > self.sma_long.Current.Value and 
            self.spy_rsi.Current.Value > 50 and self.spy_rsi.Current.Value < 70):
            return 1.0  # Full momentum allocation
        
        # Weak uptrend
        elif self.sma_short.Current.Value > self.sma_long.Current.Value:
            return 0.7  # Reduced momentum allocation
        
        # Downtrend but oversold (potential bounce)
        elif self.spy_rsi.Current.Value < 30:
            return 0.4  # Small position for bounce
        
        # Downtrend
        else:
            return 0.2  # Minimal allocation
    
    def CalculateReversionWeight(self):
        """Calculate mean reversion component weight"""
        vti_price = self.Securities[self.vti].Price
        
        # Oversold - maximum reversion bet
        if (vti_price < self.vti_bb.LowerBand.Current.Value and 
            self.vti_rsi.Current.Value < 30):
            return 1.0
        
        # Moderately oversold
        elif vti_price < self.vti_bb.LowerBand.Current.Value:
            return 0.7
        
        # Overbought - reduce or short
        elif (vti_price > self.vti_bb.UpperBand.Current.Value and 
              self.vti_rsi.Current.Value > 70):
            return -0.3  # Small short position
        
        # Neutral
        else:
            return 0.5
    
    def CalculateFactorAllocation(self):
        """Calculate factor component allocation (QQQ vs IWM)"""
        qqq_price = self.Securities[self.qqq].Price
        iwm_price = self.Securities[self.iwm].Price
        
        base_allocation = self.FACTOR_ALLOCATION * 0.5  # Split between QQQ and IWM
        
        # Tech outperformance (QQQ > SMA, IWM < SMA)
        if (qqq_price > self.qqq_sma.Current.Value and 
            iwm_price < self.iwm_sma.Current.Value):
            return (self.FACTOR_ALLOCATION * 0.8, self.FACTOR_ALLOCATION * 0.2)
        
        # Small cap outperformance (IWM > SMA, QQQ < SMA)
        elif (iwm_price > self.iwm_sma.Current.Value and 
              qqq_price < self.qqq_sma.Current.Value):
            return (self.FACTOR_ALLOCATION * 0.2, self.FACTOR_ALLOCATION * 0.8)
        
        # Both trending up
        elif (qqq_price > self.qqq_sma.Current.Value and 
              iwm_price > self.iwm_sma.Current.Value):
            return (base_allocation, base_allocation)
        
        # Both trending down - reduce factor exposure
        else:
            return (self.FACTOR_ALLOCATION * 0.3, self.FACTOR_ALLOCATION * 0.3)
    
    def OnData(self, data):
        pass
    
    def OnEndOfAlgorithm(self):
        """Log final performance vs targets"""
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - 100000) / 100000
        years = 3.0  # 2021-2023
        actual_cagr = ((final_value / 100000) ** (1/years) - 1) * 100
        
        self.Debug(f"FINAL RESULTS vs TARGETS:")
        self.Debug(f"Actual CAGR: {actual_cagr:.2f}% (Target: {self.TARGET_CAGR}%)")
        self.Debug(f"Final Value: ${final_value:,.2f}")'''
    
    print(f"📊 Strategy: Complete Multi-Component System")
    print(f"📅 Period: Jan 2021 - Dec 2023 (3 years)")
    print(f"💰 Starting Capital: $100,000")
    print(f"⚖️ Max Leverage: 1.2x (realistic constraint)")
    print(f"🎯 Target CAGR: 12.0%")
    print(f"📈 Components:")
    print(f"   • 50% Momentum (SPY with SMA + RSI)")
    print(f"   • 30% Mean Reversion (VTI with Bollinger Bands)")  
    print(f"   • 20% Factor (QQQ/IWM rotation)")
    
    # Deploy the strategy
    print(f"\n🔧 DEPLOYING FINAL SYSTEM...")
    
    try:
        result = api.deploy_strategy("Final12PercentTarget", final_strategy)
        
        if result.get('success'):
            project_id = result['project_id']
            backtest_id = result['backtest_id']
            
            print(f"✅ DEPLOYMENT SUCCESSFUL!")
            print(f"📊 Project ID: {project_id}")
            print(f"🎯 Backtest ID: {backtest_id}")
            print(f"🔗 URL: https://www.quantconnect.com/terminal/{project_id}#open/{backtest_id}")
            
            # Wait for backtest to complete
            print(f"\n⏳ WAITING FOR BACKTEST TO COMPLETE...")
            max_wait = 360  # 6 minutes for complex 3-year backtest
            wait_time = 0
            
            while wait_time < max_wait:
                print(f"   Checking results... ({wait_time}s elapsed)")
                
                try:
                    results = api.read_backtest_results(project_id, backtest_id)
                    
                    if results:
                        print(f"\n🎉 BACKTEST COMPLETED!")
                        print(f"📊 FINAL SYSTEM RESULTS:")
                        print(f"="*50)
                        
                        # Display key metrics with targets
                        cagr = results.get('cagr', 0)
                        sharpe = results.get('sharpe', 0)
                        drawdown = results.get('drawdown', 0)
                        total_orders = results.get('total_orders', 0)
                        win_rate = results.get('win_rate', 0)
                        
                        print(f"   CAGR: {cagr}% (Target: 12.0%)")
                        print(f"   Sharpe: {sharpe} (Target: 1.0)")
                        print(f"   Max Drawdown: {drawdown}% (Target: ≤20%)")
                        print(f"   Total Orders: {total_orders}")
                        print(f"   Win Rate: {win_rate}%")
                        
                        # Performance assessment
                        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
                        print(f"="*30)
                        
                        targets_met = 0
                        total_targets = 3
                        
                        if cagr >= 12.0:
                            print(f"   ✅ CAGR Target MET: {cagr}% ≥ 12%")
                            targets_met += 1
                        else:
                            print(f"   ❌ CAGR Target MISSED: {cagr}% < 12%")
                        
                        if sharpe >= 1.0:
                            print(f"   ✅ Sharpe Target MET: {sharpe} ≥ 1.0")
                            targets_met += 1
                        else:
                            print(f"   ❌ Sharpe Target MISSED: {sharpe} < 1.0")
                        
                        if drawdown <= 20.0:
                            print(f"   ✅ Drawdown Target MET: {drawdown}% ≤ 20%")
                            targets_met += 1
                        else:
                            print(f"   ❌ Drawdown Target MISSED: {drawdown}% > 20%")
                        
                        success_rate = (targets_met / total_targets) * 100
                        print(f"\n📊 OVERALL SUCCESS: {targets_met}/{total_targets} targets met ({success_rate:.0f}%)")
                        
                        if targets_met == total_targets:
                            print(f"🏆 COMPLETE SUCCESS - All targets achieved with realistic leverage!")
                        elif targets_met >= 2:
                            print(f"✅ PARTIAL SUCCESS - Most targets met, system is viable")
                        else:
                            print(f"⚠️ NEEDS IMPROVEMENT - Majority of targets missed")
                        
                        # Save comprehensive results
                        timestamp = int(time.time())
                        filename = f"/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/FINAL_12_PERCENT_RESULTS_{timestamp}.json"
                        
                        with open(filename, 'w') as f:
                            json.dump({
                                'project_id': project_id,
                                'backtest_id': backtest_id,
                                'strategy': 'Final12PercentTarget',
                                'description': 'Complete multi-component system with 1.2x max leverage',
                                'targets': {
                                    'cagr': 12.0,
                                    'sharpe': 1.0,
                                    'max_drawdown': 20.0
                                },
                                'components': {
                                    'momentum': '50% - SPY with SMA crossover + RSI',
                                    'reversion': '30% - VTI with Bollinger Bands + RSI',
                                    'factor': '20% - QQQ/IWM rotation based on momentum'
                                },
                                'results': results,
                                'targets_met': targets_met,
                                'success_rate': success_rate,
                                'timestamp': timestamp
                            }, f, indent=2)
                        
                        print(f"\n💾 Results saved to: FINAL_12_PERCENT_RESULTS_{timestamp}.json")
                        print(f"🏁 FINAL SYSTEM TESTING COMPLETE!")
                        
                        return results
                        
                except Exception as e:
                    print(f"   ❌ Error reading results: {e}")
                
                # Wait 20 seconds before checking again
                time.sleep(20)
                wait_time += 20
            
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
    deploy_final_system()