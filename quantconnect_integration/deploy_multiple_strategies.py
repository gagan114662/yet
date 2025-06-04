#!/usr/bin/env python3
"""
Deploy Multiple High-Performance Strategies to QuantConnect Cloud
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from working_qc_api import QuantConnectCloudAPI

def get_strategy_codes():
    """Get collection of high-performance strategy codes"""
    
    strategies = {
        "MomentumTrend": '''from AlgorithmImports import *

class MomentumTrendStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Momentum instruments
        self.tqqq = self.AddEquity("TQQQ", Resolution.Hour).Symbol
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
        
        # Indicators
        self.qqq_momentum = self.MOMP("QQQ", 20, Resolution.Daily)
        self.spy_rsi = self.RSI("SPY", 14, Resolution.Daily)
        self.qqq_sma = self.SMA("QQQ", 50, Resolution.Daily)
        
        self.Schedule.On(self.DateRules.EveryDay("QQQ"), 
                        self.TimeRules.AfterMarketOpen("QQQ", 30), 
                        self.Rebalance)
    
    def Rebalance(self):
        if not self.qqq_momentum.IsReady or not self.spy_rsi.IsReady:
            return
        
        # Strong momentum conditions
        momentum_strong = self.qqq_momentum.Current.Value > 5
        market_healthy = self.spy_rsi.Current.Value > 30 and self.spy_rsi.Current.Value < 70
        uptrend = self.Securities[self.qqq].Price > self.qqq_sma.Current.Value
        
        if momentum_strong and market_healthy and uptrend:
            self.SetHoldings(self.tqqq, 1.5)  # 150% leveraged tech
        elif momentum_strong:
            self.SetHoldings(self.qqq, 1.0)   # 100% tech
        else:
            self.SetHoldings(self.spy, 0.6)   # 60% defensive
    
    def OnData(self, data):
        pass''',

        "VolatilityBreakout": '''from AlgorithmImports import *

class VolatilityBreakoutStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Volatility instruments
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.vxx = self.AddEquity("VXX", Resolution.Hour).Symbol
        self.uvxy = self.AddEquity("UVXY", Resolution.Hour).Symbol
        
        # Indicators
        self.spy_bb = self.BB("SPY", 20, 2, Resolution.Daily)
        self.spy_atr = self.ATR("SPY", 14, Resolution.Daily)
        self.vxx_sma = self.SMA("VXX", 10, Resolution.Daily)
        
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.CheckBreakout)
    
    def CheckBreakout(self):
        if not self.spy_bb.IsReady or not self.spy_atr.IsReady:
            return
        
        spy_price = self.Securities[self.spy].Price
        
        # Volatility breakout conditions
        upper_breakout = spy_price > self.spy_bb.UpperBand.Current.Value
        lower_breakout = spy_price < self.spy_bb.LowerBand.Current.Value
        high_volatility = self.spy_atr.Current.Value > self.spy_atr.Window[5].Value
        vxx_spike = self.vxx_sma.IsReady and self.Securities[self.vxx].Price > self.vxx_sma.Current.Value * 1.1
        
        if upper_breakout and not high_volatility:
            # Upward breakout with low vol = momentum
            self.SetHoldings(self.spy, 1.2)
            
        elif lower_breakout and vxx_spike:
            # Downward breakout with vol spike = volatility play
            self.SetHoldings(self.uvxy, 0.8)
            self.SetHoldings(self.spy, -0.4)  # Short equity
            
        else:
            # Neutral
            self.SetHoldings(self.spy, 0.5)
    
    def OnData(self, data):
        pass''',

        "MeanReversionPro": '''from AlgorithmImports import *

class MeanReversionProStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Mean reversion universe
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Hour).Symbol
        
        # Indicators
        self.spy_rsi = self.RSI("SPY", 7, Resolution.Daily)
        self.qqq_rsi = self.RSI("QQQ", 7, Resolution.Daily)
        self.iwm_rsi = self.RSI("IWM", 7, Resolution.Daily)
        
        self.spy_bb = self.BB("SPY", 20, 2, Resolution.Daily)
        
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.FindMeanReversion)
    
    def FindMeanReversion(self):
        if not all([self.spy_rsi.IsReady, self.qqq_rsi.IsReady, self.iwm_rsi.IsReady]):
            return
        
        # Oversold conditions
        spy_oversold = self.spy_rsi.Current.Value < 25
        qqq_oversold = self.qqq_rsi.Current.Value < 25
        iwm_oversold = self.iwm_rsi.Current.Value < 25
        
        # Bollinger band support
        spy_at_lower = self.Securities[self.spy].Price < self.spy_bb.LowerBand.Current.Value * 1.02
        
        total_allocation = 0
        
        if spy_oversold and spy_at_lower:
            self.SetHoldings(self.spy, 0.8)
            total_allocation += 0.8
            
        if qqq_oversold and total_allocation < 1.0:
            weight = min(0.6, 1.0 - total_allocation)
            self.SetHoldings(self.qqq, weight)
            total_allocation += weight
            
        if iwm_oversold and total_allocation < 1.0:
            weight = min(0.4, 1.0 - total_allocation)
            self.SetHoldings(self.iwm, weight)
    
    def OnData(self, data):
        pass''',

        "DiversifiedMomentum": '''from AlgorithmImports import *

class DiversifiedMomentumStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Diversified universe
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol  # US Large Cap
        self.eem = self.AddEquity("EEM", Resolution.Daily).Symbol  # Emerging Markets
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol  # Gold
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol  # Long Bonds
        
        # Momentum indicators
        self.indicators = {}
        for symbol in [self.spy, self.eem, self.gld, self.tlt]:
            self.indicators[symbol] = self.MOMP(symbol, 60, Resolution.Daily)
        
        self.Schedule.On(self.DateRules.MonthStart("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.RebalancePortfolio)
    
    def RebalancePortfolio(self):
        if not all(indicator.IsReady for indicator in self.indicators.values()):
            return
        
        # Calculate momentum scores
        momentum_scores = {}
        for symbol, indicator in self.indicators.items():
            momentum_scores[symbol] = indicator.Current.Value
        
        # Sort by momentum (highest first)
        sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Equal weight top 3 performers with leverage
        top_performers = sorted_assets[:3]
        weight_per_asset = 1.0 / len(top_performers)
        
        # Clear all positions first
        self.Liquidate()
        
        # Allocate to top performers
        for symbol, momentum in top_performers:
            if momentum > 0:  # Only positive momentum
                self.SetHoldings(symbol, weight_per_asset * 1.2)  # 20% leverage
    
    def OnData(self, data):
        pass'''
    }
    
    return strategies

def main():
    """Deploy multiple strategies to QuantConnect Cloud"""
    # Credentials
    USER_ID = "357130"
    API_TOKEN = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    # Initialize API
    api = QuantConnectCloudAPI(USER_ID, API_TOKEN)
    
    print("🚀 QUANTCONNECT MULTIPLE STRATEGY DEPLOYMENT")
    print("=" * 70)
    
    strategies = get_strategy_codes()
    successful_deployments = []
    failed_deployments = []
    
    for strategy_name, strategy_code in strategies.items():
        print(f"\n{'='*60}")
        print(f"🎯 Deploying: {strategy_name}")
        print(f"{'='*60}")
        
        try:
            result = api.deploy_strategy(strategy_name, strategy_code)
            
            if result['success']:
                successful_deployments.append({
                    'name': strategy_name,
                    'project_id': result['project_id'],
                    'backtest_id': result['backtest_id'],
                    'url': result['url']
                })
                print(f"✅ {strategy_name} deployed successfully!")
            else:
                failed_deployments.append({
                    'name': strategy_name,
                    'error': result['error']
                })
                print(f"❌ {strategy_name} deployment failed: {result['error']}")
                
        except Exception as e:
            failed_deployments.append({
                'name': strategy_name,
                'error': str(e)
            })
            print(f"❌ {strategy_name} deployment error: {e}")
        
        # Wait between deployments to avoid rate limits
        if strategy_name != list(strategies.keys())[-1]:  # Not the last one
            print("⏳ Waiting 30 seconds before next deployment...")
            time.sleep(30)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 DEPLOYMENT SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n✅ SUCCESSFUL DEPLOYMENTS ({len(successful_deployments)}):")
    for deployment in successful_deployments:
        print(f"  • {deployment['name']}")
        print(f"    Project ID: {deployment['project_id']}")
        print(f"    Backtest ID: {deployment['backtest_id']}")
        print(f"    URL: {deployment['url']}")
        print()
    
    if failed_deployments:
        print(f"\n❌ FAILED DEPLOYMENTS ({len(failed_deployments)}):")
        for deployment in failed_deployments:
            print(f"  • {deployment['name']}: {deployment['error']}")
    
    print(f"\n🎉 TOTAL: {len(successful_deployments)}/{len(strategies)} strategies deployed successfully!")

if __name__ == "__main__":
    main()