#!/usr/bin/env python3
"""
Deploy 15-Year Backtest Strategies to QuantConnect
All strategies configured for 2010-2024 (15 years) testing period
"""

import asyncio
import sys
import os
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI
import time

def get_15_year_strategies():
    """Get strategies optimized for 15-year backtesting (2010-2024)"""
    
    strategies = {
        "Momentum15Y": '''from AlgorithmImports import *

class Momentum15YStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Core ETFs available since 2010
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        
        # 15-year optimized indicators
        self.spy_momentum = self.MOMP("SPY", 60, Resolution.Daily)  # 3-month momentum
        self.spy_sma_long = self.SMA("SPY", 200, Resolution.Daily)  # Long-term trend
        
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.Rebalance)
        
        self.Debug("15-Year Momentum Strategy: 2010-2024")
    
    def Rebalance(self):
        if not self.spy_momentum.IsReady or not self.spy_sma_long.IsReady:
            return
        
        # 15-year momentum strategy
        strong_momentum = self.spy_momentum.Current.Value > 5
        long_term_trend = self.Securities[self.spy].Price > self.spy_sma_long.Current.Value
        
        if strong_momentum and long_term_trend:
            self.SetHoldings(self.spy, 0.7)
            self.SetHoldings(self.qqq, 0.3)  # Growth tilt
        elif strong_momentum:
            self.SetHoldings(self.spy, 0.8)
            self.SetHoldings(self.qqq, 0.2)
        elif long_term_trend:
            self.SetHoldings(self.spy, 0.6)
        else:
            self.SetHoldings(self.spy, 0.3)  # Defensive
    
    def OnData(self, data):
        pass''',

        "CrisisAlpha15Y": '''from AlgorithmImports import *

class CrisisAlpha15YStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS - Includes 2008 recovery, 2020 COVID
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Crisis-tested instruments
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol  # Long bonds
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol  # Gold
        
        # Crisis detection over 15 years
        self.spy_rsi = self.RSI("SPY", 14, Resolution.Daily)
        self.vix_proxy = self.RSI("SPY", 5, Resolution.Daily)  # Fast RSI as volatility proxy
        
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.ManageCrisis)
        
        self.Debug("15-Year Crisis Alpha: 2010-2024 (includes multiple crises)")
    
    def ManageCrisis(self):
        if not self.spy_rsi.IsReady or not self.vix_proxy.IsReady:
            return
        
        # Crisis detection across 15 years
        extreme_oversold = self.spy_rsi.Current.Value < 20
        high_volatility = self.vix_proxy.Current.Value < 15 or self.vix_proxy.Current.Value > 85
        
        if extreme_oversold and high_volatility:
            # Crisis mode - flight to quality
            self.SetHoldings(self.tlt, 0.4)  # Bonds
            self.SetHoldings(self.gld, 0.3)  # Gold
            self.SetHoldings(self.spy, 0.3)  # Some equity
        elif extreme_oversold:
            # Oversold bounce opportunity
            self.SetHoldings(self.spy, 1.2)  # Leverage up on oversold
        elif self.spy_rsi.Current.Value > 75:
            # Overbought - reduce risk
            self.SetHoldings(self.spy, 0.5)
            self.SetHoldings(self.tlt, 0.3)
            self.SetHoldings(self.gld, 0.2)
        else:
            # Normal allocation
            self.SetHoldings(self.spy, 0.8)
            self.SetHoldings(self.tlt, 0.1)
            self.SetHoldings(self.gld, 0.1)
    
    def OnData(self, data):
        pass''',

        "TrendFollowing15Y": '''from AlgorithmImports import *

class TrendFollowing15YStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS - Multiple market cycles
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Trend following universe
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Daily).Symbol  # Small caps
        
        # Multi-timeframe trend following for 15 years
        self.spy_sma_50 = self.SMA("SPY", 50, Resolution.Daily)
        self.spy_sma_200 = self.SMA("SPY", 200, Resolution.Daily)
        self.qqq_sma_50 = self.SMA("QQQ", 50, Resolution.Daily)
        
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.FollowTrends)
        
        self.Debug("15-Year Trend Following: 2010-2024")
    
    def FollowTrends(self):
        if not all([self.spy_sma_50.IsReady, self.spy_sma_200.IsReady, self.qqq_sma_50.IsReady]):
            return
        
        # 15-year trend analysis
        spy_uptrend = self.spy_sma_50.Current.Value > self.spy_sma_200.Current.Value
        spy_strong = self.Securities[self.spy].Price > self.spy_sma_50.Current.Value
        qqq_strong = self.Securities[self.qqq].Price > self.qqq_sma_50.Current.Value
        
        if spy_uptrend and spy_strong and qqq_strong:
            # All trends aligned - aggressive growth
            self.SetHoldings(self.spy, 0.5)
            self.SetHoldings(self.qqq, 0.4)
            self.SetHoldings(self.iwm, 0.1)  # Small cap boost
        elif spy_uptrend and spy_strong:
            # SPY trend strong
            self.SetHoldings(self.spy, 0.8)
            self.SetHoldings(self.qqq, 0.2)
        elif spy_uptrend:
            # Weak uptrend
            self.SetHoldings(self.spy, 0.6)
        else:
            # Downtrend or sideways - defensive
            self.SetHoldings(self.spy, 0.2)
    
    def OnData(self, data):
        pass''',

        "VolatilityHarvester15Y": '''from AlgorithmImports import *

class VolatilityHarvester15YStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS - Multiple volatility cycles
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Volatility harvesting instruments
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        
        # 15-year volatility indicators
        self.spy_bb = self.BB("SPY", 20, 2, Resolution.Daily)
        self.spy_atr = self.ATR("SPY", 14, Resolution.Daily)
        self.qqq_bb = self.BB("QQQ", 20, 2, Resolution.Daily)
        
        self.volatility_history = RollingWindow[float](252)  # 1 year
        
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.HarvestVolatility)
        
        self.Debug("15-Year Volatility Harvesting: 2010-2024")
    
    def OnData(self, data):
        if self.spy in data and data[self.spy] is not None:
            # Track daily volatility over 15 years
            if hasattr(self, 'prev_spy_price'):
                daily_return = abs((data[self.spy].Close - self.prev_spy_price) / self.prev_spy_price)
                self.volatility_history.Add(daily_return)
            self.prev_spy_price = data[self.spy].Close
    
    def HarvestVolatility(self):
        if not all([self.spy_bb.IsReady, self.spy_atr.IsReady, self.qqq_bb.IsReady]):
            return
        
        spy_price = self.Securities[self.spy].Price
        qqq_price = self.Securities[self.qqq].Price
        
        # Volatility analysis over 15 years
        spy_squeeze = (self.spy_bb.UpperBand.Current.Value - self.spy_bb.LowerBand.Current.Value) / self.spy_bb.MiddleBand.Current.Value < 0.1
        spy_breakout_up = spy_price > self.spy_bb.UpperBand.Current.Value
        spy_breakout_down = spy_price < self.spy_bb.LowerBand.Current.Value
        qqq_breakout_up = qqq_price > self.qqq_bb.UpperBand.Current.Value
        
        if spy_squeeze:
            # Low volatility - prepare for breakout
            self.SetHoldings(self.spy, 0.5)
            self.SetHoldings(self.qqq, 0.5)
        elif spy_breakout_up and qqq_breakout_up:
            # Upward volatility breakout
            self.SetHoldings(self.spy, 0.6)
            self.SetHoldings(self.qqq, 0.6)  # 120% exposure
        elif spy_breakout_up:
            # SPY breakout only
            self.SetHoldings(self.spy, 1.0)
        elif spy_breakout_down:
            # Downward breakout - defensive
            self.SetHoldings(self.spy, 0.2)
        else:
            # Normal volatility
            self.SetHoldings(self.spy, 0.7)
            self.SetHoldings(self.qqq, 0.3)
    
    def OnData(self, data):
        pass''',

        "DiversifiedRotation15Y": '''from AlgorithmImports import *

class DiversifiedRotation15YStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS - Full economic cycles
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Diversified asset universe (all available since 2010)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol  # Large cap US
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol  # Tech
        self.iwm = self.AddEquity("IWM", Resolution.Daily).Symbol  # Small cap
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol  # Long bonds
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol  # Gold
        
        # 15-year momentum indicators for each asset
        self.momentum_indicators = {}
        for symbol in [self.spy, self.qqq, self.iwm, self.tlt, self.gld]:
            self.momentum_indicators[symbol] = self.MOMP(symbol, 126, Resolution.Daily)  # 6-month momentum
        
        # Monthly rebalancing for 15-year strategy
        self.Schedule.On(self.DateRules.MonthStart("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.RotateAssets)
        
        self.Debug("15-Year Diversified Rotation: 2010-2024")
    
    def RotateAssets(self):
        if not all(indicator.IsReady for indicator in self.momentum_indicators.values()):
            return
        
        # Calculate momentum scores for all assets
        momentum_scores = {}
        for symbol, indicator in self.momentum_indicators.items():
            momentum_scores[symbol] = indicator.Current.Value
        
        # Sort by momentum (best first)
        sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 15-year diversified allocation strategy
        top_3 = sorted_assets[:3]
        
        # Dynamic allocation based on momentum strength
        total_weight = 0
        self.Liquidate()  # Clear all positions
        
        for i, (symbol, momentum) in enumerate(top_3):
            if momentum > 0:  # Only invest in positive momentum assets
                weight = 0.4 if i == 0 else 0.3 if i == 1 else 0.3  # 40-30-30 split
                self.SetHoldings(symbol, weight)
                total_weight += weight
        
        # If not fully invested, allocate to cash (defensive)
        if total_weight < 1.0:
            self.Debug(f"Defensive allocation: {(1-total_weight)*100:.1f}% cash")
    
    def OnData(self, data):
        pass'''
    }
    
    return strategies

async def deploy_15_year_strategies():
    """Deploy all 15-year strategies to QuantConnect"""
    
    # Initialize API
    api = QuantConnectCloudAPI("357130", "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912")
    
    print("🕐 DEPLOYING 15-YEAR BACKTEST STRATEGIES")
    print("=" * 70)
    print("📅 Period: January 1, 2010 → December 31, 2024 (15 YEARS)")
    print("📊 Includes: 2010-2012 recovery, 2008 crisis aftermath, COVID-19, etc.")
    print("⏱️  Rate Limit: 60 seconds between deployments")
    print("=" * 70)
    
    strategies = get_15_year_strategies()
    deployed_strategies = []
    
    for i, (strategy_name, strategy_code) in enumerate(strategies.items(), 1):
        print(f"\n🚀 [{i}/{len(strategies)}] Deploying: {strategy_name}")
        print("-" * 50)
        
        try:
            result = api.deploy_strategy(strategy_name, strategy_code)
            
            if result['success']:
                deployed_strategies.append({
                    'name': strategy_name,
                    'project_id': result['project_id'],
                    'backtest_id': result['backtest_id'],
                    'url': result['url']
                })
                print(f"✅ SUCCESS: {strategy_name}")
                print(f"   Project: {result['project_id']}")
                print(f"   Backtest: {result['backtest_id']}")
                print(f"   URL: {result['url']}")
            else:
                print(f"❌ FAILED: {result['error']}")
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
        
        # Rate limiting between deployments
        if i < len(strategies):
            print(f"\n⏳ Waiting 60 seconds before next deployment...")
            await asyncio.sleep(60)
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 15-YEAR DEPLOYMENT SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successfully deployed: {len(deployed_strategies)}/{len(strategies)} strategies")
    
    for strategy in deployed_strategies:
        print(f"\n🏆 {strategy['name']}")
        print(f"   📊 15-Year Backtest: {strategy['url']}")
        print(f"   🆔 Project ID: {strategy['project_id']}")
    
    print(f"\n🎯 All strategies now running 15-year backtests (2010-2024)")
    print(f"📈 Check results in QuantConnect terminal for complete metrics")
    
    return deployed_strategies

if __name__ == "__main__":
    asyncio.run(deploy_15_year_strategies())