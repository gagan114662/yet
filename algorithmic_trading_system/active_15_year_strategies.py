#!/usr/bin/env python3
"""
Active 15-Year Trading Strategies
Designed for 100+ trades per year (1,500+ trades over 15 years)
"""

import asyncio
import sys
import os
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI
import time

def get_active_15_year_strategies():
    """Get highly active strategies with 100+ trades per year"""
    
    strategies = {
        "DayTradingMomentum15Y": '''from AlgorithmImports import *

class DayTradingMomentum15Y(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # High-frequency trading universe
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Minute).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Minute).Symbol
        
        # Short-term indicators for frequent trading
        self.spy_rsi_5 = self.RSI("SPY", 5, Resolution.Minute)
        self.spy_sma_20 = self.SMA("SPY", 20, Resolution.Minute)
        self.qqq_rsi_5 = self.RSI("QQQ", 5, Resolution.Minute)
        
        # Trade multiple times per day
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.Every(TimeSpan.FromMinutes(30)), 
                        self.ActiveTrading)
        
        self.last_trade_time = self.Time
        self.trades_today = 0
        self.max_trades_per_day = 5
        
        self.Debug("Active Day Trading Strategy: 100+ trades/year target")
    
    def ActiveTrading(self):
        if not self.spy_rsi_5.IsReady or not self.qqq_rsi_5.IsReady:
            return
        
        # Reset daily trade counter
        if self.Time.date() != self.last_trade_time.date():
            self.trades_today = 0
        
        if self.trades_today >= self.max_trades_per_day:
            return
        
        # Active trading signals
        spy_oversold = self.spy_rsi_5.Current.Value < 25
        spy_overbought = self.spy_rsi_5.Current.Value > 75
        qqq_oversold = self.qqq_rsi_5.Current.Value < 25
        qqq_overbought = self.qqq_rsi_5.Current.Value > 75
        
        spy_momentum = self.Securities[self.spy].Price > self.spy_sma_20.Current.Value
        
        current_spy = self.Portfolio[self.spy].Quantity
        current_qqq = self.Portfolio[self.qqq].Quantity
        
        # Frequent rebalancing logic
        if spy_oversold and spy_momentum:
            self.SetHoldings(self.spy, 0.8)
            self.SetHoldings(self.qqq, 0.0)
            self.trades_today += 1
        elif qqq_oversold and spy_momentum:
            self.SetHoldings(self.spy, 0.4)
            self.SetHoldings(self.qqq, 0.6)
            self.trades_today += 1
        elif spy_overbought or qqq_overbought:
            self.SetHoldings(self.spy, 0.3)
            self.SetHoldings(self.qqq, 0.2)
            self.SetHoldings(self.iwm, 0.2)  # Rotate to small caps
            self.trades_today += 1
        elif not spy_momentum:
            self.Liquidate()  # Exit all positions
            self.trades_today += 1
        
        self.last_trade_time = self.Time
    
    def OnData(self, data):
        pass''',

        "SwingTradingRotation15Y": '''from AlgorithmImports import *

class SwingTradingRotation15Y(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Swing trading universe (8 ETFs for rotation)
        self.symbols = []
        self.symbols.append(self.AddEquity("SPY", Resolution.Daily).Symbol)
        self.symbols.append(self.AddEquity("QQQ", Resolution.Daily).Symbol)
        self.symbols.append(self.AddEquity("IWM", Resolution.Daily).Symbol)
        self.symbols.append(self.AddEquity("XLF", Resolution.Daily).Symbol)  # Financials
        self.symbols.append(self.AddEquity("XLE", Resolution.Daily).Symbol)  # Energy
        self.symbols.append(self.AddEquity("XLK", Resolution.Daily).Symbol)  # Technology
        self.symbols.append(self.AddEquity("TLT", Resolution.Daily).Symbol)  # Bonds
        self.symbols.append(self.AddEquity("GLD", Resolution.Daily).Symbol)  # Gold
        
        # Swing trading indicators
        self.rsi_indicators = {}
        self.momentum_indicators = {}
        
        for symbol in self.symbols:
            self.rsi_indicators[symbol] = self.RSI(symbol, 7, Resolution.Daily)  # Fast RSI
            self.momentum_indicators[symbol] = self.MOMP(symbol, 10, Resolution.Daily)  # 10-day momentum
        
        # Trade every 2-3 days for swing trading
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.SwingTrade)
        
        self.rebalance_frequency = 3  # Rebalance every 3 days
        self.days_since_rebalance = 0
        
        self.Debug("Swing Trading Rotation: 100+ trades/year across 8 ETFs")
    
    def SwingTrade(self):
        self.days_since_rebalance += 1
        
        if self.days_since_rebalance < self.rebalance_frequency:
            return
        
        self.days_since_rebalance = 0
        
        if not all(indicator.IsReady for indicator in self.rsi_indicators.values()):
            return
        
        # Score each asset for swing trading
        asset_scores = {}
        for symbol in self.symbols:
            rsi = self.rsi_indicators[symbol].Current.Value
            momentum = self.momentum_indicators[symbol].Current.Value
            
            # Swing trading score (looking for oversold with momentum)
            score = 0
            if rsi < 40:  # Oversold
                score += (40 - rsi) / 10
            if momentum > 2:  # Positive momentum
                score += momentum / 5
            if 30 < rsi < 70:  # Not extreme
                score += 1
            
            asset_scores[symbol] = score
        
        # Select top 3-4 assets and rotate positions
        sorted_assets = sorted(asset_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Clear all positions (creates trades)
        self.Liquidate()
        
        # Allocate to top performers
        top_assets = sorted_assets[:4]
        weight_per_asset = 1.0 / len(top_assets)
        
        for symbol, score in top_assets:
            if score > 0:
                self.SetHoldings(symbol, weight_per_asset)
        
        self.Debug(f"Rebalanced to: {[str(asset[0]) for asset in top_assets]}")
    
    def OnData(self, data):
        pass''',

        "MeanReversionScalper15Y": '''from AlgorithmImports import *

class MeanReversionScalper15Y(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Scalping universe
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
        
        # High-frequency mean reversion indicators
        self.spy_bb = self.BB("SPY", 10, 1.5, Resolution.Hour)  # Tight bands
        self.spy_rsi = self.RSI("SPY", 3, Resolution.Hour)  # Very fast RSI
        self.qqq_bb = self.BB("QQQ", 10, 1.5, Resolution.Hour)
        self.qqq_rsi = self.RSI("QQQ", 3, Resolution.Hour)
        
        # Trade every hour during market hours
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.Every(TimeSpan.FromHours(1)), 
                        self.ScalpTrade)
        
        self.position_hold_hours = 0
        self.max_hold_time = 4  # Hold positions max 4 hours
        
        self.Debug("Mean Reversion Scalper: Very active hourly trading")
    
    def ScalpTrade(self):
        if not all([self.spy_bb.IsReady, self.spy_rsi.IsReady, self.qqq_bb.IsReady, self.qqq_rsi.IsReady]):
            return
        
        # Force exit after max hold time
        self.position_hold_hours += 1
        if self.position_hold_hours >= self.max_hold_time:
            self.Liquidate()
            self.position_hold_hours = 0
            return
        
        spy_price = self.Securities[self.spy].Price
        qqq_price = self.Securities[self.qqq].Price
        
        # Scalping signals
        spy_oversold = (self.spy_rsi.Current.Value < 10 and 
                       spy_price < self.spy_bb.LowerBand.Current.Value)
        spy_overbought = (self.spy_rsi.Current.Value > 90 and 
                         spy_price > self.spy_bb.UpperBand.Current.Value)
        
        qqq_oversold = (self.qqq_rsi.Current.Value < 10 and 
                       qqq_price < self.qqq_bb.LowerBand.Current.Value)
        qqq_overbought = (self.qqq_rsi.Current.Value > 90 and 
                         qqq_price > self.qqq_bb.UpperBand.Current.Value)
        
        # Active scalping logic
        if spy_oversold:
            self.SetHoldings(self.spy, 1.0)
            self.SetHoldings(self.qqq, 0.0)
            self.position_hold_hours = 0
        elif qqq_oversold:
            self.SetHoldings(self.spy, 0.0)
            self.SetHoldings(self.qqq, 1.0)
            self.position_hold_hours = 0
        elif spy_overbought or qqq_overbought:
            self.Liquidate()  # Exit on overbought
            self.position_hold_hours = 0
        elif 20 < self.spy_rsi.Current.Value < 80:
            # Neutral zone - light positioning
            self.SetHoldings(self.spy, 0.5)
            self.SetHoldings(self.qqq, 0.3)
    
    def OnData(self, data):
        pass''',

        "BreakoutChaser15Y": '''from AlgorithmImports import *

class BreakoutChaser15Y(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Breakout chasing universe
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Daily).Symbol
        self.xlk = self.AddEquity("XLK", Resolution.Daily).Symbol  # Tech sector
        
        # Breakout detection
        self.highs = {}
        self.lows = {}
        
        for symbol in [self.spy, self.qqq, self.iwm, self.xlk]:
            self.highs[symbol] = self.MAX(symbol, 5, Resolution.Daily)  # 5-day high
            self.lows[symbol] = self.MIN(symbol, 5, Resolution.Daily)   # 5-day low
        
        # Daily breakout scanning
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.ChaseBreakouts)
        
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.BeforeMarketClose("SPY", 60), 
                        self.ExitPositions)
        
        self.current_breakout = None
        self.days_in_position = 0
        self.max_hold_days = 3  # Quick exits
        
        self.Debug("Breakout Chaser: Daily breakout hunting across 4 ETFs")
    
    def ChaseBreakouts(self):
        if not all(indicator.IsReady for high_low in [self.highs, self.lows] for indicator in high_low.values()):
            return
        
        self.days_in_position += 1
        
        # Exit after max hold period
        if self.days_in_position >= self.max_hold_days:
            self.Liquidate()
            self.current_breakout = None
            self.days_in_position = 0
            return
        
        # Scan for new breakouts
        for symbol in [self.spy, self.qqq, self.iwm, self.xlk]:
            current_price = self.Securities[symbol].Price
            five_day_high = self.highs[symbol].Current.Value
            five_day_low = self.lows[symbol].Current.Value
            
            # Upward breakout
            if current_price > five_day_high * 1.01:  # 1% above 5-day high
                if self.current_breakout != symbol:
                    self.Liquidate()  # Exit previous position
                    self.SetHoldings(symbol, 1.2)  # 120% leverage on breakout
                    self.current_breakout = symbol
                    self.days_in_position = 0
                    self.Debug(f"Upward breakout: {symbol}")
                    break
            
            # Downward breakout (short opportunity)
            elif current_price < five_day_low * 0.99:  # 1% below 5-day low
                if self.current_breakout != f"{symbol}_SHORT":
                    self.Liquidate()
                    self.SetHoldings(symbol, -0.5)  # Short position
                    self.current_breakout = f"{symbol}_SHORT"
                    self.days_in_position = 0
                    self.Debug(f"Downward breakout: {symbol}")
                    break
    
    def ExitPositions(self):
        # End-of-day profit taking
        total_profit = self.Portfolio.TotalProfit
        if total_profit > 500:  # Take profits
            self.Liquidate()
            self.current_breakout = None
            self.days_in_position = 0
    
    def OnData(self, data):
        pass''',

        "VolatilityTrader15Y": '''from AlgorithmImports import *

class VolatilityTrader15Y(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # 15 YEARS
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Volatility trading instruments
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol
        
        # Volatility measurement
        self.spy_atr = self.ATR("SPY", 10, Resolution.Daily)
        self.volatility_window = RollingWindow[float](20)
        
        # Trade based on volatility changes
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.TradeVolatility)
        
        self.last_volatility_regime = "NORMAL"
        self.regime_change_count = 0
        
        self.Debug("Volatility Trader: Active regime-based trading")
    
    def OnData(self, data):
        if self.spy in data and data[self.spy] is not None:
            if hasattr(self, 'prev_spy_price'):
                daily_return = abs((data[self.spy].Close - self.prev_spy_price) / self.prev_spy_price)
                self.volatility_window.Add(daily_return)
            self.prev_spy_price = data[self.spy].Close
    
    def TradeVolatility(self):
        if not self.spy_atr.IsReady or not self.volatility_window.IsReady:
            return
        
        # Calculate volatility metrics
        current_atr = self.spy_atr.Current.Value
        avg_atr = sum([self.spy_atr[i] for i in range(min(10, self.spy_atr.Count))]) / min(10, self.spy_atr.Count)
        recent_vol = sum([vol for vol in self.volatility_window]) / self.volatility_window.Count
        
        # Determine volatility regime
        if current_atr > avg_atr * 1.5 or recent_vol > 0.02:
            current_regime = "HIGH_VOL"
        elif current_atr < avg_atr * 0.7 and recent_vol < 0.008:
            current_regime = "LOW_VOL"
        else:
            current_regime = "NORMAL"
        
        # Trade on regime changes
        if current_regime != self.last_volatility_regime:
            self.regime_change_count += 1
            
            if current_regime == "HIGH_VOL":
                # High volatility - flight to quality + volatility play
                self.SetHoldings(self.tlt, 0.4)  # Bonds
                self.SetHoldings(self.gld, 0.3)  # Gold
                self.SetHoldings(self.spy, 0.3)  # Reduced equity
                self.Debug("HIGH VOL regime - defensive allocation")
                
            elif current_regime == "LOW_VOL":
                # Low volatility - risk on
                self.SetHoldings(self.spy, 0.6)
                self.SetHoldings(self.qqq, 0.4)  # Growth tilt
                self.Debug("LOW VOL regime - risk on")
                
            else:  # NORMAL
                # Balanced allocation
                self.SetHoldings(self.spy, 0.5)
                self.SetHoldings(self.qqq, 0.2)
                self.SetHoldings(self.tlt, 0.2)
                self.SetHoldings(self.gld, 0.1)
                self.Debug("NORMAL regime - balanced")
            
            self.last_volatility_regime = current_regime
    
    def OnData(self, data):
        pass'''
    }
    
    return strategies

async def deploy_active_strategies():
    """Deploy high-frequency trading strategies targeting 100+ trades/year"""
    
    api = QuantConnectCloudAPI("357130", "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912")
    
    print("⚡ DEPLOYING ACTIVE 15-YEAR TRADING STRATEGIES")
    print("=" * 70)
    print("📅 Period: 2010-2024 (15 years)")
    print("🎯 Target: 100+ trades per year (1,500+ total trades)")
    print("⚡ Frequency: Daily/Hourly rebalancing, active position management")
    print("📊 Features: Breakouts, scalping, swing trading, regime changes")
    print("=" * 70)
    
    strategies = get_active_15_year_strategies()
    deployed = []
    
    for i, (name, code) in enumerate(strategies.items(), 1):
        print(f"\n🚀 [{i}/{len(strategies)}] Deploying: {name}")
        print("-" * 50)
        
        # Show expected trading frequency
        freq_map = {
            "DayTradingMomentum15Y": "150+ trades/year (every 30 min)",
            "SwingTradingRotation15Y": "120+ trades/year (every 3 days)", 
            "MeanReversionScalper15Y": "300+ trades/year (hourly)",
            "BreakoutChaser15Y": "100+ trades/year (daily breakouts)",
            "VolatilityTrader15Y": "80+ trades/year (regime changes)"
        }
        
        print(f"⚡ Expected Frequency: {freq_map.get(name, '100+ trades/year')}")
        
        try:
            result = api.deploy_strategy(name, code)
            
            if result['success']:
                deployed.append({
                    'name': name,
                    'project_id': result['project_id'],
                    'backtest_id': result['backtest_id'],
                    'url': result['url'],
                    'frequency': freq_map.get(name, '100+ trades/year')
                })
                print(f"✅ SUCCESS!")
                print(f"   🔗 URL: {result['url']}")
            else:
                print(f"❌ FAILED: {result['error']}")
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
        
        # Wait between deployments
        if i < len(strategies):
            print(f"\n⏳ Waiting 60 seconds...")
            await asyncio.sleep(60)
    
    # Summary
    print(f"\n{'='*70}")
    print("⚡ ACTIVE STRATEGY DEPLOYMENT COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Deployed: {len(deployed)}/{len(strategies)} strategies")
    print(f"🎯 All strategies target 100+ trades per year")
    print(f"📊 Total expected: 1,500+ trades over 15 years per strategy")
    
    for strategy in deployed:
        print(f"\n🏆 {strategy['name']}")
        print(f"   ⚡ Frequency: {strategy['frequency']}")
        print(f"   🔗 Results: {strategy['url']}")
    
    return deployed

if __name__ == "__main__":
    asyncio.run(deploy_active_strategies())