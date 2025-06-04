from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta

class BattleTestedAlpha(QCAlgorithm):
    """
    Battle-tested high performance strategy
    Simplified approach focusing on what actually works in backtests
    """
    
    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        
        # Basic configuration
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Risk parameters
        self.leverage = 3.0
        self.stop_loss = 0.02
        self.take_profit = 0.05
        
        # Core instruments only - avoid data issues
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        
        # Only add other symbols if they exist
        try:
            self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
            self.iwm = self.AddEquity("IWM", Resolution.Hour).Symbol
            self.tlt = self.AddEquity("TLT", Resolution.Hour).Symbol
        except:
            self.Debug("Some symbols not available, using SPY only")
            self.qqq = None
            self.iwm = None
            self.tlt = None
        
        # Simple indicators
        self.fast_ema = self.EMA(self.spy, 10, Resolution.Daily)
        self.slow_ema = self.EMA(self.spy, 30, Resolution.Daily)
        self.rsi = self.RSI(self.spy, 14, Resolution.Daily)
        self.bb = self.BB(self.spy, 20, 2, Resolution.Daily)
        self.atr = self.ATR(self.spy, 14, Resolution.Daily)
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.AfterMarketOpen(self.spy, 30),
                        self.Trade)
        
        self.SetWarmUp(30, Resolution.Daily)
    
    def Trade(self):
        """Main trading logic"""
        if self.IsWarmingUp or not self.fast_ema.IsReady:
            return
        
        # Get current positions
        spy_holding = self.Portfolio[self.spy].Quantity
        
        # Momentum signal
        bullish = self.fast_ema.Current.Value > self.slow_ema.Current.Value
        bearish = self.fast_ema.Current.Value < self.slow_ema.Current.Value
        
        # RSI confirmation
        oversold = self.rsi.Current.Value < 30
        overbought = self.rsi.Current.Value > 70
        
        # Bollinger Band position
        price = self.Securities[self.spy].Price
        bb_position = (price - self.bb.LowerBand.Current.Value) / (self.bb.UpperBand.Current.Value - self.bb.LowerBand.Current.Value)
        
        # Position sizing based on volatility
        if self.atr.Current.Value > 0:
            position_size = (self.Portfolio.TotalPortfolioValue * 0.02) / (self.atr.Current.Value * 2)
            position_size = min(position_size, self.Portfolio.TotalPortfolioValue * self.leverage / price)
        else:
            position_size = self.Portfolio.TotalPortfolioValue * 0.5 / price
        
        # Trading logic
        if bullish and (oversold or bb_position < 0.2):
            # Strong buy signal
            if spy_holding <= 0:
                self.SetHoldings(self.spy, self.leverage * 0.8)
                self.Debug(f"BUY signal: RSI={self.rsi.Current.Value:.1f}, BB={bb_position:.2f}")
        
        elif bearish and (overbought or bb_position > 0.8):
            # Sell signal
            if spy_holding > 0:
                self.Liquidate(self.spy)
                self.Debug(f"SELL signal: RSI={self.rsi.Current.Value:.1f}, BB={bb_position:.2f}")
                
                # Go defensive if available
                if self.tlt:
                    self.SetHoldings(self.tlt, 0.5)
        
        # Stop loss and take profit
        if self.Portfolio[self.spy].UnrealizedProfitPercent < -self.stop_loss:
            self.Liquidate(self.spy)
            self.Debug("Stop loss triggered")
        elif self.Portfolio[self.spy].UnrealizedProfitPercent > self.take_profit:
            self.SetHoldings(self.spy, self.Portfolio[self.spy].Quantity * 0.5 / self.Portfolio.TotalPortfolioValue)
            self.Debug("Taking partial profits")
        
        # Diversification if other symbols available
        if self.qqq and bullish:
            self.SetHoldings(self.qqq, self.leverage * 0.2)
        
    def OnData(self, data):
        """Additional data handling"""
        pass