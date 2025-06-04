from AlgorithmImports import *
import numpy as np

class FinalAlphaMonster(QCAlgorithm):
    """
    Final simplified strategy focusing on what actually works
    """
    
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Use SPY as primary instrument
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Simple momentum indicators
        self.fast = self.SMA(self.spy, 10, Resolution.Daily)
        self.slow = self.SMA(self.spy, 30, Resolution.Daily)
        self.rsi = self.RSI(self.spy, 14, Resolution.Daily)
        
        # Position sizing
        self.leverage = 2.0
        
        self.SetWarmUp(30)
    
    def OnData(self, data):
        if self.IsWarmingUp or not self.fast.IsReady:
            return
        
        if self.spy not in data:
            return
        
        holdings = self.Portfolio[self.spy].Quantity
        
        # Simple trend following
        if self.fast.Current.Value > self.slow.Current.Value and self.rsi.Current.Value > 30:
            if holdings <= 0:
                self.SetHoldings(self.spy, self.leverage)
        elif self.fast.Current.Value < self.slow.Current.Value or self.rsi.Current.Value > 70:
            if holdings > 0:
                self.Liquidate()