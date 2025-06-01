from AlgorithmImports import *

class MomentumTradingAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2015, 1, 1)  # Set Start Date
        self.SetEndDate(2025, 1, 1)    # Set End Date
        self.SetCash(100000)            # Set Strategy Cash
        self.AddEquity("SPY", Resolution.Daily)  # Add S&P 500 ETF

        self.short_sma = self.SMA("SPY", 50, Resolution.Daily)
        self.long_sma = self.SMA("SPY", 200, Resolution.Daily)

    def OnData(self, data):
        if not self.short_sma.IsReady or not self.long_sma.IsReady:
            return

        if self.short_sma.Current.Value > self.long_sma.Current.Value and not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1)  # Buy SPY
        elif self.short_sma.Current.Value < self.long_sma.Current.Value and self.Portfolio.Invested:
            self.Liquidate("SPY")  # Sell SPY

    def OnEndOfDay(self):
        self.Plot("SMA", "Short SMA", self.short_sma.Current.Value)
        self.Plot("SMA", "Long SMA", self.long_sma.Current.Value)
