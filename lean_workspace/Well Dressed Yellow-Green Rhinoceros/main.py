from AlgorithmImports import *

class MomentumTradingAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)  # Set Start Date
        self.SetEndDate(2025, 1, 1)    # Set End Date
        self.SetCash(100000)            # Set Strategy Cash
        self.symbol = "SPY"             # Use S&P 500 ETF
        self.AddEquity(self.symbol, Resolution.Daily)

        # Define indicators
        self.short_sma = self.SMA(self.symbol, 20, Resolution.Daily)
        self.long_sma = self.SMA(self.symbol, 50, Resolution.Daily)
        self.rsi = self.RSI(self.symbol, 14, MovingAverageType.Wilders, Resolution.Daily)

        # Initialize variables
        self.lastAction = None

    def OnData(self, data):
        if not self.short_sma.IsReady or not self.long_sma.IsReady or not self.rsi.IsReady:
            return

        # Check for buy signal
        if self.short_sma.Current.Value > self.long_sma.Current.Value and self.rsi.Current.Value < 30:
            if not self.Portfolio.Invested:
                self.SetHoldings(self.symbol, 1)  # Buy SPY
                self.lastAction = self.Time

        # Check for sell signal
        elif self.short_sma.Current.Value < self.long_sma.Current.Value and self.rsi.Current.Value > 70:
            if self.Portfolio.Invested:
                self.Liquidate(self.symbol)  # Sell SPY
                self.lastAction = self.Time

    def OnEndOfDay(self):
        self.Plot("SMA", "Short SMA", self.short_sma.Current.Value)
        self.Plot("SMA", "Long SMA", self.long_sma.Current.Value)
        self.Plot("RSI", "RSI", self.rsi.Current.Value)

    def OnEndOfAlgorithm(self):
        # Calculate performance metrics
        self.Log(f"CAGR: {self.CAGR:.2%}")
        self.Log(f"Sharpe Ratio: {self.SharpeRatio:.2f}")
        self.Log(f"Max Drawdown: {self.MaxDrawdown:.2%}")
        self.Log(f"Average Profit: {self.AverageProfit:.2%}")
