"""
Basic SPY Strategy
A very simple strategy that just buys and holds SPY
"""
from QuantConnect import Resolution
from QuantConnect.Algorithm import QCAlgorithm

class BasicSPYStrategy(QCAlgorithm):
    
    def Initialize(self):
        # We won't set dates since they're enforced by the platform
        self.SetCash(100000)
        
        # Add SPY at daily resolution
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        
        # Set benchmark
        self.SetBenchmark("SPY")
        
        # Log initialization
        self.Log("Basic SPY Strategy initialized")
        self.Log(f"Algorithm start date: {self.StartDate}")
        self.Log(f"Algorithm end date: {self.EndDate}")
        
        # Schedule buying SPY at the start
        self.Schedule.On(self.DateRules.MonthStart("SPY"), self.TimeRules.AfterMarketOpen("SPY"), self.RebalancePortfolio)

    def OnData(self, data):
        """Event fired each time new data arrives"""
        # Buy SPY on the first day
        if not self.Portfolio.Invested and not self.IsWarmingUp:
            self.SetHoldings("SPY", 1.0)
            self.Log(f"Buying SPY at {self.Time}")

    def RebalancePortfolio(self):
        """Monthly rebalancing to ensure we're fully invested"""
        self.Log(f"Monthly rebalance at {self.Time}")
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1.0)
            self.Log(f"Buying SPY at {self.Time}")
        else:
            self.Log(f"Already invested in SPY, current holdings: {self.Portfolio['SPY'].Quantity} shares")

    def OnEndOfAlgorithm(self):
        """Summarize strategy performance at the end of the backtest"""
        self.Log(f"Algorithm completed at {self.Time}")
        self.Log(f"Final SPY holdings: {self.Portfolio['SPY'].Quantity} shares")
        self.Log(f"Final portfolio value: ${self.Portfolio.TotalPortfolioValue}")
        self.Log(f"Total profit: {(self.Portfolio.TotalPortfolioValue / 100000 - 1) * 100:.2f}%")
