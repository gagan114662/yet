"""
Basic SPY Strategy
A very simple strategy that just buys and holds SPY with extensive logging
"""
from QuantConnect import Resolution
from QuantConnect.Algorithm import QCAlgorithm
import datetime

class BasicSPYStrategy(QCAlgorithm):
    
    def Initialize(self):
        # Explicitly set dates
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add SPY at daily resolution
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        
        # Set benchmark
        self.SetBenchmark("SPY")
        
        # Log initialization with detailed information
        self.Log("INITIALIZATION: Basic SPY Strategy initialized")
        self.Log(f"INITIALIZATION: Algorithm start date: {self.StartDate}")
        self.Log(f"INITIALIZATION: Algorithm end date: {self.EndDate}")
        self.Log(f"INITIALIZATION: Initial cash: ${self.Portfolio.Cash}")
        
        # Schedule daily check
        self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.At(9, 30), self.DailyCheck)
        
        # Schedule buying SPY at the start of each month
        self.Schedule.On(self.DateRules.MonthStart("SPY"), self.TimeRules.AfterMarketOpen("SPY"), self.RebalancePortfolio)
        
        # Schedule weekly status check
        self.Schedule.On(self.DateRules.WeekStart("SPY"), self.TimeRules.AfterMarketOpen("SPY"), self.WeeklyStatus)

    def OnData(self, data):
        """Event fired each time new data arrives"""
        # Log data receipt
        if "SPY" in data:
            self.Log(f"ONDATA: Received SPY data at {self.Time}, Price: ${data['SPY'].Close}")
        
        # Buy SPY on the first day
        if not self.Portfolio.Invested and not self.IsWarmingUp:
            self.Log(f"ONDATA: Not invested yet, buying SPY with all available cash: ${self.Portfolio.Cash}")
            self.SetHoldings("SPY", 1.0)
            self.Log(f"ONDATA: Buy order submitted for SPY at {self.Time}")
        elif self.Portfolio.Invested:
            self.Log(f"ONDATA: Currently holding SPY: {self.Portfolio['SPY'].Quantity} shares, Value: ${self.Portfolio['SPY'].HoldingsValue}")

    def DailyCheck(self):
        """Daily portfolio check"""
        self.Log(f"DAILY CHECK: Date: {self.Time}")
        self.Log(f"DAILY CHECK: Portfolio Value: ${self.Portfolio.TotalPortfolioValue}")
        self.Log(f"DAILY CHECK: Cash: ${self.Portfolio.Cash}")
        
        if self.Portfolio.Invested:
            self.Log(f"DAILY CHECK: SPY Holdings: {self.Portfolio['SPY'].Quantity} shares, Value: ${self.Portfolio['SPY'].HoldingsValue}")
        else:
            self.Log(f"DAILY CHECK: Not invested in SPY")

    def RebalancePortfolio(self):
        """Monthly rebalancing to ensure we're fully invested"""
        self.Log(f"MONTHLY REBALANCE: Monthly rebalance at {self.Time}")
        if not self.Portfolio.Invested:
            self.Log(f"MONTHLY REBALANCE: Not invested, buying SPY with all available cash: ${self.Portfolio.Cash}")
            self.SetHoldings("SPY", 1.0)
            self.Log(f"MONTHLY REBALANCE: Buy order submitted for SPY at {self.Time}")
        else:
            self.Log(f"MONTHLY REBALANCE: Already invested in SPY, current holdings: {self.Portfolio['SPY'].Quantity} shares, Value: ${self.Portfolio['SPY'].HoldingsValue}")
    
    def WeeklyStatus(self):
        """Weekly status report"""
        self.Log(f"WEEKLY STATUS: Week starting {self.Time}")
        self.Log(f"WEEKLY STATUS: Portfolio Value: ${self.Portfolio.TotalPortfolioValue}")
        self.Log(f"WEEKLY STATUS: Cash: ${self.Portfolio.Cash}")
        self.Log(f"WEEKLY STATUS: SPY Price: ${self.Securities['SPY'].Price}")
        
        if self.Portfolio.Invested:
            self.Log(f"WEEKLY STATUS: SPY Holdings: {self.Portfolio['SPY'].Quantity} shares, Value: ${self.Portfolio['SPY'].HoldingsValue}")
        else:
            self.Log(f"WEEKLY STATUS: Not invested in SPY")

    def OnOrderEvent(self, orderEvent):
        """Log all order events"""
        self.Log(f"ORDER EVENT: {orderEvent}")

    def OnEndOfAlgorithm(self):
        """Summarize strategy performance at the end of the backtest"""
        self.Log(f"FINAL SUMMARY: Algorithm completed at {self.Time}")
        self.Log(f"FINAL SUMMARY: Final SPY holdings: {self.Portfolio['SPY'].Quantity} shares")
        self.Log(f"FINAL SUMMARY: Final portfolio value: ${self.Portfolio.TotalPortfolioValue}")
        self.Log(f"FINAL SUMMARY: Total profit: {(self.Portfolio.TotalPortfolioValue / 100000 - 1) * 100:.2f}%")
