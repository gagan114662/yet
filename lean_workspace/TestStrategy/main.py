from AlgorithmImports import *

class BasicSPYAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(100000)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.Debug("Test strategy initialized")

    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings(self.spy, 1)
            self.Debug(f"Buying SPY at {self.Securities[self.spy].Price}")

        # Log portfolio value to ensure it's running
        # self.Debug(f"Portfolio Value: {self.Portfolio.TotalPortfolioValue}")

    def OnEndOfAlgorithm(self):
        self.Debug(f"Algorithm ended. Final portfolio value: {self.Portfolio.TotalPortfolioValue}")
