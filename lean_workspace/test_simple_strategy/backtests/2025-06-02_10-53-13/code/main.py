from AlgorithmImports import *

class SimpleTestStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(100000)
        
        # Add SPY
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        
    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1.0)