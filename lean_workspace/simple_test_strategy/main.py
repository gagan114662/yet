from AlgorithmImports import *

class SimpleTestStrategy(QCAlgorithm):
    """
    Simple test strategy to verify cloud connectivity
    """
    
    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.spy = self.add_equity("SPY", Resolution.DAILY)
        
    def on_data(self, data):
        if not self.portfolio.invested:
            self.set_holdings("SPY", 1.0)