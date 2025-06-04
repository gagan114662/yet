from AlgorithmImports import *

class BaseAgent(QCAlgorithm):
    """Base trading agent for DGM evolution"""
    
    def initialize(self):
        self.set_start_date(2015, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Basic setup
        self.symbol = self.add_equity("SPY", Resolution.DAILY)
        self.symbol.set_leverage(2.0)  # Conservative start
        
        # Simple indicators
        self.sma_fast = self.sma("SPY", 10)
        self.sma_slow = self.sma("SPY", 30)
        
    def on_data(self, data):
        if not self.sma_fast.is_ready:
            return
            
        # Simple trend following
        if self.sma_fast.current.value > self.sma_slow.current.value:
            self.set_holdings("SPY", 1.0)
        else:
            self.liquidate()