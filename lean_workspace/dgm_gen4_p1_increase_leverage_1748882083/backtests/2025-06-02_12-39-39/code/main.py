from AlgorithmImports import *

class DGMAgent(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("SPY", Resolution.DAILY)
        self.symbol.set_leverage(5.0)
        
        self.sma_fast = self.sma("SPY", 10)
        self.sma_slow = self.sma("SPY", 30)
        self.rsi = self.rsi('SPY', 14)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready:
            return
            
        # Trade frequency control
        if (self.time - self.last_trade).days < 7:
            return
            
        self.last_trade = self.time
        
        if (self.sma_fast.current.value > self.sma_slow.current.value and 
            self.rsi.is_ready and self.rsi.current.value < 70):
            self.set_holdings("SPY", 1.0)
        else:
            self.liquidate()
