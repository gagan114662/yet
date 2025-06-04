from AlgorithmImports import *

class TargetCrusherStrategy(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("TQQQ", Resolution.DAILY)
        self.symbol.set_leverage(8.0)
        
        self.sma_fast = self.sma("TQQQ", 8)
        self.sma_slow = self.sma("TQQQ", 21)
        self.rsi = self.rsi("TQQQ", 14)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        if (self.time - self.last_trade).days < 1:
            return
            
        self.last_trade = self.time
        
        if (self.sma_fast.current.value > self.sma_slow.current.value and \
            self.rsi.is_ready and self.rsi.current.value < 65):
            self.set_holdings("TQQQ", 1.5)
        else:
            self.liquidate()
