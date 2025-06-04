from AlgorithmImports import *

class AdvancedStrategy(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Asset and leverage
        self.symbol = self.add_equity("QQQ", Resolution.DAILY)
        self.symbol.set_leverage(15.0)
        
        # Technical indicators
        self.sma_fast = self.sma("QQQ", 10)
        self.sma_slow = self.sma("QQQ", 30)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        # Trade frequency control
        if (self.time - self.last_trade).days < 7:
            return
            
        self.last_trade = self.time
        
        # Trading logic
        if self.sma_fast.current.value > self.sma_slow.current.value:
            self.set_holdings("QQQ", 2.0)
        else:
            self.liquidate()
