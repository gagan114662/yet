from AlgorithmImports import *

class TrustworthyStrategy(QCAlgorithm):
    """
    Trustworthy Strategy - Agent: gen20_switch_to_tqqq
    Config: TQQQ 5.6x leverage, 1.0x position, SMA(10,30)
    """
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("TQQQ", Resolution.DAILY)
        self.symbol.set_leverage(5.6)
        
        self.sma_fast = self.sma("TQQQ", 10)
        self.sma_slow = self.sma("TQQQ", 30)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        if (self.time - self.last_trade).days < 1:
            return
            
        self.last_trade = self.time
        
        if self.sma_fast.current.value > self.sma_slow.current.value:
            self.set_holdings("TQQQ", 1.0)
        else:
            self.liquidate()
