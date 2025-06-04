from AlgorithmImports import *

class TrustworthyStrategy(QCAlgorithm):
    """
    Trustworthy Strategy - Agent: gen1_options_strategy_spy
    Config: SPY 35.0x leverage, 3.0x position, SMA(3,12)
    """
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("SPY", Resolution.DAILY)
        self.symbol.set_leverage(35.0)
        
        self.sma_fast = self.sma("SPY", 3)
        self.sma_slow = self.sma("SPY", 12)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        if (self.time - self.last_trade).days < 1:
            return
            
        self.last_trade = self.time
        
        if self.sma_fast.current.value > self.sma_slow.current.value:
            self.set_holdings("SPY", 3.0)
        else:
            self.liquidate()
