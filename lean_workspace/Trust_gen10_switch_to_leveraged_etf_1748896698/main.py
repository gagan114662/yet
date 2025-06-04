from AlgorithmImports import *

class TrustworthyStrategy(QCAlgorithm):
    """
    Trustworthy Strategy - Agent: gen10_switch_to_leveraged_etf
    Config: UPRO 8.0x leverage, 2.0x position, SMA(10,30)
    """
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("UPRO", Resolution.DAILY)
        self.symbol.set_leverage(8.0)
        
        self.sma_fast = self.sma("UPRO", 10)
        self.sma_slow = self.sma("UPRO", 30)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        if (self.time - self.last_trade).days < 1:
            return
            
        self.last_trade = self.time
        
        if self.sma_fast.current.value > self.sma_slow.current.value:
            self.set_holdings("UPRO", 2.0)
        else:
            self.liquidate()
