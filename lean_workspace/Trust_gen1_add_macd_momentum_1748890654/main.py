from AlgorithmImports import *

class TrustworthyStrategy(QCAlgorithm):
    """
    Trustworthy Strategy - Agent: gen1_add_macd_momentum
    Config: QQQ 12.0x leverage, 1.8x position, SMA(8,21)
    """
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("QQQ", Resolution.DAILY)
        self.symbol.set_leverage(12.0)
        
        self.sma_fast = self.sma("QQQ", 8)
        self.sma_slow = self.sma("QQQ", 21)
        self.macd = self.macd("QQQ", 12, 26, 9)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        if (self.time - self.last_trade).days < 1:
            return
            
        self.last_trade = self.time
        
        if (self.sma_fast.current.value > self.sma_slow.current.value and \
            self.macd.is_ready and self.macd.current.value > 0):
            self.set_holdings("QQQ", 1.8)
        else:
            self.liquidate()
