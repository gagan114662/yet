from AlgorithmImports import *

class Target25Strategy(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Start with TQQQ for higher growth potential
        self.symbol = self.add_equity("TQQQ", Resolution.DAILY)
        self.symbol.set_leverage(3.0)
        
        # Technical indicators
        self.sma_fast = self.sma("TQQQ", 5)  # Faster signals
        self.sma_slow = self.sma("TQQQ", 15)
        self.rsi = self.rsi('TQQQ', 14)
        self.macd = self.macd('TQQQ', 12, 26, 9)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.macd.is_ready:
            return
            
        # Daily trading for more opportunities
        if (self.time - self.last_trade).days < 1:
            return
            
        self.last_trade = self.time
        
        # Aggressive entry: SMA crossover + MACD momentum + RSI not oversold
        if (self.sma_fast.current.value > self.sma_slow.current.value and 
            self.macd.current.value > 0 and
            self.rsi.is_ready and self.rsi.current.value > 30):
            self.set_holdings("TQQQ", 1.0)
        else:
            self.liquidate()