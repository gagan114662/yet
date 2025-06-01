# region imports
from AlgorithmImports import *
# endregion

class TestStrategy(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2023, 1, 1)
        self.set_cash(100000)
        self.symbol = self.add_equity("SPY", Resolution.DAILY).symbol
        
        # Create moving average indicators
        self.sma_short = self.sma(self.symbol, 20, Resolution.DAILY)
        self.sma_long = self.sma(self.symbol, 50, Resolution.DAILY)
        
        self.set_warm_up(50)

    def on_data(self, data: Slice):
        if self.is_warming_up:
            return
            
        if not (self.sma_short.is_ready and self.sma_long.is_ready):
            return
        
        short_ma = self.sma_short.current.value
        long_ma = self.sma_long.current.value
        
        if short_ma > long_ma and not self.portfolio[self.symbol].invested:
            self.set_holdings(self.symbol, 1.0)
            self.debug(f"BUY: SMA20={short_ma:.2f} > SMA50={long_ma:.2f}")
            
        elif short_ma < long_ma and self.portfolio[self.symbol].invested:
            self.liquidate(self.symbol)
            self.debug(f"SELL: SMA20={short_ma:.2f} < SMA50={long_ma:.2f}")
