from AlgorithmImports import *

class Target25Strategy(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Use TQQQ (3x leveraged QQQ) for higher returns
        self.tqqq = self.add_equity("TQQQ", Resolution.DAILY)
        self.tqqq.set_leverage(5.0)  # 5x leverage on 3x leveraged ETF = 15x effective
        
        # Simple but effective momentum strategy
        self.sma_short = self.sma("TQQQ", 5)
        self.sma_long = self.sma("TQQQ", 20)
        
        self.last_trade_date = None
        
    def on_data(self, data):
        if not self.sma_short.is_ready or not self.sma_long.is_ready:
            return
            
        # Trade daily for maximum opportunities
        current_date = self.time.date()
        if self.last_trade_date == current_date:
            return
        
        self.last_trade_date = current_date
        
        # Aggressive momentum strategy
        if self.sma_short.current.value > self.sma_long.current.value:
            # Strong uptrend - go long with maximum position
            if not self.portfolio.invested:
                self.set_holdings("TQQQ", 1.0)
        else:
            # Exit on any sign of weakness
            if self.portfolio.invested:
                self.liquidate()