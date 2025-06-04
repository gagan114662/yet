from AlgorithmImports import *

class Ultra25CAGR(QCAlgorithm):
    """
    ULTRA 25% CAGR - MAXIMUM AGGRESSION
    """
    
    def initialize(self):
        self.set_start_date(2011, 1, 1)  # TQQQ inception
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # 3x leveraged NASDAQ ETF
        self.tqqq = self.add_equity("TQQQ", Resolution.DAILY)
        self.tqqq.set_leverage(2.0)  # 2x on 3x = 6x total
        
        # Ultra fast signals
        self.sma3 = self.sma("TQQQ", 3)
        self.sma8 = self.sma("TQQQ", 8)
        
        # Trade counter
        self.days_since_trade = 0
        
    def on_data(self, data):
        if not self.sma3.is_ready:
            return
            
        self.days_since_trade += 1
        
        # Trade every day for maximum opportunities
        if self.days_since_trade < 1:
            return
            
        fast = self.sma3.current.value
        slow = self.sma8.current.value
        
        if fast > slow * 1.005:  # 0.5% above = strong bull
            self.set_holdings("TQQQ", 2.0)  # 200% = 6x leverage
            self.days_since_trade = 0
        elif fast > slow:  # Mild bull
            self.set_holdings("TQQQ", 1.5)  # 150% = 4.5x leverage
            self.days_since_trade = 0
        elif fast < slow * 0.995:  # Bear
            self.liquidate()
            self.days_since_trade = 0
            
    def on_end_of_algorithm(self):
        final = self.portfolio.total_portfolio_value
        years = (self.end_date - self.start_date).days / 365.25
        cagr = ((final / 100000) ** (1/years)) - 1
        
        self.log(f"ULTRA 25% RESULTS: CAGR = {cagr*100:.1f}%")
        if cagr >= 0.25:
            self.log("TARGET ACHIEVED!")