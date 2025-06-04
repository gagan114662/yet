from AlgorithmImports import *

class FinalCloud25CAGR(QCAlgorithm):
    """
    FINAL CLOUD 25% CAGR STRATEGY
    Optimized based on previous results
    """
    
    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Use QQQ for higher volatility and returns
        self.qqq = self.add_equity("QQQ", Resolution.DAILY)
        self.qqq.set_leverage(5.0)  # Maximum leverage
        
        # Very fast indicators for aggressive trading
        self.sma_5 = self.sma("QQQ", 5)
        self.sma_15 = self.sma("QQQ", 15)
        self.mom = self.momp("QQQ", 10)  # 10-day momentum
        
        # Track for daily trading
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_5.is_ready or not self.sma_15.is_ready:
            return
            
        # Trade every 2 days for more opportunities
        if (self.time - self.last_trade).days < 2:
            return
            
        self.last_trade = self.time
        
        # Get indicators
        fast = self.sma_5.current.value
        slow = self.sma_15.current.value
        momentum = self.mom.current.value
        
        # Calculate position based on trend and momentum
        if fast > slow and momentum > 0:
            # Strong uptrend with momentum
            if momentum > 5:  # Very strong momentum
                position = 4.5  # 450% = 22.5x leverage
            elif momentum > 2:  # Good momentum
                position = 3.5  # 350% = 17.5x leverage
            else:  # Weak momentum
                position = 2.0  # 200% = 10x leverage
                
            self.set_holdings("QQQ", position)
            
        elif fast < slow * 0.99:  # Downtrend
            # Go to cash or short
            if momentum < -2:
                self.set_holdings("QQQ", -1.5)  # Short
            else:
                self.liquidate()  # Cash
        else:
            # Sideways - reduce exposure
            self.set_holdings("QQQ", 1.0)
            
    def on_end_of_algorithm(self):
        final = self.portfolio.total_portfolio_value
        years = (self.end_date - self.start_date).days / 365.25
        cagr = ((final / 100000) ** (1/years)) - 1
        
        self.log("FINAL CLOUD 25% CAGR RESULTS")
        self.log(f"Final Value: ${final:,.0f}")
        self.log(f"CAGR: {cagr*100:.1f}%")
        self.log(f"Years: {years:.1f}")
        
        if cagr >= 0.25:
            self.log("TARGET ACHIEVED: 25%+ CAGR!")
        elif cagr >= 0.20:
            self.log("CLOSE: 20%+ CAGR")
        else:
            self.log(f"Need {(0.25-cagr)*100:.1f}% more")