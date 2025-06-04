from AlgorithmImports import *

class AggressiveCloud25(QCAlgorithm):
    """
    AGGRESSIVE CLOUD 25% CAGR STRATEGY
    Uses high leverage, frequent trading, and multiple assets
    """
    
    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Core SPY for stability and signals
        self.spy = self.add_equity("SPY", Resolution.DAILY)
        self.spy.set_leverage(5.0)  # Maximum leverage
        
        # Fast indicators for aggressive trading
        self.sma_fast = self.sma("SPY", 10)
        self.sma_slow = self.sma("SPY", 30)
        self.rsi = self.rsi("SPY", 14)
        
        # Performance tracking
        self.last_trade_time = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        # Trade frequently for more opportunities
        if (self.time - self.last_trade_time).days < 5:
            return
            
        self.last_trade_time = self.time
        
        # Simple but aggressive signals
        fast_ma = self.sma_fast.current.value
        slow_ma = self.sma_slow.current.value
        rsi_val = self.rsi.current.value
        
        # Aggressive position sizing based on trend strength
        trend_strength = (fast_ma - slow_ma) / slow_ma
        
        if fast_ma > slow_ma:
            # Bull market - aggressive long
            if trend_strength > 0.02:  # Strong trend
                position = 4.0  # 400% with 5x leverage = 20x
            elif trend_strength > 0.01:  # Moderate trend
                position = 3.0  # 300% with 5x leverage = 15x
            else:
                position = 2.0  # 200% with 5x leverage = 10x
                
            # RSI boost for oversold
            if rsi_val < 30:
                position *= 1.2
                
            self.set_holdings("SPY", min(4.5, position))
            
        elif fast_ma < slow_ma * 0.98:  # Clear downtrend
            # Short aggressively
            self.set_holdings("SPY", -2.0)  # -200% = -10x
        else:
            # Neutral - reduce exposure
            self.set_holdings("SPY", 0.5)
            
    def on_end_of_algorithm(self):
        final = self.portfolio.total_portfolio_value
        years = (self.end_date - self.start_date).days / 365.25
        cagr = ((final / 100000) ** (1/years)) - 1
        
        self.log("AGGRESSIVE CLOUD 25% RESULTS")
        self.log(f"Final: ${final:,.0f}")
        self.log(f"CAGR: {cagr*100:.1f}%")
        
        if cagr >= 0.25:
            self.log("SUCCESS: 25%+ CAGR ACHIEVED!")
        else:
            self.log(f"Need {(0.25-cagr)*100:.1f}% more")