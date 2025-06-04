from AlgorithmImports import *
import numpy as np

class BreakthroughStrategy(QCAlgorithm):
    """
    Breakthrough Strategy - Agent: gen16_agent1
    Type: HYBRID | Asset: BITO | Leverage: 5.8x
    Multi-dimensional evolution for 25%+ CAGR breakthrough
    """
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Primary asset
        self.symbol = self.add_equity("BITO", Resolution.Daily)
        self.symbol.set_leverage(5.7546794395759795)
        
        # Initialize tracking for Sharpe calculation
        self.daily_returns = []
        self.last_portfolio_value = self.portfolio.total_portfolio_value
        
        self.sma_fast = self.sma('BITO', 9)
        self.sma_slow = self.sma('BITO', 29)
        self.macd = self.macd('BITO', 12, 26, 9)
        self.bb = self.bb('BITO', 20, 2)
        self.atr = self.atr('BITO', 14)
        self.vix = self.add_index('VIX', Resolution.Daily)
        
        # Risk management
        self.stop_loss = 0.09423014137350896
        self.take_profit = 0.1846836578858127
        self.last_trade_time = self.time
        
        # Position tracking
        self.entry_price = 0
        self.position_value = 0
        
    def on_data(self, data):
        # Calculate daily returns for Sharpe ratio
        current_value = self.portfolio.total_portfolio_value
        if self.last_portfolio_value > 0:
            daily_return = (current_value - self.last_portfolio_value) / self.last_portfolio_value
            self.daily_returns.append(daily_return)
        self.last_portfolio_value = current_value
        
        if not self._indicators_ready():
            return
            
        
        # HYBRID STRATEGY (Momentum + Mean Reversion)
        current_price = self.securities["BITO"].price
        momentum_signal = self.sma_fast.current.value > self.sma_slow.current.value
        
        if momentum_signal:
            # Momentum phase
            position_size = 1.3116224781136203 * 0.8
            if hasattr(self, 'macd') and self.macd.is_ready:
                if self.macd.current.value > 0:
                    position_size *= 1.4
            self.set_holdings("BITO", position_size)
            self.entry_price = current_price
        else:
            # Mean reversion phase
            sma_mid = (self.sma_fast.current.value + self.sma_slow.current.value) / 2
            if current_price < sma_mid * 0.97:
                self.set_holdings("BITO", 1.3116224781136203 * 0.6)
                self.entry_price = current_price
            else:
                self.liquidate()
                self.entry_price = 0
        
        # RISK MANAGEMENT
        if self.portfolio.invested and self.entry_price > 0:
            current_price = self.securities[self.symbol].price
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Stop loss
            if pnl_pct < -0.09423014137350896:
                self.liquidate()
                self.log(f"Stop loss triggered: {pnl_pct:.2%}")
            
            # Take profit
            elif pnl_pct > 0.1846836578858127:
                self.liquidate()
                self.log(f"Take profit triggered: {pnl_pct:.2%}")
    
    def _indicators_ready(self):
        """Check if all indicators are ready"""
        ready = True
        if hasattr(self, 'sma_fast'):
            ready &= self.sma_fast.is_ready
        if hasattr(self, 'sma_slow'):
            ready &= self.sma_slow.is_ready
        if hasattr(self, 'rsi'):
            ready &= self.rsi.is_ready
        if hasattr(self, 'macd'):
            ready &= self.macd.is_ready
        return ready
    
    def on_end_of_algorithm(self):
        """Calculate final Sharpe ratio"""
        if len(self.daily_returns) > 252:  # At least 1 year of data
            returns_array = np.array(self.daily_returns)
            if np.std(returns_array) > 0:
                sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                self.log(f"Calculated Sharpe Ratio: {sharpe:.3f}")
            else:
                self.log("Sharpe Ratio: 0.000 (no variance)")
        else:
            self.log("Insufficient data for Sharpe calculation")
