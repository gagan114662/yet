from AlgorithmImports import *
import numpy as np

class BreakthroughStrategy(QCAlgorithm):
    """
    Breakthrough Strategy - Agent: gen6_agent3
    Type: BREAKOUT | Asset: BITO | Leverage: 15.6x
    Multi-dimensional evolution for 25%+ CAGR breakthrough
    """
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Primary asset
        self.symbol = self.add_equity("BITO", Resolution.Daily)
        self.symbol.set_leverage(15.612854795340471)
        
        # Initialize tracking for Sharpe calculation
        self.daily_returns = []
        self.last_portfolio_value = self.portfolio.total_portfolio_value
        
        self.sma_fast = self.sma('BITO', 14)
        self.sma_slow = self.sma('BITO', 23)
        self.bb = self.bb('BITO', 20, 2)
        self.atr = self.atr('BITO', 14)
        
        # Risk management
        self.stop_loss = 0.07954897506411791
        self.take_profit = 0.2170716369255179
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
            
        
        # BREAKOUT STRATEGY
        high_20 = max([self.history(self.symbol, 20, Resolution.Daily)['high'].max()])
        current_price = self.securities["BITO"].price
        
        # Breakout above 20-day high
        if current_price > high_20 * 1.01:
            position_size = 1.1405404461014992
            if hasattr(self, 'atr') and self.atr.is_ready:
                # Increase position on high volatility
                if self.atr.current.value > self.atr.window[10]:
                    position_size *= 1.3
            self.set_holdings("BITO", position_size)
            self.entry_price = current_price
        # Exit on breakdown
        elif current_price < self.sma_slow.current.value:
            self.liquidate()
            self.entry_price = 0
        
        # RISK MANAGEMENT
        if self.portfolio.invested and self.entry_price > 0:
            current_price = self.securities[self.symbol].price
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Stop loss
            if pnl_pct < -0.07954897506411791:
                self.liquidate()
                self.log(f"Stop loss triggered: {pnl_pct:.2%}")
            
            # Take profit
            elif pnl_pct > 0.2170716369255179:
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
