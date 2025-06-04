from AlgorithmImports import *
import numpy as np

class BreakthroughStrategy(QCAlgorithm):
    """
    Breakthrough Strategy - Agent: gen11_agent3
    Type: MOMENTUM | Asset: UPRO | Leverage: 10.5x
    Multi-dimensional evolution for 25%+ CAGR breakthrough
    """
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Primary asset
        self.symbol = self.add_equity("UPRO", Resolution.Daily)
        self.symbol.set_leverage(10.481412005803934)
        
        # Initialize tracking for Sharpe calculation
        self.daily_returns = []
        self.last_portfolio_value = self.portfolio.total_portfolio_value
        
        self.sma_fast = self.sma('UPRO', 7)
        self.sma_slow = self.sma('UPRO', 32)
        self.atr = self.atr('UPRO', 14)
        
        # Risk management
        self.stop_loss = 0.04152732755827261
        self.take_profit = 0.05610022248374833
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
            
        
        # MOMENTUM STRATEGY
        if self.sma_fast.current.value > self.sma_slow.current.value:
            # Strong momentum signal
            position_size = 2.4308207919700218
            if hasattr(self, 'rsi') and self.rsi.is_ready:
                if self.rsi.current.value < 70:  # Not overbought
                    position_size *= 1.2
            self.set_holdings("UPRO", position_size)
            self.entry_price = self.securities["UPRO"].price
        elif self.sma_fast.current.value < self.sma_slow.current.value * 0.98:
            # Exit on momentum reversal
            self.liquidate()
            self.entry_price = 0
        
        # RISK MANAGEMENT
        if self.portfolio.invested and self.entry_price > 0:
            current_price = self.securities[self.symbol].price
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Stop loss
            if pnl_pct < -0.04152732755827261:
                self.liquidate()
                self.log(f"Stop loss triggered: {pnl_pct:.2%}")
            
            # Take profit
            elif pnl_pct > 0.05610022248374833:
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
