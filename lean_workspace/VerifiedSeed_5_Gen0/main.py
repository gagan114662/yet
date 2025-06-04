from AlgorithmImports import *
import numpy as np

class VerifiedStrategy(QCAlgorithm):
    """
    Verified Strategy - Variant 5
    Type: MOMENTUM | Asset: QQQ | Leverage: 25.7x
    Evolution system with verified 15-year backtest
    """
    
    def initialize(self):
        # VERIFIED 15-YEAR PERIOD
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Primary asset with leverage
        self.symbol = self.add_equity("QQQ", Resolution.Daily)
        self.symbol.set_leverage(25.7)
        
        # Tracking for performance calculation
        self.daily_returns = []
        self.last_portfolio_value = self.portfolio.total_portfolio_value
        self.trade_count = 0
        
        # Technical indicators
        self.sma_fast = self.sma('QQQ', 7)
        self.sma_slow = self.sma('QQQ', 60)
        self.rsi = self.rsi('QQQ', 22)
        self.atr = self.atr('QQQ', 14)
        
        # Risk management parameters
        self.stop_loss = 0.04
        self.take_profit = 0.25
        self.position_size = 2.6
        
        # Position tracking
        self.entry_price = 0
        
    def on_data(self, data):
        # Calculate daily returns for Sharpe ratio
        current_value = self.portfolio.total_portfolio_value
        if self.last_portfolio_value > 0:
            daily_return = (current_value - self.last_portfolio_value) / self.last_portfolio_value
            self.daily_returns.append(daily_return)
        self.last_portfolio_value = current_value
        
        if not self._indicators_ready():
            return
        
        # HIGH-FREQUENCY MOMENTUM STRATEGY
        current_price = self.securities["QQQ"].price
        
        # Entry logic with momentum confirmation
        if (self.sma_fast.current.value > self.sma_slow.current.value and 
            not self.portfolio.invested):
            
            # Additional RSI filter for momentum
            if self.rsi.current.value < 75:  # Not severely overbought
                self.set_holdings("QQQ", self.position_size)
                self.entry_price = current_price
                self.trade_count += 1
                self.log(f"ENTRY: Trade #{self.trade_count} at ${current_price:.2f}")
        
        # Exit logic - momentum reversal
        elif (self.sma_fast.current.value < self.sma_slow.current.value * 0.995 and 
              self.portfolio.invested):
            self.liquidate()
            self.trade_count += 1
            self.log(f"EXIT: Trade #{self.trade_count} at ${current_price:.2f}")
            self.entry_price = 0
        
        # RISK MANAGEMENT
        if self.portfolio.invested and self.entry_price > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Stop loss
            if pnl_pct < -self.stop_loss:
                self.liquidate()
                self.trade_count += 1
                self.log(f"STOP LOSS: Trade #{self.trade_count}, Loss: {pnl_pct:.2%}")
                self.entry_price = 0
            
            # Take profit  
            elif pnl_pct > self.take_profit:
                self.liquidate()
                self.trade_count += 1
                self.log(f"TAKE PROFIT: Trade #{self.trade_count}, Gain: {pnl_pct:.2%}")
                self.entry_price = 0
                
        # Additional rebalancing for higher trade frequency
        if self.time.day % 5 == 0 and self.portfolio.invested:
            # Weekly position adjustment based on momentum strength
            momentum_strength = (self.sma_fast.current.value - self.sma_slow.current.value) / self.sma_slow.current.value
            
            if momentum_strength > 0.05:  # Very strong momentum
                adjusted_size = min(self.position_size * 1.1, 3.0)
                if abs(self.portfolio["QQQ"].holdings_value / self.portfolio.total_portfolio_value - adjusted_size) > 0.1:
                    self.set_holdings("QQQ", adjusted_size)
                    self.trade_count += 1
    
    def _indicators_ready(self):
        """Check if all indicators are ready"""
        return (self.sma_fast.is_ready and self.sma_slow.is_ready and 
                self.rsi.is_ready and self.atr.is_ready)
    
    def on_end_of_algorithm(self):
        """Calculate final statistics"""
        years = (self.end_date - self.start_date).days / 365.25
        
        if len(self.daily_returns) > 252:
            returns_array = np.array(self.daily_returns)
            if np.std(returns_array) > 0:
                sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                self.log(f"Final Sharpe Ratio: {sharpe:.3f}")
            else:
                self.log("Sharpe Ratio: 0.000 (no variance)")
        
        trades_per_year = self.trade_count / years if years > 0 else 0
        self.log("FINAL STATISTICS:")
        self.log(f"  Total Trades: {self.trade_count}")
        self.log(f"  Trades/Year: {trades_per_year:.1f}")
        self.log(f"  Portfolio Value: ${self.portfolio.total_portfolio_value:,.2f}")
        
        # Calculate CAGR
        if years > 0:
            total_return = (self.portfolio.total_portfolio_value / 100000) ** (1/years) - 1
            cagr_pct = total_return * 100
            self.log(f"  CAGR: {cagr_pct:.2f}%")