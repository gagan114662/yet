# region imports
from AlgorithmImports import *
# endregion

class EvolutionMomentum(QCAlgorithm):
    
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Strategy parameters from evolution - optimized for 25% CAGR target
        self.leverage = 2.8  # Increased from 2.0
        self.position_size = 0.25  # Increased from 0.2
        self.stop_loss_pct = 0.06  # Tightened from 0.08
        
        # Add SPY with minute resolution for better entries
        self.spy = self.add_equity("SPY", Resolution.MINUTE).symbol
        
        # Technical indicators
        self.rsi = self.RSI(self.spy, 12, MovingAverageType.SIMPLE, Resolution.DAILY)
        self.macd = self.MACD(self.spy, 12, 26, 9, MovingAverageType.EXPONENTIAL, Resolution.DAILY)
        self.bb = self.BB(self.spy, 20, 2.0, MovingAverageType.SIMPLE, Resolution.DAILY)
        self.atr = self.ATR(self.spy, 14, MovingAverageType.SIMPLE, Resolution.DAILY)
        
        # Risk management
        self.stop_loss_price = None
        self.trailing_stop_pct = 0.04  # 4% trailing stop
        self.highest_price = None
        
        # Set leverage
        self.securities[self.spy].set_leverage(self.leverage)
        
        # Schedule rebalancing
        self.schedule.on(self.date_rules.every_day(self.spy), 
                        self.time_rules.after_market_open(self.spy, 30), 
                        self.check_signals)
        
        # Schedule end of day check
        self.schedule.on(self.date_rules.every_day(self.spy), 
                        self.time_rules.before_market_close(self.spy, 15), 
                        self.end_of_day_check)
        
        # Track performance
        self.benchmark_start = None
        self.trades_count = 0
        self.winning_trades = 0
        
        # Warm up indicators
        self.set_warm_up(50, Resolution.DAILY)
    
    def check_signals(self):
        """Check for entry/exit signals"""
        if self.is_warming_up:
            return
            
        if not self.rsi.is_ready or not self.macd.is_ready or not self.bb.is_ready:
            return
        
        current_price = self.securities[self.spy].price
        
        # Enhanced momentum signals
        rsi_bullish = self.rsi.current.value > 55 and self.rsi.current.value < 80
        macd_bullish = self.macd.current.value > self.macd.signal.current.value
        macd_momentum = self.macd.current.value > 0
        price_above_bb_middle = current_price > self.bb.middle_band.current.value
        volatility_favorable = self.atr.current.value > self.atr.current.value * 0.8
        
        # Position management
        holdings = self.portfolio[self.spy].quantity
        
        if holdings == 0:
            # Entry conditions - multiple confirmations
            if (rsi_bullish and macd_bullish and macd_momentum and 
                price_above_bb_middle and volatility_favorable):
                
                # Calculate position size with Kelly criterion adjustment
                portfolio_value = self.portfolio.total_portfolio_value
                win_rate = 0.6 if self.trades_count > 0 else 0.55  # Estimate
                kelly_fraction = (win_rate - (1 - win_rate)) / 1  # Simplified Kelly
                adjusted_position_size = min(self.position_size, kelly_fraction * 0.25)
                
                position_value = portfolio_value * adjusted_position_size * self.leverage
                quantity = int(position_value / current_price)
                
                if quantity > 0:
                    self.market_order(self.spy, quantity)
                    self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                    self.highest_price = current_price
                    self.debug(f"BUY: {quantity} shares at ${current_price:.2f}, Stop: ${self.stop_loss_price:.2f}")
                    self.trades_count += 1
        
        elif holdings > 0:
            # Update trailing stop
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.stop_loss_price = max(self.stop_loss_price, 
                                         self.highest_price * (1 - self.trailing_stop_pct))
            
            # Exit conditions
            exit_signal = False
            exit_reason = ""
            
            # Stop loss hit
            if current_price <= self.stop_loss_price:
                exit_signal = True
                exit_reason = "STOP_LOSS"
            
            # Technical exit signals
            elif (self.rsi.current.value > 75 or  # Overbought
                  self.macd.current.value < self.macd.signal.current.value or  # MACD cross
                  current_price > self.bb.upper_band.current.value):  # Above BB upper
                exit_signal = True
                exit_reason = "TECHNICAL"
            
            if exit_signal:
                self.liquidate(self.spy)
                
                # Track trade outcome
                entry_value = holdings * (self.highest_price - (self.highest_price * self.trailing_stop_pct))
                exit_value = holdings * current_price
                if exit_value > entry_value:
                    self.winning_trades += 1
                
                self.debug(f"SELL: {holdings} shares at ${current_price:.2f} ({exit_reason})")
                self.stop_loss_price = None
                self.highest_price = None
    
    def end_of_day_check(self):
        """End of day position check"""
        if self.portfolio[self.spy].quantity > 0:
            current_price = self.securities[self.spy].price
            self.debug(f"EOD: Position open at ${current_price:.2f}, Stop: ${self.stop_loss_price:.2f}")
    
    def on_data(self, data: Slice):
        """Process incoming data"""
        # Track initial benchmark
        if self.benchmark_start is None and self.spy in data:
            self.benchmark_start = data[self.spy].close
        
        # Additional intraday checks for risk management
        if self.portfolio[self.spy].quantity > 0 and self.spy in data:
            current_price = data[self.spy].close
            if current_price <= self.stop_loss_price:
                self.liquidate(self.spy)
                self.debug(f"INTRADAY STOP: Exited at ${current_price:.2f}")
    
    def on_end_of_algorithm(self):
        """Final performance logging"""
        final_value = self.portfolio.total_portfolio_value
        
        self.debug(f"=== FINAL PERFORMANCE ===")
        self.debug(f"Final Portfolio Value: ${final_value:.2f}")
        self.debug(f"Total Return: {(final_value - 100000) / 100000:.2%}")
        
        if self.benchmark_start and self.securities[self.spy].close > 0:
            spy_return = (self.securities[self.spy].close - self.benchmark_start) / self.benchmark_start
            self.debug(f"SPY Return: {spy_return:.2%}")
            
            # Calculate alpha
            strategy_return = (final_value - 100000) / 100000
            alpha = strategy_return - spy_return
            self.debug(f"Alpha: {alpha:.2%}")
        
        if self.trades_count > 0:
            win_rate = self.winning_trades / self.trades_count
            self.debug(f"Total Trades: {self.trades_count}")
            self.debug(f"Win Rate: {win_rate:.2%}")
            
        # Calculate annualized return
        years = (self.end_date - self.start_date).days / 365.25
        annual_return = ((final_value / 100000) ** (1/years)) - 1
        self.debug(f"Annualized Return: {annual_return:.2%}")