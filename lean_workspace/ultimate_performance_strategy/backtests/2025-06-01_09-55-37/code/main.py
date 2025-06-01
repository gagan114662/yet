from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class UltimatePerformanceStrategy(QCAlgorithm):
    """
    Ultimate Performance Strategy - Aggressive momentum trading
    
    TARGETS:
    - CAGR > 25%
    - Sharpe Ratio > 1.0
    - Max Drawdown < 20%
    - Average Profit > 0.75% per trade
    
    APPROACH:
    - Aggressive momentum trading with leverage
    - Quick profit taking and strict stop losses
    - Focus on SPY/QQQ for liquidity
    """
    
    def Initialize(self):
        # Set backtest period - using recent years for better data
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Enable margin trading
        self.Settings.FreePortfolioValue = 0.0
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # Core instruments - focus on highest liquidity
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Minute).Symbol
        
        # Schedule resolution for minute data
        self.SetSecurityInitializer(lambda x: x.SetDataNormalizationMode(DataNormalizationMode.Adjusted))
        
        # Technical indicators
        self.spy_rsi = self.RSI(self.spy, 14, MovingAverageType.Wilders, Resolution.Hour)
        self.qqq_rsi = self.RSI(self.qqq, 14, MovingAverageType.Wilders, Resolution.Hour)
        
        self.spy_macd = self.MACD(self.spy, 12, 26, 9, MovingAverageType.Exponential, Resolution.Hour)
        self.qqq_macd = self.MACD(self.qqq, 12, 26, 9, MovingAverageType.Exponential, Resolution.Hour)
        
        self.spy_bb = self.BB(self.spy, 20, 2, MovingAverageType.Simple, Resolution.Hour)
        self.qqq_bb = self.BB(self.qqq, 20, 2, MovingAverageType.Simple, Resolution.Hour)
        
        self.spy_atr = self.ATR(self.spy, 14, MovingAverageType.Simple, Resolution.Hour)
        self.qqq_atr = self.ATR(self.qqq, 14, MovingAverageType.Simple, Resolution.Hour)
        
        self.spy_ema_fast = self.EMA(self.spy, 8, Resolution.Hour)
        self.spy_ema_slow = self.EMA(self.spy, 21, Resolution.Hour)
        
        self.qqq_ema_fast = self.EMA(self.qqq, 8, Resolution.Hour)
        self.qqq_ema_slow = self.EMA(self.qqq, 21, Resolution.Hour)
        
        # Additional indicators for better signals
        self.spy_momentum = self.MOM(self.spy, 10, Resolution.Hour)
        self.qqq_momentum = self.MOM(self.qqq, 10, Resolution.Hour)
        
        # Portfolio tracking
        self.portfolio_high = self.Portfolio.TotalPortfolioValue
        self.trades = []
        self.consecutive_losses = 0
        self.last_trade_time = {}
        
        # Aggressive parameters
        self.base_leverage = 3.0          # Start with 3x leverage
        self.max_leverage = 5.0           # Up to 5x when confident
        self.position_size = 0.5          # 50% per position
        self.profit_target = 0.03         # 3% profit target
        self.stop_loss = 0.015            # 1.5% stop loss
        self.max_portfolio_dd = 0.15      # 15% max drawdown
        
        # Track entry prices
        self.entry_prices = {}
        
        # Schedule functions
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.AfterMarketOpen(self.spy, 5), 
                        self.OpeningTrade)
        
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(timedelta(minutes=30)), 
                        self.IntradayTrade)
        
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.BeforeMarketClose(self.spy, 10), 
                        self.ClosingTrade)
        
        # Warm up indicators
        self.SetWarmUp(timedelta(days=30))
        
        self.Debug("Ultimate Performance Strategy Initialized")
    
    def OpeningTrade(self):
        """Execute opening momentum trades"""
        if self.IsWarmingUp:
            return
        
        # Check portfolio risk
        if self.CheckPortfolioRisk():
            return
        
        # SPY momentum trade
        if self.ShouldEnterPosition(self.spy):
            self.ExecuteTrade(self.spy)
        
        # QQQ momentum trade
        if self.ShouldEnterPosition(self.qqq):
            self.ExecuteTrade(self.qqq)
    
    def IntradayTrade(self):
        """Monitor and manage positions"""
        if self.IsWarmingUp:
            return
        
        # Check exits for existing positions
        for symbol in [self.spy, self.qqq]:
            if self.Portfolio[symbol].Invested:
                if self.ShouldExitPosition(symbol):
                    self.ExitPosition(symbol)
        
        # Look for new opportunities if not fully invested
        total_holdings = abs(self.Portfolio[self.spy].HoldingsValue) + abs(self.Portfolio[self.qqq].HoldingsValue)
        portfolio_value = self.Portfolio.TotalPortfolioValue
        
        if total_holdings < portfolio_value * 0.8:  # Less than 80% invested
            if not self.Portfolio[self.spy].Invested and self.ShouldEnterPosition(self.spy):
                self.ExecuteTrade(self.spy)
            
            if not self.Portfolio[self.qqq].Invested and self.ShouldEnterPosition(self.qqq):
                self.ExecuteTrade(self.qqq)
    
    def ClosingTrade(self):
        """End of day position management"""
        if self.IsWarmingUp:
            return
        
        # Take profits on winning positions
        for symbol in [self.spy, self.qqq]:
            if self.Portfolio[symbol].Invested:
                profit_pct = self.Portfolio[symbol].UnrealizedProfitPercent
                
                # Take profits if above 2%
                if profit_pct > 0.02:
                    self.ExitPosition(symbol, "EOD profit taking")
    
    def ShouldEnterPosition(self, symbol):
        """Determine if should enter position"""
        # Check if recently traded
        if symbol in self.last_trade_time:
            if (self.Time - self.last_trade_time[symbol]).total_seconds() < 3600:  # 1 hour cooldown
                return False
        
        # Get indicators
        if symbol == self.spy:
            rsi = self.spy_rsi.Current.Value
            macd = self.spy_macd.Current.Value
            signal = self.spy_macd.Signal.Current.Value
            price = self.Securities[symbol].Price
            upper_band = self.spy_bb.UpperBand.Current.Value
            middle_band = self.spy_bb.MiddleBand.Current.Value
            lower_band = self.spy_bb.LowerBand.Current.Value
            momentum = self.spy_momentum.Current.Value
            ema_fast = self.spy_ema_fast.Current.Value
            ema_slow = self.spy_ema_slow.Current.Value
        else:
            rsi = self.qqq_rsi.Current.Value
            macd = self.qqq_macd.Current.Value
            signal = self.qqq_macd.Signal.Current.Value
            price = self.Securities[symbol].Price
            upper_band = self.qqq_bb.UpperBand.Current.Value
            middle_band = self.qqq_bb.MiddleBand.Current.Value
            lower_band = self.qqq_bb.LowerBand.Current.Value
            momentum = self.qqq_momentum.Current.Value
            ema_fast = self.qqq_ema_fast.Current.Value
            ema_slow = self.qqq_ema_slow.Current.Value
        
        # Strong momentum conditions
        bullish_conditions = 0
        
        if 40 < rsi < 70:  # Momentum but not overbought
            bullish_conditions += 1
        
        if macd > signal and macd > 0:  # MACD bullish and positive
            bullish_conditions += 1
        
        if price > middle_band and price < upper_band:  # Above middle BB
            bullish_conditions += 1
        
        if momentum > 0:  # Positive momentum
            bullish_conditions += 1
        
        if ema_fast > ema_slow:  # EMA crossover
            bullish_conditions += 1
        
        # Mean reversion opportunity
        if rsi < 30 and price < lower_band:
            return True  # Oversold bounce
        
        # Strong momentum
        if bullish_conditions >= 3:
            return True
        
        return False
    
    def ShouldExitPosition(self, symbol):
        """Determine if should exit position"""
        if not self.Portfolio[symbol].Invested:
            return False
        
        profit_pct = self.Portfolio[symbol].UnrealizedProfitPercent
        
        # Hit profit target
        if profit_pct >= self.profit_target:
            return True
        
        # Hit stop loss
        if profit_pct <= -self.stop_loss:
            return True
        
        # Get indicators
        if symbol == self.spy:
            rsi = self.spy_rsi.Current.Value
            macd = self.spy_macd.Current.Value
            signal = self.spy_macd.Signal.Current.Value
        else:
            rsi = self.qqq_rsi.Current.Value
            macd = self.qqq_macd.Current.Value
            signal = self.qqq_macd.Signal.Current.Value
        
        # Exit on extreme overbought
        if rsi > 80:
            return True
        
        # Exit on MACD bearish cross with profit
        if macd < signal and profit_pct > 0.005:  # 0.5% profit
            return True
        
        return False
    
    def ExecuteTrade(self, symbol):
        """Execute trade with dynamic sizing"""
        # Calculate position size based on volatility
        atr = self.spy_atr.Current.Value if symbol == self.spy else self.qqq_atr.Current.Value
        price = self.Securities[symbol].Price
        
        if atr == 0 or price == 0:
            return
        
        volatility = atr / price
        
        # Dynamic leverage based on conditions
        leverage = self.base_leverage
        
        # Increase leverage if we've had consecutive wins
        if len(self.trades) > 0 and self.consecutive_losses == 0:
            leverage = min(self.max_leverage, leverage * 1.2)
        
        # Reduce leverage after losses
        if self.consecutive_losses > 1:
            leverage = max(2.0, leverage * 0.8)
        
        # Adjust for volatility
        if volatility > 0.02:  # High volatility
            leverage *= 0.7
        
        # Calculate final position size
        position_size = self.position_size * leverage
        
        # Ensure we don't over-leverage
        max_position = min(position_size, 0.8)  # Max 80% per position
        
        # Execute trade
        self.SetHoldings(symbol, max_position)
        
        # Track entry
        self.entry_prices[symbol] = price
        self.last_trade_time[symbol] = self.Time
        
        self.Debug(f"Entered {symbol} - Size: {max_position:.2f}, Leverage: {leverage:.1f}x, Price: ${price:.2f}")
    
    def ExitPosition(self, symbol, reason=""):
        """Exit position and track results"""
        if not self.Portfolio[symbol].Invested:
            return
        
        # Calculate profit
        entry_price = self.entry_prices.get(symbol, self.Portfolio[symbol].AveragePrice)
        exit_price = self.Securities[symbol].Price
        profit_pct = (exit_price - entry_price) / entry_price
        
        # Track trade
        self.trades.append({
            'symbol': symbol,
            'entry': entry_price,
            'exit': exit_price,
            'profit': profit_pct,
            'time': self.Time
        })
        
        # Update consecutive losses
        if profit_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Exit position
        self.Liquidate(symbol)
        
        # Clean up
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]
        
        self.Debug(f"Exited {symbol} - Profit: {profit_pct:.2%} {reason}")
    
    def CheckPortfolioRisk(self):
        """Check if portfolio is at risk limit"""
        current_value = self.Portfolio.TotalPortfolioValue
        
        # Update portfolio high
        if current_value > self.portfolio_high:
            self.portfolio_high = current_value
        
        # Calculate drawdown
        drawdown = (self.portfolio_high - current_value) / self.portfolio_high
        
        if drawdown > self.max_portfolio_dd:
            self.Debug(f"Portfolio risk limit hit: {drawdown:.2%} drawdown")
            self.Liquidate()
            return True
        
        return False
    
    def OnData(self, data):
        """Process incoming data"""
        pass  # All logic in scheduled functions
    
    def OnEndOfAlgorithm(self):
        """Final performance report"""
        if len(self.trades) == 0:
            self.Debug("No trades executed")
            return
        
        # Calculate metrics
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades)
        avg_profit = sum(t['profit'] for t in self.trades) / len(self.trades)
        
        # Final portfolio value
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - 100000) / 100000
        
        self.Debug(f"\n=== FINAL PERFORMANCE ===")
        self.Debug(f"Total Trades: {len(self.trades)}")
        self.Debug(f"Win Rate: {win_rate:.2%}")
        self.Debug(f"Average Profit: {avg_profit:.2%}")
        self.Debug(f"Total Return: {total_return:.2%}")
        self.Debug(f"Final Value: ${final_value:,.2f}")