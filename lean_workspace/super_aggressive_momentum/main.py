from AlgorithmImports import *
import numpy as np

class SuperAggressiveMomentum(QCAlgorithm):
    """
    Super Aggressive Momentum Strategy
    Based on the best performing strategy but enhanced for higher returns
    
    TARGETS:
    - CAGR > 25%
    - Sharpe Ratio > 1.0
    - Max Drawdown < 20%
    - Average Profit > 0.75% per trade
    """
    
    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Enable margin for leverage
        self.Settings.FreePortfolioValue = 0.0
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # Trade SPY and QQQ for liquidity and reliable data
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
        
        # Fast indicators for SPY
        self.spy_mom_ultra_fast = self.MOMP(self.spy, 3)
        self.spy_mom_fast = self.MOMP(self.spy, 5)
        self.spy_mom_slow = self.MOMP(self.spy, 15)
        self.spy_rsi = self.RSI(self.spy, 14)
        self.spy_rsi_fast = self.RSI(self.spy, 7)
        self.spy_bb = self.BB(self.spy, 20, 2)
        self.spy_ema_fast = self.EMA(self.spy, 8)
        self.spy_ema_slow = self.EMA(self.spy, 21)
        self.spy_atr = self.ATR(self.spy, 14)
        self.spy_adx = self.ADX(self.spy, 14)
        
        # Fast indicators for QQQ
        self.qqq_mom_ultra_fast = self.MOMP(self.qqq, 3)
        self.qqq_mom_fast = self.MOMP(self.qqq, 5)
        self.qqq_mom_slow = self.MOMP(self.qqq, 15)
        self.qqq_rsi = self.RSI(self.qqq, 14)
        self.qqq_rsi_fast = self.RSI(self.qqq, 7)
        self.qqq_bb = self.BB(self.qqq, 20, 2)
        self.qqq_ema_fast = self.EMA(self.qqq, 8)
        self.qqq_ema_slow = self.EMA(self.qqq, 21)
        self.qqq_atr = self.ATR(self.qqq, 14)
        self.qqq_adx = self.ADX(self.qqq, 14)
        
        # Position tracking
        self.trade_count = 0
        self.win_count = 0
        self.total_profit = 0
        self.entry_prices = {}
        self.position_times = {}
        
        # Risk parameters
        self.base_leverage = 2.0
        self.max_leverage = 4.0
        self.position_size = 0.95  # 95% allocation per trade
        self.profit_target = 0.025  # 2.5% profit target
        self.stop_loss = 0.01      # 1% stop loss
        self.trailing_stop = 0.015  # 1.5% trailing stop
        
        # Portfolio tracking
        self.portfolio_high = self.Portfolio.TotalPortfolioValue
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        self.SetWarmUp(30)
        
    def OnData(self, data):
        if self.IsWarmingUp:
            return
            
        # Check portfolio risk
        if self.CheckPortfolioRisk():
            return
            
        # Trade SPY
        self.TradeSymbol(self.spy, data)
        
        # Trade QQQ
        self.TradeSymbol(self.qqq, data)
        
    def TradeSymbol(self, symbol, data):
        """Execute trades for a specific symbol"""
        if symbol == self.spy:
            # SPY indicators
            if not (self.spy_mom_fast.IsReady and self.spy_rsi.IsReady and self.spy_bb.IsReady):
                return
                
            ultra_fast_mom = self.spy_mom_ultra_fast.Current.Value
            fast_mom = self.spy_mom_fast.Current.Value
            slow_mom = self.spy_mom_slow.Current.Value
            rsi = self.spy_rsi.Current.Value
            rsi_fast = self.spy_rsi_fast.Current.Value
            bb_upper = self.spy_bb.UpperBand.Current.Value
            bb_lower = self.spy_bb.LowerBand.Current.Value
            bb_middle = self.spy_bb.MiddleBand.Current.Value
            ema_fast = self.spy_ema_fast.Current.Value
            ema_slow = self.spy_ema_slow.Current.Value
            atr = self.spy_atr.Current.Value
            adx = self.spy_adx.Current.Value
        else:
            # QQQ indicators
            if not (self.qqq_mom_fast.IsReady and self.qqq_rsi.IsReady and self.qqq_bb.IsReady):
                return
                
            ultra_fast_mom = self.qqq_mom_ultra_fast.Current.Value
            fast_mom = self.qqq_mom_fast.Current.Value
            slow_mom = self.qqq_mom_slow.Current.Value
            rsi = self.qqq_rsi.Current.Value
            rsi_fast = self.qqq_rsi_fast.Current.Value
            bb_upper = self.qqq_bb.UpperBand.Current.Value
            bb_lower = self.qqq_bb.LowerBand.Current.Value
            bb_middle = self.qqq_bb.MiddleBand.Current.Value
            ema_fast = self.qqq_ema_fast.Current.Value
            ema_slow = self.qqq_ema_slow.Current.Value
            atr = self.qqq_atr.Current.Value
            adx = self.qqq_adx.Current.Value
        
        price = self.Securities[symbol].Price
        holdings = self.Portfolio[symbol].Quantity
        
        # Exit logic first
        if holdings != 0:
            # Check for exit conditions
            entry_price = self.entry_prices.get(symbol, self.Portfolio[symbol].AveragePrice)
            profit_pct = (price - entry_price) / entry_price if holdings > 0 else (entry_price - price) / entry_price
            
            exit_signal = False
            
            # Profit target hit
            if profit_pct >= self.profit_target:
                exit_signal = True
                self.win_count += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                
            # Stop loss hit
            elif profit_pct <= -self.stop_loss:
                exit_signal = True
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                
            # Trailing stop
            elif profit_pct > 0.01 and profit_pct < self.trailing_stop:
                if holdings > 0 and (ultra_fast_mom < -0.003 or rsi_fast > 80):
                    exit_signal = True
                elif holdings < 0 and (ultra_fast_mom > 0.003 or rsi_fast < 20):
                    exit_signal = True
                    
            # Momentum reversal exit
            if holdings > 0 and (fast_mom < -0.005 or rsi > 85):
                exit_signal = True
            elif holdings < 0 and (fast_mom > 0.005 or rsi < 15):
                exit_signal = True
                
            if exit_signal:
                self.Liquidate(symbol)
                self.total_profit += profit_pct
                self.trade_count += 1
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                if symbol in self.position_times:
                    del self.position_times[symbol]
                return
        
        # Entry logic - multiple strategies
        leverage = self.CalculateLeverage()
        position_size = self.position_size * leverage
        
        # 1. STRONG MOMENTUM ENTRY
        if holdings == 0:
            strong_long = (
                fast_mom > slow_mom and
                fast_mom > 0.003 and
                ultra_fast_mom > 0.002 and
                rsi < 75 and
                rsi_fast > 50 and
                ema_fast > ema_slow and
                price > bb_middle and
                adx > 20  # Trending market
            )
            
            strong_short = (
                fast_mom < slow_mom and
                fast_mom < -0.003 and
                ultra_fast_mom < -0.002 and
                rsi > 25 and
                rsi_fast < 50 and
                ema_fast < ema_slow and
                price < bb_middle and
                adx > 20  # Trending market
            )
            
            if strong_long:
                self.SetHoldings(symbol, position_size)
                self.entry_prices[symbol] = price
                self.position_times[symbol] = self.Time
                
            elif strong_short:
                self.SetHoldings(symbol, -position_size)
                self.entry_prices[symbol] = price
                self.position_times[symbol] = self.Time
                
        # 2. MEAN REVERSION TRADES
        if holdings == 0:
            # Oversold bounce
            if price < bb_lower and rsi < 25 and rsi_fast < 20:
                self.SetHoldings(symbol, position_size * 0.7)  # Smaller size for mean reversion
                self.entry_prices[symbol] = price
                self.position_times[symbol] = self.Time
                
            # Overbought reversal
            elif price > bb_upper and rsi > 75 and rsi_fast > 80:
                self.SetHoldings(symbol, -position_size * 0.7)
                self.entry_prices[symbol] = price
                self.position_times[symbol] = self.Time
                
        # 3. BREAKOUT TRADES
        if holdings == 0:
            # Bullish breakout
            if (price > bb_upper * 1.01 and 
                fast_mom > 0.005 and 
                rsi > 50 and rsi < 70 and
                adx > 25):
                self.SetHoldings(symbol, position_size)
                self.entry_prices[symbol] = price
                self.position_times[symbol] = self.Time
                
            # Bearish breakdown
            elif (price < bb_lower * 0.99 and 
                  fast_mom < -0.005 and 
                  rsi < 50 and rsi > 30 and
                  adx > 25):
                self.SetHoldings(symbol, -position_size)
                self.entry_prices[symbol] = price
                self.position_times[symbol] = self.Time
    
    def CalculateLeverage(self):
        """Calculate dynamic leverage based on performance"""
        base = self.base_leverage
        
        # Increase leverage on winning streaks
        if self.consecutive_wins >= 3:
            base = min(self.max_leverage, base * 1.5)
        elif self.consecutive_wins >= 2:
            base = min(self.max_leverage, base * 1.2)
            
        # Decrease leverage on losing streaks
        if self.consecutive_losses >= 2:
            base = max(1.5, base * 0.7)
            
        # Adjust for portfolio performance
        current_value = self.Portfolio.TotalPortfolioValue
        if current_value > self.portfolio_high * 1.05:  # 5% above high
            base = min(self.max_leverage, base * 1.1)
            
        return base
    
    def CheckPortfolioRisk(self):
        """Check portfolio risk limits"""
        current_value = self.Portfolio.TotalPortfolioValue
        
        # Update portfolio high
        if current_value > self.portfolio_high:
            self.portfolio_high = current_value
            
        # Calculate drawdown
        drawdown = (self.portfolio_high - current_value) / self.portfolio_high
        
        # Risk management
        if drawdown > 0.15:  # 15% drawdown
            self.Liquidate()
            self.Debug(f"Portfolio protection triggered at {drawdown:.2%} drawdown")
            return True
            
        # Daily loss limit
        start_value = 100000  # Initial capital
        daily_loss = (start_value - current_value) / start_value
        if daily_loss > 0.05:  # 5% daily loss limit
            self.Liquidate()
            self.Debug(f"Daily loss limit triggered at {daily_loss:.2%} loss")
            return True
            
        return False
    
    def OnEndOfAlgorithm(self):
        """Performance summary"""
        if self.trade_count == 0:
            self.Debug("No trades executed")
            return
            
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0
        avg_profit = self.total_profit / self.trade_count if self.trade_count > 0 else 0
        
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - 100000) / 100000
        years = (self.EndDate - self.StartDate).days / 365.25
        cagr = (final_value / 100000) ** (1 / years) - 1 if years > 0 else 0
        
        self.Debug(f"\n=== PERFORMANCE SUMMARY ===")
        self.Debug(f"Total Trades: {self.trade_count}")
        self.Debug(f"Win Rate: {win_rate:.2%}")
        self.Debug(f"Average Profit per Trade: {avg_profit:.2%}")
        self.Debug(f"Total Return: {total_return:.2%}")
        self.Debug(f"CAGR: {cagr:.2%}")
        self.Debug(f"Final Portfolio Value: ${final_value:,.2f}")