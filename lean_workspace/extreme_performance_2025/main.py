from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class ExtremePerformance2025(QCAlgorithm):
    """
    Extreme Performance Strategy 2025
    
    TARGETS:
    - CAGR > 25%
    - Sharpe Ratio > 1.0  
    - Max Drawdown < 20%
    - Average Profit > 0.75% per trade
    
    APPROACH:
    - Aggressive momentum with 3x leveraged ETFs
    - Mean reversion on extreme moves
    - Volatility-based position sizing
    - Strict risk management
    """
    
    def Initialize(self):
        # Set backtest period
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Enable high leverage for aggressive returns
        self.Settings.FreePortfolioValue = 0.0
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # Core trading instruments - focus on high liquidity
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Hour).Symbol
        
        # 3x Leveraged ETFs for aggressive gains
        self.tqqq = self.AddEquity("TQQQ", Resolution.Hour).Symbol  # 3x Nasdaq
        self.upro = self.AddEquity("UPRO", Resolution.Hour).Symbol  # 3x S&P 500
        self.tna = self.AddEquity("TNA", Resolution.Hour).Symbol    # 3x Small Cap
        
        # Inverse ETFs for shorting
        self.sqqq = self.AddEquity("SQQQ", Resolution.Hour).Symbol  # 3x Inverse Nasdaq
        self.spxu = self.AddEquity("SPXU", Resolution.Hour).Symbol  # 3x Inverse S&P
        
        # All symbols for tracking
        self.symbols = [self.spy, self.qqq, self.iwm, self.tqqq, self.upro, self.tna, self.sqqq, self.spxu]
        
        # Technical indicators
        self.indicators = {}
        for symbol in self.symbols:
            self.indicators[symbol] = {
                'rsi': self.RSI(symbol, 14, MovingAverageType.Wilders, Resolution.Hour),
                'rsi_fast': self.RSI(symbol, 5, MovingAverageType.Wilders, Resolution.Hour),
                'macd': self.MACD(symbol, 12, 26, 9, MovingAverageType.Exponential, Resolution.Hour),
                'bb': self.BB(symbol, 20, 2, MovingAverageType.Simple, Resolution.Hour),
                'atr': self.ATR(symbol, 14, MovingAverageType.Simple, Resolution.Hour),
                'ema_fast': self.EMA(symbol, 8, Resolution.Hour),
                'ema_slow': self.EMA(symbol, 21, Resolution.Hour),
                'momentum': self.MOM(symbol, 10, Resolution.Hour),
                'roc': self.ROC(symbol, 10, Resolution.Hour)
            }
        
        # Portfolio tracking
        self.portfolio_high = self.Portfolio.TotalPortfolioValue
        self.max_drawdown = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.total_profit = 0.0
        self.last_trade_time = self.Time
        
        # Risk parameters
        self.max_portfolio_risk = 0.15  # 15% max portfolio loss
        self.position_size = 0.25       # 25% per position base
        self.profit_target = 0.08       # 8% profit target per trade
        self.stop_loss = 0.04           # 4% stop loss per trade
        self.leverage_factor = 2.0      # Base leverage factor
        
        # Schedule functions
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(timedelta(hours=1)), 
                        self.HourlyTrading)
        
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.At(9, 35), 
                        self.OpeningTrades)
        
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.At(15, 45), 
                        self.ClosingTrades)
        
        self.Schedule.On(self.DateRules.WeekStart(), 
                        self.TimeRules.At(9, 30), 
                        self.WeeklyRebalance)
        
        # Warm up indicators
        self.SetWarmUp(timedelta(days=30))
        
        self.Debug("Extreme Performance 2025 Strategy Initialized")
    
    def HourlyTrading(self):
        """Execute hourly momentum and mean reversion trades"""
        if self.IsWarmingUp:
            return
        
        # Check portfolio risk first
        if self.CheckPortfolioRisk():
            return
        
        # Momentum trades on leveraged ETFs
        for symbol in [self.tqqq, self.upro, self.tna]:
            if self.Portfolio[symbol].Invested:
                # Check exit conditions
                if self.ShouldExitPosition(symbol):
                    self.Liquidate(symbol, "Exit signal")
            else:
                # Check entry conditions
                if self.ShouldEnterLong(symbol):
                    size = self.CalculatePositionSize(symbol)
                    if size > 0:
                        self.SetHoldings(symbol, size, False, f"Momentum long entry")
                        self.trade_count += 1
        
        # Mean reversion trades on inverse ETFs
        for symbol in [self.sqqq, self.spxu]:
            if self.Portfolio[symbol].Invested:
                if self.ShouldExitPosition(symbol):
                    self.Liquidate(symbol, "Exit signal")
            else:
                if self.ShouldEnterMeanReversion(symbol):
                    size = self.CalculatePositionSize(symbol) * 0.5  # Half size for inverse
                    if size > 0:
                        self.SetHoldings(symbol, size, False, f"Mean reversion entry")
                        self.trade_count += 1
    
    def OpeningTrades(self):
        """Execute opening momentum trades"""
        if self.IsWarmingUp:
            return
        
        # Strong opening momentum
        for symbol in [self.spy, self.qqq, self.iwm]:
            indicators = self.indicators[symbol]
            
            # Check for strong opening gap
            if indicators['momentum'].Current.Value > 2:
                leverage_symbol = self.GetLeveragedSymbol(symbol)
                if not self.Portfolio[leverage_symbol].Invested:
                    size = self.position_size * self.leverage_factor
                    self.SetHoldings(leverage_symbol, size, False, "Opening momentum")
                    self.trade_count += 1
    
    def ClosingTrades(self):
        """Execute closing trades and profit taking"""
        if self.IsWarmingUp:
            return
        
        # Take profits on winning positions
        for symbol in self.symbols:
            if self.Portfolio[symbol].Invested:
                profit_pct = self.Portfolio[symbol].UnrealizedProfitPercent
                
                # Take profits above target
                if profit_pct > self.profit_target:
                    self.Liquidate(symbol, f"Profit target hit: {profit_pct:.2%}")
                    self.win_count += 1
                    self.total_profit += profit_pct
                
                # Cut losses
                elif profit_pct < -self.stop_loss:
                    self.Liquidate(symbol, f"Stop loss hit: {profit_pct:.2%}")
                    self.total_profit += profit_pct
    
    def WeeklyRebalance(self):
        """Weekly portfolio rebalance and risk adjustment"""
        if self.IsWarmingUp:
            return
        
        # Calculate performance metrics
        current_value = self.Portfolio.TotalPortfolioValue
        
        # Update max drawdown
        if current_value > self.portfolio_high:
            self.portfolio_high = current_value
        
        drawdown = (self.portfolio_high - current_value) / self.portfolio_high
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Adjust leverage based on performance
        if drawdown < 0.05:  # Less than 5% drawdown
            self.leverage_factor = min(3.0, self.leverage_factor * 1.1)
        elif drawdown > 0.10:  # More than 10% drawdown
            self.leverage_factor = max(1.5, self.leverage_factor * 0.9)
        
        # Log performance
        win_rate = self.win_count / max(1, self.trade_count)
        avg_profit = self.total_profit / max(1, self.trade_count)
        
        self.Debug(f"Weekly Stats - Trades: {self.trade_count}, Win Rate: {win_rate:.2%}, " + 
                  f"Avg Profit: {avg_profit:.2%}, Drawdown: {drawdown:.2%}, Leverage: {self.leverage_factor:.1f}")
    
    def ShouldEnterLong(self, symbol):
        """Determine if should enter long position"""
        indicators = self.indicators[symbol]
        
        # Wait for indicators to be ready
        if not all(ind.IsReady for ind in indicators.values()):
            return False
        
        # Strong momentum conditions
        rsi = indicators['rsi'].Current.Value
        rsi_fast = indicators['rsi_fast'].Current.Value
        macd = indicators['macd'].Current.Value
        signal = indicators['macd'].Signal.Current.Value
        momentum = indicators['momentum'].Current.Value
        
        # Multiple confirmation signals
        conditions = [
            rsi > 50 and rsi < 80,              # Not overbought
            rsi_fast > 60,                      # Short-term momentum
            macd > signal,                      # MACD bullish
            momentum > 0,                       # Positive momentum
            indicators['ema_fast'].Current.Value > indicators['ema_slow'].Current.Value,  # EMA cross
            self.Securities[symbol].Price > indicators['bb'].MiddleBand.Current.Value     # Above BB middle
        ]
        
        return sum(conditions) >= 4  # Need 4 out of 6 signals
    
    def ShouldEnterMeanReversion(self, symbol):
        """Determine if should enter mean reversion position"""
        indicators = self.indicators[symbol]
        
        if not all(ind.IsReady for ind in indicators.values()):
            return False
        
        # Oversold conditions for inverse ETFs
        rsi = indicators['rsi'].Current.Value
        price = self.Securities[symbol].Price
        lower_band = indicators['bb'].LowerBand.Current.Value
        
        conditions = [
            rsi < 30,                           # Oversold
            price < lower_band,                 # Below lower band
            indicators['roc'].Current.Value < -5  # Strong negative rate of change
        ]
        
        return all(conditions)
    
    def ShouldExitPosition(self, symbol):
        """Determine if should exit position"""
        if not self.Portfolio[symbol].Invested:
            return False
        
        indicators = self.indicators[symbol]
        profit_pct = self.Portfolio[symbol].UnrealizedProfitPercent
        
        # Exit conditions
        exit_conditions = [
            profit_pct > self.profit_target,                    # Hit profit target
            profit_pct < -self.stop_loss,                       # Hit stop loss
            indicators['rsi'].Current.Value > 85,                # Extremely overbought
            indicators['rsi'].Current.Value < 15,                # Extremely oversold
            indicators['macd'].Current.Value < indicators['macd'].Signal.Current.Value and profit_pct > 0  # MACD bearish with profit
        ]
        
        return any(exit_conditions)
    
    def CalculatePositionSize(self, symbol):
        """Calculate position size based on volatility and risk"""
        if symbol not in self.indicators:
            return 0
        
        atr = self.indicators[symbol]['atr'].Current.Value
        price = self.Securities[symbol].Price
        
        if atr == 0 or price == 0:
            return 0
        
        # Volatility-based sizing
        volatility = atr / price
        base_size = self.position_size * self.leverage_factor
        
        # Reduce size for high volatility
        if volatility > 0.03:  # 3% volatility
            base_size *= 0.5
        elif volatility > 0.02:  # 2% volatility
            base_size *= 0.75
        
        # Ensure we don't over-leverage
        max_size = 0.5  # Max 50% per position
        return min(base_size, max_size)
    
    def CheckPortfolioRisk(self):
        """Check if portfolio is at risk limit"""
        current_value = self.Portfolio.TotalPortfolioValue
        initial_value = 100000  # Starting cash
        
        # Calculate total portfolio loss
        total_loss = (initial_value - current_value) / initial_value
        
        if total_loss > self.max_portfolio_risk:
            self.Debug(f"Portfolio risk limit hit: {total_loss:.2%} loss")
            self.Liquidate()  # Close all positions
            return True
        
        return False
    
    def GetLeveragedSymbol(self, symbol):
        """Get corresponding leveraged ETF symbol"""
        if symbol == self.spy:
            return self.upro
        elif symbol == self.qqq:
            return self.tqqq
        elif symbol == self.iwm:
            return self.tna
        return symbol
    
    def OnData(self, data):
        """Process incoming data"""
        pass  # All logic handled in scheduled functions
    
    def OnOrderEvent(self, orderEvent):
        """Track order fills"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order Filled: {orderEvent.Symbol} - {orderEvent.Direction} - " + 
                      f"Quantity: {orderEvent.FillQuantity} @ ${orderEvent.FillPrice}")