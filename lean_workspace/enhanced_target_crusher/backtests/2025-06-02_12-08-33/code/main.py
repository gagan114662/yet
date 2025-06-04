from AlgorithmImports import *

class EnhancedTargetCrusher(QCAlgorithm):
    """
    ENHANCED 25% TARGET CRUSHER with ADVANCED SIGNALS
    
    Multi-factor momentum strategy with regime detection, volatility scaling,
    and dynamic position sizing to achieve 25%+ CAGR with SPY.
    """
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # SPY with MAXIMUM leverage
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy.SetLeverage(5.0)  # 5x leverage for aggressive returns
        
        # Multi-timeframe indicators for better signals
        self.sma_5 = self.SMA("SPY", 5)    # Short-term trend
        self.sma_10 = self.SMA("SPY", 10)  # Medium-term trend
        self.sma_20 = self.SMA("SPY", 20)  # Intermediate trend
        self.sma_50 = self.SMA("SPY", 50)  # Long-term trend
        
        # Momentum indicators
        self.rsi = self.RSI("SPY", 14)
        self.macd = self.MACD("SPY", 12, 26, 9)
        
        # Volatility indicator for position sizing
        self.atr = self.ATR("SPY", 14)
        self.bb = self.BB("SPY", 20, 2)
        
        # Performance tracking
        self.rebalance_time = self.Time
        self.high_water_mark = 100000
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
    def OnData(self, data):
        if not self.AllIndicatorsReady():
            return
            
        if "SPY" not in data:
            return
            
        # Rebalance every 3 days for optimal performance
        if (self.Time - self.rebalance_time).days < 3:
            return
            
        self.rebalance_time = self.Time
        
        # Determine market regime and optimal position
        regime = self.DetectMarketRegime()
        position_size = self.CalculateOptimalPosition(regime, data["SPY"].Close)
        
        # Execute position
        if abs(position_size) > 0.1:  # Only trade if meaningful position change
            self.SetHoldings("SPY", position_size)
            direction = "LONG" if position_size > 0 else "SHORT"
            self.Log(f"{regime}: {direction} {abs(position_size)*100:.0f}% SPY @ ${data['SPY'].Close}")
        else:
            self.Liquidate()
            self.Log(f"{regime}: CASH @ ${data['SPY'].Close}")
            
    def AllIndicatorsReady(self):
        """Check if all indicators are ready"""
        return (self.sma_5.IsReady and self.sma_10.IsReady and 
                self.sma_20.IsReady and self.sma_50.IsReady and
                self.rsi.IsReady and self.macd.IsReady and 
                self.atr.IsReady and self.bb.IsReady)
    
    def DetectMarketRegime(self):
        """Advanced regime detection using multiple indicators"""
        price = self.Securities["SPY"].Price
        
        # Trend alignment across timeframes
        uptrend_short = self.sma_5.Current.Value > self.sma_10.Current.Value
        uptrend_medium = self.sma_10.Current.Value > self.sma_20.Current.Value
        uptrend_long = self.sma_20.Current.Value > self.sma_50.Current.Value
        
        # Momentum confirmation
        rsi_bullish = self.rsi.Current.Value > 50
        macd_bullish = self.macd.Current.Value > self.macd.Signal.Current.Value
        
        # Price vs trend
        above_short_ma = price > self.sma_5.Current.Value
        above_long_ma = price > self.sma_50.Current.Value
        
        # Count bullish signals
        bullish_signals = sum([
            uptrend_short, uptrend_medium, uptrend_long,
            rsi_bullish, macd_bullish, above_short_ma, above_long_ma
        ])
        
        if bullish_signals >= 6:
            return "BULL_STRONG"
        elif bullish_signals >= 4:
            return "BULL_MODERATE"
        elif bullish_signals >= 3:
            return "BULL_WEAK"
        elif bullish_signals <= 1:
            return "BEAR_STRONG"
        elif bullish_signals <= 3:
            return "BEAR_MODERATE"
        else:
            return "SIDEWAYS"
    
    def CalculateOptimalPosition(self, regime, current_price):
        """Calculate optimal position size based on regime and risk factors"""
        
        # Base position sizes by regime
        base_positions = {
            "BULL_STRONG": 4.5,    # 450% position = 22.5x with leverage
            "BULL_MODERATE": 3.0,  # 300% position = 15x with leverage
            "BULL_WEAK": 1.5,      # 150% position = 7.5x with leverage
            "SIDEWAYS": 0.0,       # Stay in cash
            "BEAR_MODERATE": -1.5, # Short 150% = -7.5x
            "BEAR_STRONG": -3.0    # Short 300% = -15x
        }
        
        base_position = base_positions.get(regime, 0.0)
        
        # Volatility adjustment
        volatility_adj = 1.0
        if self.atr.IsReady:
            current_volatility = self.atr.Current.Value / current_price
            if current_volatility > 0.025:  # High volatility - reduce size
                volatility_adj = 0.7
            elif current_volatility < 0.015:  # Low volatility - increase size
                volatility_adj = 1.3
        
        # RSI extremes adjustment
        rsi_adj = 1.0
        if self.rsi.IsReady:
            rsi_val = self.rsi.Current.Value
            if rsi_val < 20 and base_position > 0:  # Oversold + bullish
                rsi_adj = 1.5
            elif rsi_val > 80 and base_position > 0:  # Overbought + bullish
                rsi_adj = 0.6
            elif rsi_val > 80 and base_position < 0:  # Overbought + bearish
                rsi_adj = 1.3
        
        # Drawdown protection
        current_value = self.Portfolio.TotalPortfolioValue
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value
            
        drawdown = (self.high_water_mark - current_value) / self.high_water_mark
        drawdown_adj = 1.0
        if drawdown > 0.15:  # 15% drawdown
            drawdown_adj = 0.5
        elif drawdown > 0.10:  # 10% drawdown
            drawdown_adj = 0.7
        
        # Momentum boost for winning streaks
        momentum_adj = 1.0
        if self.consecutive_wins >= 3:
            momentum_adj = 1.2  # Increase position on hot streak
        elif self.consecutive_losses >= 2:
            momentum_adj = 0.8  # Reduce position on losing streak
        
        # Final calculation
        final_position = (base_position * volatility_adj * 
                         rsi_adj * drawdown_adj * momentum_adj)
        
        # Safety caps
        return max(-4.0, min(4.0, final_position))
    
    def OnOrderEvent(self, orderEvent):
        """Track wins/losses for momentum adjustment"""
        if orderEvent.Status == OrderStatus.Filled:
            # Simplified win/loss tracking based on portfolio performance
            current_value = self.Portfolio.TotalPortfolioValue
            if current_value > self.high_water_mark * 0.98:  # Within 2% of high
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
            
    def OnEndOfAlgorithm(self):
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value / 100000) - 1
        days = (self.EndDate - self.StartDate).days
        cagr = ((final_value / 100000) ** (365.25 / days)) - 1
        
        # Calculate max drawdown
        max_drawdown = (self.high_water_mark - final_value) / self.high_water_mark
        max_drawdown = max(0, max_drawdown)
        
        self.Log("=" * 60)
        self.Log("🚀 ENHANCED TARGET CRUSHER RESULTS")
        self.Log("=" * 60)
        self.Log(f"Final Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return*100:.1f}%")
        self.Log(f"CAGR: {cagr*100:.1f}%")
        self.Log(f"High Water Mark: ${self.high_water_mark:,.2f}")
        self.Log(f"Max Drawdown: {max_drawdown*100:.1f}%")
        self.Log(f"Consecutive Wins: {self.consecutive_wins}")
        self.Log("=" * 60)
        
        if cagr >= 0.25:
            self.Log("🏆 25% TARGET CRUSHED!")
        elif cagr >= 0.18:
            self.Log("🎯 Close to target!")
        else:
            self.Log(f"📈 Need {(0.25-cagr)*100:.1f}% more for target")