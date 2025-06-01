from AlgorithmImports import *

class Enhanced_High_Return_Strategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Trade sector ETFs for higher returns and diversification
        self.symbols = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "GLD", "TLT"]
        self.securities = {}
        
        for symbol in self.symbols:
            security = self.AddEquity(symbol, Resolution.Hour)
            security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            self.securities[symbol] = security
            
        # Multi-timeframe indicators for each symbol
        self.indicators = {}
        for symbol in self.symbols:
            self.indicators[symbol] = {
                "momentum_fast": self.MOMP(symbol, 3),    # Very fast
                "momentum_med": self.MOMP(symbol, 10),    # Medium
                "momentum_slow": self.MOMP(symbol, 21),   # Slower
                "rsi_fast": self.RSI(symbol, 7),          # Fast RSI
                "rsi_slow": self.RSI(symbol, 21),         # Slow RSI
                "bb": self.BB(symbol, 20, 2),
                "ema_fast": self.EMA(symbol, 5),
                "ema_slow": self.EMA(symbol, 15),
                "atr": self.ATR(symbol, 14),
            }
            
        # Market regime detection
        self.market_regime = "NEUTRAL"
        
        # Position tracking
        self.trade_count = 0
        self.last_rebalance = self.StartDate
        
        # Risk management - More aggressive for higher returns
        self.max_position_size = 0.5  # 50% per position for higher returns
        self.max_total_leverage = 3.0  # 300% total exposure
        
        # Schedule frequent rebalancing for more trades
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(timedelta(hours=2)),
                        self.Rebalance)
                        
    def Rebalance(self):
        """Main rebalancing logic"""
        # Update market regime
        self.UpdateMarketRegime()
        
        # Get signals for all symbols
        signals = {}
        for symbol in self.symbols:
            if self.AllIndicatorsReady(symbol):
                signals[symbol] = self.GenerateEnhancedSignal(symbol)
                
        if not signals:
            return
            
        # Execute trades based on market regime and signals
        self.ExecuteEnhancedTrades(signals)
        
    def UpdateMarketRegime(self):
        """Detect market regime for position sizing"""
        if not self.indicators["SPY"]["momentum_med"].IsReady:
            return
            
        # Simple regime detection based on SPY momentum
        spy_momentum = self.indicators["SPY"]["momentum_med"].Current.Value
        spy_rsi = self.indicators["SPY"]["rsi_slow"].Current.Value
        
        if spy_momentum > 0.015 and spy_rsi > 50:
            self.market_regime = "BULL"
        elif spy_momentum < -0.015 and spy_rsi < 50:
            self.market_regime = "BEAR" 
        else:
            self.market_regime = "NEUTRAL"
            
    def AllIndicatorsReady(self, symbol):
        """Check if all indicators for a symbol are ready"""
        return all(indicator.IsReady for indicator in self.indicators[symbol].values())
        
    def GenerateEnhancedSignal(self, symbol):
        """Generate enhanced signal with multiple factors"""
        indicators = self.indicators[symbol]
        price = self.Securities[symbol].Price
        
        # Get indicator values
        mom_fast = indicators["momentum_fast"].Current.Value
        mom_med = indicators["momentum_med"].Current.Value  
        mom_slow = indicators["momentum_slow"].Current.Value
        rsi_fast = indicators["rsi_fast"].Current.Value
        rsi_slow = indicators["rsi_slow"].Current.Value
        ema_fast = indicators["ema_fast"].Current.Value
        ema_slow = indicators["ema_slow"].Current.Value
        bb_upper = indicators["bb"].UpperBand.Current.Value
        bb_lower = indicators["bb"].LowerBand.Current.Value
        bb_middle = indicators["bb"].MiddleBand.Current.Value
        
        signal_strength = 0
        
        # AGGRESSIVE MOMENTUM SIGNALS (for higher returns)
        if mom_fast > 0.008 and mom_med > 0.004:  # Strong uptrend
            signal_strength += 4
        elif mom_fast > 0.004 and mom_med > 0:    # Moderate uptrend
            signal_strength += 2
        elif mom_fast < -0.008 and mom_med < -0.004:  # Strong downtrend
            signal_strength -= 4
        elif mom_fast < -0.004 and mom_med < 0:    # Moderate downtrend
            signal_strength -= 2
            
        # TREND SIGNALS
        if ema_fast > ema_slow and price > ema_fast:
            signal_strength += 2
        elif ema_fast < ema_slow and price < ema_fast:
            signal_strength -= 2
            
        # MEAN REVERSION SIGNALS (for additional trades)
        if price < bb_lower and rsi_fast < 25:  # Very oversold
            signal_strength += 3
        elif price > bb_upper and rsi_fast > 75:  # Very overbought
            signal_strength -= 3
            
        # BREAKOUT SIGNALS
        if price > bb_upper and mom_fast > 0.004:
            signal_strength += 2
        elif price < bb_lower and mom_fast < -0.004:
            signal_strength -= 2
            
        # Normalize signal (-1 to 1)
        return max(-1, min(1, signal_strength / 10))
        
    def ExecuteEnhancedTrades(self, signals):
        """Execute trades with enhanced position sizing for higher returns"""
        
        # Filter for strong signals only
        strong_signals = {symbol: signal for symbol, signal in signals.items() 
                         if abs(signal) > 0.2}
                         
        if not strong_signals:
            # No strong signals - reduce positions but don't liquidate all
            for symbol in self.symbols:
                if self.Portfolio[symbol].Invested:
                    current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    if abs(current_weight) > 0.15:  # Reduce large positions
                        new_weight = current_weight * 0.5  # Reduce by 50%
                        self.SetHoldings(symbol, new_weight)
                        self.trade_count += 1
            return
            
        # Calculate total signal strength
        total_abs_signal = sum(abs(signal) for signal in strong_signals.values())
        
        if total_abs_signal == 0:
            return
            
        # AGGRESSIVE position sizing based on market regime
        base_leverage = {
            "BULL": 2.8,     # Very high leverage in bull markets
            "NEUTRAL": 2.0,  # High leverage in neutral markets
            "BEAR": 1.2      # Still leveraged in bear markets for short opportunities
        }.get(self.market_regime, 1.5)
        
        # Calculate target positions
        target_positions = {}
        for symbol, signal in strong_signals.items():
            # Weight by signal strength
            signal_weight = abs(signal) / total_abs_signal
            
            # AGGRESSIVE position size with leverage
            position_size = signal * signal_weight * base_leverage * 0.9  # Use 90% of capacity
            
            # Cap individual positions at higher levels for more returns
            position_size = max(-self.max_position_size, 
                              min(self.max_position_size, position_size))
                              
            target_positions[symbol] = position_size
            
        # Execute trades
        for symbol, target_weight in target_positions.items():
            current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
            
            # Trade on smaller changes for more frequent trading
            if abs(target_weight - current_weight) > 0.01:  # 1% threshold
                self.SetHoldings(symbol, target_weight)
                self.trade_count += 1
                
        # Risk management - check total leverage
        total_leverage = sum(abs(self.Portfolio[symbol].HoldingsValue) 
                           for symbol in self.symbols) / self.Portfolio.TotalPortfolioValue
                           
        if total_leverage > self.max_total_leverage:
            # Reduce all positions proportionally
            reduction_factor = self.max_total_leverage / total_leverage
            for symbol in self.symbols:
                if self.Portfolio[symbol].Invested:
                    current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    new_weight = current_weight * reduction_factor
                    self.SetHoldings(symbol, new_weight)
                    self.trade_count += 1
                    
    def OnData(self, data):
        """Handle intraday momentum opportunities for more trades"""
        # Quick momentum scalps for additional trades
        for symbol in self.symbols:
            if symbol in data and self.indicators[symbol]["momentum_fast"].IsReady:
                mom_fast = self.indicators[symbol]["momentum_fast"].Current.Value
                current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                
                # AGGRESSIVE momentum reversals for more trades
                if abs(mom_fast) > 0.012:  # Strong momentum
                    
                    # Add to momentum if position is small
                    if abs(current_weight) < 0.2:
                        quick_position = 0.2 if mom_fast > 0 else -0.2  # Larger quick positions
                        new_weight = current_weight + quick_position
                        
                        # Cap at position limits
                        new_weight = max(-self.max_position_size, 
                                       min(self.max_position_size, new_weight))
                        
                        if abs(new_weight - current_weight) > 0.05:  # Only if significant change
                            self.SetHoldings(symbol, new_weight)
                            self.trade_count += 1
                            
    def OnEndOfAlgorithm(self):
        """Log final statistics"""
        years = (self.EndDate - self.StartDate).days / 365.25
        trades_per_year = self.trade_count / years
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - self.StartingCapital) / self.StartingCapital
        cagr = (final_value / self.StartingCapital) ** (1/years) - 1
        
        self.Log(f"=== FINAL RESULTS ===")
        self.Log(f"Total Return: {total_return:.2%}")
        self.Log(f"CAGR: {cagr:.2%}")
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
        self.Log(f"Final Portfolio Value: ${final_value:,.2f}")
        self.Log(f"Market Regime: {self.market_regime}")