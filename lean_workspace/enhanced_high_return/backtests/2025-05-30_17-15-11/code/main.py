
from AlgorithmImports import *

class Enhanced_High_Return_Strategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Enable leverage for higher returns
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        
        # Trade sector ETFs and leveraged ETFs for higher returns
        self.symbols = {
            "SPY": self.AddEquity("SPY", Resolution.Hour),    # S&P 500
            "QQQ": self.AddEquity("QQQ", Resolution.Hour),    # Nasdaq
            "IWM": self.AddEquity("IWM", Resolution.Hour),    # Small cap
            "XLF": self.AddEquity("XLF", Resolution.Hour),    # Financials
            "XLK": self.AddEquity("XLK", Resolution.Hour),    # Technology
            "XLE": self.AddEquity("XLE", Resolution.Hour),    # Energy
            "GLD": self.AddEquity("GLD", Resolution.Hour),    # Gold
            "TLT": self.AddEquity("TLT", Resolution.Hour),    # Long bonds
        }
        
        # Set normalization
        for symbol, security in self.symbols.items():
            security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            
        # Multi-timeframe indicators for each symbol
        self.indicators = {}
        for symbol in self.symbols.keys():
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
                "volume": self.SMA(self.Volume(symbol), 20)
            }
            
        # Market regime detection
        self.vix = self.AddEquity("VIX", Resolution.Daily)  # Volatility
        self.market_regime = "NEUTRAL"
        
        # Position tracking
        self.trade_count = 0
        self.last_rebalance = self.StartDate
        self.position_sizes = {}
        
        # Risk management
        self.max_position_size = 0.4  # 40% per position
        self.max_total_leverage = 2.5  # 250% total exposure
        
        # Schedule frequent rebalancing
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(timedelta(hours=3)),
                        self.Rebalance)
                        
    def Rebalance(self):
        """Main rebalancing logic"""
        # Update market regime
        self.UpdateMarketRegime()
        
        # Get signals for all symbols
        signals = {}
        for symbol in self.symbols.keys():
            if self.AllIndicatorsReady(symbol):
                signals[symbol] = self.GenerateEnhancedSignal(symbol)
                
        if not signals:
            return
            
        # Execute trades based on market regime and signals
        self.ExecuteEnhancedTrades(signals)
        
    def UpdateMarketRegime(self):
        """Detect market regime for position sizing"""
        if self.Securities["SPY"].Price == 0:
            return
            
        # Simple regime detection based on momentum and volatility
        spy_momentum = self.indicators["SPY"]["momentum_med"].Current.Value
        spy_rsi = self.indicators["SPY"]["rsi_slow"].Current.Value
        
        if spy_momentum > 0.02 and spy_rsi > 50:
            self.market_regime = "BULL"
        elif spy_momentum < -0.02 and spy_rsi < 50:
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
        
        # MOMENTUM SIGNALS (strongest weight)
        if mom_fast > 0.01 and mom_med > 0.005:
            signal_strength += 3
        elif mom_fast > 0.005 and mom_med > 0:
            signal_strength += 2
        elif mom_fast < -0.01 and mom_med < -0.005:
            signal_strength -= 3
        elif mom_fast < -0.005 and mom_med < 0:
            signal_strength -= 2
            
        # TREND SIGNALS
        if ema_fast > ema_slow and price > ema_fast:
            signal_strength += 2
        elif ema_fast < ema_slow and price < ema_fast:
            signal_strength -= 2
            
        # MEAN REVERSION SIGNALS
        if price < bb_lower and rsi_fast < 30:
            signal_strength += 2
        elif price > bb_upper and rsi_fast > 70:
            signal_strength -= 2
            
        # BREAKOUT SIGNALS
        if price > bb_upper and mom_fast > 0.005:
            signal_strength += 1
        elif price < bb_lower and mom_fast < -0.005:
            signal_strength -= 1
            
        # Normalize signal (-1 to 1)
        return max(-1, min(1, signal_strength / 8))
        
    def ExecuteEnhancedTrades(self, signals):
        """Execute trades with enhanced position sizing"""
        
        # Filter for strong signals only
        strong_signals = {symbol: signal for symbol, signal in signals.items() 
                         if abs(signal) > 0.25}
                         
        if not strong_signals:
            # No strong signals - reduce positions
            for symbol in self.symbols.keys():
                if self.Portfolio[symbol].Invested:
                    current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    if abs(current_weight) > 0.1:  # Only liquidate significant positions
                        self.Liquidate(symbol)
                        self.trade_count += 1
            return
            
        # Calculate total signal strength
        total_abs_signal = sum(abs(signal) for signal in strong_signals.values())
        
        if total_abs_signal == 0:
            return
            
        # Position sizing based on market regime and signal strength
        base_leverage = {
            "BULL": 2.0,     # Higher leverage in bull markets
            "NEUTRAL": 1.5,  # Moderate leverage
            "BEAR": 1.0      # Conservative in bear markets
        }.get(self.market_regime, 1.0)
        
        # Calculate target positions
        target_positions = {}
        for symbol, signal in strong_signals.items():
            # Weight by signal strength
            signal_weight = abs(signal) / total_abs_signal
            
            # Position size with leverage
            position_size = signal * signal_weight * base_leverage * 0.8  # 80% of available
            
            # Cap individual positions
            position_size = max(-self.max_position_size, 
                              min(self.max_position_size, position_size))
                              
            target_positions[symbol] = position_size
            
        # Execute trades
        for symbol, target_weight in target_positions.items():
            current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
            
            # Only trade if change is significant
            if abs(target_weight - current_weight) > 0.02:  # 2% threshold
                self.SetHoldings(symbol, target_weight)
                self.trade_count += 1
                
        # Risk management - check total leverage
        total_leverage = sum(abs(self.Portfolio[symbol].HoldingsValue) 
                           for symbol in self.symbols.keys()) / self.Portfolio.TotalPortfolioValue
                           
        if total_leverage > self.max_total_leverage:
            # Reduce all positions proportionally
            reduction_factor = self.max_total_leverage / total_leverage
            for symbol in self.symbols.keys():
                if self.Portfolio[symbol].Invested:
                    current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    new_weight = current_weight * reduction_factor
                    self.SetHoldings(symbol, new_weight)
                    self.trade_count += 1
                    
    def OnData(self, data):
        """Handle intraday momentum opportunities"""
        # Quick momentum scalps for additional trades
        for symbol in self.symbols.keys():
            if symbol in data and self.indicators[symbol]["momentum_fast"].IsReady:
                mom_fast = self.indicators[symbol]["momentum_fast"].Current.Value
                current_holdings = self.Portfolio[symbol].Quantity
                
                # Quick reversal trades
                if abs(mom_fast) > 0.015:  # Strong momentum
                    current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    
                    # Add to momentum if position is small
                    if abs(current_weight) < 0.1:
                        quick_position = 0.15 if mom_fast > 0 else -0.15
                        self.SetHoldings(symbol, current_weight + quick_position)
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
