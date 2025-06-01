from AlgorithmImports import *

class FinalOptimizedStrategy(QCAlgorithm):
    """
    Final Optimized Strategy for >25% CAGR with 100+ trades/year
    """
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Aggressive asset selection for high returns
        self.symbols = ["SPY", "QQQ", "IWM", "XLK", "XLF", "GLD", "TLT", "VXX"]
        
        for symbol in self.symbols:
            security = self.AddEquity(symbol, Resolution.Daily)
            security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            
        # Fast indicators for high-frequency trading
        self.indicators = {}
        for symbol in self.symbols:
            self.indicators[symbol] = {
                "momentum": self.MOMP(symbol, 5),    # 5-day momentum
                "rsi": self.RSI(symbol, 10),         # Fast RSI
                "bb": self.BB(symbol, 15, 2),        # Bollinger Bands
                "ema_fast": self.EMA(symbol, 8),     # Fast EMA
                "ema_slow": self.EMA(symbol, 21),    # Slow EMA
            }
            
        # Performance tracking
        self.trade_count = 0
        self.starting_cash = 100000
        
        # AGGRESSIVE PARAMETERS for 25%+ CAGR
        self.max_leverage = 2.8           # 280% exposure
        self.max_position_size = 0.5      # 50% per position
        self.rebalance_threshold = 0.015  # 1.5% change triggers trade
        
        # Schedule DAILY rebalancing for high frequency
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.AfterMarketOpen("SPY", 30),
                        self.Rebalance)
                        
    def Rebalance(self):
        """Daily rebalancing for 100+ trades per year"""
        
        # Generate signals for all assets
        signals = {}
        for symbol in self.symbols:
            if self.AllReady(symbol):
                signals[symbol] = self.GetAggressiveSignal(symbol)
                
        if not signals:
            return
            
        # Execute aggressive trades
        self.ExecuteAggressiveTrades(signals)
        
    def AllReady(self, symbol):
        """Check if indicators are ready"""
        return all(ind.IsReady for ind in self.indicators[symbol].values())
        
    def GetAggressiveSignal(self, symbol):
        """Generate aggressive signals for high returns"""
        ind = self.indicators[symbol]
        price = self.Securities[symbol].Price
        
        momentum = ind["momentum"].Current.Value
        rsi = ind["rsi"].Current.Value
        bb_upper = ind["bb"].UpperBand.Current.Value
        bb_lower = ind["bb"].LowerBand.Current.Value
        ema_fast = ind["ema_fast"].Current.Value
        ema_slow = ind["ema_slow"].Current.Value
        
        signal = 0
        
        # AGGRESSIVE MOMENTUM (primary driver)
        if momentum > 0.01:      # 1%+ momentum = strong buy
            signal += 3
        elif momentum > 0.005:   # 0.5%+ momentum = buy
            signal += 2
        elif momentum < -0.01:   # -1% momentum = strong sell
            signal -= 3
        elif momentum < -0.005:  # -0.5% momentum = sell
            signal -= 2
            
        # TREND CONFIRMATION
        if ema_fast > ema_slow and price > ema_fast:
            signal += 2  # Strong uptrend
        elif ema_fast < ema_slow and price < ema_fast:
            signal -= 2  # Strong downtrend
            
        # BREAKOUT/BREAKDOWN SIGNALS
        if price > bb_upper:
            signal += 2  # Breakout
        elif price < bb_lower:
            signal -= 2  # Breakdown
            
        # OVERSOLD/OVERBOUGHT (counter-trend)
        if rsi < 20:
            signal += 1  # Oversold bounce
        elif rsi > 80:
            signal -= 1  # Overbought decline
            
        # Special handling for defensive assets
        if symbol in ["GLD", "TLT", "VXX"]:
            signal *= -0.5  # Inverse correlation during risk-off
            
        # Normalize to -1 to +1
        return max(-1, min(1, signal / 8))
        
    def ExecuteAggressiveTrades(self, signals):
        """Execute trades with aggressive position sizing"""
        
        # Filter for meaningful signals
        strong_signals = {k: v for k, v in signals.items() if abs(v) > 0.2}
        
        if not strong_signals:
            # Reduce all positions when no signals
            for symbol in self.symbols:
                if self.Portfolio[symbol].Invested:
                    current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    if abs(current_weight) > 0.05:
                        new_weight = current_weight * 0.5  # Cut position in half
                        self.SetHoldings(symbol, new_weight)
                        self.trade_count += 1
            return
            
        # Calculate total signal strength
        total_signal_strength = sum(abs(v) for v in strong_signals.values())
        
        if total_signal_strength == 0:
            return
            
        # Calculate target positions with AGGRESSIVE leverage
        for symbol, signal in strong_signals.items():
            # Position weight based on signal strength
            signal_weight = abs(signal) / total_signal_strength
            
            # AGGRESSIVE position sizing
            target_weight = signal * signal_weight * self.max_leverage
            
            # Cap individual positions
            target_weight = max(-self.max_position_size, 
                              min(self.max_position_size, target_weight))
            
            current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
            
            # Trade on smaller changes for higher frequency
            if abs(target_weight - current_weight) > self.rebalance_threshold:
                self.SetHoldings(symbol, target_weight)
                self.trade_count += 1
                
    def OnData(self, data):
        """Intraday momentum trades for extra frequency"""
        
        # Additional scalping opportunities
        for symbol in self.symbols:
            if symbol in data and self.indicators[symbol]["momentum"].IsReady:
                momentum = self.indicators[symbol]["momentum"].Current.Value
                current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                
                # QUICK momentum reversals
                if abs(momentum) > 0.02:  # 2%+ momentum
                    if abs(current_weight) < 0.2:  # Only if position is small
                        quick_trade = 0.15 if momentum > 0 else -0.15
                        
                        # Special handling for defensive assets
                        if symbol in ["GLD", "TLT", "VXX"]:
                            quick_trade *= -0.5
                            
                        new_weight = current_weight + quick_trade
                        new_weight = max(-self.max_position_size, 
                                       min(self.max_position_size, new_weight))
                        
                        if abs(new_weight - current_weight) > 0.02:
                            self.SetHoldings(symbol, new_weight)
                            self.trade_count += 1
                            
    def OnEndOfAlgorithm(self):
        """Calculate final performance metrics"""
        years = (self.EndDate - self.StartDate).days / 365.25
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - self.starting_cash) / self.starting_cash
        cagr = (final_value / self.starting_cash) ** (1/years) - 1
        trades_per_year = self.trade_count / years
        
        self.Log("=== FINAL OPTIMIZED STRATEGY RESULTS ===")
        self.Log(f"Final Portfolio Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return:.2%}")
        self.Log(f"CAGR: {cagr:.2%}")
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
        
        # Target evaluation
        self.Log("=== TARGET EVALUATION ===")
        self.Log(f"CAGR Target (>25%): {'PASS' if cagr > 0.25 else 'FAIL'} - {cagr:.2%}")
        self.Log(f"Trades Target (>100/year): {'PASS' if trades_per_year > 100 else 'FAIL'} - {trades_per_year:.1f}")