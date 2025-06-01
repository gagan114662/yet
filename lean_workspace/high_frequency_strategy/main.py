from AlgorithmImports import *

class HighFrequencyStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add leveraged ETFs for higher returns
        self.symbols = ["SPY", "QQQ", "UPRO", "TQQQ"]  # Include leveraged ETFs
        self.securities = {}
        
        for symbol in self.symbols:
            try:
                security = self.AddEquity(symbol, Resolution.DAILY)
                self.securities[symbol] = security
            except:
                continue  # Skip if symbol not available
        
        # Simple but effective indicators
        self.indicators = {}
        for symbol in self.securities.keys():
            self.indicators[symbol] = {
                "momentum": self.MOMP(symbol, 5),
                "rsi": self.RSI(symbol, 14),
                "sma_fast": self.SMA(symbol, 10),
                "sma_slow": self.SMA(symbol, 30)
            }
        
        self.trade_count = 0
        self.rebalance_days = 0
        
        # Schedule frequent rebalancing for high trade frequency
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.AfterMarketOpen("SPY", 30),
                        self.Rebalance)
    
    def Rebalance(self):
        """High-frequency rebalancing logic"""
        
        self.rebalance_days += 1
        
        # Only rebalance every 2-3 days for manageable frequency
        if self.rebalance_days % 2 != 0:
            return
            
        signals = {}
        for symbol in self.securities.keys():
            if self.AllReady(symbol):
                signals[symbol] = self.GetSignal(symbol)
        
        if not signals:
            return
            
        # Execute trades based on signals
        self.ExecuteTrades(signals)
    
    def AllReady(self, symbol):
        """Check if all indicators are ready"""
        return all(ind.IsReady for ind in self.indicators[symbol].values())
    
    def GetSignal(self, symbol):
        """Generate trading signal"""
        ind = self.indicators[symbol]
        
        momentum = ind["momentum"].Current.Value
        rsi = ind["rsi"].Current.Value
        sma_fast = ind["sma_fast"].Current.Value
        sma_slow = ind["sma_slow"].Current.Value
        price = self.Securities[symbol].Price
        
        signal = 0
        
        # Momentum signal
        if momentum > 0.01:  # 1% momentum
            signal += 1
        elif momentum < -0.01:
            signal -= 1
            
        # Trend signal
        if sma_fast > sma_slow and price > sma_fast:
            signal += 1
        elif sma_fast < sma_slow and price < sma_fast:
            signal -= 1
            
        # RSI signal (contrarian for reversion)
        if rsi < 30:
            signal += 1
        elif rsi > 70:
            signal -= 1
            
        return signal
    
    def ExecuteTrades(self, signals):
        """Execute trades with position sizing"""
        
        # Count strong signals
        strong_signals = {k: v for k, v in signals.items() if abs(v) >= 2}
        
        if not strong_signals:
            # Reduce positions when no strong signals
            for symbol in self.securities.keys():
                if self.Portfolio[symbol].Invested:
                    current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    if abs(current_weight) > 0.05:
                        new_weight = current_weight * 0.8
                        self.SetHoldings(symbol, new_weight)
                        self.trade_count += 1
            return
        
        # Calculate position sizes
        total_signals = sum(abs(v) for v in strong_signals.values())
        
        for symbol, signal in strong_signals.items():
            # Position size based on signal strength
            weight = (signal / total_signals) * 0.8  # Use 80% of capital
            
            # Apply leverage for leveraged ETFs
            if symbol in ["UPRO", "TQQQ"]:
                weight *= 0.5  # Reduce size for leveraged ETFs
            
            current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
            
            # Only trade if significant change
            if abs(weight - current_weight) > 0.03:
                self.SetHoldings(symbol, weight)
                self.trade_count += 1
    
    def OnData(self, data):
        """Additional trading opportunities"""
        
        # Additional momentum trades for higher frequency
        for symbol in self.securities.keys():
            if symbol in data and self.indicators[symbol]["momentum"].IsReady:
                momentum = self.indicators[symbol]["momentum"].Current.Value
                current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                
                # Quick momentum trades
                if abs(momentum) > 0.02 and abs(current_weight) < 0.1:
                    quick_weight = 0.1 if momentum > 0 else -0.1
                    
                    # Smaller positions for leveraged ETFs
                    if symbol in ["UPRO", "TQQQ"]:
                        quick_weight *= 0.5
                        
                    self.SetHoldings(symbol, current_weight + quick_weight)
                    self.trade_count += 1
    
    def OnEndOfAlgorithm(self):
        """Final statistics"""
        years = (self.EndDate - self.StartDate).days / 365.25
        trades_per_year = self.trade_count / years if years > 0 else 0
        
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
        self.Log(f"Final Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        
        if self.Portfolio.TotalPortfolioValue > 0:
            total_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            cagr = (self.Portfolio.TotalPortfolioValue / 100000) ** (1/years) - 1
            self.Log(f"Total Return: {total_return:.2%}")
            self.Log(f"CAGR: {cagr:.2%}")