
from AlgorithmImports import *

class MultiTimeframeReversal(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Trade liquid ETFs for reliable data
        self.symbols = ["SPY", "QQQ", "IWM"]
        self.securities = {}
        self.indicators = {}
        
        for symbol in self.symbols:
            security = self.AddEquity(symbol, Resolution.Hour)
            security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            self.securities[symbol] = security
            
            # Multiple timeframe indicators
            self.indicators[symbol] = {
                "rsi_short": self.RSI(symbol, 6),   # Very short-term
                "rsi_medium": self.RSI(symbol, 14), # Medium-term
                "bb": self.BB(symbol, 20, 2),
                "momentum": self.MOMP(symbol, 10),
                "ema_fast": self.EMA(symbol, 8),
                "ema_slow": self.EMA(symbol, 21)
            }
            
        self.trade_count = 0
        self.rebalance_frequency = 4  # Rebalance every 4 hours for more trades
        self.last_rebalance = self.StartDate
        
    def OnData(self, data):
        # Rebalance frequently for more trade opportunities
        hours_since_rebalance = (self.Time - self.last_rebalance).total_seconds() / 3600
        if hours_since_rebalance < self.rebalance_frequency:
            return
            
        self.last_rebalance = self.Time
        
        # Check if all indicators are ready
        ready_symbols = []
        for symbol in self.symbols:
            if all(indicator.IsReady for indicator in self.indicators[symbol].values()):
                ready_symbols.append(symbol)
                
        if not ready_symbols:
            return
            
        # Generate signals for each symbol
        signals = {}
        for symbol in ready_symbols:
            signals[symbol] = self.GenerateSignal(symbol)
            
        # Execute trades based on signals
        self.ExecuteTrades(signals)
        
    def GenerateSignal(self, symbol):
        indicators = self.indicators[symbol]
        price = self.Securities[symbol].Price
        
        rsi_short = indicators["rsi_short"].Current.Value
        rsi_medium = indicators["rsi_medium"].Current.Value
        momentum = indicators["momentum"].Current.Value
        ema_fast = indicators["ema_fast"].Current.Value
        ema_slow = indicators["ema_slow"].Current.Value
        bb_upper = indicators["bb"].UpperBand.Current.Value
        bb_lower = indicators["bb"].LowerBand.Current.Value
        
        # AGGRESSIVE MEAN REVERSION SIGNALS
        signal_strength = 0
        
        # RSI oversold/overbought
        if rsi_short < 25:
            signal_strength += 2  # Strong buy
        elif rsi_short < 35:
            signal_strength += 1  # Moderate buy
        elif rsi_short > 75:
            signal_strength -= 2  # Strong sell
        elif rsi_short > 65:
            signal_strength -= 1  # Moderate sell
            
        # Bollinger Band reversals
        if price < bb_lower:
            signal_strength += 1
        elif price > bb_upper:
            signal_strength -= 1
            
        # EMA crossover for momentum
        if ema_fast > ema_slow and momentum > 0:
            signal_strength += 1
        elif ema_fast < ema_slow and momentum < 0:
            signal_strength -= 1
            
        # Return normalized signal (-1 to 1)
        return max(-1, min(1, signal_strength / 3))
        
    def ExecuteTrades(self, signals):
        # Equal weight allocation with frequent rebalancing
        total_weight = sum(abs(signal) for signal in signals.values())
        
        if total_weight > 0:
            for symbol, signal in signals.items():
                if abs(signal) > 0.2:  # Only trade if signal is strong enough
                    weight = (signal / total_weight) * 0.9  # Use 90% of capital
                    current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    
                    # Only trade if position change is significant
                    if abs(weight - current_weight) > 0.05:
                        self.SetHoldings(symbol, weight)
                        self.trade_count += 1
                        
        else:
            # No strong signals, reduce positions
            for symbol in self.symbols:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    
    def OnEndOfAlgorithm(self):
        years = (self.EndDate - self.StartDate).days / 365.25
        trades_per_year = self.trade_count / years
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
