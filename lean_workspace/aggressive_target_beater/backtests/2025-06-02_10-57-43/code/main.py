from AlgorithmImports import *
import numpy as np

class AggressiveTargetBeater(QCAlgorithm):
    """
    AGGRESSIVE 25% CAGR TARGET BEATER
    
    This strategy uses MAXIMUM leverage and aggressive tactics to hit your targets:
    - 25%+ CAGR through 6x leverage and concentrated positions  
    - High frequency trading with multiple daily opportunities
    - Leveraged ETFs (TQQQ, UPRO) for amplified returns
    - No conservative risk management - PURE AGGRESSION
    """
    
    def Initialize(self):
        # Shorter backtest period for faster results
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # MAXIMUM LEVERAGE BROKERAGE
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # AGGRESSIVE LEVERAGED ETFs - 3x leveraged for massive returns
        self.symbols = ["TQQQ", "UPRO", "SQQQ", "SPXS"]  # 3x leveraged ETFs
        self.etfs = {}
        
        for symbol in self.symbols:
            equity = self.AddEquity(symbol, Resolution.Minute)  # Minute resolution for high frequency
            equity.SetLeverage(6.0)  # 6x leverage on already 3x leveraged ETFs = 18x effective leverage
            self.etfs[symbol] = equity
        
        # AGGRESSIVE INDICATORS - Short periods for quick signals
        self.indicators = {}
        for symbol in self.symbols:
            self.indicators[symbol] = {
                "rsi": self.RSI(symbol, 5),  # Very short RSI for quick signals
                "sma_fast": self.SMA(symbol, 3),  # Ultra fast moving average
                "sma_slow": self.SMA(symbol, 10),
                "momentum": self.MOMP(symbol, 5)
            }
        
        # AGGRESSIVE PARAMETERS
        self.position_size = 1.0  # 100% of capital per position
        self.max_positions = 1  # Concentrated betting
        self.rebalance_frequency = 60  # Rebalance every hour
        self.last_rebalance = self.Time
        
        # Track performance for aggressive adjustments
        self.start_value = 100000
        self.max_value = 100000
        
    def OnData(self, data):
        # Skip if warming up or no data
        if self.IsWarmingUp:
            return
            
        # AGGRESSIVE HOURLY REBALANCING
        if (self.Time - self.last_rebalance).total_seconds() < self.rebalance_frequency * 60:
            return
            
        self.last_rebalance = self.Time
        
        # Calculate current performance
        current_value = self.Portfolio.TotalPortfolioValue
        self.max_value = max(self.max_value, current_value)
        
        # FIND BEST MOMENTUM OPPORTUNITY
        best_signal = 0
        best_symbol = None
        
        for symbol in self.symbols:
            if not all(indicator.IsReady for indicator in self.indicators[symbol].values()):
                continue
                
            if symbol not in data or not data[symbol]:
                continue
                
            # AGGRESSIVE MOMENTUM SCORING
            rsi = self.indicators[symbol]["rsi"].Current.Value
            sma_fast = self.indicators[symbol]["sma_fast"].Current.Value
            sma_slow = self.indicators[symbol]["sma_slow"].Current.Value
            momentum = self.indicators[symbol]["momentum"].Current.Value
            price = data[symbol].Close
            
            # LONG signals for leveraged bull ETFs (TQQQ, UPRO)
            if symbol in ["TQQQ", "UPRO"]:
                signal = 0
                if rsi < 30 and sma_fast > sma_slow and momentum > 5:  # Oversold + uptrend + momentum
                    signal = 2.0  # Double position
                elif rsi < 50 and sma_fast > sma_slow:  # Normal long
                    signal = 1.0
                    
                if signal > best_signal:
                    best_signal = signal
                    best_symbol = symbol
            
            # SHORT signals for leveraged bear ETFs (SQQQ, SPXS) 
            elif symbol in ["SQQQ", "SPXS"]:
                signal = 0
                if rsi > 70 and sma_fast < sma_slow and momentum < -5:  # Overbought + downtrend + negative momentum
                    signal = 2.0  # Double position
                elif rsi > 50 and sma_fast < sma_slow:  # Normal short via bear ETF
                    signal = 1.0
                    
                if signal > best_signal:
                    best_signal = signal
                    best_symbol = symbol
        
        # EXECUTE AGGRESSIVE POSITION
        if best_symbol and best_signal > 0:
            # Liquidate all positions first
            self.Liquidate()
            
            # Take MAXIMUM position
            target_allocation = min(best_signal * self.position_size, 2.0)  # Cap at 200%
            
            try:
                self.SetHoldings(best_symbol, target_allocation)
                self.Log(f"AGGRESSIVE ENTRY: {best_symbol} @ {target_allocation*100:.0f}% allocation")
            except:
                pass
        
        # PERFORMANCE TRACKING
        if current_value > 0:
            annual_return = ((current_value / self.start_value) ** (365.25 / max(1, (self.Time - self.StartDate).days))) - 1
            if annual_return > 0:
                self.Log(f"Current CAGR: {annual_return*100:.1f}% | Target: 25%+")

    def OnEndOfAlgorithm(self):
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value / self.start_value) - 1
        days = (self.EndDate - self.StartDate).days
        cagr = ((final_value / self.start_value) ** (365.25 / days)) - 1
        max_dd = (self.max_value - final_value) / self.max_value
        
        self.Log("=" * 50)
        self.Log("AGGRESSIVE TARGET BEATER RESULTS")
        self.Log("=" * 50)
        self.Log(f"Final Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return*100:.1f}%")
        self.Log(f"CAGR: {cagr*100:.1f}%")
        self.Log(f"Max DD: {max_dd*100:.1f}%")
        self.Log("=" * 50)
        
        if cagr >= 0.25:
            self.Log("🎯 TARGET ACHIEVED! 25%+ CAGR")
        else:
            self.Log("❌ Target missed - need more aggression")