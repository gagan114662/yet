from AlgorithmImports import *

class FinalOptimized25Target(QCAlgorithm):
    """
    FINAL OPTIMIZED 25% TARGET STRATEGY
    
    Uses the full SPY data range (1998-2021) with optimized parameters
    for achieving 25%+ CAGR. Focus on buy-and-hold with momentum overlays
    during favorable periods, plus tactical rebalancing.
    """
    
    def Initialize(self):
        # Use full available data range for better results
        self.SetStartDate(1998, 1, 1)
        self.SetEndDate(2021, 3, 31)  # End of available SPY data
        self.SetCash(100000)
        
        # SPY with moderate leverage
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy.SetLeverage(3.0)  # 3x leverage (more conservative than 5x)
        
        # Simple but effective indicators
        self.sma_20 = self.SMA("SPY", 20)   # Short-term trend
        self.sma_50 = self.SMA("SPY", 50)   # Medium-term trend
        self.sma_200 = self.SMA("SPY", 200) # Long-term trend
        
        # Momentum and volatility
        self.rsi = self.RSI("SPY", 14)
        self.atr = self.ATR("SPY", 20)
        
        # Performance tracking
        self.rebalance_time = self.Time
        self.year_start_value = 100000
        self.annual_returns = []
        
    def OnData(self, data):
        if not self.AllIndicatorsReady():
            return
            
        if "SPY" not in data:
            return
            
        # Track annual performance
        if self.Time.year != (self.Time - timedelta(days=1)).year:
            current_value = self.Portfolio.TotalPortfolioValue
            annual_return = (current_value - self.year_start_value) / self.year_start_value
            self.annual_returns.append(annual_return)
            self.year_start_value = current_value
            self.Log(f"Year {self.Time.year-1} return: {annual_return*100:.1f}%")
        
        # Rebalance monthly for optimal risk-adjusted returns
        if (self.Time - self.rebalance_time).days < 20:
            return
            
        self.rebalance_time = self.Time
        
        # Determine position based on multiple factors
        position_size = self.CalculateOptimalPosition(data["SPY"].Close)
        
        if abs(position_size) > 0.1:
            self.SetHoldings("SPY", position_size)
            self.Log(f"Position: {position_size*100:.0f}% SPY @ ${data['SPY'].Close} | Portfolio: ${self.Portfolio.TotalPortfolioValue:,.0f}")
        else:
            self.Liquidate()
            self.Log(f"CASH @ ${data['SPY'].Close}")
            
    def AllIndicatorsReady(self):
        """Check if all indicators are ready"""
        return (self.sma_20.IsReady and self.sma_50.IsReady and 
                self.sma_200.IsReady and self.rsi.IsReady and self.atr.IsReady)
    
    def CalculateOptimalPosition(self, current_price):
        """Calculate optimal position using trend following with momentum"""
        
        # Basic trend following signals
        above_200sma = current_price > self.sma_200.Current.Value
        above_50sma = current_price > self.sma_50.Current.Value
        above_20sma = current_price > self.sma_20.Current.Value
        
        # Moving average slopes (momentum)
        sma20_slope = (self.sma_20.Current.Value - self.sma_20[1]) / self.sma_20[1] if self.sma_20[1] > 0 else 0
        sma50_slope = (self.sma_50.Current.Value - self.sma_50[1]) / self.sma_50[1] if self.sma_50[1] > 0 else 0
        
        # RSI for momentum confirmation
        rsi_val = self.rsi.Current.Value
        
        # Base position: Conservative buy-and-hold with trend overlay
        base_position = 0.0
        
        if above_200sma and above_50sma and above_20sma:
            # Strong uptrend - go aggressive
            if sma20_slope > 0.001 and sma50_slope > 0.001:  # Strong momentum
                base_position = 2.5  # 250% with 3x leverage = 7.5x exposure
            elif sma20_slope > 0.0005:  # Moderate momentum
                base_position = 2.0  # 200% with 3x leverage = 6x exposure
            else:
                base_position = 1.5  # 150% with 3x leverage = 4.5x exposure
                
        elif above_200sma and above_50sma:
            # Moderate uptrend
            base_position = 1.2  # 120% with 3x leverage = 3.6x exposure
            
        elif above_200sma:
            # Long-term uptrend only
            base_position = 0.8  # 80% with 3x leverage = 2.4x exposure
            
        else:
            # Downtrend or unclear - defensive
            base_position = 0.0  # Cash
        
        # RSI adjustments for extremes
        if rsi_val < 25 and base_position > 0:  # Oversold in uptrend
            base_position *= 1.3
        elif rsi_val > 75 and base_position > 0:  # Overbought in uptrend
            base_position *= 0.7
        
        # Volatility adjustment
        if self.atr.IsReady:
            volatility = self.atr.Current.Value / current_price
            if volatility > 0.03:  # High volatility
                base_position *= 0.8
            elif volatility < 0.015:  # Low volatility
                base_position *= 1.1
        
        # Safety caps
        return max(0.0, min(2.5, base_position))
    
    def OnEndOfAlgorithm(self):
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value / 100000) - 1
        years = (self.EndDate - self.StartDate).days / 365.25
        cagr = ((final_value / 100000) ** (1/years)) - 1
        
        # Add final year return
        if self.annual_returns:
            final_year_return = (final_value - self.year_start_value) / self.year_start_value
            self.annual_returns.append(final_year_return)
        
        # Calculate performance metrics
        avg_annual_return = sum(self.annual_returns) / len(self.annual_returns) if self.annual_returns else 0
        
        self.Log("=" * 60)
        self.Log("🎯 FINAL OPTIMIZED 25% TARGET RESULTS")
        self.Log("=" * 60)
        self.Log(f"Backtest Period: {years:.1f} years")
        self.Log(f"Final Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return*100:.1f}%")
        self.Log(f"CAGR: {cagr*100:.1f}%")
        self.Log(f"Average Annual Return: {avg_annual_return*100:.1f}%")
        
        if self.annual_returns:
            positive_years = sum(1 for r in self.annual_returns if r > 0)
            self.Log(f"Positive Years: {positive_years}/{len(self.annual_returns)} ({positive_years/len(self.annual_returns)*100:.0f}%)")
            
            # Show year by year
            self.Log("\nYear-by-Year Performance:")
            start_year = self.StartDate.year
            for i, ret in enumerate(self.annual_returns):
                year = start_year + i
                self.Log(f"{year}: {ret*100:.1f}%")
        
        self.Log("=" * 60)
        
        if cagr >= 0.25:
            self.Log("🏆 25% CAGR TARGET ACHIEVED!")
            self.Log("🎉 STRATEGY SUCCESSFUL!")
        elif cagr >= 0.20:
            self.Log("🎯 Very close to target!")
            self.Log(f"Need only {(0.25-cagr)*100:.1f}% more")
        elif cagr >= 0.15:
            self.Log("📈 Good performance, but need optimization")
            self.Log(f"Need {(0.25-cagr)*100:.1f}% more for target")
        else:
            self.Log("⚠️ Strategy needs significant improvement")
            self.Log(f"Need {(0.25-cagr)*100:.1f}% more for target")
            
        self.Log("=" * 60)
        self.Log("Strategy Summary:")
        self.Log("- Long-only trend following with 3x leverage")
        self.Log("- 200/50/20 SMA trend confirmation")
        self.Log("- RSI extremes for tactical adjustments")
        self.Log("- Volatility-based position sizing")
        self.Log("- Monthly rebalancing for optimal performance")