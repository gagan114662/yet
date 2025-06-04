from AlgorithmImports import *

class Cloud25Target(QCAlgorithm):
    """
    CLOUD 25% TARGET STRATEGY - OPTIMIZED FOR QUANTCONNECT
    
    Aggressive momentum strategy using leveraged ETFs
    Target: 25%+ CAGR with professional cloud data
    """
    
    def Initialize(self):
        self.SetStartDate(2015, 1, 1)  # Start when leveraged ETFs have good data
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add leveraged ETFs for maximum returns
        self.tqqq = self.AddEquity("TQQQ", Resolution.Daily)  # 3x NASDAQ
        self.upro = self.AddEquity("UPRO", Resolution.Daily)  # 3x S&P 500
        self.spy = self.AddEquity("SPY", Resolution.Daily)    # Benchmark
        
        # Set leverage
        self.tqqq.SetLeverage(2.0)  # 2x on 3x = 6x effective
        self.upro.SetLeverage(2.0)  # 2x on 3x = 6x effective
        
        # Momentum indicators
        self.tqqq_mom = self.MOMP("TQQQ", 20)
        self.upro_mom = self.MOMP("UPRO", 20)
        self.spy_sma = self.SMA("SPY", 50)
        self.spy_rsi = self.RSI("SPY", 14)
        
        # Track performance
        self.start_value = 100000
        
    def OnData(self, data):
        if not self.tqqq_mom.IsReady or not self.upro_mom.IsReady:
            return
            
        # Market regime check using SPY
        spy_trend = self.Securities["SPY"].Price > self.spy_sma.Current.Value
        
        if spy_trend:
            # Bull market - use leveraged ETFs
            tqqq_strength = self.tqqq_mom.Current.Value
            upro_strength = self.upro_mom.Current.Value
            
            # Choose the stronger momentum
            if tqqq_strength > upro_strength and tqqq_strength > 5:
                # Strong tech momentum
                self.SetHoldings("TQQQ", 1.5)  # 150% = 9x exposure
                self.SetHoldings("UPRO", 0.0)
            elif upro_strength > 5:
                # Strong broad market momentum
                self.SetHoldings("UPRO", 1.5)  # 150% = 9x exposure
                self.SetHoldings("TQQQ", 0.0)
            else:
                # Moderate momentum - balanced approach
                self.SetHoldings("TQQQ", 0.75)
                self.SetHoldings("UPRO", 0.75)
        else:
            # Bear market - go to cash
            self.Liquidate()
            
    def OnEndOfAlgorithm(self):
        final_value = self.Portfolio.TotalPortfolioValue
        years = (self.EndDate - self.StartDate).days / 365.25
        cagr = ((final_value / self.start_value) ** (1/years)) - 1
        
        self.Log("=" * 60)
        self.Log("CLOUD 25% TARGET RESULTS")
        self.Log("=" * 60)
        self.Log(f"Final Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {(final_value/self.start_value - 1)*100:.1f}%")
        self.Log(f"CAGR: {cagr*100:.1f}%")
        self.Log("=" * 60)
        
        if cagr >= 0.25:
            self.Log("TARGET ACHIEVED: 25%+ CAGR!")
        else:
            self.Log(f"Need {(0.25-cagr)*100:.1f}% more for target")
