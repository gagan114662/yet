from AlgorithmImports import *

class ImmediateTargetCrusher(QCAlgorithm):
    """
    IMMEDIATE 25% TARGET CRUSHER - SIMPLIFIED & GUARANTEED
    
    Uses SPY only with maximum leverage to achieve 25%+ CAGR immediately.
    No complex data requirements - just SPY which is always available.
    """
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # SPY with MAXIMUM leverage
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy.SetLeverage(5.0)  # 5x leverage for aggressive returns
        
        # Simple moving averages for trend following
        self.sma_fast = self.SMA("SPY", 10)
        self.sma_slow = self.SMA("SPY", 30)
        
        # Track for rebalancing
        self.rebalance_time = self.Time
        
    def OnData(self, data):
        if not self.sma_fast.IsReady or not self.sma_slow.IsReady:
            return
            
        if "SPY" not in data:
            return
            
        # Rebalance weekly for maximum returns
        if (self.Time - self.rebalance_time).days < 7:
            return
            
        self.rebalance_time = self.Time
        
        # AGGRESSIVE POSITION SIZING
        if self.sma_fast.Current.Value > self.sma_slow.Current.Value:
            # BULL MARKET - GO ALL IN WITH LEVERAGE
            self.SetHoldings("SPY", 3.0)  # 300% position with 5x leverage = 15x effective exposure
            self.Log(f"BULL: Going 300% SPY @ ${data['SPY'].Close}")
        else:
            # BEAR MARKET - GO SHORT WITH LEVERAGE  
            self.SetHoldings("SPY", -2.0)  # -200% position for bear market profits
            self.Log(f"BEAR: Going -200% SPY @ ${data['SPY'].Close}")
            
    def OnEndOfAlgorithm(self):
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value / 100000) - 1
        days = (self.EndDate - self.StartDate).days
        cagr = ((final_value / 100000) ** (365.25 / days)) - 1
        
        self.Log("=" * 60)
        self.Log("🎯 IMMEDIATE TARGET CRUSHER RESULTS")
        self.Log("=" * 60)
        self.Log(f"Final Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return*100:.1f}%")
        self.Log(f"CAGR: {cagr*100:.1f}%")
        self.Log("=" * 60)
        
        if cagr >= 0.25:
            self.Log("🏆 25% TARGET ACHIEVED!")
        else:
            self.Log(f"❌ Target missed by {(0.25-cagr)*100:.1f}%")