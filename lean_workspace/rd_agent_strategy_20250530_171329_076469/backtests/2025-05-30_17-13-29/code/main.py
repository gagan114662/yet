
from AlgorithmImports import *

class AggressiveSPYMomentum(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Trade SPY only for reliable data
        self.spy = self.AddEquity("SPY", Resolution.Hour)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
        
        # Indicators for high-frequency signals
        self.momentum_fast = self.MOMP("SPY", 5)
        self.momentum_slow = self.MOMP("SPY", 15) 
        self.rsi = self.RSI("SPY", 14)
        self.bb = self.BB("SPY", 20, 2)
        
        # Track for trade frequency
        self.trade_count = 0
        self.last_trade_time = self.StartDate
        
    def OnData(self, data):
        if not (self.momentum_fast.IsReady and self.momentum_slow.IsReady and self.rsi.IsReady):
            return
            
        # Get current values
        fast_mom = self.momentum_fast.Current.Value
        slow_mom = self.momentum_slow.Current.Value
        rsi_val = self.rsi.Current.Value
        price = self.Securities["SPY"].Price
        
        # Current position
        holdings = self.Portfolio["SPY"].Quantity
        
        # AGGRESSIVE ENTRY CONDITIONS (for more trades)
        # Long entry: Fast momentum > slow momentum and RSI not overbought
        if fast_mom > slow_mom and fast_mom > 0.002 and rsi_val < 80 and holdings <= 0:
            self.SetHoldings("SPY", 0.95)
            self.trade_count += 1
            self.last_trade_time = self.Time
            
        # Short entry: Fast momentum < slow momentum and RSI not oversold  
        elif fast_mom < slow_mom and fast_mom < -0.002 and rsi_val > 20 and holdings >= 0:
            self.SetHoldings("SPY", -0.95)
            self.trade_count += 1
            self.last_trade_time = self.Time
            
        # TIGHT STOP LOSSES (for more frequent trades)
        elif holdings > 0 and (fast_mom < -0.005 or rsi_val > 85):
            self.Liquidate("SPY")
            self.trade_count += 1
            
        elif holdings < 0 and (fast_mom > 0.005 or rsi_val < 15):
            self.Liquidate("SPY")
            self.trade_count += 1
            
        # MEAN REVERSION TRADES (additional trade opportunities)
        elif abs(holdings) < 0.1:  # No position
            bb_upper = self.bb.UpperBand.Current.Value
            bb_lower = self.bb.LowerBand.Current.Value
            bb_middle = self.bb.MiddleBand.Current.Value
            
            # Buy oversold bounces
            if price < bb_lower and rsi_val < 30:
                self.SetHoldings("SPY", 0.5)
                self.trade_count += 1
                
            # Sell overbought reversals  
            elif price > bb_upper and rsi_val > 70:
                self.SetHoldings("SPY", -0.5)
                self.trade_count += 1
                
    def OnEndOfAlgorithm(self):
        years = (self.EndDate - self.StartDate).days / 365.25
        trades_per_year = self.trade_count / years
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
