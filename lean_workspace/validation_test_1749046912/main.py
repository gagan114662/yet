from AlgorithmImports import *

class ValidationStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 6, 30)
        self.SetCash(100000)
        
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Simple indicators
        self.sma_short = self.SMA("SPY", 10, Resolution.Daily)
        self.sma_long = self.SMA("SPY", 20, Resolution.Daily)
        
        # Weekly rebalancing
        self.Schedule.On(self.DateRules.WeekStart("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.Rebalance)
        
    def Rebalance(self):
        if not self.sma_short.IsReady or not self.sma_long.IsReady:
            return
            
        # Simple momentum with realistic leverage
        if self.sma_short.Current.Value > self.sma_long.Current.Value:
            self.SetHoldings(self.spy, 1.5)  # 1.5x leverage - realistic
        else:
            self.SetHoldings(self.spy, 0.5)  # Reduced position in downtrend
    
    def OnData(self, data):
        pass