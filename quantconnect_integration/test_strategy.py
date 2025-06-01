
from AlgorithmImports import *
import numpy as np
import pandas as pd

class RDAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Strategy parameters from RD-Agent
        self.lookback_period = 20
        self.rebalance_frequency = 5
        self.position_size = 0.1
        
        # Add universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.AfterMarketOpen("SPY", 10),
                        self.Rebalance)
        
        # Risk management
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))
        
        # Initialize indicators and variables
        self.symbols = []
        self.indicators = {}
        
    def CoarseSelectionFunction(self, coarse):
        """Select universe based on RD-Agent criteria"""
        # Filter by price and volume
        filtered = [x for x in coarse if x.HasFundamentalData 
                    and x.Price > 10 
                    and x.DollarVolume > 5000000]
        
        # Sort by selection criteria
        sorted_stocks = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        
        # Return top N stocks
        return [x.Symbol for x in sorted_stocks[:20]]
        
    def OnSecuritiesChanged(self, changes):
        """Handle universe changes"""
        # Remove indicators for removed securities
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.indicators:
                del self.indicators[symbol]
                
        # Add indicators for new securities
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            self.indicators[symbol] = {
                "momentum": self.MOMP(symbol, 20)
            }
            
    def Rebalance(self):
        """Execute trading logic based on RD-Agent strategy"""
        insights = []
        
        for symbol in self.indicators:
            if not self.IsWarmingUp:
                # Generate trading signal
                signal = self.GenerateSignal(symbol)
                
                if signal > 0:
                    insights.append(Insight.Price(symbol, timedelta(days=10), InsightDirection.Up))
                elif signal < 0:
                    insights.append(Insight.Price(symbol, timedelta(days=10), InsightDirection.Down))
                    
        # Execute trades based on insights
        self.SetHoldings(insights)
        
    def GenerateSignal(self, symbol):
        """Generate trading signal based on RD-Agent logic"""
        
        indicators = self.indicators[symbol]
        momentum = indicators["momentum"].Current.Value
        
        if momentum > 0.02:
            signal = 1
        elif momentum < -0.02:
            signal = -1
        else:
            signal = 0
        
        
        return signal
        
    def OnData(self, data):
        """Process incoming data"""
        # Update custom calculations if needed
        pass
