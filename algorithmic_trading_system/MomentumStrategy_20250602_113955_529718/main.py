
from AlgorithmImports import *
import numpy as np
import pandas as pd

class RDAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2004, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)

        # Strategy parameters from RD-Agent
        self.lookback_period = 15
        self.rebalance_frequency = 7
        self.position_size = 0.056

        # Add universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)

        # Schedule rebalancing
        self.Schedule.On(self.DateRules.WeekStart(),
                        self.TimeRules.AfterMarketOpen("SPY", 10),
                        self.Rebalance)

        # Risk management
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.063))

        # Initialize indicators and variables
        self.symbols = []
        self.indicators = {}

    def CoarseSelectionFunction(self, coarse):
        """Select universe based on RD-Agent criteria"""
        # Filter by price and volume
        filtered = [x for x in coarse if x.HasFundamentalData
                    and x.Price > 5.96
                    and x.DollarVolume > 5734044]

        # Sort by selection criteria
        sorted_stocks = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)

        # Return top N stocks
        return [x.Symbol for x in sorted_stocks[:30]]

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
                '"momentum": self.MOMP(symbol, self.lookback_period), "rsi": self.RSI(symbol, 14)'
            }

    def Rebalance(self):
        """Execute trading logic based on RD-Agent strategy"""
        insights = []

        for symbol in self.indicators:
            if not self.IsWarmingUp:
                # Generate trading signal
                signal = self.GenerateSignal(symbol)

                if signal > 0:
                    insights.append(Insight.Price(symbol, timedelta(days=7), InsightDirection.Up))
                elif signal < 0:
                    insights.append(Insight.Price(symbol, timedelta(days=7), InsightDirection.Down))

        # Execute trades based on insights
        self.SetHoldings(insights)

    def GenerateSignal(self, symbol):
        """Generate trading signal based on RD-Agent logic"""

indicators = self.indicators[symbol]
momentum = indicators["momentum"].Current.Value
rsi = indicators["rsi"].Current.Value
signal = 0
if self.Securities[symbol].Price > 0 and momentum > (0.01 + random.uniform(-0.005, 0.005)) and rsi < (70 + random.randint(-5, 5)): # Randomized threshold
    signal = 1
elif self.Securities[symbol].Price > 0 and momentum < (-0.01 + random.uniform(-0.005, 0.005)) and rsi > (30 + random.randint(-5, 5)): # Randomized threshold
    signal = -1


        return signal

    def OnData(self, data):
        """Process incoming data"""
        # Update custom calculations if needed
        pass
