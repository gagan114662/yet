
from AlgorithmImports import *
import numpy as np
import pandas as pd

class RDAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2004, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)

        # Strategy parameters from RD-Agent
        self.lookback_period = 36
        self.rebalance_frequency = 10
        self.position_size = 0.173

        # Add universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)

        # Schedule rebalancing
        self.Schedule.On(self.DateRules.MonthStart(),
                        self.TimeRules.AfterMarketOpen("SPY", 10),
                        self.Rebalance)

        # Risk management
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.024))

        # Initialize indicators and variables
        self.symbols = []
        self.indicators = {}

    def CoarseSelectionFunction(self, coarse):
        """Select universe based on RD-Agent criteria"""
        # Filter by price and volume
        filtered = [x for x in coarse if x.HasFundamentalData
                    and x.Price > 7.61
                    and x.DollarVolume > 3639079]

        # Sort by selection criteria
        sorted_stocks = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)

        # Return top N stocks
        return [x.Symbol for x in sorted_stocks[:61]]

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
                '"bb": self.BB(symbol, self.lookback_period, 2), "rsi": self.RSI(symbol, 10)'
            }

    def Rebalance(self):
        """Execute trading logic based on RD-Agent strategy"""
        insights = []

        for symbol in self.indicators:
            if not self.IsWarmingUp:
                # Generate trading signal
                signal = self.GenerateSignal(symbol)

                if signal > 0:
                    insights.append(Insight.Price(symbol, timedelta(days=15), InsightDirection.Up))
                elif signal < 0:
                    insights.append(Insight.Price(symbol, timedelta(days=15), InsightDirection.Down))

        # Execute trades based on insights
        self.SetHoldings(insights)

    def GenerateSignal(self, symbol):
        """Generate trading signal based on RD-Agent logic"""

indicators = self.indicators[symbol]
price = self.Securities[symbol].Price
upper_band = indicators["bb"].UpperBand.Current.Value
lower_band = indicators["bb"].LowerBand.Current.Value
rsi = indicators["rsi"].Current.Value
signal = 0
if self.Securities[symbol].Price > 0 and upper_band > 0 and lower_band > 0: # Ensure bands are valid
    if price < lower_band and rsi < (35 + random.randint(-5,5)): # Randomized threshold
        signal = 1
    elif price > upper_band and rsi > (65 + random.randint(-5,5)): # Randomized threshold
        signal = -1


        return signal

    def OnData(self, data):
        """Process incoming data"""
        # Update custom calculations if needed
        pass
