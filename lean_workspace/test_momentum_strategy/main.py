from AlgorithmImports import *

class RDAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Strategy parameters
        self.lookback_period = 20
        self.rebalance_frequency = 5
        self.position_size = 0.1
        self.universe_size = 20
        
        # Add SPY for timing
        self.AddEquity("SPY", Resolution.Daily)
        
        # Add universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Schedule rebalancing every 5 days
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Friday), 
                        self.TimeRules.At(10, 0),
                        self.Rebalance)
        
        # Initialize indicators and variables
        self.symbols = []
        self.indicators = {}
        self.last_rebalance = self.StartDate
        
    def CoarseSelectionFunction(self, coarse):
        """Select universe based on criteria"""
        # Filter by price and volume
        filtered = [x for x in coarse if x.HasFundamentalData 
                    and x.Price > 10 
                    and x.DollarVolume > 5000000]
        
        # Sort by dollar volume
        sorted_stocks = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        
        # Return top stocks
        return [x.Symbol for x in sorted_stocks[:self.universe_size]]
        
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
            if symbol not in self.indicators:
                self.indicators[symbol] = {
                    "momentum": self.MOMP(symbol, self.lookback_period),
                    "rsi": self.RSI(symbol, 14)
                }
            
    def Rebalance(self):
        """Execute trading logic"""
        if (self.Time - self.last_rebalance).days < self.rebalance_frequency:
            return
            
        self.last_rebalance = self.Time
        
        # Get signals for all symbols
        signals = {}
        for symbol in self.indicators:
            if self.indicators[symbol]["momentum"].IsReady and self.indicators[symbol]["rsi"].IsReady:
                signals[symbol] = self.GenerateSignal(symbol)
        
        # Filter for buy signals
        buy_signals = [symbol for symbol, signal in signals.items() if signal > 0]
        
        # Equal weight allocation
        if buy_signals:
            weight = 1.0 / len(buy_signals)
            
            # Liquidate positions not in buy signals
            for symbol in self.Portfolio.Keys:
                if symbol not in buy_signals and self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
            
            # Set new positions
            for symbol in buy_signals:
                self.SetHoldings(symbol, weight * self.position_size)
        else:
            # No signals, liquidate all
            self.Liquidate()
        
    def GenerateSignal(self, symbol):
        """Generate trading signal"""
        indicators = self.indicators[symbol]
        momentum = indicators["momentum"].Current.Value
        rsi = indicators["rsi"].Current.Value
        
        # Buy: positive momentum, not overbought
        if momentum > 0.02 and rsi < 70:
            return 1
        # Sell: negative momentum or overbought  
        elif momentum < -0.02 or rsi > 80:
            return -1
        else:
            return 0
            
    def OnData(self, data):
        """Process incoming data"""
        pass