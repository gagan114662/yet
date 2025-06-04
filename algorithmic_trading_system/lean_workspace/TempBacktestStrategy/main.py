
from AlgorithmImports import *

class EnhancedStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020,1,1)
        self.SetEndDate(2023,12,31)
        self.SetCash(100000)
        
        # Enhanced parameters
        self.position_size = 0.205
        self.leverage = 3.9
        self.universe_size = 100
        self.min_volume = 13545148
        self.stop_loss = 0.13
        
        # Universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        self.indicators = {}
        self.last_rebalance = datetime.min
        self.entry_prices = {}
        
    def CoarseSelectionFunction(self, coarse):
        filtered = [x for x in coarse if x.Price > 10 and x.DollarVolume > self.min_volume]
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:self.universe_size]]
    
    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.indicators:
                self.indicators[symbol] = {'"vix": self.RSI("VIX", 14), "bb": self.BB("VXX", 20, 2.5)'}
        
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.indicators:
                del self.indicators[symbol]
    
    def OnData(self, data):
        # Rebalance every few days
        if (self.Time - self.last_rebalance).days < 1:
            return
            
        signals = {}
        for symbol in self.indicators.keys():
            if symbol in data and self.indicators[symbol]["rsi"].IsReady:
                try:

vix_rsi = indicators.get("vix", {}).get("Current", {}).get("Value", 50)
vxx_price = self.Securities.get("VXX", {}).get("Price", 0)
bb_upper = indicators.get("bb", {}).get("UpperBand", {}).get("Current", {}).get("Value", 0)
bb_lower = indicators.get("bb", {}).get("LowerBand", {}).get("Current", {}).get("Value", 0)
trade_signal = 0

# Volatility premium harvesting logic
if vix_rsi > 70 and vxx_price > bb_upper:  # High volatility, short VXX
    trade_signal = -1  
elif vix_rsi < 30 and vxx_price < bb_lower:  # Low volatility, long SVXY
    trade_signal = 1

                    signals[symbol] = trade_signal
                except:
                    signals[symbol] = 0
        
        # Risk management and execution
        self.ExecuteSignals(signals, data)
        self.last_rebalance = self.Time
    
    def ExecuteSignals(self, signals, data):
        # Stop loss management
        for symbol in list(self.Portfolio.Keys):
            if self.Portfolio[symbol].Invested and symbol in data:
                entry_price = self.entry_prices.get(symbol, data[symbol].Price)
                current_price = data[symbol].Price
                
                if self.Portfolio[symbol].IsLong and current_price < entry_price * (1 - self.stop_loss):
                    self.Liquidate(symbol)
                elif self.Portfolio[symbol].IsShort and current_price > entry_price * (1 + self.stop_loss):
                    self.Liquidate(symbol)
        
        # Position sizing with leverage
        positive_signals = [s for s in signals.values() if s > 0]
        if positive_signals:
            position_per_stock = (self.position_size * self.leverage) / len(positive_signals)
            position_per_stock = min(position_per_stock, 0.25)  # Max 25% per position
            
            for symbol, signal in signals.items():
                if signal > 0 and not self.Portfolio[symbol].IsLong:
                    self.SetHoldings(symbol, position_per_stock)
                    self.entry_prices[symbol] = data[symbol].Price
                elif signal <= 0 and self.Portfolio[symbol].IsLong:
                    self.Liquidate(symbol)
        else:
            self.Liquidate()
