# region imports
from AlgorithmImports import *
# endregion
from QuantConnect import *
from QuantConnect.Algorithm import *

class VandanStrategyBurke(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 3)  # First Wednesday of 2024
        self.SetEndDate(2024, 1, 3)    # Same day
        self.SetCash(100000)
        
        # Add SPX index and its options
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        self.spx_options = self.AddIndexOption(self.spx.Symbol, Resolution.Minute)
        
        # Wide filter to ensure we get options
        self.spx_options.SetFilter(lambda u: u.Strikes(-50, +50)
                                           .Expiration(0, 30))
                                           
        self.Debug("Initialize completed")

    def OnData(self, slice: slice):
        self.Debug(f"\n[{self.Time}] === New Data Bar ===")
        
        if not slice.Bars:
            self.Debug("No bars data")
            return
            
        spx_price = slice.Bars[self.spx.Symbol].Close
        self.Debug(f"SPX Price: ${spx_price:0.2f}")
        
        if slice.OptionChains:
            chain = slice.OptionChains[self.spx_options.Symbol]
            self.Debug(f"Options found: {len(chain)} contracts")
            
            # Print first 5 puts and calls to avoid spam
            for contract in chain:
                self.Debug(f"Option: {contract.Symbol.Value}, Strike: ${contract.Strike:0.2f}, Bid: ${contract.BidPrice:0.2f}, Ask: ${contract.AskPrice:0.2f}")
        else:
            self.Debug("No option chain data")
            
            "now, i want you to take the code from @/Strategies/VandanStrategyBurke/old/main_old_5.py and refactor it for SPX using the knowledge from @/Strategies/VandanStrategyBurke/main.py"
