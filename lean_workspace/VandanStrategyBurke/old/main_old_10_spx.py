from AlgorithmImports import *
from datetime import time, timedelta
from typing import Optional
# endregion

class VandanStrategyBurke(QCAlgorithm):
    """Burke Strategy Implementation
    - Uses market direction strength (-5 to +5)
    - Trades options based on ATR distance
    - Includes stop losses and hedging
    """
    def Initialize(self) -> None:
        """Initialize the strategy"""
        self.set_time_zone(TimeZones.NEW_YORK)
        self.set_start_date(2023, 10, 1)  # Set start date
        self.set_end_date(2023, 10, 31)    # Set end date
        # Basic settings
        self.SetCash(200_000)
        
        # Add option chain with pricing model
        option = self.add_index_option("SPX", Resolution.MINUTE)
        self.option = option
        self.option.PriceModel = OptionPriceModels.CrankNicolsonFD()
        self.option_symbol = option.Symbol
        
        # Add underlying reference
        equity = self.add_index("SPX", Resolution.MINUTE)
        self.underlying_symbol = equity.Symbol
        self.set_benchmark(self.underlying_symbol)
        
        # Configure option filter for OTM puts
        option.SetFilter(lambda u: u.Strikes(-60, 60)   # Wider range for OTM puts
                                  .Expiration(1, 1)      # 0-1 days to expiration
                                  .IncludeWeeklys())     # Include weekly options
                                  
        self.debug_count = 0
        self.max_debug_count = 20
        self.SetWarmUp(timedelta(days=1))
    def OnData(self, slice: Slice) -> None:
        # Print OTM calls and puts (I want to see the min and max strikes with their bid/ask)
        if self.is_warming_up or not slice.OptionChains:
            return
        if self.Time.minute % 15 == 0:
            for chain in slice.OptionChains.values():
                if chain.Symbol != self.option_symbol:
                    continue
                
                underlying_price = self.Securities[self.underlying_symbol].Price
                
                # Get and print OTM puts
                puts = [x for x in chain if x.Right == OptionRight.Put and x.Strike < underlying_price]
                if puts:
                    min_put = min(puts, key=lambda x: x.Strike)
                    max_put = max(puts, key=lambda x: x.Strike)
                    self.debug(f"[{self.Time}|{underlying_price}] OTM Puts: "
                             f"Min Strike: {min_put.Strike} (Bid: {min_put.BidPrice:.2f}, Ask: {min_put.AskPrice:.2f}), "
                             f"Max Strike: {max_put.Strike} (Bid: {max_put.BidPrice:.2f}, Ask: {max_put.AskPrice:.2f})")
                
                # Get and print OTM calls
                calls = [x for x in chain if x.Right == OptionRight.Call and x.Strike > underlying_price]
                if calls:
                    min_call = min(calls, key=lambda x: x.Strike)
                    max_call = max(calls, key=lambda x: x.Strike)
                    self.debug(f"[{self.Time}|{underlying_price}] OTM Calls: "
                             f"Min Strike: {min_call.Strike} (Bid: {min_call.BidPrice:.2f}, Ask: {min_call.AskPrice:.2f}), "
                             f"Max Strike: {max_call.Strike} (Bid: {max_call.BidPrice:.2f}, Ask: {max_call.AskPrice:.2f})")
