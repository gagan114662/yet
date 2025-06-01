# region imports
from AlgorithmImports import *
# endregion

class OptionsStrategyDev(QCAlgorithm): 
    def initialize(self) -> None:
        self.set_start_date(2024, 4, 1)
        self.set_end_date(2024, 4, 30)
        self.set_cash(100000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.universe_settings.asynchronous = True
        option = self.add_option("SPY", Resolution.MINUTE)
        self._symbol = option.symbol
        # option.set_filter(-10, 10, 0, 10)
        option.SetFilter(self.OptionFilter)
        self.print_bool = True
        self.debug_count = 0
        self.max_debug_count = 5

    def OptionFilter(self, universe: OptionFilterUniverse):
        # Include both standard monthlys and weeklies, expiring in 0–7 days,
        # strikes within ±2 levels
        return (
            universe
                .IncludeWeeklys()       # include non‑standard (weekly) series
                .Expiration(0, 2)       # expiring within 0–7 days
                .Strikes(-15, 15)         # within 2 strikes of the underlying
        )
    
    def on_data(self, slice: Slice) -> None:
        # if self.debug_count > self.max_debug_count * 2:
        #     return 
        
        # Check if we already have 2 positions
        positions = [x for x in self.Portfolio.Values if x.Invested]
        if len(positions) >= 2:
            return

        # Get the OptionChain
        chain = slice.option_chains.get(self._symbol, None)
        if not chain:
            return
        
        # Find options expiring today (0DTE)
        today = self.Time.date()
        today_expiry = [x.expiry.date() for x in chain if x.expiry.date() == today]
        
        if not today_expiry:
            return
            
        # Select the call and put contracts that expire today
        calls = [x for x in chain if x.right == OptionRight.CALL and x.expiry.date() == today]
        puts = [x for x in chain if x.right == OptionRight.PUT and x.expiry.date() == today]
        # if self.print_bool:
        #     self.Debug({'calls': len(calls), 'puts': len(puts)})
        #     for i, put in enumerate(puts):
        #         self.Debug(f"{self.Time}| Put {i}: {put.symbol} | Strike: {put.strike} | IV: {put.implied_volatility:.2f} | $: {put.last_price} | SPY: {chain.underlying.price}")
            
        #     for i, call in enumerate(calls):
        #         self.Debug(f"{self.Time}| Call {i}: {call.symbol} | Strike: {call.strike} | IV: {call.implied_volatility:.2f} | $: {call.last_price} | SPY: {chain.underlying.price}")
        
        if not calls or not puts:
            return

        # Debug first 5 puts in one line (only first 5 times)
        if self.print_bool and self.debug_count < self.max_debug_count:
            # Filter for OTM puts and sort by distance from underlying
            otm_puts = [p for p in puts if p.strike < chain.underlying.price]
            sorted_puts = sorted(otm_puts,
                               key=lambda x: (abs(x.strike - chain.underlying.price), -x.strike))[:5]
            put_info = [f"[S:{p.strike},P(b):${p.bid_price:.2f},D:{((chain.underlying.price - p.strike)/chain.underlying.price)*100:.1f}%,E:{p.expiry.strftime('%d%m')}]" for p in sorted_puts]
            self.Debug(f"{self.Time}| SPY: {chain.underlying.price:.2f} | Puts: {' | '.join(put_info)}")
            self.debug_count += 1

        # Select the OTM contracts
        # Filter puts for short leg (1.5% OTM)
        valid_short_puts = [x for x in puts if
                          x.strike <= chain.underlying.price * 0.995 and  # At least 1.5% OTM
                          x.last_price >= 0.2]                           # Price >= 2
        
        # Filter puts for long leg (4.5% OTM)
        valid_long_puts = [x for x in puts if
                         x.strike <= chain.underlying.price * 0.975]  # Around 4.5% OTM
        
        if not valid_short_puts or not valid_long_puts:
            return
            
        # Get puts closest to our target percentages
        short_put = min(valid_short_puts,
                       key = lambda x: abs(x.strike - chain.underlying.price * 0.995))  # Get closest to 1.5% OTM
        long_put = min(valid_long_puts,
                      key = lambda x: abs(x.strike - chain.underlying.price * 0.975))   # Get closest to 4.5% OTM

        legs = [
            Leg.create(short_put.symbol, -1),  # Sell the 1% OTM put
            Leg.create(long_put.symbol, 1)     # Buy the 4% OTM put
        ]
        self.combo_market_order(legs, 1)
