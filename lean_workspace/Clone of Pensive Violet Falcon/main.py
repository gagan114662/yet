# region imports
from AlgorithmImports import *
# endregion

class OptionsStrategyDev(QCAlgorithm): 
    def initialize(self) -> None:
        self.set_start_date(2020, 4, 1)
        self.set_end_date(2021, 4, 30)
        self.set_cash(100000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.universe_settings.asynchronous = True
        option = self.add_option("SPY", Resolution.MINUTE)
        self._symbol = option.symbol
        option.set_filter(-10, 10, 0, 10)
        self.print_bool = True

    def on_data(self, slice: Slice) -> None:
        
        # Check if we already have 2 positions
        positions = [x for x in self.Portfolio.Values if x.Invested]
        if len(positions) >= 2:
            return

        # Get the OptionChain
        chain = slice.option_chains.get(self._symbol, None)
        if not chain:
            return
        
        # Select an expiry date
        # Select options expiring today (0DTE)
        expiry = min([x.expiry for x in chain])
        if self.print_bool:
            self.Debug(f"{self.Time}| Min Expiry: {expiry}")

        # Select the call and put contracts that expire on the selected date
        calls = [x for x in chain if x.right == OptionRight.CALL and x.expiry == expiry]
        puts = [x for x in chain if x.right == OptionRight.PUT and x.expiry == expiry]
        if self.print_bool:
            self.Debug({'calls': len(calls), 'puts': len(puts)})
            for i, put in enumerate(puts):
                self.Debug(f"{self.Time}| Put {i}: {put.symbol} | Strike: {put.strike} | IV: {put.implied_volatility:.2f} | $: {put.last_price} | SPY: {chain.underlying.price}")
            
            for i, call in enumerate(calls):
                self.Debug(f"{self.Time}| Call {i}: {call.symbol} | Strike: {call.strike} | IV: {call.implied_volatility:.2f} | $: {call.last_price} | SPY: {chain.underlying.price}")
        
        if not calls or not puts:
            return

        # Select the OTM contracts
        # Filter puts that are at least 1% away from current price and price >= 2.5
        valid_puts = [x for x in puts if
                     x.strike <= chain.underlying.price * 0.99 and  # At least 1% OTM
                     x.last_price >= 2.5]                           # Price >= 2.5
        
        if not valid_puts:
            if self.print_bool:
                self.Debug("No valid puts found matching criteria")
            return
            
        put = sorted(valid_puts, key = lambda x: x.strike)[0]  # Get the lowest strike from valid puts

        if self.print_bool:
            self.Debug(f"Selected put: Strike: {put.strike}, Price: {put.last_price}, Distance from underlying: {((chain.underlying.price - put.strike)/chain.underlying.price)*100:.2f}%")

        legs = [
        Leg.create(put.symbol, -1),
        # Leg.create(chain.underlying.symbol, 2)
        ]
        self.combo_market_order(legs, 1)
        self.print_bool = False        
