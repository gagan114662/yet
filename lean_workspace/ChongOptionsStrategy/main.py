# region imports
from AlgorithmImports import *
# endregion

class BearCallSpreadStrategy(QCAlgorithm): 
    def initialize(self) -> None:
        self.set_start_date(2017, 4, 1)
        self.set_end_date(2017, 4, 30)
        self.set_cash(100000)

        self.universe_settings.asynchronous = True
        option = self.add_option("SPY", Resolution.MINUTE)
        self._symbol = option.symbol
        option.set_filter(lambda universe: universe.include_weeklys().protective_collar(30, -1, -10))

    def on_data(self, slice: Slice) -> None:
        if self.portfolio.invested:
            return

        # Get the OptionChain
        chain = slice.option_chains.get(self._symbol, None)
        if not chain:
            return

        # Select an expiry date
        expiry = max([x.expiry for x in chain])

        # Select the call and put contracts that expire on the selected date
        calls = [x for x in chain if x.right == OptionRight.CALL and x.expiry == expiry]
        puts = [x for x in chain if x.right == OptionRight.PUT and x.expiry == expiry]
        if not calls or not puts:
            return

        # Select the OTM contracts
        call = sorted(calls, key = lambda x: x.strike)[-1]
        put = sorted(puts, key = lambda x: x.strike)[0]

        legs = [
        Leg.create(call.symbol, 1),
        Leg.create(put.symbol, -1),
        # Leg.create(chain.underlying.symbol, 2)
        ]
        self.combo_market_order(legs, 1)
