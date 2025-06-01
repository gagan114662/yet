from AlgorithmImports import *

class SPYCallCreditSpreadReddit(QCAlgorithm):

    def Initialize(self):
        self.SetCash(10000)  # Set initial cash
        self.SetStartDate(2022, 1, 1)  # Set Start Date (adjust as needed)
        self.SetEndDate(2023, 1, 1)    # Set End Date (adjust as needed)
        self.SetTimeZone(TimeZones.NewYork) # Standard US Market Time Zone
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersTws) # Use a realistic brokerage model for options

        # Add SPY equity
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Raw) # Don't split-adjust equity for options alignment

        # Add SPY option chain
        self.option = self.AddOption(self.spy.Symbol)
        # Set filter to only get call options and within the desired DTE range
        # Strikes(0, 100) is a wide range filter, the specific strike selection happens later
        self.option.SetFilter(lambda universe: universe.Strikes(0, 100).Expiration(self.min_dte, self.max_dte).CallsOnly())

        # Strategy parameters
        self.spread_width = 2.0      # $2 wide spread
        self.min_dte = 10            # Minimum days to expiration
        self.max_dte = 30            # Maximum days to expiration
        self.min_premium_target = 0.20 # Minimum premium per share ($20 per contract)

        # Track open position status
        self._position_open = False
        self._spread_expiry = None   # Store the expiration date of the open spread
        self._spread_legs = []       # Store the symbols of the legs if position is open

        # Schedule a daily event to check for trading opportunities
        # Check around market open time (adjust if needed)
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(9, 35), # Shortly after market open
                         self.CheckAndPlaceSpread)

        # Schedule removal of expired option symbols if necessary
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(17, 00), # After market close
                         self.CleanUpExpiredOptions)


    def OnData(self, slice):
        # OnData is not strictly needed for this scheduled strategy,
        # but can be used for monitoring or more complex logic.
        pass

    def OnSecuritiesChanged(self, changes):
        # Log added/removed securities (especially options)
        # for change in changes.AddedSecurities:
        #     self.Debug(f"Added Security: {change.Symbol} Type: {change.Type}")
        # for change in changes.RemovedSecurities:
        #     self.Debug(f"Removed Security: {change.Symbol} Type: {change.Type}")
        pass # Keep base implementation


    def CheckAndPlaceSpread(self):
        '''Checks if a spread can be placed based on strategy rules.'''

        # Only open a new spread if no position is currently open
        if self._position_open:
            # Check if the existing position is still valid (not expired/closed)
            # This is a basic check, more robust management is needed for real trading
            if self.Portfolio.Invested:
                 # Check if the held positions match our spread symbols
                 held_spread_legs = [h.Symbol for h in self.Portfolio.Values if h.Quantity != 0 and h.Symbol in self._spread_legs]
                 if len(held_spread_legs) == len(self._spread_legs) and len(held_spread_legs) > 0:
                      # Position is still open
                      self.Debug(f"Position already open, expires {self._spread_expiry.date()}. Skipping new trade check.")
                      return
                 else:
                     # Portfolio is invested, but doesn't seem to be our tracked spread.
                     # Could be dividends, fees, or other untracked positions.
                     # Or the spread expired/closed and Portfolio state hasn't updated perfectly yet.
                     # Let's assume the position was closed if our tracked symbols aren't held.
                     self.Debug("Tracked position symbols not found in portfolio. Resetting position flag.")
                     self._position_open = False
                     self._spread_legs = []
                     self._spread_expiry = None
            else:
                 # Portfolio is not invested at all, so our tracked position must be closed
                 self.Debug("Portfolio not invested. Resetting position flag.")
                 self._position_open = False
                 self._spread_legs = []
                 self._spread_expiry = None


        # --- Proceed if no position is open ---

        # Get the current SPY price
        spy_price = self.Securities["SPY"].Price
        if spy_price <= 0:
            self.Debug("SPY price data not available yet.")
            return

        # Get the option chain for SPY
        chain = self.OptionChainProvider.GetOptionChain(self.spy.Symbol, self.Time)
        if not chain:
            self.Debug("No option chain found.")
            return

        # Filter the chain for Call options within the desired DTE range (already set by self.option.SetFilter)
        # Now find the expiration date closest to the middle of the DTE range
        suitable_expirations = sorted(list(set([opt.Expiry for opt in chain if self.min_dte <= (opt.Expiry.date() - self.Time.date()).days <= self.max_dte])))

        if not suitable_expirations:
            self.Debug(f"No suitable expirations found within {self.min_dte}-{self.max_dte} DTE.")
            return

        # Find the expiration closest to the midpoint (e.g., 20 days)
        target_dte = (self.min_dte + self.max_dte) // 2
        expiry = min(suitable_expirations, key=lambda exp: abs((exp.date() - self.Time.date()).days - target_dte))

        options_at_expiry = [opt for opt in chain if opt.Expiry == expiry and opt.Right == OptionRight.Call]
        if not options_at_expiry:
            self.Debug(f"No call options found for expiration {expiry}.")
            return

        # Sort strikes for this expiration
        strikes = sorted(list(set(opt.Strike for opt in options_at_expiry)))

        # Find the short strike: the first strike price greater than the current SPY price
        short_strike = None
        for strike in strikes:
            if strike > spy_price:
                short_strike = strike
                break

        if short_strike is None:
            self.Debug(f"Could not find a strike above SPY price {spy_price:.2f} for expiration {expiry.date()}.")
            return

        # Find the long strike: short strike + spread_width
        long_strike = short_strike + self.spread_width
        if long_strike not in strikes:
             self.Debug(f"Could not find the required long strike {long_strike} for expiration {expiry.date()}.")
             return

        # Find the actual option contracts for the selected strikes and expiration
        short_call = next((opt for opt in options_at_expiry if opt.Strike == short_strike), None)
        long_call = next((opt for opt in options_at_expiry if opt.Strike == long_strike), None)

        if short_call is None or long_call is None:
            self.Debug(f"Could not find both required option contracts for {expiry.date()}, strikes {short_strike}, {long_strike}.")
            return

        # Calculate the potential premium (Bid of Short Call - Ask of Long Call)
        # This is the theoretical maximum premium you could receive immediately.
        if short_call.BidPrice <= 0 or long_call.AskPrice <= 0:
             self.Debug(f"Bid or Ask price is zero for required options. Cannot calculate premium.")
             return

        premium_per_share = short_call.BidPrice - long_call.AskPrice
        premium_per_spread = premium_per_share * 100 # Options contract multiplier is 100

        self.Debug(f"Potential premium for {expiry.date()} {short_strike}/{long_strike} spread: {premium_per_share:.2f} per share ({premium_per_spread:.2f} per contract).")

        # Check if the potential premium meets the minimum target
        if premium_per_share < self.min_premium_target:
            self.Debug(f"Potential premium {premium_per_share:.2f} is below minimum target {self.min_premium_target}. Skipping trade.")
            return

        # Define the quantity (number of spread contracts)
        quantity = 1 # Trading one spread contract at a time

        # Create the legs for the spread order
        # Note: Sell leg is negative quantity, Buy leg is positive quantity
        legs = [
            Leg.Create(short_call.Symbol, -quantity), # Sell the OTM call
            Leg.Create(long_call.Symbol, quantity)   # Buy the further OTM call
        ]

        # Place a Limit Order for the spread combination
        # The limit price for a credit spread is the premium we receive (per share)
        limit_price_spread = premium_per_share # This is premium *per share*

        # Note: Limit orders might not fill immediately in backtesting depending on price movement and resolution.
        # You could use a MarketOrder if immediate entry is critical, but LimitOrder is better for capturing premium.
        ticket = self.LimitOrder(legs, quantity, limit_price_spread)

        if ticket.Status != OrderStatus.Invalid:
            self.Debug(f"Placed Call Credit Spread order for {expiry.date()} {short_strike}/{long_strike} at limit {limit_price_spread:.2f}. Ticket ID: {ticket.OrderId}")
            # Tentatively mark position as open, will confirm on fill
            self._position_open = True
            self._spread_expiry = expiry
            self._spread_legs = [short_call.Symbol, long_call.Symbol] # Store symbols for tracking
        else:
             self.Debug(f"Failed to place spread order: {ticket.Response.ErrorMessage}")


    def OnOrderEvent(self, orderEvent):
        '''Handles order fill events.'''
        order = self.Transactions.GetOrderById(orderEvent.OrderId)

        if orderEvent.Status == OrderStatus.Filled:
            if order.Type == OrderType.ComboLimit:
                 # Check if this filled order corresponds to the spread we tried to open
                 # A simple check is to see if the symbols match our tracked legs
                 # (More robust logic needed for multiple simultaneous orders)
                 if self._position_open and all(leg.Symbol in self._spread_legs for leg in order.Legs):
                      self.Debug(f"Spread Order Fully Filled. Ticket ID: {orderEvent.OrderId}, Price: {orderEvent.FillPrice}, Quantity: {orderEvent.FillQuantity}")
                      # Position is confirmed open, state variables are already set
            elif order.Type in [OrderType.Limit, OrderType.Market] and order.Quantity < 0 and len(self._spread_legs) == 1 and order.Symbol == self._spread_legs[0]:
                 # This could potentially be a partial fill of one leg if not using Combo orders correctly,
                 # or a separate closing order if implemented later.
                 # For this strategy, we expect ComboLimit fills.
                 pass # Ignore fills of individual legs if not part of combo logic

        elif orderEvent.Status == OrderStatus.Canceled or orderEvent.Status == OrderStatus.Invalid:
             # If the opening order is cancelled/invalid, reset the position flag
             if self._position_open and self._spread_legs and any(leg.Symbol == orderEvent.Symbol for leg in order.Legs):
                  self.Debug(f"Spread Order Cancelled/Invalid. Ticket ID: {orderEvent.OrderId}. Resetting position flag.")
                  self._position_open = False
                  self._spread_legs = []
                  self._spread_expiry = None


    def CleanUpExpiredOptions(self):
        '''Basic cleanup - Note: QC handles removal of expired options, this is mostly for logging/state.'''
        if self._position_open and self._spread_expiry is not None and self.Time.date() >= self._spread_expiry.date():
            # Check if the positions are still in the portfolio with quantity
            short_holding = self.Portfolio.get_security_holding(self._spread_legs[0])
            long_holding = self.Portfolio.get_security_holding(self._spread_legs[1])

            if short_holding is not None and short_holding.Quantity == 0 and \
               long_holding is not None and long_holding.Quantity == 0:
                 self.Debug(f"Spread position for {self._spread_expiry.date()} appears closed/expired. Resetting position flag.")
                 self._position_open = False
                 self._spread_legs = []
                 self._spread_expiry = None
            # If quantities are NOT zero here, it indicates the spread might still be live
            # or there was an issue with expiration handling requiring manual intervention (e.g., assignment)
            # In a real bot, this would require more sophisticated checks or alerts.


    def OnEndOfAlgorithm(self):
        '''Clean up any remaining positions at the end of backtest.'''
        if self.Portfolio.Invested:
            self.Debug(f"Algorithm ending. Liquidating all positions.")
            self.Liquidate()
