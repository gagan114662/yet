from AlgorithmImports import *
# endregion

class VandanStrategyBurke(QCAlgorithm):
    def Initialize(self) -> None:
        self.SetStartDate(2021, 5, 1)
        self.SetEndDate(2021, 5, 10)
        self.SetCash(200_000)
        self.SetTimeZone(TimeZones.NEW_YORK)
        """Initialize the strategy"""
        # Basic settings
        self.SetStartDate(2021, 5, 1)
        self.SetCash(200_000)
        self.SetTimeZone(TimeZones.NewYork)
        
        # Add our option chain
        option = self.AddOption("SPY")
        self._symbol = option.Symbol
        
        # Add underlying for reference
        equity = self.AddEquity("SPY", Resolution.Minute)
        
        # Configure option filter for OTM puts
        option.SetFilter(lambda u: u.Strikes(-100, 10)   # Wider range for OTM puts
                                  .Expiration(0, 2)      # 0-1 days to expiration
                                  .IncludeWeeklys())     # Include weekly options
                                  
        self.debug_count = 0
        self.max_debug_count = 100

    def GetLongPutLeg(self, chain, expiry, short_strike):
        """Get the long put leg - target 4% below short strike"""
        puts = [x for x in chain if x.right == OptionRight.PUT and x.expiry == expiry]
        
        # Calculate target strike (4% below short strike)
        target = short_strike * 0.96
        
        # Get puts below target and sort by strike price
        valid_puts = [x for x in puts if x.strike < target]
        valid_puts = sorted(valid_puts, key=lambda x: x.strike)
        
        if not valid_puts:
            return None
            
        # Take the highest strike (closest to but below target)
        put = valid_puts[-1]
        return Leg.create(put.symbol, 1), put

    def GetShortPutLeg(self, chain, expiry):
        """Get the short put leg - at least 1% OTM with premium >= $2.5"""
        puts = [x for x in chain if x.right == OptionRight.PUT and x.expiry == expiry]
        if not puts:
            return None
            
        # Debug chain info
        self.PrintOutput(f"[{self.Time}] SPY price: ${chain.underlying.price:0.2f}")
        self.PrintOutput(f"[{self.Time}] {len(puts)} puts available for {expiry}")
        
        # Print available puts info
        self.PrintOutput(f"[{self.Time}] Underlying price: ${chain.underlying.price:0.2f}")
        self.PrintOutput(f"[{self.Time}] Total puts: {len(puts)}")
        
        # Find OTM puts
        otm_strike = chain.underlying.price * 0.99
        otm_puts = [x for x in puts if x.strike <= otm_strike]
        self.PrintOutput(f"[{self.Time}] OTM puts (<= ${otm_strike:0.2f}): {len(otm_puts)}")
        
        # Show some OTM put prices
        if otm_puts:
            for put in sorted(otm_puts, key=lambda x: x.strike)[:3]:
                self.PrintOutput(f"[{self.Time}] Strike ${put.strike:0.2f}: Bid=${put.bid_price:0.2f}")
        
        # Filter for premium
        valid_puts = [x for x in otm_puts if x.bid_price >= 2]
        self.PrintOutput(f"[{self.Time}] Valid puts (Premium >= $2.00): {len(valid_puts)}")
        
        if not valid_puts:
            return None
            
        # Get the lowest strike (most OTM) from valid puts
        put = sorted(valid_puts, key=lambda x: x.strike)[0]
        return Leg.create(put.symbol, -1), put

    def PrintOutput(self, msg):
        if self.debug_count < self.max_debug_count:
            self.Debug(msg)
            self.debug_count += 1
        else:
            raise Exception("Debug limit reached. Please check the logs for more information.")
            
    def OnOrderEvent(self, orderEvent):
        """Handle order fills and assignments"""
        if orderEvent.Status == OrderStatus.Filled:
            # Handle option fills
            if orderEvent.Symbol.SecurityType == SecurityType.Option:
                # Check for fills after expiry
                if orderEvent.Symbol.ID.Date.date() < self.Time.date():
                    self.PrintOutput(f"[{self.Time}] WARNING: Fill after expiry for {orderEvent.Symbol}")
                
                # Log short entries
                if orderEvent.FillQuantity < 0:
                    self.PrintOutput(f"[{self.Time}] NEW SHORT - Put {orderEvent.Symbol.ID.StrikePrice} @ ${orderEvent.FillPrice:0.2f}")
            
            # Handle assignments
            elif orderEvent.Symbol.SecurityType == SecurityType.Equity:
                if orderEvent.Message.lower().startswith("option assignment"):
                    self.MarketOrder(orderEvent.Symbol, -orderEvent.FillQuantity)  # Close position immediately
                    self.PrintOutput(f"[{self.Time}] ASSIGNED & CLOSING - {abs(orderEvent.FillQuantity)} shares of {orderEvent.Symbol.Value}")
                
    def ManageStopLoss(self, slice: Slice):
        """Monitor and manage stop-loss for short put positions"""
        
        for security in self.Securities.Values:
            if not security.Holdings.Invested:
                continue
                
            if (security.Symbol.SecurityType == SecurityType.Option and
                security.Symbol.ID.OptionRight == OptionRight.PUT and
                security.Holdings.Quantity < 0):
                
                # Skip expired options
                if security.Symbol.ID.Date.date() < self.Time.date():
                    self.PrintOutput(f"[{self.Time}] Skipping expired Put {security.Symbol.ID.StrikePrice} (Expired: {security.Symbol.ID.Date.date()})")
                    continue
                
                current_price = security.Price
                unrealized_pl = security.Holdings.UnrealizedProfit
                
                # # Log every 5 minutes
                # if self.Time.minute % 5 == 0:
                #     self.PrintOutput(f"[{self.Time}] MONITOR - Put {security.Symbol.ID.StrikePrice}: ${current_price:0.2f}, P&L=${unrealized_pl:0.2f}")
                
                # Check stop-loss conditions
                if current_price >= 8.0:
                    if not self.Transactions.GetOpenOrders(security.Symbol):
                        order = self.LimitOrder(security.Symbol, -security.Holdings.Quantity, 20.0)
                        if order.Status != OrderStatus.Invalid:
                            self.PrintOutput(f"[{self.Time}] STOPLOSS - Put {security.Symbol}: {security.Symbol.ID.StrikePrice}: ${current_price:0.2f} (P&L: ${unrealized_pl:0.2f})")

    def OnData(self, slice: Slice) -> None:
        # Check stop-losses
        self.ManageStopLoss(slice)
        
        # Check if we already have 2 positions
        positions = [x for x in self.Portfolio.Securities if x.Value.Invested]
        if len(positions) >= 2:
            return

        # Get the OptionChain for new trades
        chain = slice.option_chains.get(self._symbol, None)
        if not chain:
            return
        # # Debug chain data
        # if slice.option_chains:
        #     self.PrintOutput(f"[{self.Time}] Found {len(slice.option_chains)} option chains")
        
        if chain:
            # self.PrintOutput(f"[{self.Time}] SPY chain has {len(chain)} contracts")
            puts = [x for x in chain if x.right == OptionRight.PUT]
            # self.PrintOutput(f"[{self.Time}] Chain contains {len(puts)} puts")
        # else:
        #     self.PrintOutput(f"[{self.Time}] No chain data for SPY")
        #     return
        
        # Get all expiries and sort them
        all_expiries = sorted(list(set([x.expiry for x in chain])))
        
        # # Debug closest expiries
        # self.PrintOutput(f"[{self.Time}] Current time: {self.Time}")
        # self.PrintOutput(f"[{self.Time}] Next 3 expiries:")
        # for exp in all_expiries[:3]:
        #     self.PrintOutput(f"[{self.Time}]   - {str(exp)}")
        
        # Find valid expirations (today's date and not expired)
        current_time = self.Time
        valid_expiries = [exp for exp in all_expiries
                         if exp > current_time]                     # Future expiry
        
        # # Check if market is open
        # is_market_open = self.Securities[self._symbol.Underlying].Exchange.ExchangeOpen
        # if not is_market_open:
        #     return
            
        # Log expiry details
        if valid_expiries:
            expiry = min(valid_expiries)
            hours_to_expiry = (expiry - current_time).total_seconds() / 3600
            self.PrintOutput(f"[{self.Time}] Found option expiring in {hours_to_expiry:.1f} hours")
        else:
            self.PrintOutput(f"[{self.Time}] No valid expiries options found")
        
        if not valid_expiries:
            self.PrintOutput(f"[{self.Time}] No valid expiries found for SPY")
            return
            
        # Get closest expiry
        expiry = min(valid_expiries)
        time_to_expiry = expiry - self.Time
        hours_to_expiry = time_to_expiry.total_seconds() / 3600
        
        # # Log timing details to debug expiry selection
        # self.PrintOutput(f"[{self.Time}] Current time: {self.Time.strftime('%Y-%m-%d %H:%M')}")
        # self.PrintOutput(f"[{self.Time}] Option expiry: {expiry.strftime('%Y-%m-%d %H:%M')} ({hours_to_expiry:.1f}h remaining)")
        
        # Get both legs
        short_leg_result = self.GetShortPutLeg(chain, expiry)
        if not short_leg_result:
            self.PrintOutput(f"[{self.Time}] No short leg found for expiry {expiry}")
            return
        short_leg, short_put = short_leg_result
        
        # Get the long put leg (20-point spread)
        long_leg_result = self.GetLongPutLeg(chain, expiry, short_put.strike)
        if not long_leg_result:
            self.PrintOutput(f"[{self.Time}] No long leg found for short put {short_put.strike}")
            return
        long_leg, long_put = long_leg_result
        
        legs = [short_leg, long_leg]
        
        # Check margin usage including new trade
        margin_multiplier = 3.3
        total_portfolio_value = self.Portfolio.TotalPortfolioValue
        total_available_margin = total_portfolio_value * margin_multiplier
        
        # Calculate current margin used
        current_margin_used = sum([x.Value.Holdings.AbsoluteHoldingsCost for x in self.Portfolio.Securities if x.Value.Invested])
        
        # Calculate new margin required
        # Calculate spread width
        spread_width = short_put.strike - long_put.strike
        
        # For put credit spread, margin is width of spread - credit received
        credit_received = short_put.last_price - long_put.last_price
        new_margin_required = (spread_width - credit_received) * 100
        
        # Calculate total margin after this trade
        total_margin = current_margin_used + new_margin_required
        margin_usage_pct = (total_margin / total_available_margin) * 100
        
        if margin_usage_pct > 15:
            self.PrintOutput(f"[{self.Time}] WARNING: Margin usage {margin_usage_pct:.2f}% exceeds 15% limit. Skipping trade.")
            return
            
        # Place the trades
        self.ComboMarketOrder(legs, 1)
        
        # Add options to our universe
        for contract in [short_put.symbol, long_put.symbol]:
            self.AddOptionContract(contract, Resolution.Minute)
        
        # Log the new position
        self.PrintOutput(f"[{self.Time}] NEW SPREAD - Put {short_put.strike}/{long_put.strike} @ ${credit_received:0.2f} credit")
