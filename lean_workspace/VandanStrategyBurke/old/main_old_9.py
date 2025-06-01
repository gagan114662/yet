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
        self.SetTimeZone(TimeZones.NEW_YORK)
        self.strategy_params()  # Initialize strategy parameters
        # Basic settings
        self.SetCash(200_000)
        
        # Add option chain with pricing model
        option = self.AddOption("SPY", Resolution.MINUTE)
        self.option = option
        self.option.PriceModel = OptionPriceModels.CrankNicolsonFD()
        self.option_symbol = option.Symbol
        
        # Add underlying reference
        equity = self.AddEquity("SPY", Resolution.MINUTE)
        self.underlying_symbol = equity.Symbol
        self.set_benchmark(self.underlying_symbol)
        
        # Configure option filter for OTM puts
        option.SetFilter(lambda u: u.Strikes(-200, 200)   # Wider range for OTM puts
                                  .Expiration(1, 1)      # 0-1 days to expiration
                                  .IncludeWeeklys())     # Include weekly options
                                  
        self.debug_count = 0
        self.max_debug_count = 20
        self.SetWarmUp(timedelta(days=1))

        self.last_order_time = None
        self.pending_liquidation = False
        self.exit_orders = {}  # Track exit orders for positions
        self.margin_target = 0.80  # Use only 60% of available margin
    
    def strategy_params(self) -> None:
        # Strategy parameters
        self.SetStartDate(2024, 1, 1)
        # self.SetEndDate(2025, 2, 12)
        self.market_strength_threshold = 0  # Threshold for market strength to trigger trades
        self.sell_leg_atr_multiplier = 0.5 # Multiplier for ATR distance
        self.buy_leg_insurance_atr_multiplier = 1  # Multiplier for insurance leg distance
        self.minimum_premium = 0.25
        self.buy_leg_insurance_max_premium = 0.03
        self.conditional_exit_price = self.minimum_premium * 4  # Price to trigger exit order
        self.position_size = 10  # Number of contracts to trade
    
    def margin_check(self, sell_contract, buy_contract, quantity, available_margin) -> bool:
        """
        Calculate margin requirement for a vertical spread
        For credit spreads, margin requirement is (width of spread - net credit) * number of contracts
        """
        contract_multiplier = 100
        
        # Calculate net credit
        sell_price = sell_contract.BidPrice
        buy_price = buy_contract.AskPrice
        net_credit = (sell_price - buy_price) * contract_multiplier * quantity
        
        # Calculate width of spread
        strike_width = abs(sell_contract.Strike - buy_contract.Strike)
        max_risk = strike_width * contract_multiplier * quantity
        
        # Required margin is max risk minus net credit received
        required_margin = max_risk - net_credit
        
        # Add 10% buffer for safety
        required_margin *= 1.1
        
        underlying_price = self.Securities[self.underlying_symbol].Price
        self.PrintOutput(f"[{self.Time}|${underlying_price:.2f}] MARGIN CHECK - "
                f"Required: ${required_margin:.2f}, Available: ${available_margin:.2f}, "
                f"Max Risk: ${max_risk:.2f}, Net Credit: ${net_credit:.2f}", print=False)
        
        return available_margin >= required_margin
    
    def PrintOutput(self, msg, print=False):
        # if self.Time.minute == 00 and 9 <= self.Time.hour < 16:
            if print and self.debug_count < self.max_debug_count:
                self.Debug(f"{self.debug_count} ::: {msg}")
                self.debug_count += 1
        
    def get_OTM_contracts(self, chain, slice):
        # Get all the OTM puts and sort them by strike price
        otm_puts = [x for x in chain if x.Right == OptionRight.Put and x.Strike < slice.Bars[self.underlying_symbol].Price]
        otm_puts.sort(key=lambda x: x.Strike, reverse=True)
        
        # Get all the OTM calls and sort them by strike price
        otm_calls = [x for x in chain if x.Right == OptionRight.Call and x.Strike > slice.Bars[self.underlying_symbol].Price]
        otm_calls.sort(key=lambda x: x.Strike)
        
        return otm_puts, otm_calls
    
    def get_market_strength(self):
        # THIS LINE IS VERY SLOW AND TAKES A LONG TIME TO EXECUTE
        start_time = self.Time.replace(hour=9, minute=30, second=0, microsecond=0)  # Start from the previous day
        end_time = self.Time  # current algorithm time
        underlying_price_history = self.History(self.underlying_symbol, start_time, end_time, Resolution.Minute)['close'].to_list()
        
        # underlying_price_history = underlying_price_history_31d.loc[underlying_price_history_31d.index.get_level_values(1).date == self.Time.date()]['close']
        if len(underlying_price_history) == 0:
            return None
            
        current_price = underlying_price_history[-1]  # Latest price
        current_minutes = self.Time.hour * 60 + self.Time.minute
        market_open_minutes = 9 * 60 + 30  # 9:30 AM in minutes
        
        # Initialize opening range variables if not already done
        if not hasattr(self, 'opening_range_high'):
            self.opening_range_high = None
            self.opening_range_low = None
        
        # First hour: Build opening range (9:30 - 10:30)
        # First hour: Build opening range (9:30 - 10:30)
        if current_minutes >= market_open_minutes and current_minutes < (market_open_minutes + 60):
            if self.opening_range_high is None or current_price > self.opening_range_high:
                self.opening_range_high = current_price
            if self.opening_range_low is None or current_price < self.opening_range_low:
                self.opening_range_low = current_price
                
            # Log opening range only if no positions
            if current_minutes == 10 * 60 and not any(x.Invested for x in self.Portfolio.Values):
                self.PrintOutput(f"[{self.Time}|${current_price:.2f}] Opening Range Status: "
                       f"Low=${float(self.opening_range_low):.2f}, High=${float(self.opening_range_high):.2f}")
            return None
        
        # After first hour: Use opening range and momentum
        if self.opening_range_high is None or self.opening_range_low is None:
            self.PrintOutput(f"No opening range established yet", print=True)
            return None
            
        # Calculate opening range breakout signal (-5 to +5)
        opening_range = self.opening_range_high - self.opening_range_low
        if opening_range == 0:
            self.PrintOutput(f"Opening range is zero", print=True)
            return None  # Can't calculate signal without a range
        else:
            range_midpoint = self.opening_range_low + (opening_range / 2)
            range_signal = ((current_price - range_midpoint) / (opening_range / 2)) * 5
            self.PrintOutput(f"Range Signal: {float(range_signal):.2f} (Current: {float(current_price):.2f}, Mid: {float(range_midpoint):.2f})", print=False)
            
        # Calculate momentum signal (-5 to +5)
        if len(underlying_price_history) < 15:  # Need at least 15 minutes of history
            # self.PrintOutput(f"Insufficient price history", print=True)
            return None  # Need more price history
        else:
            price_change = current_price - underlying_price_history[-15]  # 15-minute change
            momentum_signal = (price_change / current_price) * 100 * 5  # Scale to -5 to +5
            self.PrintOutput(f"Momentum Signal: {float(momentum_signal):.2f} (15min change: {float(price_change):.2f})", print=False)
            
        # Combine signals (60% range breakout, 40% momentum)
        final_signal = (range_signal * 0.4) + (momentum_signal * 0.6)
        final_signal = max(min(final_signal, 5), -5)  # Clamp between -5 and +5
        
        # Log signal info at start of each hour if no positions
        if self.Time.minute == 0 and not any(x.Invested for x in self.Portfolio.Values):
            self.PrintOutput(f"[{self.Time}|${current_price:.2f}] SIGNALS - "
                   f"Range: {range_signal:.2f}, Momentum: {momentum_signal:.2f}, Final: {final_signal:.2f}", print=False)
            
        return final_signal
    
    def get_underlying_price_history(self):
        start_time = self.Time.replace(hour=9, minute=30, second=0, microsecond=0) - timedelta(days=31)  # Start from the previous day
        end_time = self.Time  # current algorithm time
        underlying_price_history_31d = self.History(self.underlying_symbol, start_time, end_time, Resolution.Minute)
        return underlying_price_history_31d
    
    def calculate_atr_by_remaining_time(self) -> float:
        """Calculate ATR based on remaining time until market close"""
        current_time = self.Time
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        current_minutes = current_time.hour * 60 + current_time.minute
        
        if current_minutes >= 16 * 60:  # After market close
            return None
            
        # Get historical data
        price_history = self.get_underlying_price_history()
        if len(price_history) == 0:
            return None
            
        # Convert to DataFrame and add date/time columns
        df = price_history[['high', 'low', 'close']]
        df['date'] = df.index.get_level_values(1).date
        df['time'] = df.index.get_level_values(1).time
        df['minutes'] = df['time'].apply(lambda x: x.hour * 60 + x.minute)
        
        # Filter for data after current time of day
        df_filtered = df[df['minutes'] >= current_minutes]
        
        # Group by date and calculate daily high/low/close for remaining period
        daily_data = df_filtered.groupby('date').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()
        
        # Calculate true range for each day
        daily_data['tr1'] = daily_data['high'] - daily_data['low']
        daily_data['tr2'] = abs(daily_data['high'] - daily_data['close'].shift(1))
        daily_data['tr3'] = abs(daily_data['low'] - daily_data['close'].shift(1))
        daily_data['true_range'] = daily_data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Debug output
        # self.PrintOutput(f"Current time: {current_time.strftime('%H:%M')}")
        # self.PrintOutput(f"Data points per day after {current_time.strftime('%H:%M')}:")
        for date in df_filtered['date'].unique():
            count = len(df_filtered[df_filtered['date'] == date])
            # self.PrintOutput(f"{date}: {count} bars")
        
        # Calculate ATR using last 5 days
        atr = float(daily_data['true_range'].tail(5).mean())
        
        # self.PrintOutput(f"Daily true ranges: {[f'{tr:.3f}' for tr in daily_data['true_range'].tail(5).tolist()]}")
        # self.PrintOutput(f"Final ATR: {atr:.3f}")
        
        return atr
    
    def get_sell_leg(self, slice, otm_puts, otm_calls, market_strength, remaining_bars_atr):
        """Get the option contract to sell based on market strength and ATR distance"""
        if remaining_bars_atr is None or market_strength is None:
            return None
        
        current_price = slice.Bars[self.underlying_symbol].Price
        strike_distance = self.sell_leg_atr_multiplier * remaining_bars_atr
        
        # Log analysis only if no positions
        if not any(x.Invested for x in self.Portfolio.Values):
            self.PrintOutput(f"[{self.Time}|${current_price:.2f}] ANALYSIS - "
                   f"Market Strength: {market_strength:.2f}, ATR: {remaining_bars_atr:.2f}, "
                   f"Strike Distance: {strike_distance:.2f}")
        
        # Choose direction based on market strength
        if market_strength < self.market_strength_threshold * -1:
            # Bearish - Sell Call
            target_strike = current_price + strike_distance
            
            # Log calls around target strike
            if not any(x.Invested for x in self.Portfolio.Values):
                # Sort by distance from target strike
                nearby_calls = sorted(otm_calls, key=lambda x: abs(x.Strike - target_strike))
                below_target = [c for c in otm_calls if c.Strike < target_strike][-3:]  # Last 3 below target
                above_target = [c for c in otm_calls if c.Strike >= target_strike][:3]  # First 3 above target
                
                strikes_below = [f"${c.Strike:.2f}(${c.BidPrice:.2f})" for c in below_target]
                strikes_above = [f"${c.Strike:.2f}(${c.BidPrice:.2f})" for c in above_target]
                
                self.PrintOutput(f"[{self.Time}|${current_price:.2f}] BEARISH DETAILS - "
                       f"Current: ${current_price:.2f}, Target: ${target_strike:.2f}", print=False)
                self.PrintOutput(f"Below Target: {', '.join(strikes_below)}\n"
                       f"Above Target: {', '.join(strikes_above)}", print=False)
            
            # Filter pre-filtered OTM calls
            valid_options = [contract for contract in otm_calls if contract.Strike >= target_strike]
            
            # Sort and filter by premium
            if valid_options:
                valid_options.sort(key=lambda x: abs(x.BidPrice - self.minimum_premium))
                valid_before = len(valid_options)
                valid_options = [x for x in valid_options if x.BidPrice >= self.minimum_premium]
                
                # Log premium filtering details
                if not any(x.Invested for x in self.Portfolio.Values):
                    self.PrintOutput(f"[{self.Time}|${current_price:.2f}] BEARISH PREMIUM FILTER - "
                           f"Before: {valid_before}, After: {len(valid_options)}, "
                           f"Min Premium: ${self.minimum_premium:.2f}", print=False)
            
            # Log search details only if no positions
            if not any(x.Invested for x in self.Portfolio.Values):
                valid_count = len(valid_options)
                premium_range = [f"${x.BidPrice:.2f}" for x in valid_options[:3]] if valid_options else []
                self.PrintOutput(f"[{self.Time}|${current_price:.2f}] BEARISH SEARCH - "
                       f"Target Strike >= ${target_strike:.2f}, Valid Options: {valid_count}, "
                       f"Top Premiums: {', '.join(premium_range)}", print=False)
            
        elif market_strength > self.market_strength_threshold:
            # Bullish - Sell Put
            target_strike = current_price - strike_distance
            
            # Filter pre-filtered OTM puts below target strike
            valid_options = [contract for contract in otm_puts if contract.Strike <= target_strike]
            
            # Sort by how close their premium is to minimum_premium
            if valid_options:
                valid_options.sort(key=lambda x: abs(x.BidPrice - self.minimum_premium))
                # Only keep options with acceptable premium
                valid_options = [x for x in valid_options if x.BidPrice >= self.minimum_premium]
            
            # Log search details only if no positions
            if not any(x.Invested for x in self.Portfolio.Values):
                valid_count = len(valid_options)
                premium_range = [f"${x.BidPrice:.2f}" for x in valid_options[:3]] if valid_options else []
                self.PrintOutput(f"[{self.Time}|${current_price:.2f}] BULLISH SEARCH - "
                       f"Target Strike <= ${target_strike:.2f}, Valid Options: {valid_count}, "
                       f"Top Premiums: {', '.join(premium_range)}", print=False)
        else:
            return None
        
        if valid_options:
            selected = valid_options[0]  # Take the closest strike that meets criteria
            self.PrintOutput(f"Selected {selected.Right} option at strike ${selected.Strike:.2f} with bid ${selected.BidPrice:.2f}", print=False)
            return selected
            
        analysis_details = False  # Set to True to enable detailed analysis output
        if analysis_details:
            # Prepare analysis data
            rejected_by_strike = 0
            rejected_by_price = 0
            is_bullish = market_strength > self.market_strength_threshold
            option_type = "PUTS" if is_bullish else "CALLS"
            contracts = otm_puts if is_bullish else otm_calls
            total_count = len(contracts)
            
            # Get min/max strikes
            min_strike = min(c.Strike for c in contracts) if contracts else 0
            max_strike = max(c.Strike for c in contracts) if contracts else 0
            
            # Find contract closest to target strike
            closest_contract = None
            min_distance = float('inf')
            for contract in contracts:
                distance = abs(contract.Strike - target_strike)
                if distance < min_distance:
                    min_distance = distance
                    closest_contract = contract
            
            for contract in contracts:
                if is_bullish:
                    if contract.Strike > target_strike:
                        rejected_by_strike += 1
                    elif contract.BidPrice <= self.minimum_premium:
                        rejected_by_price += 1
                else:
                    if contract.Strike < target_strike:
                        rejected_by_strike += 1
                    elif contract.BidPrice <= self.minimum_premium:
                        rejected_by_price += 1
            
            closest_info = ""
            if closest_contract:
                closest_info = f"Closest={closest_contract.symbol}(Strike=${closest_contract.Strike:.2f},Bid=${closest_contract.BidPrice:.2f}), "
                        
            self.Debug(f"[{self.Time}] No valid {option_type} found: SPY=${current_price:.2f}, " +
                    f"Strikes=[${min_strike:.2f}-${max_strike:.2f}], " + closest_info +
                    f"MarketStr={market_strength:.2f}, Total={total_count}, " +
                    f"SRej={rejected_by_strike}, $Rej={rejected_by_price}, ")
        return None
        
    def get_insurance_leg(self, slice, otm_puts, otm_calls, market_strength, remaining_bars_atr):
        """Get the insurance option contract to buy - closer to current price than sell leg"""
        if remaining_bars_atr is None or market_strength is None:
            return None
        
        current_price = slice.Bars[self.underlying_symbol].Price
        strike_distance = self.buy_leg_insurance_atr_multiplier * remaining_bars_atr
        
        # Choose direction based on market strength (same direction as sell leg)
        if market_strength < -self.market_strength_threshold:
            # Bearish - Buy Call closer to current price
            target_strike = current_price + strike_distance
            # Get all deeper OTM calls as potential insurance
            valid_options = [contract for contract in otm_calls if contract.Strike >= target_strike]
            
            # Sort by how close the ask price is to our max insurance premium
            if valid_options:
                valid_options.sort(key=lambda x: abs(x.AskPrice - self.buy_leg_insurance_max_premium))
                # Only keep options with acceptable premium
                valid_options = [x for x in valid_options if x.AskPrice <= self.buy_leg_insurance_max_premium]
            
            self.PrintOutput(f"Insurance: Looking for calls >= ${target_strike:.2f} with premium near ${self.buy_leg_insurance_max_premium:.2f}", print=False)
            
        elif market_strength > self.market_strength_threshold:
            # Bullish - Buy Put closer to current price
            target_strike = current_price - strike_distance
            
            # Get all deeper OTM puts as potential insurance
            valid_options = [contract for contract in otm_puts if contract.Strike <= target_strike]
            
            # Sort by how close the ask price is to our max insurance premium
            if valid_options:
                valid_options.sort(key=lambda x: abs(x.AskPrice - self.buy_leg_insurance_max_premium))
                # Only keep options with acceptable premium
                valid_options = [x for x in valid_options if x.AskPrice <= self.buy_leg_insurance_max_premium]
            
            self.PrintOutput(f"Insurance: Looking for puts <= ${target_strike:.2f} with premium near ${self.buy_leg_insurance_max_premium:.2f}", print=False)
        else:
            return None
        
        
        if valid_options:
            selected = valid_options[0]
            self.PrintOutput(f"Selected insurance {selected.Right} at strike ${selected.Strike:.2f} for ${selected.AskPrice:.2f}", print=False)
            return selected
            
        self.PrintOutput(f"No valid insurance options found for Strike Distance: ${target_strike:.2f}", print=False)
        return None
    
    def OnAssignmentOrderEvent(self, orderEvent: OrderEvent):
        underlying_price = self.Securities[orderEvent.Symbol.Underlying].Price
        self.Debug(f"[{self.Time}|${underlying_price:.2f}] Assignment detected: {orderEvent.Symbol} | Quantity: {orderEvent.FillQuantity}")
        self.pending_liquidation = True
        self.assigned_symbol = orderEvent.Symbol.Underlying
    
    def liquidate_on_assignment(self):
        if self.pending_liquidation:
            if self.Portfolio[self.assigned_symbol].Invested:
                self.Liquidate(self.assigned_symbol)
                self.Debug(f"[{self.Time}] Liquidated assigned position: {self.assigned_symbol}")
            self.pending_liquidation = False

    def manage_positions(self, slice) -> None:
        """Check open positions and place exit orders when needed"""
        if not slice.OptionChains:
            return
            
        chain = slice.OptionChains[self.option_symbol]
        if not chain:
            return
            
        for holding in self.Portfolio.Values:
            if holding.Invested and holding.Type == SecurityType.Option and holding.Quantity < 0:
                option = holding.Symbol
                
                # Skip if we already have an exit order for this position
                if option in self.exit_orders:
                    continue
                    
                # Find our option in the chain
                contract = [x for x in chain if x.Symbol == option]
                if contract and contract[0].AskPrice > self.conditional_exit_price:
                    current_ask = contract[0].AskPrice
                    self.Debug(f"[{self.Time}|{slice.Bars[self.underlying_symbol].Price:.2f}] Position in {option} has ask > {round(self.conditional_exit_price, 2)} (Ask: ${current_ask:.2f}). Placing limit order to exit")
                    # Place limit order at $2
                    exit_order = self.LimitOrder(option, -holding.Quantity, self.conditional_exit_price, tag=f"STOPLOSS EXIT: [SPY: {slice.Bars[self.underlying_symbol].Price:.2f}]")
                    self.exit_orders[option] = exit_order

    def OnOrderEvent(self, orderEvent: OrderEvent):
        """Handle order events and clean up exit orders tracking"""
        # Remove from exit orders tracking when filled or canceled
        if orderEvent.Status in [OrderStatus.Filled, OrderStatus.Canceled]:
            if orderEvent.Symbol in self.exit_orders:
                del self.exit_orders[orderEvent.Symbol]

    def EODClosePositions(self, slice: Slice) -> None:
        """Close all positions at market price at 15:58"""
        if self.Time.hour != 15 or self.Time.minute <= 57:
            return
        self.Debug(f"[{self.Time}] EOD Position Closure Check Passed")
        
        if not any(holding.Invested for holding in self.Portfolio.Values):
            return
        
        # Run this only if there are open positions
        self.Debug(f"\n[{self.Time}|${slice.Bars[self.underlying_symbol].Price:.2f}] === Starting EOD Position Closure ===")
        
        # Close all option positions
        for holding in self.Portfolio.Values:
            if not holding.Invested or holding.Type != SecurityType.Option:
                continue
                
            symbol = holding.Symbol
            quantity = holding.Quantity
            
            if quantity >= 0:
                continue  # Only close short positions
            
            # Get current market price
            chain = slice.OptionChains[self.option_symbol]
            if chain:
                chain_contract = [x for x in chain if x.Symbol == symbol]
                if chain_contract:
                    contract = chain_contract[0]
                    market_price = contract.BidPrice if quantity < 0 else contract.AskPrice
                    
                    # First cancel any existing orders
                    existing_orders = self.Transactions.GetOpenOrders(symbol)
                    if existing_orders:
                        self.PrintOutput(f"Cancelling {len(existing_orders)} open orders for {symbol}", print=False)
                        self.Transactions.CancelOpenOrders(symbol)
                    
                    # Place market order and log the result
                    underlying_price = slice.Bars[self.underlying_symbol].Price
                    # order = self.MarketOrder(symbol, -quantity, True, tag='EOD Exiting Orders')  # True for aggressive order
                    self.liquidate(symbol, True, tag='EOD Exiting Orders')  # Ensure liquidation
                    self.Debug(f"[{self.Time}|${underlying_price:.2f}] Closed {symbol} ({quantity} contracts) (Market: ${market_price:.2f})")
                        
    def OnData(self, slice: Slice) -> None:
        # Manage existing positions
        self.liquidate_on_assignment()
        self.manage_positions(slice)  # Check positions for exit conditions
        self.EODClosePositions(slice)  # Close all positions at 15:58
        
        if self.last_order_time and self.Time - self.last_order_time < timedelta(minutes=120):
            return
        
        # Return if warmup is not complete or if it's too early
        if self.IsWarmingUp or not slice.OptionChains or self.Time.hour < 10:
            return
        
        # get the option chain and print the expiration dates available
        chain = slice.OptionChains[self.option_symbol]
        if not chain:
            return
        
        # Get the OTM contracts
        otm_puts, otm_calls = self.get_OTM_contracts(chain, slice)
        
        # Get Today's market strength
        market_strength = self.get_market_strength()
        if market_strength is None:
            return
        # self.PrintOutput(f"[{self.Time}]: Market Strength: {float(market_strength):.2f}", print=False)
        
        # Get the ATR for remaining time
        remaining_bars_atr = self.calculate_atr_by_remaining_time()
        if not remaining_bars_atr:
            return
        self.PrintOutput(f"[{self.Time}]: Remaining ATR: {float(remaining_bars_atr):.2f}", print=False)

        # Print market strength and ATR
        if remaining_bars_atr and market_strength and self.Time.minute == 00 and 9 <= self.Time.hour < 16:
            self.PrintOutput(f"[{self.Time}]: Market Strength: {float(market_strength):.2f}, ATR: {float(remaining_bars_atr):.2f}", print=False)
        
        # Get the sell leg
        sell_leg_contract = self.get_sell_leg(slice, otm_puts, otm_calls, market_strength, remaining_bars_atr)
        if not sell_leg_contract:
            return
        
        # Get insurance leg
        insurance_leg_contract = self.get_insurance_leg(slice, otm_puts, otm_calls, market_strength, remaining_bars_atr)
        if not insurance_leg_contract:
            self.Debug(f"[{self.Time}|{slice.Bars[self.underlying_symbol].Price:.2f}|Mkt:{round(market_strength, 2)}] | Sell Log Found, but no insurance leg found for {sell_leg_contract.symbol}. Skipping order.")
            return
            
        # Check if we have sufficient margin
        # Get available margin
        margin_remaining = self.Portfolio.MarginRemaining
        available_margin = margin_remaining * self.margin_target
        if not self.margin_check(sell_leg_contract, insurance_leg_contract, self.position_size, available_margin):
            self.PrintOutput(f"[{self.Time}|${slice.Bars[self.underlying_symbol].Price:.2f}] Insufficient margin - skipping trade", print=False)
            return
            
        # Place the order for all legs
        self.Debug(f"[{self.Time}|${slice.Bars[self.underlying_symbol].Price:.2f}|Mkt:{round(market_strength, 2)}|ATR:{remaining_bars_atr:.2f}]: "
                f"Leg_1-SELL: {sell_leg_contract.symbol}, Price: {sell_leg_contract.BidPrice:.2f} | "
                f"Leg_2-BUY: {insurance_leg_contract.symbol}, Price: {insurance_leg_contract.AskPrice:.2f} | "
                f"Margin: ${margin_remaining:.2f}")
        
        # Create legs
        sell_leg = Leg.create(sell_leg_contract.symbol, -1)
        insurance_leg = Leg.create(insurance_leg_contract.symbol, 1)  # Buy the insurance
        legs = [sell_leg,
                insurance_leg
                ]
        # Place the combo order
        self.combo_market_order(legs, self.position_size,
            tag=f"ENTER: [SPY: {slice.Bars[self.underlying_symbol].Price:.2f}|Mkt:{round(market_strength, 2)}|ATR:{remaining_bars_atr:.2f}|Margin:{available_margin:.2f}]")
        self.last_order_time = self.Time
