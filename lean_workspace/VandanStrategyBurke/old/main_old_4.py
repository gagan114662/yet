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
        # Basic settings
        self.SetStartDate(2021, 7, 4)
        self.SetEndDate(2022, 7, 10)
        self.SetCash(200_000)
        self.SetTimeZone(TimeZones.NEW_YORK)
        
        # Add our option chain
        option = self.AddOption("SPY")
        self._symbol = option.Symbol
        
        # Add underlying for reference
        equity = self.AddEquity("SPY", Resolution.MINUTE)
        self.spy = equity.Symbol
        
        # Trading schedule constants
        self.market_open = time(9, 30)    # Market opens at 9:30 AM ET
        self.market_close = time(16, 0)   # Market closes at 4:00 PM ET
        self.trading_start = time(10, 0)  # Start trading at 10:00 AM ET
        
        # Trading tracking
        self.last_trade_hour = None
        self.last_trade_date = None
        self.trades_today = {}  # Dictionary to track trades by hour
        
        # Market direction tracking
        self.opening_range_high = None
        self.opening_range_low = None
        self.price_history = []  # Store recent prices for momentum
        self.last_direction_date = None
        
        # Set up ATR indicator and history
        self.history_bars = 30  # Days of history
        
        # Initialize history storage
        self.minute_history = []
        
        # Request historical data for ATR calculation
        history = self.History(self.spy, 30, Resolution.DAILY)
        if history.empty:
            return
            
        # Store historical data
        for bar in history.itertuples():
            self.minute_history.append({
                'date': bar.Index[1],
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close)
            })
            
        
        # Configure option filter for OTM puts
        option.SetFilter(lambda u: u.Strikes(-100, 10)   # Wider range for OTM puts
                                  .Expiration(1, 2)      # 0-1 days to expiration
                                  .IncludeWeeklys())     # Include weekly options
                                  
        self.debug_count = 0
        self.max_debug_count = 200
        self.SetWarmUp(timedelta(days=31))

    def PrintOutput(self, msg, print=True):
        if print and self.debug_count < self.max_debug_count:
            self.Debug(f"{self.debug_count} ::: {msg}")
            self.debug_count += 1
            
    def get_market_direction(self) -> Optional[float]:
        """
        Calculate market direction strength combining opening range breakout
        and momentum on a scale of -5 (very bearish) to +5 (very bullish)
        """
        security = self.Securities[self.spy]
        current_price = security.Price
        current_time = self.Time
        
        # Reset tracking for new day
        if self.last_direction_date != current_time.date():
            self.opening_range_high = None
            self.opening_range_low = None
            self.price_history = []
            self.last_direction_date = current_time.date()
            
        # Store price for momentum calculation
        self.price_history.append(current_price)
        if len(self.price_history) > 30:  # Keep last 30 minutes
            self.price_history.pop(0)
            
        current_minutes = current_time.hour * 60 + current_time.minute
        market_open_minutes = 9 * 60 + 30
        
        # First hour: Build opening range (9:30 - 10:30)
        if current_minutes >= market_open_minutes and current_minutes < (market_open_minutes + 60):
            if self.opening_range_high is None or current_price > self.opening_range_high:
                self.opening_range_high = current_price
            if self.opening_range_low is None or current_price < self.opening_range_low:
                self.opening_range_low = current_price
            self.PrintOutput(f"Building opening range: Low={self.opening_range_low:.2f}, High={self.opening_range_high:.2f}", print=False)
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
            self.PrintOutput(f"Range Signal: {range_signal:.2f} (Current: {current_price:.2f}, Mid: {range_midpoint:.2f})", print=False)
            
        # Calculate momentum signal (-5 to +5)
        if len(self.price_history) < 15:  # Need at least 15 minutes of history
            self.PrintOutput(f"Insufficient price history", print=True)
            return None  # Need more price history
        else:
            price_change = current_price - self.price_history[-15]  # 15-minute change
            momentum_signal = (price_change / current_price) * 100 * 5  # Scale to -5 to +5
            self.PrintOutput(f"Momentum Signal: {momentum_signal:.2f} (15min change: {price_change:.2f})", print=False)
            
        # Combine signals (60% range breakout, 40% momentum)
        final_signal = (range_signal * 0.6) + (momentum_signal * 0.4)
        
        # Clamp final value between -5 and +5
        return max(min(final_signal, 5), -5)
    
    def calculate_custom_atr(self, lookback_bars: int) -> float:
        """Calculate ATR using daily data for stability"""
        if len(self.minute_history) < 2:
            return 0
            
        self.PrintOutput(f"Starting ATR calculation with {len(self.minute_history)} days of data", print=False)
        tr_values = []
        
        # Calculate True Range for each day
        for i in range(1, len(self.minute_history)):
            current = self.minute_history[i]
            prev = self.minute_history[i-1]
            
            tr = max(
                current['high'] - current['low'],  # Current day's range
                abs(current['high'] - prev['close']),  # High vs previous close
                abs(current['low'] - prev['close'])    # Low vs previous close
            )
            tr_values.append(tr)
        
        # Calculate ATR
        if not tr_values:
            return 0
            
        # Scale ATR by sqrt of time
        daily_atr = sum(tr_values) / len(tr_values)
        minutes_per_day = (self.market_close.hour - self.market_open.hour) * 60
        
        # Scale ATR to the number of bars remaining
        scaled_atr = daily_atr * (lookback_bars / minutes_per_day) ** 0.5
        
        self.PrintOutput(f"ATR Calculation: Daily ATR = {daily_atr:.4f}, Bars = {lookback_bars}, Minutes per day = {minutes_per_day}, Scaled ATR = {scaled_atr:.4f}", print=False)
        
        return scaled_atr
            
    def calculate_remaining_bars_until_expiry(self, expiry: datetime) -> int:
        """Calculate how many minute bars are left until option expiry"""
        # Get current time in market timezone
        current_time = self.Time
        
        # If we're past market close or it's expiry day after market close, return 0
        if current_time.time() >= self.market_close or (
            current_time.date() == expiry.date() and current_time.time() >= self.market_close):
            return 0
            
        # Calculate remaining bars for today
        if current_time.date() == expiry.date():
            close_minutes = self.market_close.hour * 60 + self.market_close.minute
            current_minutes = current_time.hour * 60 + current_time.minute
            return max(0, close_minutes - current_minutes)
            
        # Calculate full days plus remaining minutes in current day
        days_between = (expiry.date() - current_time.date()).days
        minutes_per_day = (self.market_close.hour - self.market_open.hour) * 60
        
        # Calculate remaining minutes in current day
        current_day_minutes = 0
        if current_time.time() < self.market_close:
            current_minutes = current_time.hour * 60 + current_time.minute
            close_minutes = self.market_close.hour * 60 + self.market_close.minute
            current_day_minutes = max(0, close_minutes - current_minutes)
        
        return days_between * minutes_per_day + current_day_minutes
    
    def can_trade_this_hour(self) -> bool:
        """Check if we can trade in the current hour"""
        current_time = self.Time
        current_hour = current_time.hour
        
        # Reset trades for new day
        if self.last_trade_date != current_time.date():
            self.PrintOutput(f"[{self.Time}] New trading day - resetting trade tracking", print=False)
            self.trades_today = {}
            self.last_trade_date = current_time.date()
        
        # Check if we've already traded this hour
        return current_hour not in self.trades_today
    
    def OnAssignmentOrderEvent(self, orderEvent: OrderEvent):
        if orderEvent.Status != OrderStatus.Filled:
            return
        
        assigned_symbol = orderEvent.Symbol
        self.Debug(f"Assigned: {assigned_symbol}")
        
        # Save to avoid double liquidation
        if assigned_symbol in self.assigned_symbols:
            return
        
        self.assigned_symbols.add(assigned_symbol)

        # If we now hold the assigned security, liquidate it
        if self.Portfolio[assigned_symbol].Invested:
            self.Debug(f"Liquidating assigned position: {assigned_symbol}")
            self.Liquidate(assigned_symbol)
    
    
    def OnData(self, slice: Slice) -> None:
        # Don't trade during warmup
        if self.IsWarmingUp:
            return
            
        # Update historical data
        if self.Time.minute == 0:  # Update history every hour
            history = self.History(self.spy, 30, Resolution.DAILY)
            if not history.empty:
                self.minute_history = []
                for bar in history.itertuples():
                    self.minute_history.append({
                        'date': bar.Index[1],
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close)
                    })
                self.PrintOutput(f"Updated history with {len(self.minute_history)} days of data", print=False)
        
        # Check if we're within trading hours
        current_time = self.Time.time()
        if current_time < self.trading_start:
            return
            
        # Get the OptionChain for new trades
        chain = slice.option_chains.get(self._symbol, None)
        if not chain:
            return
        
        # Determine market direction strength
        market_strength = self.get_market_direction()
        if market_strength is None:
            self.PrintOutput(f"[{self.Time}] Market Direction: Undefined - waiting for more data", print=False)
            return
            
        direction = "Bullish" if market_strength > 0 else "Bearish"
        self.PrintOutput(f"[{self.Time}] Market Direction: {direction} (Strength: {market_strength:.2f})", print=False)

        if chain:
            # Get all expiries and sort them
            all_expiries = sorted(list(set([x.expiry for x in chain])))[:3]
            expiry = min(all_expiries)
            self.PrintOutput(f"[{self.Time}] Next expiry: {expiry}", print=False)
        
            
            # Calculate remaining bars until expiry
            remaining_bars = self.calculate_remaining_bars_until_expiry(expiry)
            self.PrintOutput(f"[{self.Time}] Bars until expiry: {remaining_bars}", print=False)

            if remaining_bars <= 0:
                self.PrintOutput(f"[{self.Time}] No trading - too close to expiry", print=False)
                return
                
            # Calculate ATR using remaining bars
            current_atr = self.calculate_custom_atr(remaining_bars)
            self.PrintOutput(f"Raw ATR calculation complete", print=False)
            self.PrintOutput(f"[{self.Time}] Custom ATR for {remaining_bars} bars: {current_atr:.4f}", print=False)
            
            if current_atr <= 0:
                self.PrintOutput(f"[{self.Time}] No trading - invalid ATR value", print=False)
                return
                
            # Check if we can trade this hour
            if not self.can_trade_this_hour():
                self.PrintOutput(f"[{self.Time}] No trading - already traded in hour {self.Time.hour}. Today's trades: {list(self.trades_today.keys())}", print=False)
                return
                
            # Get current price for target calculations
            current_price = self.Securities[self.spy].Price
            
            # Don't trade if we don't have a clear direction
            if market_strength is None:
                return
                
            # Find target option
            is_bullish = market_strength > 0
            target_distance = current_atr * 1.5
            target_price = current_price - target_distance if is_bullish else current_price + target_distance
            self.PrintOutput(f"[{self.Time}] Target price: {target_price:.2f} ({'-' if is_bullish else '+'}{target_distance:.2f} from {current_price:.2f})", print=False)
            
            # Filter valid options
            valid_options = [x for x in chain if
                        x.expiry == expiry and
                        x.right == (OptionRight.PUT if is_bullish else OptionRight.CALL)]
                        
            if not valid_options:
                self.PrintOutput(f"[{self.Time}] No valid options found for {'puts' if is_bullish else 'calls'} at target {target_price:.2f}", print=False)
                return
                
            # Find the option closest to our target price that meets premium requirement
            valid_options = sorted(valid_options, key=lambda x: abs(x.strike - target_price))
            
            for option in valid_options:
                # Check minimum premium requirement (bid price since we're selling)
                if not option.BidPrice or option.BidPrice <= 0:
                    self.PrintOutput(f"[{self.Time}] Skipping {option.right} at strike {option.strike} - no bid price available", print=False)
                    continue
            
                premium = option.BidPrice  # Convert to actual dollars
                if premium >= 1.50:
                    # Execute the trade
                    legs = [Leg.create(option.Symbol, -1)]
                    self.combo_market_order(legs, 1)
                    # Calculate distances
                    price_distance = abs(option.Strike - current_price)
                    distance_pct = (price_distance / current_price) * 100
                    distance_atr = price_distance / current_atr if current_atr > 0 else 0
                    option_type = "CALL" if option.Right == OptionRight.Call else "PUT"
                    
                    self.PrintOutput(f"[{self.Time}] [Mkt Strength: {market_strength:.2f}] Executed: Sell {option_type}|{option.Strike}|{expiry} @ ${premium:.2f} "
                                f"(Qty: -1, Underlying: ${current_price:.2f}, Target: ${target_price:.2f}, "
                                f"Distance: ${price_distance:.2f} / {distance_pct:.1f}% / {distance_atr:.1f}x ATR)",
                                print=True)
                    self.trades_today[self.Time.hour] = option
                    break
                else:
                    self.PrintOutput(f"[{self.Time}] Skipping {option.right} at strike {option.strike} - insufficient premium ${premium:.2f}", print=False)
