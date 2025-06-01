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
        self.strategy_params()  # Initialize strategy parameters
        
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
        self.SetWarmUp(timedelta(days=31))
        
    
    def strategy_params(self) -> None:
        # Strategy parameters
        self.set_start_date(2023, 10, 1)  # Set start date
        self.set_end_date(2023, 10, 31)   # Set end date
        self.market_strength_threshold_min = 2  # Threshold for market strength to trigger trades
        self.market_strength_threshold_max = 4  # Threshold for market strength to trigger trades
        self.sell_leg_atr_multiplier = 1 # Multiplier for ATR distance
        self.buy_leg_insurance_atr_multiplier = 1  # Multiplier for insurance leg distance
        self.minimum_premium = 2.5
        self.buy_leg_insurance_max_premium = 0.15
        self.conditional_exit_price = self.minimum_premium * 4  # Price to trigger exit order
        self.position_size = 10  # Number of contracts to trade
    
    
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
        start_time = self.Time.replace(hour=0, minute=0, second=0, microsecond=0)  # Start from the previous day
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
    
    def calculate_atr_by_remaining_time(self) -> float:
        """Calculate ATR based on remaining time until market close"""
        current_time = self.Time
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        current_minutes = current_time.hour * 60 + current_time.minute
        
        if current_minutes >= 16 * 60:  # After market close
            return None
            
        # Get historical data
        price_history = self.History(self.underlying_symbol, self.Time - timedelta(days=31), self.Time, Resolution.Minute)
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
        """Get furthest OTM option with minimum premium and ATR distance requirements"""
        if remaining_bars_atr is None or market_strength is None:
            return None
            
        current_price = slice.Bars[self.underlying_symbol].Price
        min_strike_distance = self.sell_leg_atr_multiplier * remaining_bars_atr
        
        if -self.market_strength_threshold_max < market_strength < -self.market_strength_threshold_min:
            # Bearish - Sell furthest OTM call with sufficient premium
            valid_strike = current_price + min_strike_distance
            valid_calls = [c for c in otm_calls
                          if c.Strike >= valid_strike
                          and c.BidPrice >= self.minimum_premium]
            return max(valid_calls, key=lambda x: x.Strike) if valid_calls else None
                
        elif self.market_strength_threshold_max > market_strength > self.market_strength_threshold_min:
            # Bullish - Sell furthest OTM put with sufficient premium
            valid_strike = current_price - min_strike_distance
            valid_puts = [p for p in otm_puts
                         if p.Strike <= valid_strike
                         and p.BidPrice >= self.minimum_premium]
            return min(valid_puts, key=lambda x: x.Strike) if valid_puts else None
        
        return None
    
    def get_insurance_leg(self, sell_leg, otm_puts, otm_calls):
        """Get closest option to sell_leg with premium <= insurance max"""
        if not sell_leg:
            return None
            
        candidates = otm_calls if sell_leg.Right == OptionRight.Call else otm_puts
        valid_options = [opt for opt in candidates
                        if opt.AskPrice <= self.buy_leg_insurance_max_premium]
        
        if not valid_options:
            return None
            
        # Return option with strike closest to sell_leg strike
        return min(valid_options, key=lambda x: abs(x.Strike - sell_leg.Strike))
      
    def OnData(self, slice: Slice) -> None:
        # Return if warmup is not complete or if it's too early
        if self.IsWarmingUp or not slice.OptionChains or self.Time.hour < 10:
            return
        
         # get the option chain and print the expiration dates available
        chain = slice.OptionChains[self.option_symbol]
        if not chain:
            return
        
        # Get the underlying price
        underlying_price = self.Securities[self.underlying_symbol].Price
        
        # Get Market Strength
        market_strength = self.get_market_strength()
        if market_strength is None:
            return
        
        # Get the ATR for remaining time
        remaining_bars_atr = self.calculate_atr_by_remaining_time()
        if not remaining_bars_atr:
            return
        
        # Get the OTM contracts
        otm_puts, otm_calls = self.get_OTM_contracts(chain, slice)
        
        # Get the sell leg
        sell_leg_contract = self.get_sell_leg(slice, otm_puts, otm_calls, market_strength, remaining_bars_atr)
        if not sell_leg_contract:
            return
        
        # Get insurance leg
        insurance_leg_contract = self.get_insurance_leg(sell_leg_contract, otm_puts, otm_calls)
        if not insurance_leg_contract:
            self.Debug(f"[{self.Time}|{slice.Bars[self.underlying_symbol].Price:.2f}|Mkt:{round(market_strength, 2)}] | Sell Log Found, but no insurance leg found for {sell_leg_contract.symbol}. Skipping order.")
            return
        
        if self.Time.minute % 15 == 0:
            self.PrintOutput(f"[{self.Time}|{underlying_price}] Mkt Strgh: {market_strength:.2f}, curr_atr: {remaining_bars_atr:.2f}, sell-leg-sym:{sell_leg_contract.symbol}, sell_leg_dist:{(underlying_price-sell_leg_contract.strike):.2f}, Price:{sell_leg_contract.bid_price}", print=True)
            
            
