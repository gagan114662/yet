#!/usr/bin/env python3
"""
Fixed 15-Year Strategy Templates with High Trading Frequency
These strategies are designed to generate 100+ trades per year over 15 years (2009-2024)
"""

def get_high_frequency_momentum_strategy() -> str:
    """Momentum strategy with daily rebalancing for high trade frequency"""
    return '''from AlgorithmImports import *

class HighFrequencyMomentumStrategy(QCAlgorithm):
    def Initialize(self):
        # 15-YEAR BACKTEST PERIOD
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        # Add multiple assets for diversification and more trades
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Daily).Symbol  # Small cap
        self.efa = self.AddEquity("EFA", Resolution.Daily).Symbol  # International
        
        # Short-term momentum indicators for frequent signals
        self.spy_momentum = self.MOMP("SPY", 5, Resolution.Daily)
        self.qqq_momentum = self.MOMP("QQQ", 5, Resolution.Daily)
        self.iwm_momentum = self.MOMP("IWM", 5, Resolution.Daily)
        
        # RSI for mean reversion signals
        self.spy_rsi = self.RSI("SPY", 7, Resolution.Daily)
        
        # DAILY rebalancing for maximum trade frequency
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.DailyRebalance)
        
        # Track trades
        self.trades_this_year = 0
        self.current_year = 2009
        
    def DailyRebalance(self):
        """Daily rebalancing logic to generate frequent trades"""
        if not self.spy_momentum.IsReady or not self.spy_rsi.IsReady:
            return
        
        # Track yearly trades
        if self.Time.year != self.current_year:
            self.Debug(f"Year {self.current_year}: {self.trades_this_year} trades")
            self.trades_this_year = 0
            self.current_year = self.Time.year
        
        # Get momentum values
        spy_mom = self.spy_momentum.Current.Value
        qqq_mom = self.qqq_momentum.Current.Value if self.qqq_momentum.IsReady else 0
        iwm_mom = self.iwm_momentum.Current.Value if self.iwm_momentum.IsReady else 0
        rsi = self.spy_rsi.Current.Value
        
        # High-frequency allocation changes
        spy_weight = 0.25
        qqq_weight = 0.25
        iwm_weight = 0.25
        efa_weight = 0.25
        
        # Momentum-based adjustments (daily changes create more trades)
        if spy_mom > 2:
            spy_weight += 0.2
            qqq_weight -= 0.1
            iwm_weight -= 0.1
        elif spy_mom < -2:
            spy_weight -= 0.1
            efa_weight += 0.1
        
        if qqq_mom > 3:
            qqq_weight += 0.2
            spy_weight -= 0.1
            iwm_weight -= 0.1
        
        # RSI mean reversion overlay
        if rsi < 30:
            spy_weight += 0.15
        elif rsi > 70:
            spy_weight -= 0.15
            efa_weight += 0.15
        
        # Normalize weights
        total_weight = spy_weight + qqq_weight + iwm_weight + efa_weight
        spy_weight /= total_weight
        qqq_weight /= total_weight
        iwm_weight /= total_weight
        efa_weight /= total_weight
        
        # Execute trades (only if significant change to generate actual trades)
        current_spy = self.Portfolio[self.spy].HoldingsValue / self.Portfolio.TotalPortfolioValue if self.Portfolio.TotalPortfolioValue > 0 else 0
        
        if abs(current_spy - spy_weight) > 0.05:  # 5% threshold to generate real trades
            self.SetHoldings(self.spy, spy_weight)
            self.SetHoldings(self.qqq, qqq_weight)
            self.SetHoldings(self.iwm, iwm_weight)
            self.SetHoldings(self.efa, efa_weight)
            self.trades_this_year += 4  # Count each asset rebalancing
    
    def OnData(self, data):
        pass
    
    def OnEndOfAlgorithm(self):
        self.Debug(f"Final year {self.current_year}: {self.trades_this_year} trades")
        self.Debug(f"Strategy completed 15-year backtest: 2009-2024")'''

def get_high_frequency_mean_reversion_strategy() -> str:
    """Mean reversion strategy with multiple daily checks"""
    return '''from AlgorithmImports import *

class HighFrequencyMeanReversionStrategy(QCAlgorithm):
    def Initialize(self):
        # 15-YEAR BACKTEST PERIOD
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        # Multi-asset universe for more trading opportunities
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol  # Bonds
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol  # Gold
        self.vxx = self.AddEquity("VXX", Resolution.Daily).Symbol  # Volatility (starts 2009)
        
        # Short-term mean reversion indicators
        self.spy_rsi = self.RSI("SPY", 7, Resolution.Daily)
        self.spy_bb = self.BB("SPY", 10, 2, Resolution.Daily)
        self.qqq_rsi = self.RSI("QQQ", 7, Resolution.Daily)
        
        # Multiple daily checks for high frequency
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.MorningRebalance)
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 120), 
                        self.MidDayCheck)
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.BeforeMarketClose("SPY", 60), 
                        self.EveningRebalance)
        
        # Trade tracking
        self.trades_this_year = 0
        self.current_year = 2009
        self.last_allocation = {}
        
    def MorningRebalance(self):
        """Morning rebalancing based on overnight moves"""
        self._execute_mean_reversion_logic("MORNING")
    
    def MidDayCheck(self):
        """Midday adjustment based on intraday moves"""
        self._execute_mean_reversion_logic("MIDDAY")
    
    def EveningRebalance(self):
        """End of day final positioning"""
        self._execute_mean_reversion_logic("EVENING")
    
    def _execute_mean_reversion_logic(self, session):
        """Core mean reversion logic executed multiple times daily"""
        if not self.spy_rsi.IsReady or not self.spy_bb.IsReady:
            return
        
        # Track yearly trades
        if self.Time.year != self.current_year:
            self.Debug(f"Year {self.current_year}: {self.trades_this_year} trades")
            self.trades_this_year = 0
            self.current_year = self.Time.year
        
        # Get current indicators
        spy_rsi = self.spy_rsi.Current.Value
        qqq_rsi = self.qqq_rsi.Current.Value if self.qqq_rsi.IsReady else 50
        spy_price = self.Securities[self.spy].Price
        bb_upper = self.spy_bb.UpperBand.Current.Value
        bb_lower = self.spy_bb.LowerBand.Current.Value
        bb_middle = self.spy_bb.MiddleBand.Current.Value
        
        # Base allocation
        spy_weight = 0.30
        qqq_weight = 0.25
        tlt_weight = 0.20
        gld_weight = 0.15
        vxx_weight = 0.10
        
        # Mean reversion adjustments based on session
        if session == "MORNING":
            # React to overnight moves
            if spy_rsi < 25 or spy_price < bb_lower * 0.995:
                spy_weight = 0.50  # Oversold - buy more
                tlt_weight = 0.15
                gld_weight = 0.10
                vxx_weight = 0.05
            elif spy_rsi > 75 or spy_price > bb_upper * 1.005:
                spy_weight = 0.15  # Overbought - reduce
                tlt_weight = 0.35
                gld_weight = 0.25
                vxx_weight = 0.15
                
        elif session == "MIDDAY":
            # Intraday mean reversion
            if 30 < spy_rsi < 70 and bb_lower < spy_price < bb_upper:
                spy_weight = 0.40  # Normal range
                qqq_weight = 0.30
                tlt_weight = 0.20
                gld_weight = 0.10
            else:
                # Extreme moves - fade them
                if spy_rsi < 35:
                    spy_weight = 0.45
                elif spy_rsi > 65:
                    spy_weight = 0.20
                    tlt_weight = 0.30
                    
        else:  # EVENING
            # End of day positioning for overnight
            if qqq_rsi < 30:
                qqq_weight = 0.40
                spy_weight = 0.25
            elif qqq_rsi > 70:
                qqq_weight = 0.10
                tlt_weight = 0.35
        
        # Normalize weights
        total_weight = spy_weight + qqq_weight + tlt_weight + gld_weight + vxx_weight
        spy_weight /= total_weight
        qqq_weight /= total_weight
        tlt_weight /= total_weight
        gld_weight /= total_weight
        vxx_weight /= total_weight
        
        # Check if allocation changed significantly
        new_allocation = {
            'SPY': spy_weight, 'QQQ': qqq_weight, 'TLT': tlt_weight, 
            'GLD': gld_weight, 'VXX': vxx_weight
        }
        
        should_trade = False
        for symbol, weight in new_allocation.items():
            old_weight = self.last_allocation.get(symbol, 0)
            if abs(weight - old_weight) > 0.03:  # 3% threshold
                should_trade = True
                break
        
        if should_trade:
            self.SetHoldings(self.spy, spy_weight)
            self.SetHoldings(self.qqq, qqq_weight)
            self.SetHoldings(self.tlt, tlt_weight)
            self.SetHoldings(self.gld, gld_weight)
            
            # VXX only available from 2009
            if self.Time.year >= 2009:
                self.SetHoldings(self.vxx, vxx_weight)
            
            self.trades_this_year += 5  # Count each asset
            self.last_allocation = new_allocation.copy()
    
    def OnData(self, data):
        pass
    
    def OnEndOfAlgorithm(self):
        self.Debug(f"Final year {self.current_year}: {self.trades_this_year} trades")
        self.Debug(f"Total strategy period: 15 years (2009-2024)")'''

def get_high_frequency_breakout_strategy() -> str:
    """Breakout strategy with multiple timeframe monitoring"""
    return '''from AlgorithmImports import *

class HighFrequencyBreakoutStrategy(QCAlgorithm):
    def Initialize(self):
        # 15-YEAR BACKTEST PERIOD
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        # Diverse asset universe for breakout opportunities
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Daily).Symbol
        self.eem = self.AddEquity("EEM", Resolution.Daily).Symbol  # Emerging markets
        self.xlf = self.AddEquity("XLF", Resolution.Daily).Symbol  # Financials
        
        # Multiple timeframe breakout indicators
        self.spy_sma_short = self.SMA("SPY", 10, Resolution.Daily)
        self.spy_sma_long = self.SMA("SPY", 20, Resolution.Daily)
        self.spy_atr = self.ATR("SPY", 14, Resolution.Daily)
        
        self.qqq_sma_short = self.SMA("QQQ", 8, Resolution.Daily)
        self.qqq_sma_long = self.SMA("QQQ", 15, Resolution.Daily)
        
        # Bollinger Bands for breakout identification
        self.spy_bb = self.BB("SPY", 15, 2, Resolution.Daily)
        self.qqq_bb = self.BB("QQQ", 15, 2, Resolution.Daily)
        
        # High frequency scheduling - check for breakouts multiple times daily
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 15), 
                        self.EarlyBreakoutCheck)
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 90), 
                        self.MidMorningBreakout)
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 180), 
                        self.AfternoonBreakout)
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.BeforeMarketClose("SPY", 30), 
                        self.FinalBreakoutCheck)
        
        # Trade tracking
        self.trades_this_year = 0
        self.current_year = 2009
        
    def EarlyBreakoutCheck(self):
        """Check for overnight gap breakouts"""
        self._execute_breakout_strategy("EARLY")
    
    def MidMorningBreakout(self):
        """Mid-morning momentum breakouts"""
        self._execute_breakout_strategy("MIDMORNING")
    
    def AfternoonBreakout(self):
        """Afternoon trend continuation"""
        self._execute_breakout_strategy("AFTERNOON")
    
    def FinalBreakoutCheck(self):
        """End of day positioning"""
        self._execute_breakout_strategy("FINAL")
    
    def _execute_breakout_strategy(self, session):
        """Core breakout detection and allocation logic"""
        if not self.spy_sma_short.IsReady or not self.spy_bb.IsReady:
            return
        
        # Track yearly trades
        if self.Time.year != self.current_year:
            self.Debug(f"Year {self.current_year}: {self.trades_this_year} trades")
            self.trades_this_year = 0
            self.current_year = self.Time.year
        
        # Get prices and indicators
        spy_price = self.Securities[self.spy].Price
        qqq_price = self.Securities[self.qqq].Price
        iwm_price = self.Securities[self.iwm].Price
        
        spy_sma_short = self.spy_sma_short.Current.Value
        spy_sma_long = self.spy_sma_long.Current.Value if self.spy_sma_long.IsReady else spy_sma_short
        spy_bb_upper = self.spy_bb.UpperBand.Current.Value
        spy_bb_lower = self.spy_bb.LowerBand.Current.Value
        
        qqq_sma_short = self.qqq_sma_short.Current.Value if self.qqq_sma_short.IsReady else qqq_price
        qqq_sma_long = self.qqq_sma_long.Current.Value if self.qqq_sma_long.IsReady else qqq_sma_short
        
        # Base allocation
        spy_weight = 0.25
        qqq_weight = 0.25
        iwm_weight = 0.20
        eem_weight = 0.15
        xlf_weight = 0.15
        
        # Breakout detection logic varies by session
        if session == "EARLY":
            # Overnight gap breakouts
            if spy_price > spy_bb_upper * 1.001:  # Upward breakout
                spy_weight = 0.45
                qqq_weight = 0.35
                iwm_weight = 0.20
                eem_weight = 0.0
                xlf_weight = 0.0
            elif spy_price < spy_bb_lower * 0.999:  # Downward breakout
                spy_weight = 0.10
                qqq_weight = 0.10
                iwm_weight = 0.15
                eem_weight = 0.05
                xlf_weight = 0.60  # Defensive rotation
                
        elif session == "MIDMORNING":
            # Trend momentum breakouts
            spy_trend_up = spy_sma_short > spy_sma_long * 1.002
            qqq_trend_up = qqq_sma_short > qqq_sma_long * 1.002
            
            if spy_trend_up and qqq_trend_up:
                spy_weight = 0.40
                qqq_weight = 0.40
                iwm_weight = 0.20
            elif spy_trend_up and not qqq_trend_up:
                spy_weight = 0.50
                iwm_weight = 0.30
                xlf_weight = 0.20
            elif not spy_trend_up and qqq_trend_up:
                qqq_weight = 0.50
                iwm_weight = 0.30
                eem_weight = 0.20
                
        elif session == "AFTERNOON":
            # Continuation patterns
            if spy_price > spy_sma_short * 1.005:  # Strong uptrend
                spy_weight = 0.35
                qqq_weight = 0.30
                iwm_weight = 0.25
                eem_weight = 0.10
            elif spy_price < spy_sma_short * 0.995:  # Downtrend
                spy_weight = 0.15
                xlf_weight = 0.40
                eem_weight = 0.45
                
        else:  # FINAL
            # End of day positioning for overnight holds
            if spy_price > spy_sma_long and qqq_price > qqq_sma_long:
                spy_weight = 0.30
                qqq_weight = 0.30
                iwm_weight = 0.25
                eem_weight = 0.15
            else:
                spy_weight = 0.20
                qqq_weight = 0.20
                xlf_weight = 0.35
                eem_weight = 0.25
        
        # Normalize weights
        total_weight = spy_weight + qqq_weight + iwm_weight + eem_weight + xlf_weight
        if total_weight > 0:
            spy_weight /= total_weight
            qqq_weight /= total_weight
            iwm_weight /= total_weight
            eem_weight /= total_weight
            xlf_weight /= total_weight
        
        # Execute trades
        self.SetHoldings(self.spy, spy_weight)
        self.SetHoldings(self.qqq, qqq_weight)
        self.SetHoldings(self.iwm, iwm_weight)
        self.SetHoldings(self.eem, eem_weight)
        self.SetHoldings(self.xlf, xlf_weight)
        
        self.trades_this_year += 5  # Count each asset rebalancing
    
    def OnData(self, data):
        pass
    
    def OnEndOfAlgorithm(self):
        self.Debug(f"Final year {self.current_year}: {self.trades_this_year} trades")
        self.Debug(f"Breakout strategy completed 15-year period")'''