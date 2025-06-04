from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class UltraBeast(QCAlgorithm):
    """
    Ultra-aggressive strategy using maximum leverage and momentum
    """
    
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Maximum leverage
        self.Settings.FreePortfolioValuePercentage = 0.05
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Core parameters
        self.max_leverage = 6.0
        self.momentum_lookback = 20
        self.position_size = 0.33  # 33% per position with leverage
        
        # High-beta instruments
        self.tqqq = self.AddEquity("TQQQ", Resolution.Hour).Symbol
        self.upro = self.AddEquity("UPRO", Resolution.Hour).Symbol
        self.soxl = self.AddEquity("SOXL", Resolution.Hour).Symbol
        self.fas = self.AddEquity("FAS", Resolution.Hour).Symbol
        self.tna = self.AddEquity("TNA", Resolution.Hour).Symbol
        self.udow = self.AddEquity("UDOW", Resolution.Hour).Symbol
        
        # Inverse for hedging
        self.sqqq = self.AddEquity("SQQQ", Resolution.Hour).Symbol
        self.spxu = self.AddEquity("SPXU", Resolution.Hour).Symbol
        self.soxs = self.AddEquity("SOXS", Resolution.Hour).Symbol
        
        # Base indices for signals
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Hour).Symbol
        
        # Options for extra leverage
        equity = self.AddEquity("SPY", Resolution.Hour)
        option = self.AddOption("SPY", Resolution.Hour)
        option.SetFilter(-5, 5, timedelta(0), timedelta(30))
        
        # Technical indicators
        self.fast = self.SMA(self.spy, 5, Resolution.Daily)
        self.slow = self.SMA(self.spy, 20, Resolution.Daily)
        self.rsi = self.RSI(self.spy, 7, Resolution.Daily)
        self.macd = self.MACD(self.spy, 12, 26, 9, Resolution.Daily)
        
        # QQQ indicators
        self.qqq_fast = self.SMA(self.qqq, 5, Resolution.Daily)
        self.qqq_slow = self.SMA(self.qqq, 20, Resolution.Daily)
        self.qqq_rsi = self.RSI(self.qqq, 7, Resolution.Daily)
        
        # Portfolio tracking
        self.highest_value = 100000
        self.trades_today = 0
        self.last_trade_time = self.Time
        
        # Schedule functions
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.Every(timedelta(hours=1)),
                        self.AggressiveTrade)
        
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.BeforeMarketClose(self.spy, 30),
                        self.EndOfDayAdjustment)
        
        self.SetWarmUp(20)
    
    def AggressiveTrade(self):
        """Execute aggressive momentum trades"""
        if self.IsWarmingUp:
            return
        
        # Reset daily trade counter
        if self.Time.date() != self.last_trade_time.date():
            self.trades_today = 0
            self.last_trade_time = self.Time
        
        # Limit trades per day
        if self.trades_today > 10:
            return
        
        # Get market direction
        spy_trend = self.fast.Current.Value > self.slow.Current.Value
        qqq_trend = self.qqq_fast.Current.Value > self.qqq_slow.Current.Value
        
        # Calculate momentum strength
        spy_momentum = (self.Securities[self.spy].Price / self.Securities[self.spy].Close - 1) if self.Time.hour > 10 else 0
        qqq_momentum = (self.Securities[self.qqq].Price / self.Securities[self.qqq].Close - 1) if self.Time.hour > 10 else 0
        
        # Strong bullish - maximum leverage
        if spy_trend and qqq_trend and self.rsi.Current.Value > 50 and self.qqq_rsi.Current.Value > 50:
            # Tech focus
            if qqq_momentum > spy_momentum:
                self.SetHoldings(self.tqqq, self.position_size * 2)
                self.SetHoldings(self.soxl, self.position_size)
            else:
                self.SetHoldings(self.upro, self.position_size * 2)
                self.SetHoldings(self.fas, self.position_size)
            
            # Clear shorts
            for symbol in [self.sqqq, self.spxu, self.soxs]:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
            
            self.trades_today += 1
        
        # Strong bearish - short aggressively
        elif not spy_trend and not qqq_trend and self.rsi.Current.Value < 40:
            # Exit longs
            for symbol in [self.tqqq, self.upro, self.soxl, self.fas, self.tna, self.udow]:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
            
            # Short via inverse ETFs
            self.SetHoldings(self.sqqq, self.position_size)
            self.SetHoldings(self.spxu, self.position_size)
            
            self.trades_today += 1
        
        # Mixed signals - reduce exposure
        elif spy_trend != qqq_trend:
            self.ReduceExposure()
    
    def OnData(self, data):
        """Process option trades for extra leverage"""
        if self.IsWarmingUp:
            return
        
        # Trade options on strong signals
        for kvp in data.OptionChains:
            chain = kvp.Value
            if not chain:
                continue
            
            # Filter ATM options
            underlying_price = chain.Underlying.Price
            calls = [x for x in chain if x.Right == OptionRight.Call and 
                    abs(x.Strike - underlying_price) / underlying_price < 0.02]
            puts = [x for x in chain if x.Right == OptionRight.Put and 
                   abs(x.Strike - underlying_price) / underlying_price < 0.02]
            
            if self.fast.Current.Value > self.slow.Current.Value and self.rsi.Current.Value > 60:
                # Buy calls
                if calls:
                    atm_call = sorted(calls, key=lambda x: abs(x.Strike - underlying_price))[0]
                    if atm_call.AskPrice > 0 and atm_call.AskPrice < 10:
                        quantity = int(self.Portfolio.Cash * 0.1 / (atm_call.AskPrice * 100))
                        if quantity > 0:
                            self.MarketOrder(atm_call.Symbol, quantity)
            
            elif self.fast.Current.Value < self.slow.Current.Value and self.rsi.Current.Value < 40:
                # Buy puts
                if puts:
                    atm_put = sorted(puts, key=lambda x: abs(x.Strike - underlying_price))[0]
                    if atm_put.AskPrice > 0 and atm_put.AskPrice < 10:
                        quantity = int(self.Portfolio.Cash * 0.1 / (atm_put.AskPrice * 100))
                        if quantity > 0:
                            self.MarketOrder(atm_put.Symbol, quantity)
    
    def ReduceExposure(self):
        """Reduce positions during uncertain markets"""
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].Invested:
                # Keep only half of current positions
                current_value = self.Portfolio[symbol].HoldingsValue
                target_value = current_value * 0.5
                
                if abs(current_value) > 1000:  # Only adjust significant positions
                    target_pct = target_value / self.Portfolio.TotalPortfolioValue
                    self.SetHoldings(symbol, target_pct)
    
    def EndOfDayAdjustment(self):
        """End of day risk management and position adjustment"""
        current_value = self.Portfolio.TotalPortfolioValue
        
        # Update high water mark
        if current_value > self.highest_value:
            self.highest_value = current_value
        
        # Calculate drawdown
        drawdown = (self.highest_value - current_value) / self.highest_value
        
        # Emergency risk management
        if drawdown > 0.25:  # 25% drawdown
            self.Debug(f"Emergency liquidation - Drawdown: {drawdown:.2%}")
            self.Liquidate()
            return
        
        # Take profits on winners
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].UnrealizedProfitPercent > 0.10:  # 10% profit
                self.Debug(f"Taking profits on {symbol}: {self.Portfolio[symbol].UnrealizedProfitPercent:.2%}")
                self.Liquidate(symbol)
        
        # Cut losses quickly
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].UnrealizedProfitPercent < -0.03:  # 3% loss
                self.Debug(f"Cutting losses on {symbol}: {self.Portfolio[symbol].UnrealizedProfitPercent:.2%}")
                self.Liquidate(symbol)
        
        # Log daily performance
        daily_return = (current_value / self.highest_value - 1) if self.Time.date() != self.StartDate.date() else 0
        self.Debug(f"Date: {self.Time.date()}, Value: ${current_value:,.2f}, Daily: {daily_return:.2%}")
    
    def OnEndOfAlgorithm(self):
        """Final performance calculation"""
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value / 100000 - 1) * 100
        
        # Calculate CAGR
        years = (self.Time - self.StartDate).days / 365.25
        cagr = ((final_value / 100000) ** (1/years) - 1) * 100
        
        self.Debug("=== ULTRA BEAST FINAL RESULTS ===")
        self.Debug(f"Final Value: ${final_value:,.2f}")
        self.Debug(f"Total Return: {total_return:.2f}%")
        self.Debug(f"CAGR: {cagr:.2f}%")
        self.Debug(f"Target CAGR: 25%+ {'✓' if cagr > 25 else '✗'}")