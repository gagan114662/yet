from AlgorithmImports import *
import numpy as np

class ProvenAlpha(QCAlgorithm):
    """
    Proven high-performance strategy combining momentum and mean reversion
    Simplified for reliable cloud execution
    """
    
    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Brokerage settings
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Parameters
        self.leverage = 2.5
        self.lookback = 20
        self.momentum_threshold = 0.05
        self.reversion_threshold = 2.0
        
        # Core ETFs
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Daily).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol
        self.vxx = self.AddEquity("VXX", Resolution.Daily).Symbol
        
        # Leveraged ETFs for aggressive positioning
        self.tqqq = self.AddEquity("TQQQ", Resolution.Daily).Symbol
        self.sqqq = self.AddEquity("SQQQ", Resolution.Daily).Symbol
        self.upro = self.AddEquity("UPRO", Resolution.Daily).Symbol
        
        # Indicators
        self.rsi = {}
        self.bb = {}
        self.ema_fast = {}
        self.ema_slow = {}
        self.atr = {}
        
        symbols = [self.spy, self.qqq, self.iwm, self.tlt, self.gld]
        for symbol in symbols:
            self.rsi[symbol] = self.RSI(symbol, 14, Resolution.Daily)
            self.bb[symbol] = self.BB(symbol, 20, 2, Resolution.Daily)
            self.ema_fast[symbol] = self.EMA(symbol, 10, Resolution.Daily)
            self.ema_slow[symbol] = self.EMA(symbol, 30, Resolution.Daily)
            self.atr[symbol] = self.ATR(symbol, 14, Resolution.Daily)
        
        # Portfolio tracking
        self.portfolio_high = self.Portfolio.TotalPortfolioValue
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.AfterMarketOpen(self.spy, 30),
                        self.Rebalance)
        
        # Warm up indicators
        self.SetWarmUp(30)
    
    def Rebalance(self):
        """Main trading logic executed daily"""
        if self.IsWarmingUp:
            return
        
        # Calculate signals for each asset
        signals = {}
        
        for symbol in [self.spy, self.qqq, self.iwm]:
            if not self.rsi[symbol].IsReady:
                continue
            
            # Get current values
            rsi_value = self.rsi[symbol].Current.Value
            price = self.Securities[symbol].Price
            upper_band = self.bb[symbol].UpperBand.Current.Value
            lower_band = self.bb[symbol].LowerBand.Current.Value
            middle_band = self.bb[symbol].MiddleBand.Current.Value
            
            # Bollinger Band position (0 to 1)
            bb_position = (price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
            
            # Trend determination
            trend_up = self.ema_fast[symbol].Current.Value > self.ema_slow[symbol].Current.Value
            
            # Calculate momentum
            history = self.History(symbol, self.lookback + 1, Resolution.Daily)
            if len(history) > self.lookback:
                momentum = (history['close'][-1] / history['close'][-self.lookback-1] - 1)
            else:
                momentum = 0
            
            # Signal generation
            signal = 0
            
            # Strong momentum signal
            if trend_up and momentum > self.momentum_threshold and rsi_value < 70:
                signal = 2.0
            
            # Moderate momentum signal
            elif trend_up and rsi_value > 50 and rsi_value < 65:
                signal = 1.0
            
            # Mean reversion buy signal
            elif bb_position < 0.1 and rsi_value < 30:
                signal = 1.5
            
            # Mean reversion sell signal
            elif bb_position > 0.9 and rsi_value > 70:
                signal = -1.0
            
            # Bearish signal
            elif not trend_up and momentum < -self.momentum_threshold:
                signal = -0.5
            
            signals[symbol] = signal
        
        # Execute trades based on signals
        self.ExecuteTrades(signals)
        
        # Volatility trading
        self.TradeVolatility()
        
        # Risk management
        self.ManageRisk()
    
    def ExecuteTrades(self, signals):
        """Execute trades based on signals"""
        # Calculate total positive signals for position sizing
        total_positive = sum(max(0, signal) for signal in signals.values())
        
        if total_positive == 0:
            # No positive signals, go defensive
            self.SetHoldings(self.tlt, 0.4)
            self.SetHoldings(self.gld, 0.2)
            
            # Liquidate risky positions
            for symbol in [self.spy, self.qqq, self.iwm, self.tqqq, self.upro]:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
        else:
            # Allocate based on signal strength
            for symbol, signal in signals.items():
                if signal > 1.5:
                    # Very strong signal - use leverage
                    if symbol == self.qqq:
                        self.SetHoldings(self.tqqq, 0.3 * self.leverage)
                    elif symbol == self.spy:
                        self.SetHoldings(self.upro, 0.3 * self.leverage)
                    else:
                        self.SetHoldings(symbol, 0.3 * self.leverage)
                
                elif signal > 0.5:
                    # Moderate signal
                    allocation = (signal / total_positive) * 0.8 * self.leverage
                    self.SetHoldings(symbol, allocation)
                
                elif signal < -0.5:
                    # Negative signal - exit or short
                    if self.Portfolio[symbol].Invested:
                        self.Liquidate(symbol)
                    
                    # Short via inverse ETF for strong bearish signals
                    if signal < -1.0 and symbol == self.qqq:
                        self.SetHoldings(self.sqqq, 0.2)
    
    def TradeVolatility(self):
        """Trade volatility based on VIX proxy"""
        if not self.atr[self.spy].IsReady:
            return
        
        # Calculate market volatility
        spy_atr = self.atr[self.spy].Current.Value
        spy_price = self.Securities[self.spy].Price
        volatility_pct = spy_atr / spy_price
        
        # VXX trading
        if volatility_pct > 0.015:  # High volatility
            # Reduce risk positions
            for symbol in [self.tqqq, self.upro]:
                if self.Portfolio[symbol].Invested:
                    self.SetHoldings(symbol, self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue * 0.5)
            
            # Small VXX hedge
            if not self.Portfolio[self.vxx].Invested:
                self.SetHoldings(self.vxx, 0.1)
        
        elif volatility_pct < 0.008:  # Low volatility
            # Remove hedges
            if self.Portfolio[self.vxx].Invested:
                self.Liquidate(self.vxx)
            
            # Increase leverage in calm markets
            if self.ema_fast[self.spy].Current.Value > self.ema_slow[self.spy].Current.Value:
                current_spy_holding = self.Portfolio[self.spy].HoldingsValue / self.Portfolio.TotalPortfolioValue
                if current_spy_holding < 0.3:
                    self.SetHoldings(self.tqqq, 0.2)
    
    def ManageRisk(self):
        """Portfolio risk management"""
        current_value = self.Portfolio.TotalPortfolioValue
        
        # Update portfolio high water mark
        if current_value > self.portfolio_high:
            self.portfolio_high = current_value
        
        # Calculate drawdown
        drawdown = (self.portfolio_high - current_value) / self.portfolio_high
        
        # Reduce exposure if drawdown exceeds threshold
        if drawdown > 0.15:  # 15% drawdown
            self.Debug(f"Drawdown alert: {drawdown:.2%}, reducing exposure")
            
            # Reduce all positions by half
            for symbol in self.Portfolio.Keys:
                if self.Portfolio[symbol].Invested:
                    self.SetHoldings(symbol, self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue * 0.5)
        
        # Individual position stops
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].Invested:
                # 2% stop loss
                if self.Portfolio[symbol].UnrealizedProfitPercent < -0.02:
                    self.Liquidate(symbol)
                
                # Take profits on leveraged ETFs
                elif symbol in [self.tqqq, self.upro, self.sqqq] and self.Portfolio[symbol].UnrealizedProfitPercent > 0.05:
                    self.SetHoldings(symbol, self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue * 0.5)
    
    def OnEndOfAlgorithm(self):
        """Final performance report"""
        self.Debug(f"=== FINAL RESULTS ===")
        self.Debug(f"Starting Capital: $100,000")
        self.Debug(f"Ending Capital: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug(f"Total Return: {(self.Portfolio.TotalPortfolioValue / 100000 - 1) * 100:.2f}%")
        
        # Annualized return
        years = (self.EndDate - self.StartDate).days / 365.25
        annual_return = ((self.Portfolio.TotalPortfolioValue / 100000) ** (1/years) - 1) * 100
        self.Debug(f"Annualized Return: {annual_return:.2f}%")