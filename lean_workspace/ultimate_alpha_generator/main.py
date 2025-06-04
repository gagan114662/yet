from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class UltimateAlphaGenerator(QCAlgorithm):
    """
    Ultimate Alpha Generator - Simplified high-performance strategy
    Focus on proven edges: momentum, volatility premium, and mean reversion
    
    Target: 35%+ CAGR, 1.5+ Sharpe, <20% Max Drawdown
    """
    
    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        
        # Core settings
        self.UniverseSettings.Resolution = Resolution.Hour
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Risk parameters
        self.max_leverage = 4.0
        self.stop_loss_pct = 0.02
        self.profit_target_pct = 0.05
        self.max_drawdown = 0.18
        
        # Portfolio tracking
        self.portfolio_peak = self.Portfolio.TotalPortfolioValue
        self.daily_returns = []
        
        # Add primary instruments
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Hour).Symbol
        self.eem = self.AddEquity("EEM", Resolution.Hour).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Hour).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Hour).Symbol
        
        # Leveraged ETFs for momentum
        self.tqqq = self.AddEquity("TQQQ", Resolution.Hour).Symbol
        self.sqqq = self.AddEquity("SQQQ", Resolution.Hour).Symbol
        self.upro = self.AddEquity("UPRO", Resolution.Hour).Symbol
        self.spxu = self.AddEquity("SPXU", Resolution.Hour).Symbol
        
        # VIX for volatility trading
        self.vxx = self.AddEquity("VXX", Resolution.Hour).Symbol
        self.svxy = self.AddEquity("SVXY", Resolution.Hour).Symbol
        
        # Sector ETFs
        self.xlk = self.AddEquity("XLK", Resolution.Hour).Symbol  # Tech
        self.xlf = self.AddEquity("XLF", Resolution.Hour).Symbol  # Financials
        self.xle = self.AddEquity("XLE", Resolution.Hour).Symbol  # Energy
        
        # Universe selection
        self.AddUniverse(self.CoarseSelectionFunction)
        self.selected_stocks = []
        self.last_rebalance = self.Time
        
        # Technical indicators
        self.indicators = {}
        self.SetupIndicators()
        
        # Schedule trading functions
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.AfterMarketOpen(self.spy, 30),
                        self.MorningRebalance)
        
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.BeforeMarketClose(self.spy, 30),
                        self.AfternoonRebalance)
        
        # Warm up
        self.SetWarmUp(timedelta(days=30))
    
    def SetupIndicators(self):
        """Initialize indicators for all symbols"""
        symbols = [self.spy, self.qqq, self.iwm, self.tlt, self.vxx, self.xlk, self.xlf, self.xle]
        
        for symbol in symbols:
            self.indicators[symbol] = {
                'rsi': self.RSI(symbol, 14, Resolution.Daily),
                'ema_fast': self.EMA(symbol, 10, Resolution.Daily),
                'ema_slow': self.EMA(symbol, 30, Resolution.Daily),
                'bb': self.BB(symbol, 20, 2, Resolution.Daily),
                'atr': self.ATR(symbol, 14, Resolution.Daily),
                'macd': self.MACD(symbol, 12, 26, 9, Resolution.Daily),
                'adx': self.ADX(symbol, 14, Resolution.Daily)
            }
    
    def CoarseSelectionFunction(self, coarse):
        """Select high-momentum liquid stocks"""
        # Only rebalance weekly
        if self.Time - self.last_rebalance < timedelta(days=7):
            return Universe.Unchanged
        
        # Filter for liquid stocks
        filtered = [x for x in coarse if x.HasFundamentalData 
                   and x.Price > 10 
                   and x.Price < 1000
                   and x.DollarVolume > 20000000]
        
        # Sort by dollar volume and momentum
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)[:200]
        
        # Calculate momentum
        momentum_stocks = []
        for stock in sorted_by_volume:
            history = self.History(stock.Symbol, 30, Resolution.Daily)
            if len(history) > 20:
                momentum = (history['close'][-1] / history['close'][-20] - 1)
                if momentum > 0.05:  # 5% monthly momentum threshold
                    momentum_stocks.append((stock.Symbol, momentum))
        
        # Select top 20 momentum stocks
        momentum_stocks.sort(key=lambda x: x[1], reverse=True)
        self.selected_stocks = [x[0] for x in momentum_stocks[:20]]
        self.last_rebalance = self.Time
        
        return self.selected_stocks
    
    def OnData(self, data):
        """Main trading logic"""
        if self.IsWarmingUp:
            return
        
        # Risk management check
        if not self.CheckRiskLimits():
            self.EmergencyLiquidation()
            return
        
        # Execute trading strategies
        self.TradeMomentum(data)
        self.TradeVolatility(data)
        self.TradeMeanReversion(data)
        self.TradeSectorRotation(data)
    
    def TradeMomentum(self, data):
        """Momentum trading with leveraged ETFs"""
        if self.spy not in data or not self.indicators[self.spy]['macd'].IsReady:
            return
        
        spy_macd = self.indicators[self.spy]['macd']
        qqq_macd = self.indicators[self.qqq]['macd'] if self.qqq in self.indicators else None
        
        # Strong bullish signal
        if (spy_macd.Current.Value > spy_macd.Signal.Current.Value and 
            spy_macd.Current.Value > 0):
            
            # Check QQQ confirmation
            if qqq_macd and qqq_macd.Current.Value > qqq_macd.Signal.Current.Value:
                # Use leveraged ETF for strong trends
                self.SetHoldings(self.tqqq, 0.3)
                self.SetHoldings(self.upro, 0.2)
            else:
                # Regular exposure
                self.SetHoldings(self.spy, 0.25)
                self.SetHoldings(self.qqq, 0.25)
        
        # Strong bearish signal
        elif (spy_macd.Current.Value < spy_macd.Signal.Current.Value and 
              spy_macd.Current.Value < 0):
            
            # Defensive positioning
            self.SetHoldings(self.tlt, 0.3)
            self.SetHoldings(self.gld, 0.2)
            
            # Small short exposure
            if self.indicators[self.spy]['rsi'].Current.Value > 70:
                self.SetHoldings(self.spxu, 0.1)
    
    def TradeVolatility(self, data):
        """Volatility premium harvesting"""
        if self.vxx not in data or not self.indicators[self.vxx]['bb'].IsReady:
            return
        
        vxx_price = data[self.vxx].Price
        vxx_bb = self.indicators[self.vxx]['bb']
        
        # VXX mean reversion
        bb_position = (vxx_price - vxx_bb.LowerBand.Current.Value) / (
            vxx_bb.UpperBand.Current.Value - vxx_bb.LowerBand.Current.Value)
        
        # Short volatility when overbought
        if bb_position > 0.8 and self.indicators[self.vxx]['rsi'].Current.Value > 70:
            self.SetHoldings(self.vxx, -0.15)
            
            # Hedge with SVXY if available
            if self.svxy in data:
                self.SetHoldings(self.svxy, 0.15)
        
        # Close positions when normalized
        elif 0.3 < bb_position < 0.7:
            if self.Portfolio[self.vxx].Invested:
                self.Liquidate(self.vxx)
            if self.svxy in self.Portfolio and self.Portfolio[self.svxy].Invested:
                self.Liquidate(self.svxy)
    
    def TradeMeanReversion(self, data):
        """Mean reversion on oversold conditions"""
        # Trade individual stocks from universe
        for symbol in self.selected_stocks[:5]:
            if symbol not in data or symbol not in self.Securities:
                continue
            
            # Get or create indicators
            if symbol not in self.indicators:
                self.indicators[symbol] = {
                    'rsi': self.RSI(symbol, 14, Resolution.Daily),
                    'bb': self.BB(symbol, 20, 2, Resolution.Daily)
                }
            
            if not self.indicators[symbol]['rsi'].IsReady:
                continue
            
            rsi = self.indicators[symbol]['rsi'].Current.Value
            
            # Extreme oversold - buy
            if rsi < 30 and not self.Portfolio[symbol].Invested:
                # Position size based on oversold level
                size = 0.05 * (1 + (30 - rsi) / 30)  # More oversold = larger position
                self.SetHoldings(symbol, size)
                
                # Set profit target
                entry_price = data[symbol].Price
                target_price = entry_price * (1 + self.profit_target_pct)
                self.LimitOrder(symbol, -self.Portfolio[symbol].Quantity, target_price)
            
            # Take profits on overbought
            elif rsi > 70 and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)
    
    def TradeSectorRotation(self, data):
        """Rotate into strongest sectors"""
        sectors = {
            self.xlk: 'Technology',
            self.xlf: 'Financials', 
            self.xle: 'Energy'
        }
        
        sector_strength = {}
        
        for sector_symbol, sector_name in sectors.items():
            if sector_symbol in self.indicators and self.indicators[sector_symbol]['macd'].IsReady:
                macd = self.indicators[sector_symbol]['macd']
                adx = self.indicators[sector_symbol]['adx'].Current.Value
                
                # Sector strength = MACD signal + ADX trend strength
                strength = (macd.Current.Value - macd.Signal.Current.Value) * (adx / 25)
                sector_strength[sector_symbol] = strength
        
        if sector_strength:
            # Find strongest sector
            strongest_sector = max(sector_strength.items(), key=lambda x: x[1])
            
            if strongest_sector[1] > 0.5:  # Positive strength threshold
                # Allocate to strongest sector
                self.SetHoldings(strongest_sector[0], 0.2)
                
                # Reduce or exit weak sectors
                for sector_symbol, strength in sector_strength.items():
                    if strength < -0.2 and self.Portfolio[sector_symbol].Invested:
                        self.Liquidate(sector_symbol)
    
    def MorningRebalance(self):
        """Morning portfolio rebalancing"""
        # Update performance metrics
        self.UpdatePerformanceTracking()
        
        # Rebalance core positions
        total_value = self.Portfolio.TotalPortfolioValue
        
        # Target allocations based on market conditions
        if self.indicators[self.spy]['ema_fast'].Current.Value > self.indicators[self.spy]['ema_slow'].Current.Value:
            # Bullish market
            target_allocations = {
                self.spy: 0.2,
                self.qqq: 0.2,
                self.xlk: 0.15,
                self.tqqq: 0.15
            }
        else:
            # Bearish/neutral market
            target_allocations = {
                self.spy: 0.1,
                self.tlt: 0.2,
                self.gld: 0.15,
                self.iwm: 0.1
            }
        
        # Rebalance to targets
        for symbol, target in target_allocations.items():
            if symbol in self.Securities:
                current = self.Portfolio[symbol].HoldingsValue / total_value if total_value > 0 else 0
                
                if abs(current - target) > 0.05:  # 5% threshold
                    self.SetHoldings(symbol, target)
    
    def AfternoonRebalance(self):
        """Afternoon risk reduction"""
        # Close losing positions
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].UnrealizedProfitPercent < -self.stop_loss_pct:
                self.Liquidate(symbol)
        
        # Take profits on winners
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].UnrealizedProfitPercent > self.profit_target_pct:
                # Reduce position by half
                self.SetHoldings(symbol, self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue * 0.5)
        
        # Reduce leverage before close
        total_holdings = sum(abs(self.Portfolio[symbol].HoldingsValue) for symbol in self.Portfolio.Keys)
        current_leverage = total_holdings / self.Portfolio.TotalPortfolioValue if self.Portfolio.TotalPortfolioValue > 0 else 0
        
        if current_leverage > 2.0:
            # Reduce all positions proportionally
            for symbol in self.Portfolio.Keys:
                if self.Portfolio[symbol].Invested:
                    self.SetHoldings(symbol, self.Portfolio[symbol].HoldingsValue / total_holdings * 2.0)
    
    def CheckRiskLimits(self):
        """Portfolio risk management"""
        current_value = self.Portfolio.TotalPortfolioValue
        
        # Update peak
        if current_value > self.portfolio_peak:
            self.portfolio_peak = current_value
        
        # Check drawdown
        drawdown = (self.portfolio_peak - current_value) / self.portfolio_peak if self.portfolio_peak > 0 else 0
        
        if drawdown > self.max_drawdown:
            self.Debug(f"Max drawdown exceeded: {drawdown:.2%}")
            return False
        
        # Check leverage
        total_holdings = sum(abs(self.Portfolio[symbol].HoldingsValue) for symbol in self.Portfolio.Keys)
        current_leverage = total_holdings / current_value if current_value > 0 else 0
        
        if current_leverage > self.max_leverage:
            self.Debug(f"Max leverage exceeded: {current_leverage:.2f}")
            return False
        
        return True
    
    def EmergencyLiquidation(self):
        """Emergency portfolio liquidation"""
        self.Liquidate()
        self.Debug("Emergency liquidation executed - risk limits breached")
    
    def UpdatePerformanceTracking(self):
        """Track daily performance"""
        if self.Time.hour == 16:  # End of day
            daily_return = (self.Portfolio.TotalPortfolioValue / self.Portfolio.TotalPortfolioValue - 1)
            self.daily_returns.append(daily_return)
            
            # Keep last 252 days
            if len(self.daily_returns) > 252:
                self.daily_returns.pop(0)
            
            # Calculate and log metrics monthly
            if self.Time.day == 1 and len(self.daily_returns) > 20:
                returns_array = np.array(self.daily_returns)
                sharpe = np.sqrt(252) * (returns_array.mean() - 0.05/252) / (returns_array.std() + 1e-8)
                
                self.Debug(f"Monthly Update - Sharpe: {sharpe:.2f}, Avg Return: {returns_array.mean()*252:.2%}")
    
    def OnOrderEvent(self, orderEvent):
        """Log order execution"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order filled: {orderEvent.Symbol} {orderEvent.Quantity} @ {orderEvent.FillPrice}")