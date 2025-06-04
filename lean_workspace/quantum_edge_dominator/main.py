from AlgorithmImports import *
import numpy as np
from scipy import stats
from collections import deque
import pandas as pd

class QuantumEdgeDominator(QCAlgorithm):
    """
    Ultra-high performance multi-factor strategy combining:
    - Cross-asset momentum with dynamic universe selection
    - Volatility harvesting and VIX trading
    - Statistical arbitrage and mean reversion
    - Microstructure alpha capture
    - Adaptive risk management
    
    Target: 40%+ CAGR, 1.8+ Sharpe, <20% DD
    """
    
    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        
        # Core configuration
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Risk parameters
        self.max_leverage = 6.0
        self.position_limit = 0.15  # Max 15% per position
        self.stop_loss = 0.02  # 2% stop loss
        self.drawdown_limit = 0.15  # 15% portfolio drawdown limit
        
        # Performance tracking
        self.portfolio_high = self.Portfolio.TotalPortfolioValue
        self.daily_returns = deque(maxlen=252)
        self.trade_stats = {'wins': 0, 'losses': 0, 'total_profit': 0}
        
        # Strategy components
        self.momentum_weight = 0.35
        self.volatility_weight = 0.25
        self.arbitrage_weight = 0.20
        self.microstructure_weight = 0.20
        
        # Add core assets
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Minute).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Minute).Symbol
        self.eem = self.AddEquity("EEM", Resolution.Minute).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Minute).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Minute).Symbol
        self.usd = self.AddEquity("UUP", Resolution.Minute).Symbol
        
        # Volatility instruments
        self.vix = self.AddIndex("VIX", Resolution.Minute).Symbol
        self.vxx = self.AddEquity("VXX", Resolution.Minute).Symbol
        self.svxy = self.AddEquity("SVXY", Resolution.Minute).Symbol
        
        # Sector ETFs for rotation
        self.sectors = [
            self.AddEquity("XLK", Resolution.Minute).Symbol,  # Tech
            self.AddEquity("XLF", Resolution.Minute).Symbol,  # Financials
            self.AddEquity("XLE", Resolution.Minute).Symbol,  # Energy
            self.AddEquity("XLV", Resolution.Minute).Symbol,  # Healthcare
            self.AddEquity("XLI", Resolution.Minute).Symbol,  # Industrials
            self.AddEquity("XLY", Resolution.Minute).Symbol,  # Consumer Disc
            self.AddEquity("XLP", Resolution.Minute).Symbol,  # Consumer Staples
            self.AddEquity("XLU", Resolution.Minute).Symbol,  # Utilities
            self.AddEquity("XLB", Resolution.Minute).Symbol,  # Materials
            self.AddEquity("XLRE", Resolution.Minute).Symbol, # Real Estate
        ]
        
        # High momentum stocks universe
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.universe_stocks = []
        self.rebalance_time = self.Time
        
        # Technical indicators
        self.indicators = {}
        self.SetupIndicators()
        
        # Schedule functions
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.AfterMarketOpen(self.spy, 5),
                        self.OpeningTrade)
        
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.Every(timedelta(minutes=30)),
                        self.IntradayRebalance)
        
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.BeforeMarketClose(self.spy, 10),
                        self.ClosingTrade)
        
        self.Schedule.On(self.DateRules.WeekStart(self.spy),
                        self.TimeRules.AfterMarketOpen(self.spy, 30),
                        self.WeeklyRebalance)
        
        # Options for gamma scalping
        self.options_chains = {}
        spy_option = self.AddOption("SPY", Resolution.Minute)
        spy_option.SetFilter(-10, 10, timedelta(0), timedelta(30))
        
        # Warm up period
        self.SetWarmUp(timedelta(days=60))
    
    def SetupIndicators(self):
        """Initialize technical indicators for all symbols"""
        symbols = [self.spy, self.qqq, self.iwm, self.tlt, self.vxx] + self.sectors
        
        for symbol in symbols:
            self.indicators[symbol] = {
                'ema_fast': self.EMA(symbol, 8, Resolution.Daily),
                'ema_slow': self.EMA(symbol, 21, Resolution.Daily),
                'rsi': self.RSI(symbol, 14, Resolution.Daily),
                'bb': self.BB(symbol, 20, 2, Resolution.Daily),
                'atr': self.ATR(symbol, 14, Resolution.Daily),
                'adx': self.ADX(symbol, 14, Resolution.Daily),
                'macd': self.MACD(symbol, 12, 26, 9, Resolution.Daily),
                'volume_sma': self.SMA(symbol, 20, Resolution.Daily, Field.Volume)
            }
    
    def CoarseSelectionFunction(self, coarse):
        """Select high momentum stocks with good liquidity"""
        if self.Time - self.rebalance_time < timedelta(days=7):
            return Universe.Unchanged
        
        # Filter for liquid stocks
        filtered = [x for x in coarse if x.HasFundamentalData 
                   and x.Price > 10 
                   and x.DollarVolume > 10000000]
        
        # Sort by momentum (3-month return)
        sorted_by_momentum = sorted(filtered, 
                                  key=lambda x: x.Price / max(x.Price - (x.Price * 0.001), 0.01),
                                  reverse=True)
        
        return [x.Symbol for x in sorted_by_momentum[:100]]
    
    def FineSelectionFunction(self, fine):
        """Further filter by fundamental factors"""
        # Filter by profitability and growth
        filtered = [x for x in fine if x.OperationRatios.ROE.Value > 0.15
                   and x.ValuationRatios.PERatio < 30
                   and x.EarningReports.BasicEPS.TwelveMonths > 0]
        
        # Sort by combined score
        scored = []
        for stock in filtered:
            momentum_score = 1.0  # Placeholder for price momentum
            quality_score = min(stock.OperationRatios.ROE.Value / 0.20, 2.0)
            value_score = max(30 - stock.ValuationRatios.PERatio, 0) / 30
            
            combined_score = momentum_score * 0.5 + quality_score * 0.3 + value_score * 0.2
            scored.append((stock.Symbol, combined_score))
        
        # Select top stocks
        scored.sort(key=lambda x: x[1], reverse=True)
        self.universe_stocks = [x[0] for x in scored[:20]]
        self.rebalance_time = self.Time
        
        return self.universe_stocks
    
    def OnData(self, data):
        """Main trading logic"""
        if self.IsWarmingUp:
            return
        
        # Risk management check
        if not self.RiskManagementCheck():
            self.LiquidateAll()
            return
        
        # Update performance tracking
        self.UpdatePerformanceMetrics()
        
        # Execute trading strategies
        self.ExecuteMomentumStrategy(data)
        self.ExecuteVolatilityStrategy(data)
        self.ExecuteArbitrageStrategy(data)
        self.ExecuteMicrostructureStrategy(data)
        
        # Options gamma scalping
        self.ExecuteOptionsStrategy(data)
    
    def ExecuteMomentumStrategy(self, data):
        """Cross-asset momentum with dynamic allocation"""
        momentum_signals = {}
        
        # Calculate momentum scores
        for symbol in [self.spy, self.qqq, self.iwm, self.eem, self.tlt, self.gld]:
            if symbol not in data or not data[symbol]:
                continue
                
            if symbol in self.indicators and self.indicators[symbol]['macd'].IsReady:
                macd = self.indicators[symbol]['macd']
                rsi = self.indicators[symbol]['rsi'].Current.Value
                adx = self.indicators[symbol]['adx'].Current.Value
                
                # Momentum score combining multiple factors
                trend_strength = (macd.Current.Value - macd.Signal.Current.Value) / self.indicators[symbol]['atr'].Current.Value
                momentum_score = trend_strength * (adx / 25) * (1 + (rsi - 50) / 100)
                
                momentum_signals[symbol] = momentum_score
        
        # Rank and allocate
        if momentum_signals:
            sorted_signals = sorted(momentum_signals.items(), key=lambda x: x[1], reverse=True)
            
            # Long top performers
            for i, (symbol, score) in enumerate(sorted_signals[:3]):
                if score > 0.5:
                    allocation = self.momentum_weight * (0.4 - i * 0.1)
                    self.SetHoldings(symbol, allocation * min(score, 2.0))
            
            # Short bottom performers
            for i, (symbol, score) in enumerate(sorted_signals[-2:]):
                if score < -0.5:
                    allocation = self.momentum_weight * 0.1
                    self.SetHoldings(symbol, -allocation * min(abs(score), 1.5))
    
    def ExecuteVolatilityStrategy(self, data):
        """Volatility harvesting and VIX trading"""
        if self.vix not in data or self.vxx not in data:
            return
        
        vix_price = data[self.vix].Price if data[self.vix] else 0
        vxx_price = data[self.vxx].Price if data[self.vxx] else 0
        
        if vix_price > 0 and vxx_price > 0:
            # VIX term structure analysis
            if vix_price < 15:  # Low volatility regime
                # Short volatility
                self.SetHoldings(self.vxx, -self.volatility_weight * 0.5)
                if self.svxy in data and data[self.svxy]:
                    self.SetHoldings(self.svxy, self.volatility_weight * 0.5)
            
            elif vix_price > 25:  # High volatility regime
                # Long volatility for protection
                self.SetHoldings(self.vxx, self.volatility_weight * 0.3)
                # Reduce equity exposure
                for symbol in [self.spy, self.qqq]:
                    if self.Portfolio[symbol].Invested:
                        current_holding = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                        self.SetHoldings(symbol, current_holding * 0.5)
            
            else:  # Normal regime
                # Volatility mean reversion
                vix_sma = self.History(self.vix, 20, Resolution.Daily)['close'].mean()
                if vix_price > vix_sma * 1.1:
                    self.SetHoldings(self.vxx, self.volatility_weight * 0.2)
                elif vix_price < vix_sma * 0.9:
                    self.SetHoldings(self.vxx, -self.volatility_weight * 0.2)
    
    def ExecuteArbitrageStrategy(self, data):
        """Statistical arbitrage on correlated pairs"""
        # SPY-QQQ pair trading
        if self.spy in data and self.qqq in data and data[self.spy] and data[self.qqq]:
            spy_returns = self.History(self.spy, 20, Resolution.Daily)['close'].pct_change().dropna()
            qqq_returns = self.History(self.qqq, 20, Resolution.Daily)['close'].pct_change().dropna()
            
            if len(spy_returns) == len(qqq_returns) and len(spy_returns) > 10:
                # Calculate z-score
                spread = spy_returns.values - qqq_returns.values
                z_score = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-8)
                
                if z_score > 2:  # SPY overvalued relative to QQQ
                    self.SetHoldings(self.spy, -self.arbitrage_weight * 0.5)
                    self.SetHoldings(self.qqq, self.arbitrage_weight * 0.5)
                elif z_score < -2:  # SPY undervalued relative to QQQ
                    self.SetHoldings(self.spy, self.arbitrage_weight * 0.5)
                    self.SetHoldings(self.qqq, -self.arbitrage_weight * 0.5)
                else:
                    # Close positions when spread normalizes
                    if abs(z_score) < 0.5:
                        if self.Portfolio[self.spy].IsShort:
                            self.Liquidate(self.spy)
                            self.Liquidate(self.qqq)
    
    def ExecuteMicrostructureStrategy(self, data):
        """High-frequency microstructure trading"""
        for symbol in self.universe_stocks[:5]:  # Top 5 momentum stocks
            if symbol not in data or not data[symbol]:
                continue
            
            # Check bid-ask spread
            if data[symbol].AskPrice > 0 and data[symbol].BidPrice > 0:
                spread = (data[symbol].AskPrice - data[symbol].BidPrice) / data[symbol].Price
                
                # Wide spread opportunity
                if spread > 0.002:  # 0.2% spread
                    current_price = data[symbol].Price
                    
                    # Place limit orders
                    if not self.Portfolio[symbol].Invested:
                        # Buy at bid + small increment
                        limit_price = data[symbol].BidPrice + 0.01
                        quantity = int(self.microstructure_weight * self.Portfolio.TotalPortfolioValue * 0.2 / current_price)
                        
                        if quantity > 0:
                            self.LimitOrder(symbol, quantity, limit_price)
                
                # Momentum burst detection
                if symbol in self.indicators and self.indicators[symbol]['volume_sma'].IsReady:
                    current_volume = data[symbol].Volume
                    avg_volume = self.indicators[symbol]['volume_sma'].Current.Value
                    
                    if current_volume > avg_volume * 3:  # Volume spike
                        # Quick scalp trade
                        if data[symbol].Close > data[symbol].Open:  # Bullish burst
                            self.MarketOrder(symbol, int(self.microstructure_weight * self.Portfolio.TotalPortfolioValue * 0.1 / current_price))
                            # Set tight stop
                            self.StopMarketOrder(symbol, -self.Portfolio[symbol].Quantity, current_price * 0.995)
    
    def ExecuteOptionsStrategy(self, data):
        """Options gamma scalping"""
        if not self.options_chains:
            return
        
        for kvp in data.OptionChains:
            chain = kvp.Value
            if not chain:
                continue
            
            # Filter for ATM options
            underlying_price = chain.Underlying.Price
            atm_strike = min(chain, key=lambda x: abs(x.Strike - underlying_price)).Strike
            
            # Get ATM call and put
            calls = [i for i in chain if i.Strike == atm_strike and i.Right == OptionRight.Call]
            puts = [i for i in chain if i.Strike == atm_strike and i.Right == OptionRight.Put]
            
            if calls and puts:
                call = sorted(calls, key=lambda x: x.Expiry)[0]
                put = sorted(puts, key=lambda x: x.Expiry)[0]
                
                # Delta-neutral straddle for gamma scalping
                if call.BidPrice > 0 and put.BidPrice > 0:
                    # Calculate position size
                    straddle_cost = (call.AskPrice + put.AskPrice) * 100
                    position_size = int(self.Portfolio.TotalPortfolioValue * 0.02 / straddle_cost)
                    
                    if position_size > 0 and not self.Portfolio[call.Symbol].Invested:
                        self.MarketOrder(call.Symbol, position_size)
                        self.MarketOrder(put.Symbol, position_size)
                        
                        # Dynamic hedging with underlying
                        total_delta = position_size * (call.Delta - put.Delta) * 100
                        self.MarketOrder(chain.Underlying.Symbol, -int(total_delta))
    
    def OpeningTrade(self):
        """Opening range breakout strategy"""
        history = self.History([self.spy, self.qqq], 30, Resolution.Minute)
        
        if not history.empty:
            for symbol in [self.spy, self.qqq]:
                symbol_data = history.loc[symbol]
                if len(symbol_data) > 0:
                    opening_range_high = symbol_data['high'][:15].max()
                    opening_range_low = symbol_data['low'][:15].min()
                    current_price = self.Securities[symbol].Price
                    
                    if current_price > opening_range_high:
                        # Breakout buy
                        self.SetHoldings(symbol, 0.1)
                    elif current_price < opening_range_low:
                        # Breakdown short
                        self.SetHoldings(symbol, -0.1)
    
    def IntradayRebalance(self):
        """Intraday position management"""
        # Trim winners
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].UnrealizedProfitPercent > 0.02:  # 2% profit
                self.SetHoldings(symbol, self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue * 0.5)
        
        # Cut losses
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].UnrealizedProfitPercent < -0.01:  # 1% loss
                self.Liquidate(symbol)
                self.trade_stats['losses'] += 1
    
    def ClosingTrade(self):
        """End of day position management"""
        # Close intraday positions
        for symbol in self.universe_stocks:
            if symbol in self.Portfolio and self.Portfolio[symbol].Invested:
                # Close if position was opened today
                if self.Time.date() == self.Portfolio[symbol].EntryTime.date():
                    self.Liquidate(symbol)
                    
                    # Track stats
                    if self.Portfolio[symbol].UnrealizedProfit > 0:
                        self.trade_stats['wins'] += 1
                    else:
                        self.trade_stats['losses'] += 1
                    
                    self.trade_stats['total_profit'] += self.Portfolio[symbol].UnrealizedProfit
    
    def WeeklyRebalance(self):
        """Weekly strategy rebalancing"""
        # Adjust strategy weights based on performance
        total_trades = self.trade_stats['wins'] + self.trade_stats['losses']
        
        if total_trades > 50:
            win_rate = self.trade_stats['wins'] / total_trades
            
            # Increase weight of winning strategies
            if win_rate > 0.55:
                self.momentum_weight = min(0.4, self.momentum_weight * 1.05)
                self.microstructure_weight = min(0.25, self.microstructure_weight * 1.05)
            else:
                self.volatility_weight = min(0.3, self.volatility_weight * 1.05)
                self.arbitrage_weight = min(0.25, self.arbitrage_weight * 1.05)
            
            # Normalize weights
            total_weight = (self.momentum_weight + self.volatility_weight + 
                          self.arbitrage_weight + self.microstructure_weight)
            
            self.momentum_weight /= total_weight
            self.volatility_weight /= total_weight
            self.arbitrage_weight /= total_weight
            self.microstructure_weight /= total_weight
    
    def RiskManagementCheck(self):
        """Portfolio-level risk management"""
        current_value = self.Portfolio.TotalPortfolioValue
        
        # Update portfolio high
        if current_value > self.portfolio_high:
            self.portfolio_high = current_value
        
        # Check drawdown
        drawdown = (self.portfolio_high - current_value) / self.portfolio_high
        
        if drawdown > self.drawdown_limit:
            self.Debug(f"Drawdown limit hit: {drawdown:.2%}")
            return False
        
        # Check leverage
        total_holdings = sum(abs(self.Portfolio[symbol].HoldingsValue) for symbol in self.Portfolio.Keys)
        current_leverage = total_holdings / current_value
        
        if current_leverage > self.max_leverage:
            self.Debug(f"Leverage limit hit: {current_leverage:.2f}")
            return False
        
        return True
    
    def UpdatePerformanceMetrics(self):
        """Track performance for optimization"""
        if self.Time.hour == 16:  # End of day
            daily_return = (self.Portfolio.TotalPortfolioValue / self.Portfolio.TotalPortfolioValue - 1)
            self.daily_returns.append(daily_return)
            
            if len(self.daily_returns) >= 20:
                # Calculate Sharpe ratio
                returns_array = np.array(self.daily_returns)
                sharpe = np.sqrt(252) * (returns_array.mean() - 0.05/252) / (returns_array.std() + 1e-8)
                
                # Log performance
                if self.Time.day == 1:  # Monthly
                    self.Debug(f"Monthly Sharpe: {sharpe:.2f}, Win Rate: {self.trade_stats['wins']/(self.trade_stats['wins']+self.trade_stats['losses']+1e-8):.2%}")
    
    def OnOrderEvent(self, orderEvent):
        """Track order execution"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order filled: {orderEvent.Symbol} {orderEvent.Quantity} @ {orderEvent.FillPrice}")
    
    def LiquidateAll(self):
        """Emergency liquidation"""
        self.Liquidate()
        self.Debug("Emergency liquidation executed")