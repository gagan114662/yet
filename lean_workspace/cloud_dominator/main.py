from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class CloudDominator(QCAlgorithm):
    """
    High-performance strategy optimized for cloud execution
    Target: 30%+ CAGR, 1.5+ Sharpe, <20% Max Drawdown
    """
    
    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        
        # Configuration
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Risk parameters
        self.leverage = 3.0
        self.stop_loss = 0.015  # 1.5%
        self.profit_target = 0.04  # 4%
        self.max_positions = 10
        
        # Core instruments
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Hour).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Hour).Symbol
        self.dia = self.AddEquity("DIA", Resolution.Hour).Symbol
        self.eem = self.AddEquity("EEM", Resolution.Hour).Symbol
        self.efa = self.AddEquity("EFA", Resolution.Hour).Symbol
        
        # Bonds for diversification
        self.tlt = self.AddEquity("TLT", Resolution.Hour).Symbol
        self.ief = self.AddEquity("IEF", Resolution.Hour).Symbol
        self.shy = self.AddEquity("SHY", Resolution.Hour).Symbol
        
        # Commodities
        self.gld = self.AddEquity("GLD", Resolution.Hour).Symbol
        self.slv = self.AddEquity("SLV", Resolution.Hour).Symbol
        self.uso = self.AddEquity("USO", Resolution.Hour).Symbol
        
        # VIX ETFs
        self.vxx = self.AddEquity("VXX", Resolution.Hour).Symbol
        self.svxy = self.AddEquity("SVXY", Resolution.Hour).Symbol
        
        # Sector ETFs
        self.sectors = [
            self.AddEquity("XLK", Resolution.Hour).Symbol,  # Technology
            self.AddEquity("XLF", Resolution.Hour).Symbol,  # Financials
            self.AddEquity("XLE", Resolution.Hour).Symbol,  # Energy
            self.AddEquity("XLV", Resolution.Hour).Symbol,  # Healthcare
            self.AddEquity("XLI", Resolution.Hour).Symbol,  # Industrials
            self.AddEquity("XLY", Resolution.Hour).Symbol,  # Consumer Discretionary
        ]
        
        # Technical indicators
        self.indicators = {}
        self.SetupIndicators()
        
        # Universe selection
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.universe_stocks = []
        self.rebalance_time = self.Time
        
        # Scheduling
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.AfterMarketOpen(self.spy, 30),
                        self.DailyRebalance)
        
        self.Schedule.On(self.DateRules.MonthStart(self.spy),
                        self.TimeRules.AfterMarketOpen(self.spy, 60),
                        self.MonthlyRebalance)
        
        # Warm up period
        self.SetWarmUp(timedelta(days=60))
    
    def SetupIndicators(self):
        """Initialize technical indicators"""
        symbols = [self.spy, self.qqq, self.iwm, self.tlt, self.gld] + self.sectors
        
        for symbol in symbols:
            self.indicators[symbol] = {
                'ema_fast': self.EMA(symbol, 10, Resolution.Daily),
                'ema_slow': self.EMA(symbol, 30, Resolution.Daily),
                'rsi': self.RSI(symbol, 14, Resolution.Daily),
                'bb': self.BB(symbol, 20, 2, Resolution.Daily),
                'atr': self.ATR(symbol, 14, Resolution.Daily),
                'macd': self.MACD(symbol, 12, 26, 9, Resolution.Daily),
                'adx': self.ADX(symbol, 14, Resolution.Daily),
                'stoch': self.STO(symbol, 14, 3, 3, Resolution.Daily)
            }
    
    def CoarseSelectionFunction(self, coarse):
        """Select liquid momentum stocks"""
        if self.Time - self.rebalance_time < timedelta(days=30):
            return Universe.Unchanged
        
        # Filter for liquid stocks
        filtered = [x for x in coarse if x.HasFundamentalData 
                   and x.Price > 10 
                   and x.Price < 500
                   and x.DollarVolume > 10000000]
        
        # Sort by dollar volume
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)[:500]
        
        # Calculate momentum
        selected = []
        for stock in sorted_by_volume[:200]:
            history = self.History(stock.Symbol, 30, Resolution.Daily)
            if len(history) >= 20:
                returns = history['close'].pct_change().dropna()
                momentum = (history['close'][-1] / history['close'][-20] - 1)
                volatility = returns.std() * np.sqrt(252)
                
                # Select high momentum, reasonable volatility
                if momentum > 0.05 and volatility < 0.5:
                    selected.append(stock.Symbol)
        
        return selected[:100]
    
    def FineSelectionFunction(self, fine):
        """Further filter by fundamentals"""
        # Filter by quality metrics
        filtered = [x for x in fine if x.OperationRatios.ROE.Value > 0.10
                   and x.ValuationRatios.PERatio > 0
                   and x.ValuationRatios.PERatio < 50]
        
        # Sort by quality score
        scored = []
        for stock in filtered:
            roe = min(x.OperationRatios.ROE.Value, 0.5)
            pe_score = max(0, 1 - x.ValuationRatios.PERatio / 50)
            score = roe * 0.6 + pe_score * 0.4
            scored.append((stock.Symbol, score))
        
        # Select top stocks
        scored.sort(key=lambda x: x[1], reverse=True)
        self.universe_stocks = [x[0] for x in scored[:20]]
        self.rebalance_time = self.Time
        
        return self.universe_stocks
    
    def OnData(self, data):
        """Main trading logic"""
        if self.IsWarmingUp:
            return
        
        # Trade core strategies
        self.TradeMomentum(data)
        self.TradeMeanReversion(data)
        self.TradeVolatility(data)
        self.TradeSectorRotation(data)
        
        # Risk management
        self.ManageRisk()
    
    def TradeMomentum(self, data):
        """Momentum trading strategy"""
        # Trade major indices
        for symbol in [self.spy, self.qqq, self.iwm]:
            if symbol not in data or symbol not in self.indicators:
                continue
            
            if not self.indicators[symbol]['macd'].IsReady:
                continue
            
            macd = self.indicators[symbol]['macd']
            rsi = self.indicators[symbol]['rsi'].Current.Value
            adx = self.indicators[symbol]['adx'].Current.Value
            
            # Strong bullish signal
            if (macd.Current.Value > macd.Signal.Current.Value and 
                macd.Current.Value > 0 and 
                rsi > 50 and 
                adx > 25):
                
                # Size based on signal strength
                signal_strength = min((macd.Current.Value / self.indicators[symbol]['atr'].Current.Value), 2)
                position_size = 0.15 * signal_strength
                
                if not self.Portfolio[symbol].Invested:
                    self.SetHoldings(symbol, position_size * self.leverage)
            
            # Exit signals
            elif (macd.Current.Value < macd.Signal.Current.Value or rsi > 70):
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
        
        # Trade momentum stocks
        for symbol in self.universe_stocks[:5]:
            if symbol not in data or not data[symbol]:
                continue
            
            # Get or create indicators
            if symbol not in self.indicators:
                self.indicators[symbol] = {
                    'ema_fast': self.EMA(symbol, 10, Resolution.Daily),
                    'ema_slow': self.EMA(symbol, 30, Resolution.Daily),
                    'rsi': self.RSI(symbol, 14, Resolution.Daily)
                }
            
            if not self.indicators[symbol]['ema_fast'].IsReady:
                continue
            
            # Entry signal
            if (self.indicators[symbol]['ema_fast'].Current.Value > 
                self.indicators[symbol]['ema_slow'].Current.Value and
                self.indicators[symbol]['rsi'].Current.Value < 65):
                
                if not self.Portfolio[symbol].Invested:
                    self.SetHoldings(symbol, 0.05 * self.leverage)
    
    def TradeMeanReversion(self, data):
        """Mean reversion on oversold conditions"""
        for symbol in [self.spy, self.qqq, self.iwm, self.tlt]:
            if symbol not in data or symbol not in self.indicators:
                continue
            
            if not self.indicators[symbol]['bb'].IsReady:
                continue
            
            price = data[symbol].Price
            bb = self.indicators[symbol]['bb']
            rsi = self.indicators[symbol]['rsi'].Current.Value
            
            # Calculate Bollinger Band position
            bb_position = (price - bb.LowerBand.Current.Value) / (bb.UpperBand.Current.Value - bb.LowerBand.Current.Value)
            
            # Oversold - buy
            if bb_position < 0.1 and rsi < 30:
                if not self.Portfolio[symbol].Invested:
                    self.SetHoldings(symbol, 0.1 * self.leverage)
            
            # Overbought - sell
            elif bb_position > 0.9 and rsi > 70:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
    
    def TradeVolatility(self, data):
        """Trade volatility expansion/contraction"""
        if self.vxx not in data or self.svxy not in data:
            return
        
        # Get SPY volatility
        if self.spy in self.indicators and self.indicators[self.spy]['atr'].IsReady:
            spy_atr = self.indicators[self.spy]['atr'].Current.Value
            spy_price = data[self.spy].Price if self.spy in data else 200
            
            # Normalized ATR
            atr_pct = spy_atr / spy_price
            
            # High volatility - hedge
            if atr_pct > 0.02:  # 2% daily range
                if not self.Portfolio[self.vxx].Invested:
                    self.SetHoldings(self.vxx, 0.1)
                if self.Portfolio[self.svxy].Invested:
                    self.Liquidate(self.svxy)
            
            # Low volatility - short vol
            elif atr_pct < 0.01:  # 1% daily range
                if not self.Portfolio[self.svxy].Invested:
                    self.SetHoldings(self.svxy, 0.1)
                if self.Portfolio[self.vxx].Invested:
                    self.Liquidate(self.vxx)
    
    def TradeSectorRotation(self, data):
        """Rotate into strongest sectors"""
        sector_scores = {}
        
        for sector in self.sectors:
            if sector not in self.indicators or not self.indicators[sector]['macd'].IsReady:
                continue
            
            # Calculate sector strength
            macd = self.indicators[sector]['macd']
            rsi = self.indicators[sector]['rsi'].Current.Value
            adx = self.indicators[sector]['adx'].Current.Value
            
            # Trend strength score
            trend_score = 0
            if macd.Current.Value > macd.Signal.Current.Value:
                trend_score += 1
            if rsi > 50 and rsi < 70:
                trend_score += 1
            if adx > 25:
                trend_score += 1
            
            sector_scores[sector] = trend_score
        
        # Invest in top 2 sectors
        if sector_scores:
            sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Long top sectors
            for i, (sector, score) in enumerate(sorted_sectors[:2]):
                if score >= 2:  # At least 2 positive signals
                    if not self.Portfolio[sector].Invested:
                        self.SetHoldings(sector, 0.1)
            
            # Exit weak sectors
            for sector, score in sorted_sectors[3:]:
                if score <= 1 and self.Portfolio[sector].Invested:
                    self.Liquidate(sector)
    
    def DailyRebalance(self):
        """Daily portfolio rebalancing"""
        # Update stop losses
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].Invested:
                # Trailing stop loss
                if self.Portfolio[symbol].UnrealizedProfitPercent > 0.02:
                    cost_basis = self.Portfolio[symbol].AveragePrice
                    current_price = self.Securities[symbol].Price
                    stop_price = current_price * (1 - self.stop_loss)
                    
                    # Only update if stop price is higher than cost
                    if stop_price > cost_basis:
                        self.StopMarketOrder(symbol, -self.Portfolio[symbol].Quantity, stop_price)
    
    def MonthlyRebalance(self):
        """Monthly strategic rebalancing"""
        # Calculate portfolio allocation based on market regime
        spy_trend = "UP" if self.indicators[self.spy]['ema_fast'].Current.Value > self.indicators[self.spy]['ema_slow'].Current.Value else "DOWN"
        
        if spy_trend == "UP":
            # Risk-on allocation
            target_allocation = {
                self.spy: 0.2,
                self.qqq: 0.2,
                self.iwm: 0.1,
                self.eem: 0.1,
                self.gld: 0.05,
                self.tlt: 0.05
            }
        else:
            # Risk-off allocation
            target_allocation = {
                self.tlt: 0.3,
                self.ief: 0.2,
                self.gld: 0.15,
                self.shy: 0.1,
                self.spy: 0.1
            }
        
        # Rebalance to targets
        for symbol, target in target_allocation.items():
            if symbol in self.Securities:
                self.SetHoldings(symbol, target)
    
    def ManageRisk(self):
        """Portfolio-level risk management"""
        # Check total exposure
        total_holdings = sum(abs(self.Portfolio[s].HoldingsValue) for s in self.Portfolio.Keys)
        total_value = self.Portfolio.TotalPortfolioValue
        
        if total_value > 0:
            current_leverage = total_holdings / total_value
            
            # Reduce positions if over-leveraged
            if current_leverage > self.leverage * 1.2:
                for symbol in self.Portfolio.Keys:
                    if self.Portfolio[symbol].Invested:
                        self.SetHoldings(symbol, self.Portfolio[symbol].HoldingsValue / total_holdings * self.leverage)
        
        # Cut losses
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].UnrealizedProfitPercent < -self.stop_loss:
                self.Liquidate(symbol)
    
    def OnOrderEvent(self, orderEvent):
        """Track order execution"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order filled: {orderEvent.Symbol} {orderEvent.Quantity} @ {orderEvent.FillPrice}")
    
    def OnEndOfAlgorithm(self):
        """Final reporting"""
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        total_return = (self.Portfolio.TotalPortfolioValue / 100000 - 1) * 100
        self.Debug(f"Total Return: {total_return:.2f}%")