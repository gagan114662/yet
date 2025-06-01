# STRATEGY 1: GAMMA FLOW & POSITIONING MASTER
# Target: 40%+ CAGR, Sharpe > 1.2 via options gamma scalping and flow analysis

from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class GammaFlowStrategy(QCAlgorithm):
    
    def Initialize(self):
        # Aggressive cloud-based setup
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # High leverage setup - up to 5x exposure
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Primary gamma scalping universe
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Minute).Symbol  
        self.iwm = self.AddEquity("IWM", Resolution.Minute).Symbol
        
        # Options for gamma exposure analysis
        self.AddOption("SPY", Resolution.Minute)
        self.AddOption("QQQ", Resolution.Minute)
        
        # VIX proxy using VXX for volatility regime analysis
        # VXX tracks VIX futures and provides similar volatility signals
        self.vix_proxy = self.AddEquity("VXX", Resolution.Minute).Symbol  # Already added above
        
        # SPY volatility calculation for backup VIX estimation
        self.spy_returns_window = RollingWindow[float](20)
        self.calculated_vix = 20.0
        
        # Advanced indicators for gamma flow
        self.spy_rsi = self.RSI("SPY", 14, Resolution.Minute)
        # Use VXX moving average instead of VIX
        self.vxx_sma = self.SMA("VXX", 20, Resolution.Daily)
        self.gamma_exposure = {}
        self.dealer_positioning = {}
        
        # Microstructure tracking
        self.order_flow = {}
        self.tick_pressure = {}
        
        # Schedule aggressive rebalancing
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.Every(TimeSpan.FromMinutes(15)), 
                        self.GammaScalpRebalance)
        
        # Risk management with high leverage
        self.max_leverage = 5.0
        self.position_heat = 0.15  # 15% risk per position
        
        # Gamma flow state tracking
        self.gamma_flip_level = None
        self.dealer_gamma = {}
        
    def OnData(self, data):
        # Real-time gamma flow analysis
        self.UpdateGammaExposure(data)
        self.AnalyzeOrderFlow(data)
        
        # Only trade during market hours for liquidity
        if not self.IsMarketOpen("SPY"):
            return
            
        # Check for gamma squeeze opportunities
        if self.DetectGammaSqueezeSetup():
            self.ExecuteGammaScalp()
            
        # Volatility surface arbitrage
        if self.DetectVolSurfaceArb():
            self.ExecuteVolArbitrage()
    
    def UpdateGammaExposure(self, data):
        """Calculate real-time dealer gamma exposure"""
        for symbol in [self.spy, self.qqq]:
            if symbol in data and data[symbol] is not None:
                price = data[symbol].Close
                
                # Estimate dealer gamma (simplified model)
                # In practice, would use options flow data
                vix_level = self.CalculateVixFromVXX()
                
                # Higher VIX = more negative dealer gamma
                estimated_gamma = -1.0 * (vix_level - 15) / 100
                self.dealer_gamma[symbol] = estimated_gamma
                
                # Track gamma flip levels (where dealers become long gamma)
                if abs(estimated_gamma) < 0.05:
                    self.gamma_flip_level = price
    
    def DetectGammaSqueezeSetup(self):
        """Identify gamma squeeze conditions"""
        if not self.spy_rsi.IsReady:
            return False
            
        spy_price = self.Securities[self.spy].Price
        vix_level = self.CalculateVixFromVXX()
        
        # Gamma squeeze conditions:
        # 1. Low VIX (dealers short gamma)
        # 2. Price near gamma flip level
        # 3. Strong momentum
        
        low_vix = vix_level < 18
        near_flip = (self.gamma_flip_level and 
                    abs(spy_price - self.gamma_flip_level) / spy_price < 0.02)
        strong_momentum = self.spy_rsi.Current.Value > 70 or self.spy_rsi.Current.Value < 30
        
        return low_vix and near_flip and strong_momentum
    
    def ExecuteGammaScalp(self):
        """Execute gamma scalping strategy with high leverage"""
        spy_price = self.Securities[self.spy].Price
        
        # Determine direction based on gamma exposure
        dealer_gamma_spy = self.dealer_gamma.get(self.spy, 0)
        
        if dealer_gamma_spy < -0.1:  # Dealers short gamma
            # Momentum strategy - trend will continue
            if self.spy_rsi.Current.Value > 70:
                target_weight = 2.0  # 2x long
            elif self.spy_rsi.Current.Value < 30:
                target_weight = -1.5  # 1.5x short
            else:
                target_weight = 0
        else:  # Dealers long gamma
            # Mean reversion strategy
            if self.spy_rsi.Current.Value > 65:
                target_weight = -1.0  # Short into strength
            elif self.spy_rsi.Current.Value < 35:
                target_weight = 1.5   # Long into weakness
            else:
                target_weight = 0
                
        # Execute with position sizing
        if target_weight != 0:
            self.SetHoldings(self.spy, target_weight)
            
            # Hedge with QQQ for diversification
            qqq_weight = target_weight * 0.3
            self.SetHoldings(self.qqq, qqq_weight)
    
    def DetectVolSurfaceArb(self):
        """Detect volatility surface arbitrage opportunities"""
        # Simplified - would use real options chain analysis
        vix_level = self.CalculateVixFromVXX()
        
        # Look for VIX backwardation/contango extremes using VXX
        if hasattr(self, 'vxx_sma') and self.vxx_sma.IsReady:
            vxx_price = self.Securities[self.vix_proxy].Price if self.vix_proxy in self.Securities else 20
            vxx_vs_ma = vxx_price / self.vxx_sma.Current.Value
            # Convert VXX ratio to VIX-like interpretation
            vix_vs_ma = vxx_vs_ma
            
            # Extreme vol conditions
            return vix_vs_ma > 1.3 or vix_vs_ma < 0.7
        
        return False
    
    def ExecuteVolArbitrage(self):
        """Execute volatility surface arbitrage"""
        vxx_price = self.Securities[self.vix_proxy].Price if self.vix_proxy in self.Securities else 20
        
        if self.vxx_sma.IsReady and vxx_price > self.vxx_sma.Current.Value * 1.3:
            # VIX spike - short vol, long realized vol
            self.SetHoldings(self.spy, 1.0)  # Long for realized vol
            # In practice, would short VIX futures/options
            
        elif self.vxx_sma.IsReady and vxx_price < self.vxx_sma.Current.Value * 0.7:
            # VIX too low - long vol protection
            self.SetHoldings(self.spy, 0.5)  # Reduced exposure
            # In practice, would buy VIX calls/puts
    
    def AnalyzeOrderFlow(self, data):
        """Analyze order flow for microstructure signals"""
        # Track bid-ask pressure and volume imbalance
        for symbol in [self.spy, self.qqq]:
            if symbol in data and hasattr(data[symbol], 'Volume'):
                volume = data[symbol].Volume
                price_change = data[symbol].Close - data[symbol].Open
                
                # Estimate order flow pressure
                if volume > 0:
                    flow_pressure = price_change * volume
                    self.order_flow[symbol] = flow_pressure
    
    def GammaScalpRebalance(self):
        """Aggressive 15-minute rebalancing for gamma opportunities"""
        if not self.spy_rsi.IsReady:
            return
            
        # Risk management check
        total_leverage = sum([abs(x.HoldingsValue) for x in self.Portfolio.Values]) / self.Portfolio.TotalPortfolioValue
        
        if total_leverage > self.max_leverage:
            # Reduce positions if over-leveraged
            for holding in self.Portfolio.Values:
                if holding.Invested:
                    current_weight = holding.HoldingsValue / self.Portfolio.TotalPortfolioValue
                    new_weight = current_weight * 0.8  # Reduce by 20%
                    self.SetHoldings(holding.Symbol, new_weight)
                    
        # Look for new gamma opportunities
        self.CheckForNewGammaSetups()
    
    def CheckForNewGammaSetups(self):
        """Scan for emerging gamma flow opportunities"""
        spy_price = self.Securities[self.spy].Price
        
        # Calculate intraday momentum
        if len(self.spy_rsi.Window) > 0:
            momentum_score = self.spy_rsi.Current.Value - 50
            
            # High-frequency gamma scalping
            if abs(momentum_score) > 15:  # Strong momentum
                scalp_weight = np.sign(momentum_score) * 1.5
                self.SetHoldings(self.spy, scalp_weight)
                
    def OnEndOfDay(self, symbol):
        """Daily risk management and position sizing"""
        # Calculate daily performance
        if self.Portfolio.TotalPortfolioValue != 0:
            daily_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            
            # Dynamic position sizing based on recent performance
            if daily_return > 0.02:  # Good day, increase exposure
                self.position_heat = min(0.20, self.position_heat * 1.1)
            elif daily_return < -0.02:  # Bad day, reduce exposure
                self.position_heat = max(0.10, self.position_heat * 0.9)
    
    def CalculateVixFromVXX(self):
        """Calculate VIX proxy from VXX price and SPY volatility"""
        # Method 1: Use VXX as VIX proxy
        if self.vix_proxy in self.Securities:
            vxx_price = self.Securities[self.vix_proxy].Price
            # Convert VXX to VIX-like scale (VXX typically 10-30% of VIX level)
            vix_from_vxx = vxx_price * 2.5  # Rough conversion factor
            
            # Method 2: Calculate from SPY volatility
            vix_from_spy = self.CalculateImpliedVolFromSPY()
            
            # Combine methods
            if vix_from_spy > 0:
                combined_vix = (vix_from_vxx * 0.7) + (vix_from_spy * 0.3)
            else:
                combined_vix = vix_from_vxx
                
            return max(10, min(80, combined_vix))
        else:
            return self.CalculateImpliedVolFromSPY()
    
    def CalculateImpliedVolFromSPY(self):
        """Calculate implied VIX from SPY returns"""
        if self.spy in self.Securities:
            spy_price = self.Securities[self.spy].Price
            
            # Update returns window
            if hasattr(self, 'previous_spy_price') and self.previous_spy_price > 0:
                daily_return = (spy_price - self.previous_spy_price) / self.previous_spy_price
                self.spy_returns_window.Add(daily_return)
                
            self.previous_spy_price = spy_price
            
            # Calculate realized volatility
            if self.spy_returns_window.IsReady:
                returns = [self.spy_returns_window[i] for i in range(self.spy_returns_window.Count)]
                if len(returns) > 5:
                    import numpy as np
                    volatility = np.std(returns) * np.sqrt(252)
                    implied_vix = volatility * 100 * 1.3  # Risk premium
                    self.calculated_vix = max(10, min(60, implied_vix))
                    return self.calculated_vix
                    
        return self.calculated_vix