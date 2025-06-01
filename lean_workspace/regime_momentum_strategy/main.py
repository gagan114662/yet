# STRATEGY 2: CROSS-ASSET REGIME MOMENTUM MASTER  
# Target: 35%+ CAGR, Sharpe > 1.5 via regime detection and cross-asset momentum

from AlgorithmImports import *
import numpy as np
from datetime import timedelta
from scipy import stats

class RegimeMomentumStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Ultra-high leverage for regime momentum
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Cross-asset universe for regime momentum
        # Equities
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.eem = self.AddEquity("EEM", Resolution.Hour).Symbol  # Emerging markets
        self.efa = self.AddEquity("EFA", Resolution.Hour).Symbol  # Europe
        
        # Commodities  
        self.gld = self.AddEquity("GLD", Resolution.Hour).Symbol  # Gold
        self.uso = self.AddEquity("USO", Resolution.Hour).Symbol  # Oil
        self.dba = self.AddEquity("DBA", Resolution.Hour).Symbol  # Agriculture
        
        # Bonds
        self.tlt = self.AddEquity("TLT", Resolution.Hour).Symbol  # Long-term Treasury
        self.hyg = self.AddEquity("HYG", Resolution.Hour).Symbol  # High yield
        
        # Currencies  
        self.fxe = self.AddEquity("FXE", Resolution.Hour).Symbol  # Euro
        self.fxb = self.AddEquity("FXB", Resolution.Hour).Symbol  # British Pound
        
        # Crypto exposure via GBTC (when available)
        try:
            self.gbtc = self.AddEquity("GBTC", Resolution.Hour).Symbol
        except:
            self.gbtc = None
        
        # Volatility proxy using VXX (tracks VIX futures)
        # VXX already exists if added above, otherwise add it
        if not hasattr(self, 'vxx_added'):
            self.vxx = self.AddEquity("VXX", Resolution.Hour).Symbol
            self.vxx_added = True
        self.vix_proxy = self.vxx  # Use VXX as VIX proxy
        
        # SPY volatility calculation for regime detection
        self.spy_returns_window = RollingWindow[float](30)  # Longer window for regime detection
        self.calculated_vix = 20.0
        
        # Multi-timeframe momentum indicators
        self.momentum_fast = {}  # 4-hour momentum
        self.momentum_slow = {}  # 24-hour momentum 
        self.momentum_trend = {}  # 7-day momentum
        
        # Regime detection indicators
        self.volatility_regime = {}
        self.correlation_regime = {}
        self.risk_parity_weights = {}
        
        # Risk parity and volatility targeting
        self.target_volatility = 0.25  # 25% annual volatility target
        self.lookback_days = 30
        self.rebalance_frequency = 4  # Every 4 hours
        
        # Initialize momentum tracking for all assets
        self.assets = [self.spy, self.eem, self.efa, self.gld, self.uso, 
                      self.dba, self.tlt, self.hyg, self.fxe, self.fxb]
        if self.gbtc:
            self.assets.append(self.gbtc)
            
        for asset in self.assets:
            self.momentum_fast[asset] = self.MOMP(asset, 4, Resolution.Hour)
            self.momentum_slow[asset] = self.MOMP(asset, 24, Resolution.Hour) 
            self.momentum_trend[asset] = self.MOMP(asset, 168, Resolution.Hour)  # 7 days
            
        # Volatility and correlation tracking
        self.price_history = {}
        self.volatility_history = {}
        
        # Schedule aggressive rebalancing
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(TimeSpan.FromHours(4)), 
                        self.RegimeMomentumRebalance)
        
        # Regime states
        self.current_regime = "UNKNOWN"
        self.regime_confidence = 0.0
        
    def OnData(self, data):
        # Update price history for regime analysis
        self.UpdatePriceHistory(data)
        
        # Detect market regime every hour
        if self.Time.minute == 0:
            self.DetectMarketRegime()
            
        # Execute regime-specific momentum strategies
        if self.current_regime != "UNKNOWN":
            self.ExecuteRegimeMomentum()
    
    def UpdatePriceHistory(self, data):
        """Track price history for regime detection"""
        for asset in self.assets:
            if asset in data and data[asset] is not None:
                if asset not in self.price_history:
                    self.price_history[asset] = []
                    
                self.price_history[asset].append(data[asset].Close)
                
                # Keep only recent history
                if len(self.price_history[asset]) > 500:
                    self.price_history[asset] = self.price_history[asset][-500:]
    
    def DetectMarketRegime(self):
        """Advanced regime detection using multiple signals"""
        # 1. Volatility Regime using VXX proxy
        vix_level = self.CalculateVixFromVXX()
        vol_regime = self.ClassifyVolatilityRegime(vix_level)
        
        # 2. Correlation Regime  
        corr_regime = self.ClassifyCorrelationRegime()
        
        # 3. Momentum Regime
        momentum_regime = self.ClassifyMomentumRegime()
        
        # 4. Risk-on/Risk-off Regime
        risk_regime = self.ClassifyRiskRegime()
        
        # Combine regime signals
        regime_scores = {
            "BULL_MOMENTUM": 0,
            "BEAR_MOMENTUM": 0, 
            "LOW_VOL_GRIND": 0,
            "HIGH_VOL_CRASH": 0,
            "ROTATION": 0,
            "CRISIS": 0
        }
        
        # Score regimes based on multiple factors
        if vol_regime == "LOW" and momentum_regime == "STRONG_UP":
            regime_scores["BULL_MOMENTUM"] += 2
            regime_scores["LOW_VOL_GRIND"] += 1
            
        elif vol_regime == "HIGH" and momentum_regime == "STRONG_DOWN":
            regime_scores["BEAR_MOMENTUM"] += 2
            regime_scores["HIGH_VOL_CRASH"] += 1
            
        elif corr_regime == "HIGH":
            regime_scores["CRISIS"] += 2
            
        elif corr_regime == "LOW":
            regime_scores["ROTATION"] += 2
            
        # Select dominant regime
        self.current_regime = max(regime_scores, key=regime_scores.get)
        self.regime_confidence = max(regime_scores.values()) / sum(regime_scores.values()) if sum(regime_scores.values()) > 0 else 0
        
    def ClassifyVolatilityRegime(self, vix_level):
        """Classify current volatility regime"""
        if vix_level < 15:
            return "VERY_LOW"
        elif vix_level < 20:
            return "LOW"
        elif vix_level < 30:
            return "MEDIUM"
        elif vix_level < 40:
            return "HIGH"
        else:
            return "EXTREME"
    
    def ClassifyCorrelationRegime(self):
        """Estimate cross-asset correlation regime"""
        if len(self.price_history) < 3:
            return "UNKNOWN"
            
        # Calculate rolling correlations between major asset classes
        correlations = []
        
        try:
            # SPY vs EEM correlation
            if (self.spy in self.price_history and self.eem in self.price_history and 
                len(self.price_history[self.spy]) > 20 and len(self.price_history[self.eem]) > 20):
                
                spy_returns = np.diff(self.price_history[self.spy][-20:])
                eem_returns = np.diff(self.price_history[self.eem][-20:])
                
                if len(spy_returns) > 5 and len(eem_returns) > 5:
                    corr, _ = stats.pearsonr(spy_returns, eem_returns)
                    correlations.append(abs(corr))
                    
            # Add more correlations as needed
            avg_correlation = np.mean(correlations) if correlations else 0.5
            
            if avg_correlation > 0.8:
                return "HIGH"
            elif avg_correlation < 0.3:
                return "LOW"
            else:
                return "MEDIUM"
                
        except:
            return "UNKNOWN"
    
    def ClassifyMomentumRegime(self):
        """Classify momentum regime across assets"""
        if not self.momentum_fast[self.spy].IsReady:
            return "UNKNOWN"
            
        momentum_scores = []
        for asset in self.assets:
            if self.momentum_fast[asset].IsReady:
                fast_mom = self.momentum_fast[asset].Current.Value
                momentum_scores.append(fast_mom)
                
        if not momentum_scores:
            return "UNKNOWN"
            
        avg_momentum = np.mean(momentum_scores)
        momentum_strength = np.std(momentum_scores)
        
        if avg_momentum > 0.02 and momentum_strength > 0.01:
            return "STRONG_UP"
        elif avg_momentum < -0.02 and momentum_strength > 0.01:
            return "STRONG_DOWN"
        elif momentum_strength < 0.005:
            return "STAGNANT"
        else:
            return "MIXED"
    
    def ClassifyRiskRegime(self):
        """Determine risk-on vs risk-off environment"""
        # Risk-on: Stocks up, bonds down, commodities up, dollar down
        # Risk-off: Stocks down, bonds up, commodities down, dollar up
        
        if not all([self.momentum_fast[asset].IsReady for asset in [self.spy, self.tlt, self.gld]]):
            return "UNKNOWN"
            
        equity_momentum = self.momentum_fast[self.spy].Current.Value
        bond_momentum = self.momentum_fast[self.tlt].Current.Value
        gold_momentum = self.momentum_fast[self.gld].Current.Value
        
        risk_score = equity_momentum - bond_momentum + gold_momentum
        
        if risk_score > 0.03:
            return "RISK_ON"
        elif risk_score < -0.03:
            return "RISK_OFF"
        else:
            return "NEUTRAL"
    
    def ExecuteRegimeMomentum(self):
        """Execute regime-specific momentum strategies"""
        if self.regime_confidence < 0.3:
            return  # Low confidence, don't trade
            
        # Calculate volatility-adjusted position sizes
        target_positions = self.CalculateTargetPositions()
        
        # Execute trades with high leverage
        for asset, target_weight in target_positions.items():
            if abs(target_weight) > 0.01:  # Only trade significant positions
                self.SetHoldings(asset, target_weight)
    
    def CalculateTargetPositions(self):
        """Calculate target positions based on regime and momentum"""
        positions = {}
        
        if self.current_regime == "BULL_MOMENTUM":
            # Aggressive long equities, short bonds
            positions[self.spy] = 2.0
            positions[self.eem] = 1.5  
            positions[self.efa] = 1.0
            positions[self.tlt] = -1.0
            if self.gbtc:
                positions[self.gbtc] = 0.5
                
        elif self.current_regime == "BEAR_MOMENTUM":
            # Short equities, long bonds and gold
            positions[self.spy] = -1.5
            positions[self.eem] = -2.0
            positions[self.tlt] = 2.0
            positions[self.gld] = 1.5
            
        elif self.current_regime == "LOW_VOL_GRIND":
            # Risk parity with momentum tilt
            positions = self.CalculateRiskParityPositions()
            
        elif self.current_regime == "CRISIS":
            # Crisis alpha - long volatility, short risk assets
            positions[self.tlt] = 3.0
            positions[self.gld] = 2.0
            positions[self.spy] = -1.0
            positions[self.eem] = -1.5
            
        elif self.current_regime == "ROTATION":
            # Cross-asset momentum
            positions = self.CalculateCrossAssetMomentum()
            
        # Apply volatility targeting
        positions = self.ApplyVolatilityTargeting(positions)
        
        return positions
    
    def CalculateRiskParityPositions(self):
        """Calculate risk parity positions with momentum overlay"""
        positions = {}
        
        # Equal risk contribution with momentum tilt
        base_weight = 1.0 / len(self.assets)
        
        for asset in self.assets:
            if self.momentum_trend[asset].IsReady:
                momentum_score = self.momentum_trend[asset].Current.Value
                momentum_multiplier = 1.0 + np.tanh(momentum_score * 10)  # Scale momentum
                
                positions[asset] = base_weight * momentum_multiplier
                
        return positions
    
    def CalculateCrossAssetMomentum(self):
        """Pure cross-asset momentum strategy"""
        positions = {}
        momentum_scores = []
        
        # Collect momentum scores
        for asset in self.assets:
            if self.momentum_slow[asset].IsReady:
                score = self.momentum_slow[asset].Current.Value
                momentum_scores.append((asset, score))
                
        # Sort by momentum and go long/short extremes
        momentum_scores.sort(key=lambda x: x[1], reverse=True)
        
        n_assets = len(momentum_scores)
        if n_assets >= 4:
            # Long top quartile, short bottom quartile
            top_quartile = momentum_scores[:n_assets//4]
            bottom_quartile = momentum_scores[-n_assets//4:]
            
            for asset, score in top_quartile:
                positions[asset] = 2.0 / len(top_quartile)
                
            for asset, score in bottom_quartile:
                positions[asset] = -1.5 / len(bottom_quartile)
                
        return positions
    
    def ApplyVolatilityTargeting(self, positions):
        """Scale positions to target volatility"""
        # Estimate portfolio volatility (simplified)
        total_risk = sum([abs(weight) for weight in positions.values()])
        
        if total_risk > 0:
            vol_scalar = self.target_volatility / (total_risk * 0.15)  # Assume 15% asset vol
            vol_scalar = min(vol_scalar, 3.0)  # Cap at 3x leverage
            
            return {asset: weight * vol_scalar for asset, weight in positions.items()}
        
        return positions
    
    def RegimeMomentumRebalance(self):
        """4-hour rebalancing routine"""
        # Update regime detection
        self.DetectMarketRegime()
        
        # Risk management
        total_exposure = sum([abs(x.HoldingsValue) for x in self.Portfolio.Values])
        if total_exposure > self.Portfolio.TotalPortfolioValue * 5.0:
            # Over-leveraged, reduce positions
            for holding in self.Portfolio.Values:
                if holding.Invested:
                    current_weight = holding.HoldingsValue / self.Portfolio.TotalPortfolioValue
                    self.SetHoldings(holding.Symbol, current_weight * 0.8)
                    
        # Log regime information
        self.Debug(f"Current Regime: {self.current_regime}, Confidence: {self.regime_confidence:.2f}")
    
    def OnEndOfDay(self, symbol):
        """Daily performance and risk analysis"""
        if self.Portfolio.TotalPortfolioValue > 0:
            daily_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            
            # Adjust target volatility based on performance
            if abs(daily_return) > 0.05:  # High volatility day
                self.target_volatility = max(0.20, self.target_volatility * 0.95)
            else:  # Low volatility day
                self.target_volatility = min(0.30, self.target_volatility * 1.02)
    
    def CalculateVixFromVXX(self):
        """Calculate VIX proxy from VXX and SPY volatility for regime detection"""
        # Method 1: Use VXX as VIX proxy
        if hasattr(self, 'vxx') and self.vxx in self.Securities:
            vxx_price = self.Securities[self.vxx].Price
            # Convert VXX to VIX-like scale for regime classification
            vix_from_vxx = vxx_price * 2.5
            
            # Method 2: Calculate from SPY volatility
            vix_from_spy = self.CalculateImpliedVolFromSPY()
            
            # For regime detection, combine both with preference for stability
            if vix_from_spy > 0:
                # Weight VXX more for regime stability
                combined_vix = (vix_from_vxx * 0.8) + (vix_from_spy * 0.2)
            else:
                combined_vix = vix_from_vxx
                
            return max(10, min(80, combined_vix))
        else:
            return self.CalculateImpliedVolFromSPY()
    
    def CalculateImpliedVolFromSPY(self):
        """Calculate implied VIX from SPY returns for regime analysis"""
        if self.spy in self.Securities:
            spy_price = self.Securities[self.spy].Price
            
            # Update returns window
            if hasattr(self, 'previous_spy_price') and self.previous_spy_price > 0:
                hourly_return = (spy_price - self.previous_spy_price) / self.previous_spy_price
                self.spy_returns_window.Add(hourly_return)
                
            self.previous_spy_price = spy_price
            
            # Calculate realized volatility for regime detection
            if self.spy_returns_window.IsReady:
                returns = [self.spy_returns_window[i] for i in range(self.spy_returns_window.Count)]
                if len(returns) > 10:
                    import numpy as np
                    # Convert hourly to annual volatility
                    hourly_vol = np.std(returns)
                    annual_vol = hourly_vol * np.sqrt(365 * 24)  # 365 days * 24 hours
                    
                    # Convert to VIX-like scale
                    implied_vix = annual_vol * 100 * 1.2  # Moderate risk premium for regime detection
                    
                    self.calculated_vix = max(10, min(60, implied_vix))
                    return self.calculated_vix
                    
        return self.calculated_vix