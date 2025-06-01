# MASTER STRATEGY ROTATOR: Dynamic Multi-Strategy Allocation
# Target: 50%+ CAGR, Sharpe > 2.0 via optimal strategy rotation and risk management

from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class MasterStrategyRotator(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Ultra-aggressive multi-strategy setup
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Core market indicators for regime detection
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Minute).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Minute).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Minute).Symbol
        self.vxx = self.AddEquity("VXX", Resolution.Minute).Symbol
        
        # VIX proxy using VXX for volatility regime
        # VXX is a VIX ETF that tracks short-term VIX futures
        # We'll use VXX price movements to estimate VIX levels
        self.vix_proxy = self.vxx  # Use VXX as VIX proxy
        
        # SPY volatility tracking for backup VIX calculation
        self.spy_returns_window = RollingWindow[float](20)  # 20-day rolling window
        self.calculated_vix = 20.0  # Default VIX level
        
        # Strategy performance tracking
        self.strategy_performance = {
            "GAMMA_FLOW": {"returns": [], "sharpe": 0.0, "max_dd": 0.0, "active": True},
            "REGIME_MOMENTUM": {"returns": [], "sharpe": 0.0, "max_dd": 0.0, "active": True},
            "CRISIS_ALPHA": {"returns": [], "sharpe": 0.0, "max_dd": 0.0, "active": True},
            "EARNINGS_MOMENTUM": {"returns": [], "sharpe": 0.0, "max_dd": 0.0, "active": True},
            "MICROSTRUCTURE": {"returns": [], "sharpe": 0.0, "max_dd": 0.0, "active": True}
        }
        
        # Current strategy allocations (sum to 100%)
        self.strategy_allocations = {
            "GAMMA_FLOW": 0.20,
            "REGIME_MOMENTUM": 0.20,
            "CRISIS_ALPHA": 0.20,
            "EARNINGS_MOMENTUM": 0.20,
            "MICROSTRUCTURE": 0.20
        }
        
        # Market regime detection
        self.current_market_regime = "UNKNOWN"
        self.volatility_regime = "NORMAL"
        self.trend_regime = "SIDEWAYS"
        self.risk_regime = "NEUTRAL"
        
        # Regime indicators
        self.regime_indicators = {
            "vix_level": 20.0,
            "spy_momentum": 0.0,
            "correlation": 0.5,
            "volatility": 0.15,
            "volume": 1.0
        }
        
        # Multi-timeframe indicators
        self.spy_rsi_fast = self.RSI("SPY", 14, Resolution.Minute)
        self.spy_rsi_slow = self.RSI("SPY", 60, Resolution.Minute)
        self.spy_momentum = self.MOMP("SPY", 1440, Resolution.Minute)  # Daily momentum
        
        # Strategy optimization parameters
        self.lookback_days = 30
        self.rebalance_frequency = 6  # Hours
        self.min_allocation = 0.05    # 5% minimum per strategy
        self.max_allocation = 0.50    # 50% maximum per strategy
        
        # Performance targets for strategy selection
        self.target_sharpe = 1.5
        self.target_return = 0.25    # 25% annual
        self.max_drawdown_tolerance = 0.15  # 15%
        
        # Risk management
        self.total_leverage_limit = 8.0
        self.strategy_correlation_threshold = 0.7
        
        # Schedule strategy rotation
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.Every(TimeSpan.FromHours(6)), 
                        self.RotateStrategies)
        
        # Track strategy signals
        self.strategy_signals = {}
        self.last_rotation_time = self.Time
        
        # Portfolio tracking for each strategy (simulated)
        self.strategy_portfolios = {
            "GAMMA_FLOW": 20000,      # Starting with 20% each
            "REGIME_MOMENTUM": 20000,
            "CRISIS_ALPHA": 20000,
            "EARNINGS_MOMENTUM": 20000,
            "MICROSTRUCTURE": 20000
        }
        
    def OnData(self, data):
        # Update market regime indicators
        self.UpdateMarketRegime(data)
        
        # Generate strategy signals
        self.GenerateStrategySignals(data)
        
        # Execute current strategy allocation
        self.ExecuteStrategyAllocation(data)
        
        # Monitor strategy performance
        self.MonitorStrategyPerformance()
    
    def UpdateMarketRegime(self, data):
        """Update comprehensive market regime analysis"""
        
        # 1. Volatility Regime using VXX proxy and SPY volatility
        vix_level = self.CalculateVixFromVXX()
        self.regime_indicators["vix_level"] = vix_level
        
        if vix_level < 15:
            self.volatility_regime = "VERY_LOW"
        elif vix_level < 20:
            self.volatility_regime = "LOW"
        elif vix_level < 30:
            self.volatility_regime = "NORMAL"
        elif vix_level < 40:
            self.volatility_regime = "HIGH"
        else:
            self.volatility_regime = "EXTREME"
        
        # 2. Trend Regime
        if self.spy_momentum.IsReady:
            momentum = self.spy_momentum.Current.Value
            self.regime_indicators["spy_momentum"] = momentum
            
            if momentum > 0.05:
                self.trend_regime = "STRONG_UP"
            elif momentum > 0.02:
                self.trend_regime = "UP"
            elif momentum < -0.05:
                self.trend_regime = "STRONG_DOWN"
            elif momentum < -0.02:
                self.trend_regime = "DOWN"
            else:
                self.trend_regime = "SIDEWAYS"
        
        # 3. Risk Regime (using SPY vs TLT)
        if (self.spy in data and self.tlt in data and 
            data[self.spy] is not None and data[self.tlt] is not None):
            
            spy_price = data[self.spy].Close
            tlt_price = data[self.tlt].Close
            
            # Calculate relative performance
            if hasattr(self, 'spy_prev') and hasattr(self, 'tlt_prev'):
                spy_return = (spy_price - self.spy_prev) / self.spy_prev
                tlt_return = (tlt_price - self.tlt_prev) / self.tlt_prev
                
                risk_score = spy_return - tlt_return
                
                if risk_score > 0.01:
                    self.risk_regime = "RISK_ON"
                elif risk_score < -0.01:
                    self.risk_regime = "RISK_OFF"
                else:
                    self.risk_regime = "NEUTRAL"
            
            self.spy_prev = spy_price
            self.tlt_prev = tlt_price
        
        # 4. Combined Market Regime
        self.current_market_regime = self.DetermineOverallRegime()
    
    def DetermineOverallRegime(self):
        """Determine overall market regime from individual components"""
        regime_score = {
            "BULL_MOMENTUM": 0,
            "BEAR_MOMENTUM": 0,
            "LOW_VOL_GRIND": 0,
            "HIGH_VOL_CRASH": 0,
            "SIDEWAYS_GRIND": 0,
            "CRISIS": 0,
            "ROTATION": 0,
            "EARNINGS_SEASON": 0
        }
        
        # Score regimes based on current conditions
        if self.volatility_regime in ["VERY_LOW", "LOW"] and self.trend_regime in ["UP", "STRONG_UP"]:
            regime_score["BULL_MOMENTUM"] += 3
            regime_score["LOW_VOL_GRIND"] += 2
            
        elif self.volatility_regime in ["HIGH", "EXTREME"] and self.trend_regime in ["DOWN", "STRONG_DOWN"]:
            regime_score["BEAR_MOMENTUM"] += 3
            regime_score["HIGH_VOL_CRASH"] += 2
            
        elif self.volatility_regime == "EXTREME":
            regime_score["CRISIS"] += 4
            
        elif self.trend_regime == "SIDEWAYS":
            regime_score["SIDEWAYS_GRIND"] += 2
            regime_score["ROTATION"] += 1
            
        # Check for earnings season (simplified - based on month)
        if self.Time.month in [1, 4, 7, 10]:  # Earnings months
            regime_score["EARNINGS_SEASON"] += 2
            
        # Return dominant regime
        return max(regime_score, key=regime_score.get)
    
    def CalculateVixFromVXX(self):
        """Calculate VIX proxy from VXX price and SPY volatility"""
        # Method 1: Use VXX as VIX proxy (VXX typically trades 10-30% of VIX level)
        if self.vxx in self.Securities:
            vxx_price = self.Securities[self.vxx].Price
            # Approximate VIX from VXX (rough conversion based on typical relationship)
            vix_from_vxx = vxx_price * 2.5  # Rough multiplier, calibrate as needed
            
            # Method 2: Calculate implied volatility from SPY returns
            vix_from_spy = self.CalculateImpliedVolFromSPY()
            
            # Combine both methods with weighting
            if vix_from_spy > 0:
                combined_vix = (vix_from_vxx * 0.7) + (vix_from_spy * 0.3)
            else:
                combined_vix = vix_from_vxx
                
            # Sanity check and bounds
            combined_vix = max(10, min(80, combined_vix))
            return combined_vix
        else:
            return self.CalculateImpliedVolFromSPY()
    
    def CalculateImpliedVolFromSPY(self):
        """Calculate implied VIX from SPY price movements"""
        if self.spy in self.Securities:
            spy_price = self.Securities[self.spy].Price
            
            # Update SPY returns window
            if hasattr(self, 'previous_spy_price') and self.previous_spy_price > 0:
                daily_return = (spy_price - self.previous_spy_price) / self.previous_spy_price
                self.spy_returns_window.Add(daily_return)
                
            self.previous_spy_price = spy_price
            
            # Calculate realized volatility from SPY returns
            if self.spy_returns_window.IsReady:
                returns = [self.spy_returns_window[i] for i in range(self.spy_returns_window.Count)]
                if len(returns) > 5:
                    import numpy as np
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
                    # Convert to VIX-like scale (multiply by 100)
                    implied_vix = volatility * 100
                    
                    # Apply volatility risk premium (VIX typically higher than realized vol)
                    implied_vix *= 1.3  # Risk premium adjustment
                    
                    self.calculated_vix = max(10, min(60, implied_vix))
                    return self.calculated_vix
                    
        return self.calculated_vix  # Return last calculated value or default
    
    def GenerateStrategySignals(self, data):
        """Generate signals for each strategy based on current regime"""
        
        # GAMMA FLOW Strategy - best in low vol, trending markets
        gamma_signal = 0.5  # Baseline
        if self.volatility_regime in ["LOW", "VERY_LOW"] and self.trend_regime != "SIDEWAYS":
            gamma_signal = 0.8
        elif self.volatility_regime in ["HIGH", "EXTREME"]:
            gamma_signal = 0.3
        self.strategy_signals["GAMMA_FLOW"] = gamma_signal
        
        # REGIME MOMENTUM Strategy - best in clear trending environments
        momentum_signal = 0.5
        if self.trend_regime in ["STRONG_UP", "STRONG_DOWN"]:
            momentum_signal = 0.9
        elif self.trend_regime == "SIDEWAYS":
            momentum_signal = 0.2
        self.strategy_signals["REGIME_MOMENTUM"] = momentum_signal
        
        # CRISIS ALPHA Strategy - best during high vol and crisis
        crisis_signal = 0.2  # Low baseline
        if self.volatility_regime in ["HIGH", "EXTREME"] or self.current_market_regime == "CRISIS":
            crisis_signal = 0.9
        elif self.volatility_regime == "VERY_LOW":
            crisis_signal = 0.1
        self.strategy_signals["CRISIS_ALPHA"] = crisis_signal
        
        # EARNINGS MOMENTUM Strategy - best during earnings season
        earnings_signal = 0.3
        if self.current_market_regime == "EARNINGS_SEASON":
            earnings_signal = 0.8
        elif self.volatility_regime in ["HIGH", "EXTREME"]:
            earnings_signal = 0.6  # Volatility helps earnings plays
        self.strategy_signals["EARNINGS_MOMENTUM"] = earnings_signal
        
        # MICROSTRUCTURE Strategy - best in normal to high vol, good for all regimes
        micro_signal = 0.6  # Higher baseline - always some opportunities
        if self.volatility_regime in ["NORMAL", "HIGH"]:
            micro_signal = 0.8
        elif self.volatility_regime == "VERY_LOW":
            micro_signal = 0.4
        self.strategy_signals["MICROSTRUCTURE"] = micro_signal
    
    def RotateStrategies(self):
        """Main strategy rotation logic"""
        
        # Calculate strategy performance metrics
        self.CalculateStrategyMetrics()
        
        # Determine optimal allocation based on regime and performance
        new_allocations = self.OptimizeStrategyAllocation()
        
        # Apply allocation changes gradually to avoid whipsaws
        self.UpdateAllocations(new_allocations)
        
        # Log rotation decision
        self.LogRotationDecision()
        
        # Risk management check
        self.ApplyRiskManagement()
    
    def CalculateStrategyMetrics(self):
        """Calculate performance metrics for each strategy"""
        
        for strategy in self.strategy_performance.keys():
            perf = self.strategy_performance[strategy]
            
            if len(perf["returns"]) > 5:
                returns = np.array(perf["returns"][-30:])  # Last 30 observations
                
                # Sharpe ratio
                if np.std(returns) > 0:
                    perf["sharpe"] = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    perf["sharpe"] = 0.0
                
                # Maximum drawdown
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                perf["max_dd"] = abs(np.min(drawdown))
            
            # Simulate strategy returns based on regime performance
            strategy_return = self.SimulateStrategyReturn(strategy)
            perf["returns"].append(strategy_return)
            
            # Keep rolling window
            if len(perf["returns"]) > 100:
                perf["returns"] = perf["returns"][-100:]
    
    def SimulateStrategyReturn(self, strategy):
        """Simulate strategy return based on market conditions and strategy signal"""
        
        # Base return from market
        if self.spy in self.Securities:
            market_return = 0.001  # Default small positive
            # Would calculate actual return in real implementation
        else:
            market_return = 0.001
        
        # Strategy-specific performance multipliers based on regime
        multipliers = {
            "GAMMA_FLOW": {
                "BULL_MOMENTUM": 1.5, "LOW_VOL_GRIND": 2.0, "SIDEWAYS_GRIND": 0.8,
                "BEAR_MOMENTUM": -0.5, "HIGH_VOL_CRASH": 0.2, "CRISIS": 0.3
            },
            "REGIME_MOMENTUM": {
                "BULL_MOMENTUM": 2.5, "BEAR_MOMENTUM": 1.8, "LOW_VOL_GRIND": 1.2,
                "SIDEWAYS_GRIND": 0.3, "HIGH_VOL_CRASH": 0.5, "CRISIS": 0.8
            },
            "CRISIS_ALPHA": {
                "CRISIS": 3.0, "HIGH_VOL_CRASH": 2.5, "BEAR_MOMENTUM": 2.0,
                "BULL_MOMENTUM": -0.5, "LOW_VOL_GRIND": -0.2, "SIDEWAYS_GRIND": 0.1
            },
            "EARNINGS_MOMENTUM": {
                "EARNINGS_SEASON": 2.5, "HIGH_VOL_CRASH": 1.5, "BULL_MOMENTUM": 1.3,
                "BEAR_MOMENTUM": 1.1, "LOW_VOL_GRIND": 0.8, "SIDEWAYS_GRIND": 0.9
            },
            "MICROSTRUCTURE": {
                "HIGH_VOL_CRASH": 2.0, "SIDEWAYS_GRIND": 1.8, "BULL_MOMENTUM": 1.3,
                "BEAR_MOMENTUM": 1.2, "LOW_VOL_GRIND": 1.0, "CRISIS": 1.5
            }
        }
        
        regime_multiplier = multipliers[strategy].get(self.current_market_regime, 1.0)
        signal_strength = self.strategy_signals.get(strategy, 0.5)
        
        # Add noise and leverage effect
        noise = np.random.normal(0, 0.005)  # Random noise
        strategy_return = market_return * regime_multiplier * signal_strength * 2.0 + noise
        
        return strategy_return
    
    def OptimizeStrategyAllocation(self):
        """Optimize strategy allocation based on regime and performance"""
        
        # Start with equal weights
        new_allocations = {k: 0.2 for k in self.strategy_allocations.keys()}
        
        # Adjust based on regime signals
        total_signal = sum(self.strategy_signals.values())
        if total_signal > 0:
            for strategy, signal in self.strategy_signals.items():
                # Weight by signal strength
                new_allocations[strategy] = signal / total_signal
        
        # Adjust based on recent performance
        for strategy in new_allocations.keys():
            perf = self.strategy_performance[strategy]
            
            # Boost allocation for high-performing strategies
            if perf["sharpe"] > self.target_sharpe:
                new_allocations[strategy] *= 1.3
            elif perf["sharpe"] < 0.5:
                new_allocations[strategy] *= 0.7
                
            # Reduce allocation for high drawdown strategies
            if perf["max_dd"] > self.max_drawdown_tolerance:
                new_allocations[strategy] *= 0.6
        
        # Apply constraints
        for strategy in new_allocations.keys():
            new_allocations[strategy] = max(self.min_allocation, 
                                          min(self.max_allocation, new_allocations[strategy]))
        
        # Normalize to sum to 1.0
        total_allocation = sum(new_allocations.values())
        if total_allocation > 0:
            new_allocations = {k: v/total_allocation for k, v in new_allocations.items()}
        
        return new_allocations
    
    def UpdateAllocations(self, new_allocations):
        """Gradually update strategy allocations to avoid whipsaws"""
        
        # Use exponential smoothing for gradual transitions
        alpha = 0.3  # Adjustment speed
        
        for strategy in self.strategy_allocations.keys():
            current = self.strategy_allocations[strategy]
            target = new_allocations[strategy]
            
            # Smooth transition
            self.strategy_allocations[strategy] = alpha * target + (1 - alpha) * current
        
        # Renormalize
        total = sum(self.strategy_allocations.values())
        if total > 0:
            self.strategy_allocations = {k: v/total for k, v in self.strategy_allocations.items()}
    
    def ExecuteStrategyAllocation(self, data):
        """Execute the current strategy allocation via position management"""
        
        # This is a simplified execution - in practice, each strategy would run independently
        # Here we simulate by taking positions that represent each strategy's approach
        
        total_value = self.Portfolio.TotalPortfolioValue
        
        # GAMMA FLOW allocation - SPY momentum with leverage
        gamma_allocation = self.strategy_allocations["GAMMA_FLOW"]
        if gamma_allocation > 0.05 and self.spy in data:
            gamma_position = gamma_allocation * 3.0  # 3x leverage
            if self.spy_rsi_fast.IsReady:
                if self.spy_rsi_fast.Current.Value > 60:
                    self.SetHoldings(self.spy, gamma_position)
                elif self.spy_rsi_fast.Current.Value < 40:
                    self.SetHoldings(self.spy, -gamma_position * 0.5)
        
        # REGIME MOMENTUM allocation - Cross-asset momentum
        momentum_allocation = self.strategy_allocations["REGIME_MOMENTUM"]
        if momentum_allocation > 0.05:
            if self.trend_regime in ["STRONG_UP", "UP"]:
                # Risk-on assets
                self.SetHoldings(self.qqq, momentum_allocation * 2.0)
            elif self.trend_regime in ["STRONG_DOWN", "DOWN"]:
                # Risk-off assets
                self.SetHoldings(self.tlt, momentum_allocation * 2.5)
                self.SetHoldings(self.gld, momentum_allocation * 1.5)
        
        # CRISIS ALPHA allocation - Tail hedging
        crisis_allocation = self.strategy_allocations["CRISIS_ALPHA"]
        if crisis_allocation > 0.05:
            if self.volatility_regime in ["HIGH", "EXTREME"]:
                # Crisis mode - long vol, long bonds, short equities
                self.SetHoldings(self.vxx, crisis_allocation * 2.0)
                self.SetHoldings(self.tlt, crisis_allocation * 1.5)
                self.SetHoldings(self.spy, -crisis_allocation * 1.0)
            else:
                # Tail hedge mode
                self.SetHoldings(self.vxx, crisis_allocation * 0.5)
                self.SetHoldings(self.gld, crisis_allocation * 0.3)
        
        # EARNINGS MOMENTUM allocation - Sector rotation during earnings
        earnings_allocation = self.strategy_allocations["EARNINGS_MOMENTUM"]
        if earnings_allocation > 0.05 and self.current_market_regime == "EARNINGS_SEASON":
            # Simplified earnings momentum via QQQ (tech-heavy)
            earnings_leverage = 4.0 if self.volatility_regime in ["NORMAL", "HIGH"] else 2.0
            self.SetHoldings(self.qqq, earnings_allocation * earnings_leverage)
        
        # MICROSTRUCTURE allocation - Mean reversion
        micro_allocation = self.strategy_allocations["MICROSTRUCTURE"]
        if micro_allocation > 0.05 and self.spy_rsi_fast.IsReady:
            rsi = self.spy_rsi_fast.Current.Value
            if rsi > 75:  # Overbought - short
                self.SetHoldings(self.spy, -micro_allocation * 2.0)
            elif rsi < 25:  # Oversold - long
                self.SetHoldings(self.spy, micro_allocation * 3.0)
    
    def ApplyRiskManagement(self):
        """Apply portfolio-level risk management"""
        
        # Total leverage check
        total_leverage = sum([abs(x.HoldingsValue) for x in self.Portfolio.Values]) / self.Portfolio.TotalPortfolioValue
        
        if total_leverage > self.total_leverage_limit:
            # Scale down all positions
            scale_factor = self.total_leverage_limit / total_leverage * 0.9
            
            for holding in self.Portfolio.Values:
                if holding.Invested:
                    current_weight = holding.HoldingsValue / self.Portfolio.TotalPortfolioValue
                    new_weight = current_weight * scale_factor
                    self.SetHoldings(holding.Symbol, new_weight)
        
        # Drawdown protection
        if self.Portfolio.TotalPortfolioValue > 0:
            total_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            
            if total_return < -0.20:  # 20% drawdown
                # Reduce all allocations
                for strategy in self.strategy_allocations.keys():
                    self.strategy_allocations[strategy] *= 0.5
                
                self.Debug("DRAWDOWN PROTECTION: Reducing all allocations by 50%")
    
    def LogRotationDecision(self):
        """Log current strategy rotation decision"""
        
        allocation_str = ", ".join([f"{k}: {v:.1%}" for k, v in self.strategy_allocations.items()])
        
        self.Debug(f"ROTATION: Regime={self.current_market_regime}, Vol={self.volatility_regime}")
        self.Debug(f"ALLOCATION: {allocation_str}")
        
        # Log top strategies
        sorted_strategies = sorted(self.strategy_allocations.items(), key=lambda x: x[1], reverse=True)
        top_strategy = sorted_strategies[0]
        self.Debug(f"DOMINANT STRATEGY: {top_strategy[0]} ({top_strategy[1]:.1%})")
    
    def MonitorStrategyPerformance(self):
        """Monitor real-time strategy performance"""
        
        # Update strategy portfolio values (simplified tracking)
        total_value = self.Portfolio.TotalPortfolioValue
        
        for strategy, allocation in self.strategy_allocations.items():
            # Simulate strategy performance
            strategy_value = total_value * allocation
            self.strategy_portfolios[strategy] = strategy_value
    
    def OnEndOfDay(self, symbol):
        """Daily strategy analysis and performance tracking"""
        
        if self.Portfolio.TotalPortfolioValue > 0:
            daily_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            
            # Log daily performance by dominant strategy
            dominant_strategy = max(self.strategy_allocations, key=self.strategy_allocations.get)
            dominant_allocation = self.strategy_allocations[dominant_strategy]
            
            self.Debug(f"DAILY: Return {daily_return:.2%}, Dominant: {dominant_strategy} ({dominant_allocation:.1%})")
            
            # Performance attribution
            if abs(daily_return) > 0.02:  # Significant daily move
                regime_performance = f"Regime: {self.current_market_regime}, Vol: {self.volatility_regime}"
                self.Debug(f"ATTRIBUTION: {regime_performance}")
                
            # Adjust rotation frequency based on regime stability
            if self.current_market_regime in ["CRISIS", "HIGH_VOL_CRASH"]:
                self.rebalance_frequency = 2  # More frequent during crisis
            else:
                self.rebalance_frequency = 6  # Normal frequency