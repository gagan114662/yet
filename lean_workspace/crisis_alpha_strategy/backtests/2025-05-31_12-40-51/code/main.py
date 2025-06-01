# STRATEGY 3: CRISIS ALPHA & TAIL RISK MASTER
# Target: 50%+ CAGR during crises, Sharpe > 2.0 via tail hedging and crisis alpha

from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class CrisisAlphaStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Extreme leverage for crisis opportunities
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Crisis monitoring universe
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Minute).Symbol  # Long bonds
        self.gld = self.AddEquity("GLD", Resolution.Minute).Symbol  # Gold
        
        # Volatility instruments
        self.vxx = self.AddEquity("VXX", Resolution.Minute).Symbol  # VIX ETF
        
        # Credit spreads monitoring
        self.hyg = self.AddEquity("HYG", Resolution.Minute).Symbol  # High yield
        self.lqd = self.AddEquity("LQD", Resolution.Minute).Symbol  # Investment grade
        
        # International for contagion
        self.eem = self.AddEquity("EEM", Resolution.Minute).Symbol  # Emerging markets
        self.efa = self.AddEquity("EFA", Resolution.Minute).Symbol  # Europe
        
        # Dollar strength (crisis indicator)
        self.uup = self.AddEquity("UUP", Resolution.Minute).Symbol  # Dollar bull
        
        # Alternative crisis hedge
        try:
            self.gbtc = self.AddEquity("GBTC", Resolution.Minute).Symbol  # Bitcoin proxy
        except:
            self.gbtc = None
            
        # VIX data for volatility analysis
        self.vix = self.AddData(QuandlVix, "CBOE/VIX", Resolution.Daily).Symbol
        
        # Crisis detection indicators
        self.crisis_indicators = {}
        self.tail_risk_score = 0.0
        self.crisis_mode = False
        self.crisis_confidence = 0.0
        
        # Volatility surface analysis
        self.vol_term_structure = {}
        self.vol_skew = {}
        
        # Multi-timeframe crisis detection
        self.crisis_fast = self.RSI("SPY", 5, Resolution.Minute)   # Flash crash detection
        self.crisis_medium = self.RSI("SPY", 60, Resolution.Minute) # Intraday stress
        self.crisis_slow = self.RSI("SPY", 1440, Resolution.Minute) # Daily stress
        
        # Credit spreads
        self.credit_spread = None
        self.credit_trend = []
        
        # Tail risk hedging parameters
        self.tail_hedge_allocation = 0.05  # 5% baseline hedge
        self.crisis_allocation = 0.90      # 90% during crisis
        self.max_leverage = 10.0           # Extreme leverage for tail events
        
        # Crisis alpha thresholds
        self.vix_spike_threshold = 30
        self.drawdown_threshold = 0.05     # 5% SPY drawdown
        self.volume_spike_threshold = 2.0   # 2x average volume
        
        # Schedule crisis monitoring
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.Every(TimeSpan.FromMinutes(5)), 
                        self.MonitorCrisisSignals)
        
        # Track market stress indicators
        self.stress_indicators = {
            "vix_spike": False,
            "credit_widening": False,
            "equity_crash": False,
            "vol_surface_stress": False,
            "liquidity_crunch": False
        }
        
        # Historical data for tail risk modeling
        self.price_history = {}
        self.volatility_history = {}
        self.max_lookback = 252  # 1 year
        
    def OnData(self, data):
        # Update price and volatility history
        self.UpdateMarketData(data)
        
        # Real-time crisis detection
        self.DetectCrisisConditions(data)
        
        # Calculate tail risk score
        self.CalculateTailRiskScore()
        
        # Execute crisis alpha strategies
        if self.crisis_mode:
            self.ExecuteCrisisAlpha()
        else:
            self.ExecuteTailHedging()
            
        # Monitor for flash crash opportunities
        self.MonitorFlashCrash(data)
    
    def UpdateMarketData(self, data):
        """Update historical data for crisis modeling"""
        for symbol in [self.spy, self.tlt, self.gld, self.vxx, self.hyg]:
            if symbol in data and data[symbol] is not None:
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                    
                self.price_history[symbol].append(data[symbol].Close)
                
                # Keep rolling window
                if len(self.price_history[symbol]) > self.max_lookback:
                    self.price_history[symbol] = self.price_history[symbol][-self.max_lookback:]
        
        # Update credit spreads
        if (self.hyg in data and self.lqd in data and 
            data[self.hyg] is not None and data[self.lqd] is not None):
            
            hyg_price = data[self.hyg].Close
            lqd_price = data[self.lqd].Close
            
            # Approximate credit spread (inverted price relationship)
            if lqd_price > 0 and hyg_price > 0:
                self.credit_spread = (1/hyg_price) - (1/lqd_price)
                self.credit_trend.append(self.credit_spread)
                
                if len(self.credit_trend) > 50:
                    self.credit_trend = self.credit_trend[-50:]
    
    def DetectCrisisConditions(self, data):
        """Advanced crisis detection using multiple signals"""
        crisis_signals = 0
        max_signals = 7
        
        # 1. VIX Spike Detection
        vix_level = self.Securities["CBOE/VIX"].Price if "CBOE/VIX" in self.Securities else 20
        if vix_level > self.vix_spike_threshold:
            self.stress_indicators["vix_spike"] = True
            crisis_signals += 1
        else:
            self.stress_indicators["vix_spike"] = False
            
        # 2. Equity Crash Detection
        if self.crisis_fast.IsReady and self.crisis_fast.Current.Value < 20:
            self.stress_indicators["equity_crash"] = True
            crisis_signals += 1
        else:
            self.stress_indicators["equity_crash"] = False
            
        # 3. Credit Spread Widening
        if len(self.credit_trend) > 10:
            recent_spread = np.mean(self.credit_trend[-5:])
            historical_spread = np.mean(self.credit_trend[-20:-5])
            
            if recent_spread > historical_spread * 1.3:
                self.stress_indicators["credit_widening"] = True
                crisis_signals += 1
            else:
                self.stress_indicators["credit_widening"] = False
                
        # 4. Volume Spike (Liquidity Stress)
        if self.spy in data and hasattr(data[self.spy], 'Volume'):
            current_volume = data[self.spy].Volume
            if hasattr(self, 'avg_volume'):
                if current_volume > self.avg_volume * self.volume_spike_threshold:
                    self.stress_indicators["liquidity_crunch"] = True
                    crisis_signals += 1
                else:
                    self.stress_indicators["liquidity_crunch"] = False
            else:
                self.avg_volume = current_volume
                
        # 5. Cross-asset correlation breakdown
        correlation_stress = self.DetectCorrelationBreakdown()
        if correlation_stress:
            crisis_signals += 1
            
        # 6. Volatility surface stress
        vol_surface_stress = self.DetectVolSurfaceStress()
        if vol_surface_stress:
            self.stress_indicators["vol_surface_stress"] = True
            crisis_signals += 1
        else:
            self.stress_indicators["vol_surface_stress"] = False
            
        # 7. Dollar strength spike (flight to quality)
        dollar_spike = self.DetectDollarSpike(data)
        if dollar_spike:
            crisis_signals += 1
            
        # Update crisis state
        self.crisis_confidence = crisis_signals / max_signals
        self.crisis_mode = self.crisis_confidence > 0.4  # 40% threshold
        
        # Log crisis state
        if self.crisis_mode and self.Time.minute == 0:  # Log hourly
            self.Debug(f"CRISIS MODE: Confidence {self.crisis_confidence:.2f}, Signals: {crisis_signals}")
    
    def DetectCorrelationBreakdown(self):
        """Detect correlation breakdown indicating crisis"""
        if (self.spy not in self.price_history or 
            self.tlt not in self.price_history or
            len(self.price_history[self.spy]) < 20):
            return False
            
        try:
            # Calculate rolling correlation
            spy_returns = np.diff(self.price_history[self.spy][-20:])
            tlt_returns = np.diff(self.price_history[self.tlt][-20:])
            
            if len(spy_returns) > 10 and len(tlt_returns) > 10:
                correlation = np.corrcoef(spy_returns, tlt_returns)[0,1]
                
                # Normal regime: SPY and TLT negatively correlated
                # Crisis: Correlation breaks down or becomes positive
                return correlation > -0.2  # Breakdown threshold
                
        except:
            pass
            
        return False
    
    def DetectVolSurfaceStress(self):
        """Detect volatility surface stress patterns"""
        vix_level = self.Securities["CBOE/VIX"].Price if "CBOE/VIX" in self.Securities else 20
        
        # VIX above 25 indicates stress
        if vix_level > 25:
            return True
            
        # VIX rapid increase (simplified - would use options data)
        if hasattr(self, 'previous_vix'):
            vix_change = (vix_level - self.previous_vix) / self.previous_vix
            if vix_change > 0.2:  # 20% VIX increase
                return True
                
        self.previous_vix = vix_level
        return False
    
    def DetectDollarSpike(self, data):
        """Detect dollar strength spikes"""
        if self.uup in data and data[self.uup] is not None:
            if self.uup not in self.price_history:
                return False
                
            current_price = data[self.uup].Close
            if len(self.price_history[self.uup]) > 10:
                recent_avg = np.mean(self.price_history[self.uup][-10:])
                return current_price > recent_avg * 1.02  # 2% spike
                
        return False
    
    def CalculateTailRiskScore(self):
        """Calculate comprehensive tail risk score"""
        if self.spy not in self.price_history or len(self.price_history[self.spy]) < 50:
            self.tail_risk_score = 0.5
            return
            
        spy_prices = np.array(self.price_history[self.spy])
        spy_returns = np.diff(spy_prices) / spy_prices[:-1]
        
        if len(spy_returns) > 20:
            # Calculate tail risk metrics
            returns_std = np.std(spy_returns)
            recent_vol = np.std(spy_returns[-20:])
            vol_spike = recent_vol / returns_std if returns_std > 0 else 1
            
            # Skewness (negative skew = tail risk)
            skewness = self.calculate_skewness(spy_returns[-50:])
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + spy_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
            
            # Combine into tail risk score
            vol_component = min(vol_spike, 3.0) / 3.0
            skew_component = max(0, -skewness) / 2.0
            drawdown_component = min(max_drawdown, 0.3) / 0.3
            
            self.tail_risk_score = (vol_component + skew_component + drawdown_component) / 3.0
    
    def calculate_skewness(self, returns):
        """Calculate skewness of returns"""
        if len(returns) < 10:
            return 0
            
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
            
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def ExecuteCrisisAlpha(self):
        """Execute aggressive crisis alpha strategies"""
        # Crisis portfolio allocation
        target_positions = {}
        
        # 1. Volatility long (VXX)
        target_positions[self.vxx] = 3.0  # 3x leverage on volatility
        
        # 2. Long-term treasuries (flight to quality)
        target_positions[self.tlt] = 2.5
        
        # 3. Gold (crisis hedge)
        target_positions[self.gld] = 2.0
        
        # 4. Short equities (crisis beneficiary)
        target_positions[self.spy] = -2.0
        target_positions[self.eem] = -1.5  # Short emerging markets harder
        
        # 5. Dollar strength
        target_positions[self.uup] = 1.5
        
        # 6. Short credit (HYG)
        target_positions[self.hyg] = -1.0
        
        # Scale based on crisis confidence
        leverage_multiplier = 1.0 + (self.crisis_confidence * 2.0)  # Up to 3x during extreme crisis
        
        for symbol, weight in target_positions.items():
            scaled_weight = weight * leverage_multiplier
            scaled_weight = max(-5.0, min(5.0, scaled_weight))  # Cap at 5x
            
            if abs(scaled_weight) > 0.1:
                self.SetHoldings(symbol, scaled_weight)
    
    def ExecuteTailHedging(self):
        """Execute tail hedging during normal times"""
        # Conservative tail hedge allocation
        hedge_positions = {}
        
        # Small VIX hedge
        hedge_positions[self.vxx] = 0.05
        
        # Treasury hedge
        hedge_positions[self.tlt] = 0.10
        
        # Gold hedge
        hedge_positions[self.gld] = 0.05
        
        # Maintain equity exposure with hedge overlay
        hedge_positions[self.spy] = 0.70
        
        # Scale hedge based on tail risk score
        hedge_multiplier = 1.0 + (self.tail_risk_score * 0.5)
        
        for symbol, weight in hedge_positions.items():
            scaled_weight = weight * hedge_multiplier
            self.SetHoldings(symbol, scaled_weight)
    
    def MonitorFlashCrash(self, data):
        """Monitor for flash crash opportunities"""
        if not self.crisis_fast.IsReady:
            return
            
        # Flash crash indicators
        if (self.crisis_fast.Current.Value < 10 and  # Extreme oversold
            self.spy in data and hasattr(data[self.spy], 'Volume')):
            
            current_volume = data[self.spy].Volume
            if hasattr(self, 'avg_volume') and current_volume > self.avg_volume * 3:
                # Flash crash opportunity - aggressive long
                self.SetHoldings(self.spy, 5.0)  # 5x leverage into crash
                
                # Schedule reversion trade
                self.Schedule.On(self.DateRules.Today, 
                               self.TimeRules.AfterMarketOpen("SPY", 60),
                               self.RevertFlashCrashPosition)
    
    def RevertFlashCrashPosition(self):
        """Revert flash crash position after recovery"""
        # Reduce aggressive position after potential recovery
        current_spy_holding = self.Portfolio[self.spy].HoldingsValue / self.Portfolio.TotalPortfolioValue
        
        if current_spy_holding > 3.0:
            self.SetHoldings(self.spy, 1.0)  # Reduce to normal allocation
    
    def MonitorCrisisSignals(self):
        """5-minute crisis signal monitoring"""
        # Update average volume
        if self.spy in self.Securities:
            spy_security = self.Securities[self.spy]
            if hasattr(spy_security, 'Volume') and spy_security.Volume > 0:
                if not hasattr(self, 'volume_history'):
                    self.volume_history = []
                    
                self.volume_history.append(spy_security.Volume)
                if len(self.volume_history) > 100:
                    self.volume_history = self.volume_history[-100:]
                    
                self.avg_volume = np.mean(self.volume_history)
        
        # Risk management during crisis
        if self.crisis_mode:
            total_leverage = sum([abs(x.HoldingsValue) for x in self.Portfolio.Values]) / self.Portfolio.TotalPortfolioValue
            
            if total_leverage > self.max_leverage:
                # Scale down if over-leveraged
                scale_factor = self.max_leverage / total_leverage
                for holding in self.Portfolio.Values:
                    if holding.Invested:
                        current_weight = holding.HoldingsValue / self.Portfolio.TotalPortfolioValue
                        new_weight = current_weight * scale_factor
                        self.SetHoldings(holding.Symbol, new_weight)
    
    def OnEndOfDay(self, symbol):
        """Daily crisis analysis and performance tracking"""
        if self.Portfolio.TotalPortfolioValue > 0:
            daily_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            
            # Log performance during crisis vs normal times
            if self.crisis_mode and daily_return > 0.05:
                self.Debug(f"Crisis Alpha Success: {daily_return:.2%} return during crisis")
            elif not self.crisis_mode and daily_return < -0.02:
                self.Debug(f"Tail hedge activated: {daily_return:.2%} during normal times")
                
            # Adjust sensitivity based on recent performance
            if self.crisis_mode and daily_return < -0.03:
                # Crisis strategy not working, reduce sensitivity
                self.vix_spike_threshold = min(35, self.vix_spike_threshold + 1)
            elif not self.crisis_mode and abs(daily_return) > 0.03:
                # High volatility during "normal" times, increase sensitivity
                self.vix_spike_threshold = max(25, self.vix_spike_threshold - 1)