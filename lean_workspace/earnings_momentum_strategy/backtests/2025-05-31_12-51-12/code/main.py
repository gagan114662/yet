# STRATEGY 4: EARNINGS MOMENTUM & OPTIONS FLOW MASTER
# Target: 60%+ CAGR, Sharpe > 1.8 via earnings surprises and options flow analysis

from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class EarningsMomentumStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Ultra-aggressive setup for earnings plays
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Universe of earnings-sensitive ETFs and key stocks
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Minute).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Minute).Symbol
        
        # Tech earnings momentum
        self.xlk = self.AddEquity("XLK", Resolution.Minute).Symbol  # Tech sector
        self.arkk = self.AddEquity("ARKK", Resolution.Minute).Symbol # Innovation
        
        # Individual mega-cap stocks for earnings plays
        self.aapl = self.AddEquity("AAPL", Resolution.Minute).Symbol
        self.msft = self.AddEquity("MSFT", Resolution.Minute).Symbol
        self.amzn = self.AddEquity("AMZN", Resolution.Minute).Symbol
        self.googl = self.AddEquity("GOOGL", Resolution.Minute).Symbol
        self.tsla = self.AddEquity("TSLA", Resolution.Minute).Symbol
        self.nvda = self.AddEquity("NVDA", Resolution.Minute).Symbol
        
        # Earnings-sensitive sectors
        self.xly = self.AddEquity("XLY", Resolution.Minute).Symbol  # Consumer discretionary
        self.xlf = self.AddEquity("XLF", Resolution.Minute).Symbol  # Financials
        
        # Options for flow analysis (when available)
        try:
            self.AddOption("SPY", Resolution.Minute)
            self.AddOption("QQQ", Resolution.Minute)
            self.AddOption("AAPL", Resolution.Minute)
        except:
            pass
        
        # VIX proxy using VXX for volatility analysis in earnings context
        # VXX already added above as part of volatility instruments
        if not hasattr(self, 'vxx_added'):
            self.vxx = self.AddEquity("VXX", Resolution.Minute).Symbol
            self.vxx_added = True
        self.vix_proxy = self.vxx  # Use VXX as VIX proxy for earnings volatility
        
        # SPY volatility calculation for earnings momentum analysis
        self.spy_returns_window = RollingWindow[float](20)
        self.calculated_vix = 20.0
        
        # Earnings momentum tracking
        self.earnings_universe = [self.aapl, self.msft, self.amzn, self.googl, self.tsla, self.nvda]
        self.sector_universe = [self.xlk, self.xly, self.xlf, self.spy, self.qqq, self.iwm, self.arkk]
        
        # Earnings momentum indicators
        self.earnings_momentum = {}
        self.revenue_momentum = {}
        self.estimate_revisions = {}
        self.surprise_history = {}
        
        # Options flow proxies
        self.unusual_activity = {}
        self.gamma_exposure = {}
        self.put_call_ratios = {}
        
        # Earnings calendar simulation (would use real data in production)
        self.earnings_calendar = self.GenerateEarningsCalendar()
        self.current_earnings_plays = {}
        
        # Advanced momentum indicators
        self.price_momentum = {}
        self.volume_momentum = {}
        self.analyst_momentum = {}
        
        # Initialize indicators for all assets
        all_assets = self.earnings_universe + self.sector_universe
        for asset in all_assets:
            self.price_momentum[asset] = self.MOMP(asset, 14, Resolution.Daily)
            self.volume_momentum[asset] = self.SMA(asset, 20, Resolution.Daily)  # Volume proxy
            
        # Earnings strategy parameters
        self.earnings_leverage = 8.0       # 8x leverage for earnings plays
        self.position_heat = 0.25          # 25% risk per earnings play
        self.max_earnings_positions = 5    # Max concurrent earnings plays
        
        # Schedule earnings analysis
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.AfterMarketOpen("SPY", 30),
                        self.ScanEarningsOpportunities)
        
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.BeforeMarketClose("SPY", 30),
                        self.PrepareEarningsPlays)
        
        # Track earnings performance
        self.earnings_trades = {}
        self.earnings_success_rate = 0.0
        
    def GenerateEarningsCalendar(self):
        """Generate simulated earnings calendar (would use real data in production)"""
        # Simplified earnings calendar based on typical patterns
        calendar = {}
        
        # Mega-cap earnings typically in specific weeks of quarter
        earnings_weeks = [
            # Q1 earnings (Apr-May)
            (4, 15), (4, 22), (4, 29), (5, 6),
            # Q2 earnings (Jul-Aug)  
            (7, 15), (7, 22), (7, 29), (8, 5),
            # Q3 earnings (Oct-Nov)
            (10, 15), (10, 22), (10, 29), (11, 5),
            # Q4 earnings (Jan-Feb)
            (1, 15), (1, 22), (1, 29), (2, 5)
        ]
        
        for month, day in earnings_weeks:
            for year in range(2020, 2025):
                try:
                    date = datetime(year, month, day)
                    if date not in calendar:
                        calendar[date] = []
                    
                    # Assign stocks to earnings dates (simplified rotation)
                    if month in [1, 2]:  # Q4 earnings
                        calendar[date].extend([self.aapl, self.msft])
                    elif month in [4, 5]:  # Q1 earnings
                        calendar[date].extend([self.amzn, self.googl])
                    elif month in [7, 8]:  # Q2 earnings
                        calendar[date].extend([self.tsla, self.nvda])
                    elif month in [10, 11]:  # Q3 earnings
                        calendar[date].extend([self.aapl, self.msft])
                except:
                    continue
                    
        return calendar
    
    def OnData(self, data):
        # Update momentum and flow indicators
        self.UpdateMomentumIndicators(data)
        
        # Analyze options flow (simplified)
        self.AnalyzeOptionsFlow(data)
        
        # Check for earnings momentum signals
        self.CheckEarningsMomentum(data)
        
        # Execute active earnings plays
        self.ManageEarningsPositions(data)
        
        # Look for post-earnings momentum continuation
        self.TradePostEarningsMomentum(data)
    
    def UpdateMomentumIndicators(self, data):
        """Update all momentum tracking indicators"""
        for asset in self.earnings_universe + self.sector_universe:
            if asset in data and data[asset] is not None:
                price = data[asset].Close
                volume = data[asset].Volume if hasattr(data[asset], 'Volume') else 0
                
                # Update price momentum tracking
                if asset not in self.surprise_history:
                    self.surprise_history[asset] = []
                    
                # Simulate earnings surprise based on momentum
                if self.price_momentum[asset].IsReady:
                    momentum_score = self.price_momentum[asset].Current.Value
                    
                    # Estimate earnings surprise probability based on price momentum
                    surprise_prob = self.CalculateEarningsSurpriseProb(momentum_score, asset)
                    
                    if len(self.surprise_history[asset]) == 0 or abs(surprise_prob - self.surprise_history[asset][-1]) > 0.1:
                        self.surprise_history[asset].append(surprise_prob)
                        
                        if len(self.surprise_history[asset]) > 20:
                            self.surprise_history[asset] = self.surprise_history[asset][-20:]
    
    def CalculateEarningsSurpriseProb(self, momentum_score, asset):
        """Calculate probability of earnings surprise based on momentum"""
        # Strong positive momentum often precedes positive surprises
        base_prob = 0.5  # 50% baseline
        
        # Adjust based on price momentum
        momentum_adjustment = momentum_score * 2.0  # Scale momentum
        momentum_adjustment = max(-0.3, min(0.3, momentum_adjustment))  # Cap adjustment
        
        # Adjust based on sector (tech typically more volatile)
        sector_adjustment = 0.0
        if asset in [self.xlk, self.arkk, self.nvda, self.tsla]:
            sector_adjustment = 0.1  # Tech bias
        elif asset in [self.xlf, self.xly]:
            sector_adjustment = -0.05  # More stable sectors
            
        surprise_prob = base_prob + momentum_adjustment + sector_adjustment
        return max(0.1, min(0.9, surprise_prob))
    
    def AnalyzeOptionsFlow(self, data):
        """Analyze options flow for unusual activity (simplified)"""
        # In production, would use real options flow data
        # Here we simulate based on volume and price action
        
        for asset in self.earnings_universe:
            if asset in data and data[asset] is not None:
                price = data[asset].Close
                volume = data[asset].Volume if hasattr(data[asset], 'Volume') else 0
                
                # Estimate unusual activity based on volume spikes
                if asset not in self.unusual_activity:
                    self.unusual_activity[asset] = []
                    
                # Calculate volume ratio vs recent average
                if len(self.unusual_activity[asset]) > 10:
                    avg_volume = np.mean(self.unusual_activity[asset][-10:])
                    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
                    
                    # Unusual activity threshold
                    if volume_ratio > 2.0:  # 2x normal volume
                        self.unusual_activity[asset].append(volume_ratio)
                        
                        # This indicates potential earnings positioning
                        self.FlagEarningsOpportunity(asset, volume_ratio)
                else:
                    self.unusual_activity[asset].append(volume)
                    
                # Keep rolling window
                if len(self.unusual_activity[asset]) > 50:
                    self.unusual_activity[asset] = self.unusual_activity[asset][-50:]
    
    def FlagEarningsOpportunity(self, asset, signal_strength):
        """Flag asset for potential earnings play"""
        if asset not in self.current_earnings_plays:
            self.current_earnings_plays[asset] = {
                'signal_strength': signal_strength,
                'detected_time': self.Time,
                'position_size': 0,
                'entry_price': 0
            }
    
    def CheckEarningsMomentum(self, data):
        """Check for earnings momentum patterns"""
        current_date = self.Time.date()
        
        # Look for stocks near earnings dates
        for earnings_date, stocks in self.earnings_calendar.items():
            days_to_earnings = (earnings_date.date() - current_date).days
            
            if 0 <= days_to_earnings <= 5:  # 5 days before earnings
                for stock in stocks:
                    if (stock in data and data[stock] is not None and 
                        self.price_momentum[stock].IsReady):
                        
                        momentum = self.price_momentum[stock].Current.Value
                        surprise_prob = self.surprise_history.get(stock, [0.5])[-1] if self.surprise_history.get(stock) else 0.5
                        
                        # Strong pre-earnings momentum + high surprise probability
                        if momentum > 0.03 and surprise_prob > 0.7:
                            self.PrepareEarningsLong(stock, momentum, surprise_prob)
                        elif momentum < -0.03 and surprise_prob < 0.3:
                            self.PrepareEarningsShort(stock, momentum, surprise_prob)
    
    def PrepareEarningsLong(self, stock, momentum, surprise_prob):
        """Prepare long earnings play"""
        if len(self.current_earnings_plays) >= self.max_earnings_positions:
            return
            
        signal_strength = momentum * surprise_prob * 2.0
        position_size = min(self.position_heat * signal_strength, self.position_heat)
        
        if stock not in self.current_earnings_plays:
            self.current_earnings_plays[stock] = {
                'direction': 'long',
                'signal_strength': signal_strength,
                'position_size': position_size,
                'entry_price': self.Securities[stock].Price,
                'stop_loss': self.Securities[stock].Price * 0.92,  # 8% stop
                'target': self.Securities[stock].Price * 1.15      # 15% target
            }
    
    def PrepareEarningsShort(self, stock, momentum, surprise_prob):
        """Prepare short earnings play"""
        if len(self.current_earnings_plays) >= self.max_earnings_positions:
            return
            
        signal_strength = abs(momentum) * (1 - surprise_prob) * 2.0
        position_size = min(self.position_heat * signal_strength, self.position_heat)
        
        if stock not in self.current_earnings_plays:
            self.current_earnings_plays[stock] = {
                'direction': 'short',
                'signal_strength': signal_strength,
                'position_size': -position_size,
                'entry_price': self.Securities[stock].Price,
                'stop_loss': self.Securities[stock].Price * 1.08,  # 8% stop
                'target': self.Securities[stock].Price * 0.85      # 15% target
            }
    
    def ManageEarningsPositions(self, data):
        """Manage active earnings positions"""
        positions_to_close = []
        
        for stock, play_info in self.current_earnings_plays.items():
            if stock in data and data[stock] is not None:
                current_price = data[stock].Close
                
                # Check if position is active
                if self.Portfolio[stock].Invested:
                    # Check stop loss and target
                    if play_info['direction'] == 'long':
                        if (current_price <= play_info['stop_loss'] or 
                            current_price >= play_info['target']):
                            self.Liquidate(stock)
                            positions_to_close.append(stock)
                    else:  # short
                        if (current_price >= play_info['stop_loss'] or 
                            current_price <= play_info['target']):
                            self.Liquidate(stock)
                            positions_to_close.append(stock)
                            
                # Check for entry signal
                elif abs(play_info['position_size']) > 0:
                    leverage_multiplier = min(self.earnings_leverage, 
                                            self.earnings_leverage * play_info['signal_strength'])
                    final_position = play_info['position_size'] * leverage_multiplier
                    
                    self.SetHoldings(stock, final_position)
        
        # Clean up closed positions
        for stock in positions_to_close:
            del self.current_earnings_plays[stock]
    
    def TradePostEarningsMomentum(self, data):
        """Trade post-earnings momentum continuation"""
        for stock in self.earnings_universe:
            if stock in data and data[stock] is not None:
                # Look for post-earnings gap and momentum
                current_price = data[stock].Close
                
                # Check for significant gap (simplified)
                if self.price_momentum[stock].IsReady:
                    recent_momentum = self.price_momentum[stock].Current.Value
                    
                    # Strong momentum after earnings (gap continuation)
                    if abs(recent_momentum) > 0.05:  # 5% momentum
                        # Trade momentum continuation
                        momentum_position = np.sign(recent_momentum) * 0.15 * abs(recent_momentum) * 10
                        momentum_position = max(-2.0, min(2.0, momentum_position))
                        
                        # Only if not already in earnings play
                        if stock not in self.current_earnings_plays:
                            self.SetHoldings(stock, momentum_position)
    
    def ScanEarningsOpportunities(self):
        """Daily scan for earnings opportunities"""
        # Scan for upcoming earnings and momentum setups
        current_date = self.Time.date()
        
        opportunities = []
        for earnings_date, stocks in self.earnings_calendar.items():
            days_to_earnings = (earnings_date.date() - current_date).days
            
            if 1 <= days_to_earnings <= 7:  # Scan 1-7 days ahead
                for stock in stocks:
                    if stock in self.Securities and self.price_momentum[stock].IsReady:
                        momentum = self.price_momentum[stock].Current.Value
                        price = self.Securities[stock].Price
                        
                        opportunity_score = abs(momentum) * (7 - days_to_earnings)
                        opportunities.append((stock, opportunity_score, momentum, days_to_earnings))
        
        # Sort by opportunity score and log top opportunities
        opportunities.sort(key=lambda x: x[1], reverse=True)
        
        if opportunities:
            top_3 = opportunities[:3]
            for stock, score, momentum, days in top_3:
                self.Debug(f"Earnings Opportunity: {stock} in {days} days, Momentum: {momentum:.3f}, Score: {score:.3f}")
    
    def PrepareEarningsPlays(self):
        """Prepare for next day's earnings plays"""
        # Risk management for overnight earnings positions
        total_earnings_exposure = 0
        
        for stock, play_info in self.current_earnings_plays.items():
            if self.Portfolio[stock].Invested:
                exposure = abs(self.Portfolio[stock].HoldingsValue)
                total_earnings_exposure += exposure
        
        # If over-exposed, reduce positions
        max_earnings_exposure = self.Portfolio.TotalPortfolioValue * 0.5  # 50% max in earnings
        if total_earnings_exposure > max_earnings_exposure:
            scale_factor = max_earnings_exposure / total_earnings_exposure
            
            for stock in self.current_earnings_plays.keys():
                if self.Portfolio[stock].Invested:
                    current_weight = self.Portfolio[stock].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    new_weight = current_weight * scale_factor
                    self.SetHoldings(stock, new_weight)
    
    def OnEndOfDay(self, symbol):
        """Daily earnings strategy analysis"""
        if self.Portfolio.TotalPortfolioValue > 0:
            daily_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            
            # Track earnings trade performance
            earnings_positions = len([x for x in self.current_earnings_plays.keys() 
                                    if self.Portfolio[x].Invested])
            
            if earnings_positions > 0 and abs(daily_return) > 0.03:
                self.Debug(f"Earnings Performance: {daily_return:.2%} with {earnings_positions} positions")
                
            # Adjust strategy aggressiveness based on recent performance
            if daily_return > 0.05:  # Great day
                self.earnings_leverage = min(10.0, self.earnings_leverage * 1.1)
                self.position_heat = min(0.30, self.position_heat * 1.05)
            elif daily_return < -0.05:  # Bad day
                self.earnings_leverage = max(5.0, self.earnings_leverage * 0.9)
                self.position_heat = max(0.15, self.position_heat * 0.95)
    
    def CalculateVixFromVXX(self):
        """Calculate VIX proxy from VXX for earnings volatility analysis"""
        # Method 1: Use VXX as VIX proxy
        if hasattr(self, 'vxx') and self.vxx in self.Securities:
            vxx_price = self.Securities[self.vxx].Price
            # Convert VXX to VIX-like scale for earnings analysis
            vix_from_vxx = vxx_price * 2.5
            
            # Method 2: Calculate from SPY volatility (useful for earnings timing)
            vix_from_spy = self.CalculateImpliedVolFromSPY()
            
            # For earnings, prefer more responsive SPY-based calculation
            if vix_from_spy > 0:
                combined_vix = (vix_from_vxx * 0.6) + (vix_from_spy * 0.4)
            else:
                combined_vix = vix_from_vxx
                
            return max(10, min(80, combined_vix))
        else:
            return self.CalculateImpliedVolFromSPY()
    
    def CalculateImpliedVolFromSPY(self):
        """Calculate implied VIX from SPY returns for earnings analysis"""
        if self.spy in self.Securities:
            spy_price = self.Securities[self.spy].Price
            
            # Update returns window
            if hasattr(self, 'previous_spy_price') and self.previous_spy_price > 0:
                minute_return = (spy_price - self.previous_spy_price) / self.previous_spy_price
                self.spy_returns_window.Add(minute_return)
                
            self.previous_spy_price = spy_price
            
            # Calculate realized volatility for earnings context
            if self.spy_returns_window.IsReady:
                returns = [self.spy_returns_window[i] for i in range(self.spy_returns_window.Count)]
                if len(returns) > 5:
                    import numpy as np
                    # Convert minute returns to daily, then annualize
                    minute_vol = np.std(returns)
                    daily_vol = minute_vol * np.sqrt(390)  # 390 trading minutes per day
                    annual_vol = daily_vol * np.sqrt(252)  # 252 trading days
                    
                    # Convert to VIX-like scale with earnings risk premium
                    # Earnings periods typically have higher implied vol
                    earnings_risk_premium = 1.4  # Higher premium for earnings
                    implied_vix = annual_vol * 100 * earnings_risk_premium
                    
                    self.calculated_vix = max(10, min(70, implied_vix))
                    return self.calculated_vix
                    
        return self.calculated_vix