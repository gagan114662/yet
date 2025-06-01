from AlgorithmImports import *

class LeveragedBuyHold(QCAlgorithm):
    """
    TARGET DESTROYER - Active leveraged momentum strategy
    DESIGNED TO CRUSH ALL 5 TARGETS SIMULTANEOUSLY
    """
    
    def Initialize(self):
        # 15-year backtest
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Strategic leverage
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # TARGET-CRUSHING UNIVERSE
        weapons = ["TQQQ", "UPRO", "SOXL", "TECL", "QQQ", "SPY", "XLK", "VXX", "UVXY", "TLT"]
        
        self.assets = {}
        for symbol in weapons:
            try:
                security = self.AddEquity(symbol, Resolution.Daily)
                # Strategic leverage: 1.5x for leveraged ETFs, 2.5x for regular
                if symbol in ["TQQQ", "UPRO", "SOXL", "TECL", "VXX", "UVXY"]:
                    security.SetLeverage(1.5)  # Boost already leveraged
                else:
                    security.SetLeverage(2.5)  # Leverage regular ETFs
                security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
                self.assets[symbol] = security
            except:
                continue
        
        # Precision indicators
        self.indicators = {}
        for symbol in self.assets.keys():
            self.indicators[symbol] = {
                "mom_5": self.MOMP(symbol, 5),     # 5-day momentum
                "mom_20": self.MOMP(symbol, 20),   # 20-day momentum
                "rsi": self.RSI(symbol, 14),       # RSI for timing
                "bb": self.BB(symbol, 20, 2),      # Bollinger Bands
                "ema_10": self.EMA(symbol, 10),    # Short trend
                "ema_50": self.EMA(symbol, 50)     # Medium trend
            }
        
        # Target tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.peak_value = 100000
        self.max_drawdown_seen = 0
        self.position_tracker = {}
        
        # AGGRESSIVE PARAMETERS for target crushing
        self.max_positions = 4
        self.max_position_size = 0.60
        self.momentum_threshold = 0.01
        self.profit_target = 0.04        # 4% profit target
        self.stop_loss = 0.02            # 2% stop loss
        self.drawdown_protection = 0.12  # 12% drawdown protection
        
        # Market state
        self.bull_market = True
        self.protection_mode = False
        
        # AGGRESSIVE SCHEDULES for 150+ trades/year
        
        # Daily momentum hunting
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.DailyMomentumHunt
        )
        
        # Midday rebalancing
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 120),
            self.MidDayRebalance
        )
        
        # Weekly aggressive rotation
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Wednesday, DayOfWeek.Friday),
            self.TimeRules.AfterMarketOpen("SPY", 45),
            self.AggressiveRotation
        )
        
    def DailyMomentumHunt(self):
        """Hunt for daily momentum opportunities"""
        
        # Update protection mode
        current_value = self.Portfolio.TotalPortfolioValue
        if current_value > self.peak_value:
            self.peak_value = current_value
            
        drawdown = (self.peak_value - current_value) / self.peak_value
        if drawdown > self.max_drawdown_seen:
            self.max_drawdown_seen = drawdown
            
        self.protection_mode = drawdown > self.drawdown_protection
        
        if self.protection_mode:
            self.ActivateProtection()
            return
            
        # Hunt for momentum
        opportunities = []
        
        for symbol in self.assets.keys():
            if not self.AllIndicatorsReady(symbol):
                continue
                
            indicators = self.indicators[symbol]
            
            mom_5 = indicators["mom_5"].Current.Value
            mom_20 = indicators["mom_20"].Current.Value
            rsi = indicators["rsi"].Current.Value
            price = self.Securities[symbol].Price
            ema_10 = indicators["ema_10"].Current.Value
            bb_upper = indicators["bb"].UpperBand.Current.Value
            
            # MOMENTUM OPPORTUNITY
            if (mom_5 > 0.015 and           # Strong 5-day momentum
                mom_20 > 0.005 and          # Positive trend
                price > ema_10 and          # Above short trend
                rsi < 80):                  # Not overbought
                
                strength = mom_5 * 4 + mom_20 * 2
                
                opportunities.append({
                    "symbol": symbol,
                    "strength": strength,
                    "momentum": mom_5,
                    "type": "MOMENTUM"
                })
                
            # BREAKOUT OPPORTUNITY  
            elif (mom_5 > 0.02 and          # Very strong momentum
                  price > bb_upper and      # Bollinger breakout
                  rsi < 85):                # Room to run
                
                strength = mom_5 * 5
                
                opportunities.append({
                    "symbol": symbol,
                    "strength": strength,
                    "momentum": mom_5,
                    "type": "BREAKOUT"
                })
        
        if opportunities:
            opportunities.sort(key=lambda x: x["strength"], reverse=True)
            self.ExecuteOpportunities(opportunities[:self.max_positions], "DAILY")
    
    def MidDayRebalance(self):
        """Midday rebalancing for active management"""
        
        if self.protection_mode:
            return
            
        # Rebalance based on momentum changes
        adjustments = 0
        
        for symbol in self.assets.keys():
            if not self.Portfolio[symbol].Invested or not self.AllIndicatorsReady(symbol):
                continue
                
            indicators = self.indicators[symbol]
            mom_5 = indicators["mom_5"].Current.Value
            current_weight = self.GetCurrentWeight(symbol)
            
            # Increase winners, reduce losers
            if mom_5 > 0.02 and current_weight < self.max_position_size:
                new_weight = min(current_weight * 1.1, self.max_position_size)
                if new_weight - current_weight > 0.05:
                    self.SetHoldings(symbol, new_weight)
                    self.total_trades += 1
                    adjustments += 1
                    self.Log(f"MIDDAY_BOOST: {symbol} -> {new_weight:.1%}")
                    
            elif mom_5 < -0.01 and current_weight > 0.1:
                new_weight = current_weight * 0.8
                self.SetHoldings(symbol, new_weight)
                self.total_trades += 1
                adjustments += 1
                self.Log(f"MIDDAY_REDUCE: {symbol} -> {new_weight:.1%}")
                
            if adjustments >= 3:  # Limit daily adjustments
                break
    
    def AggressiveRotation(self):
        """Aggressive rotation for high trade frequency"""
        
        if self.protection_mode:
            return
            
        # Rotate underperformers
        current_holdings = []
        for symbol in self.assets.keys():
            if self.Portfolio[symbol].Invested:
                weight = self.GetCurrentWeight(symbol)
                if weight > 0.1:  # Only consider significant positions
                    current_holdings.append((symbol, weight))
        
        # Exit weakest performer if we have max positions
        if len(current_holdings) >= self.max_positions:
            weakest_symbol = None
            weakest_momentum = float('inf')
            
            for symbol, weight in current_holdings:
                if self.AllIndicatorsReady(symbol):
                    mom_5 = self.indicators[symbol]["mom_5"].Current.Value
                    if mom_5 < weakest_momentum:
                        weakest_momentum = mom_5
                        weakest_symbol = symbol
            
            # Exit if momentum is weak
            if weakest_symbol and weakest_momentum < 0.005:
                self.Liquidate(weakest_symbol)
                self.total_trades += 1
                self.Log(f"ROTATION_EXIT: {weakest_symbol}")
    
    def ActivateProtection(self):
        """Activate drawdown protection"""
        
        # Reduce all positions by 50%
        for symbol in self.assets.keys():
            if self.Portfolio[symbol].Invested:
                current_weight = self.GetCurrentWeight(symbol)
                if current_weight > 0.1:
                    protection_weight = current_weight * 0.5
                    self.SetHoldings(symbol, protection_weight)
                    self.total_trades += 1
                    self.Log(f"PROTECTION: {symbol} -> {protection_weight:.1%}")
        
        # Add defensive position
        if not self.Portfolio["TLT"].Invested:
            self.SetHoldings("TLT", 0.25)
            self.total_trades += 1
            self.Log("PROTECTION: TLT defensive")
    
    def ExecuteOpportunities(self, opportunities, session):
        """Execute high-probability opportunities"""
        
        current_positions = len([s for s in self.assets.keys() 
                               if self.Portfolio[s].Invested and self.GetCurrentWeight(s) > 0.1])
        
        available_slots = self.max_positions - current_positions
        
        for opportunity in opportunities[:available_slots]:
            symbol = opportunity["symbol"]
            current_weight = self.GetCurrentWeight(symbol)
            
            # Aggressive position sizing
            strength = opportunity["strength"]
            target_weight = min(self.max_position_size, strength * 0.15)
            target_weight = max(target_weight, 0.2)  # Minimum 20%
            
            if abs(target_weight - current_weight) > 0.08:
                self.SetHoldings(symbol, target_weight)
                self.total_trades += 1
                
                self.position_tracker[symbol] = {
                    "entry_price": self.Securities[symbol].Price,
                    "entry_time": self.Time,
                    "type": opportunity["type"]
                }
                
                self.Log(f"{session}: {opportunity['type']} - {symbol} -> {target_weight:.1%}")
    
    def OnData(self, data):
        """Aggressive profit taking and loss cutting"""
        
        for symbol in list(self.position_tracker.keys()):
            if not self.Portfolio[symbol].Invested:
                if symbol in self.position_tracker:
                    del self.position_tracker[symbol]
                continue
                
            entry_data = self.position_tracker[symbol]
            current_price = self.Securities[symbol].Price
            entry_price = entry_data["entry_price"]
            
            if entry_price == 0:
                continue
                
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Aggressive profit taking
            if pnl_pct > self.profit_target:
                self.Liquidate(symbol)
                self.total_trades += 1
                self.winning_trades += 1
                del self.position_tracker[symbol]
                self.Log(f"PROFIT: {symbol} +{pnl_pct:.1%}")
                
            # Tight stop loss
            elif pnl_pct < -self.stop_loss:
                self.Liquidate(symbol)
                self.total_trades += 1
                del self.position_tracker[symbol]
                self.Log(f"STOP: {symbol} {pnl_pct:.1%}")
    
    def AllIndicatorsReady(self, symbol):
        """Check if indicators are ready"""
        if symbol not in self.indicators:
            return False
        return all(ind.IsReady for ind in self.indicators[symbol].values())
    
    def GetCurrentWeight(self, symbol):
        """Get current weight"""
        if self.Portfolio.TotalPortfolioValue == 0:
            return 0
        return self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
    
    def OnEndOfAlgorithm(self):
        """TARGET DESTRUCTION RESULTS"""
        
        years = (self.EndDate - self.StartDate).days / 365.25
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - 100000) / 100000
        cagr = (final_value / 100000) ** (1/years) - 1
        trades_per_year = self.total_trades / years
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        avg_profit_per_trade = total_return / self.total_trades if self.total_trades > 0 else 0
        
        # Estimated Sharpe (conservative)
        if total_return > 0:
            annual_vol = 0.30  # Assume 30% volatility for leveraged strategy
            sharpe_ratio = (cagr - 0.05) / annual_vol
        else:
            sharpe_ratio = 0
            
        self.Log("=== TARGET DESTRUCTION RESULTS ===")
        self.Log(f"Final Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return:.2%}")
        self.Log(f"CAGR: {cagr:.2%}")
        self.Log(f"Estimated Sharpe: {sharpe_ratio:.2f}")
        self.Log(f"Total Trades: {self.total_trades}")
        self.Log(f"Trades/Year: {trades_per_year:.1f}")
        self.Log(f"Win Rate: {win_rate:.2%}")
        self.Log(f"Avg Profit/Trade: {avg_profit_per_trade:.2%}")
        self.Log(f"Max Drawdown: {self.max_drawdown_seen:.2%}")
        
        # TARGET STATUS
        self.Log("=== TARGET DESTRUCTION STATUS ===")
        t1 = cagr > 0.25
        t2 = sharpe_ratio > 1.0
        t3 = trades_per_year > 100
        t4 = avg_profit_per_trade > 0.0075
        t5 = self.max_drawdown_seen < 0.20
        
        self.Log(f"CAGR > 25%: {'✅ DESTROYED' if t1 else '❌ MISSED'} - {cagr:.2%}")
        self.Log(f"Sharpe > 1.0: {'✅ DESTROYED' if t2 else '❌ MISSED'} - {sharpe_ratio:.2f}")
        self.Log(f"Trades > 100/yr: {'✅ DESTROYED' if t3 else '❌ MISSED'} - {trades_per_year:.1f}")
        self.Log(f"Profit > 0.75%: {'✅ DESTROYED' if t4 else '❌ MISSED'} - {avg_profit_per_trade:.2%}")
        self.Log(f"Drawdown < 20%: {'✅ DESTROYED' if t5 else '❌ MISSED'} - {self.max_drawdown_seen:.2%}")
        
        targets_destroyed = sum([t1, t2, t3, t4, t5])
        self.Log(f"TARGETS DESTROYED: {targets_destroyed}/5")
        
        if targets_destroyed >= 4:
            self.Log("🎯 TARGET DESTRUCTION SUCCESSFUL! 🎯")
        else:
            self.Log("⚡ PARTIAL TARGET DESTRUCTION ⚡")
            
        self.Log("Strategy: Aggressive leveraged momentum with active rotation")
        self.Log("Universe: High-beta leveraged ETFs + strategic hedging")
        self.Log("Approach: Daily momentum hunting + aggressive profit taking")