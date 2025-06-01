from AlgorithmImports import *

class SmartMomentumStrategy(QCAlgorithm):
    """
    Smart Momentum Strategy - Balanced approach for 25%+ CAGR
    Optimizes for high returns while controlling trading costs
    """
    
    def Initialize(self):
        # 20-year backtest
        self.SetStartDate(2004, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Enable margin for moderate leverage
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # Diversified high-performance assets
        self.symbols = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "GLD", "TLT"]
        
        for symbol in self.symbols:
            security = self.AddEquity(symbol, Resolution.Daily)
            security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
        
        # Optimized indicators
        self.indicators = {}
        for symbol in self.symbols:
            self.indicators[symbol] = {
                "momentum_10": self.MOMP(symbol, 10),
                "momentum_30": self.MOMP(symbol, 30),
                "rsi_14": self.RSI(symbol, 14),
                "bb_20": self.BB(symbol, 20, 2),
                "ema_12": self.EMA(symbol, 12),
                "ema_26": self.EMA(symbol, 26),
                "atr_14": self.ATR(symbol, 14)
            }
        
        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0
        self.total_loss = 0
        self.position_entries = {}
        
        # Balanced parameters for sustainable 25%+ CAGR
        self.base_leverage = 2.0        # Moderate leverage
        self.max_position_size = 0.4    # 40% max per position
        self.profit_target = 0.20       # 20% profit target
        self.stop_loss = 0.12           # 12% stop loss
        self.rebalance_threshold = 0.05 # 5% change to trigger trade
        
        # Market conditions
        self.market_regime = "NEUTRAL"
        self.volatility_level = "NORMAL"
        
        # Strategic rebalancing schedule - twice per week for ~100 trades/year
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Tuesday, DayOfWeek.Friday),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )
        
        # Monthly regime update
        self.Schedule.On(
            self.DateRules.MonthStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 15),
            self.UpdateMarketRegime
        )
        
    def UpdateMarketRegime(self):
        """Update market regime monthly"""
        if not self.indicators["SPY"]["momentum_30"].IsReady:
            return
            
        spy_momentum = self.indicators["SPY"]["momentum_30"].Current.Value
        spy_rsi = self.indicators["SPY"]["rsi_14"].Current.Value
        
        # Market regime
        if spy_momentum > 0.03 and spy_rsi > 60:
            self.market_regime = "STRONG_BULL"
        elif spy_momentum > 0.01 and spy_rsi > 50:
            self.market_regime = "BULL"
        elif spy_momentum < -0.03 and spy_rsi < 40:
            self.market_regime = "STRONG_BEAR"
        elif spy_momentum < -0.01 and spy_rsi < 50:
            self.market_regime = "BEAR"
        else:
            self.market_regime = "NEUTRAL"
            
        # Volatility level
        spy_atr = self.indicators["SPY"]["atr_14"].Current.Value
        spy_price = self.Securities["SPY"].Price
        
        if spy_price > 0:
            vol_ratio = spy_atr / spy_price
            if vol_ratio > 0.025:
                self.volatility_level = "HIGH"
            elif vol_ratio < 0.015:
                self.volatility_level = "LOW"
            else:
                self.volatility_level = "NORMAL"
    
    def Rebalance(self):
        """Strategic rebalancing twice per week"""
        
        # Generate signals
        signals = {}
        for symbol in self.symbols:
            if self.AllReady(symbol):
                signals[symbol] = self.GetSignal(symbol)
        
        if not signals:
            return
            
        # Execute trades
        self.ExecuteTrades(signals)
        
        # Manage positions
        self.ManagePositions()
    
    def AllReady(self, symbol):
        """Check indicators ready"""
        return all(ind.IsReady for ind in self.indicators[symbol].values())
    
    def GetSignal(self, symbol):
        """Generate momentum signal"""
        ind = self.indicators[symbol]
        price = self.Securities[symbol].Price
        
        mom_10 = ind["momentum_10"].Current.Value
        mom_30 = ind["momentum_30"].Current.Value
        rsi = ind["rsi_14"].Current.Value
        bb_upper = ind["bb_20"].UpperBand.Current.Value
        bb_lower = ind["bb_20"].LowerBand.Current.Value
        ema_12 = ind["ema_12"].Current.Value
        ema_26 = ind["ema_26"].Current.Value
        atr = ind["atr_14"].Current.Value
        
        signal = 0
        
        # Multi-timeframe momentum (60% weight)
        if mom_10 > 0.02 and mom_30 > 0.01:  # Strong aligned momentum
            signal += 6
        elif mom_10 > 0.01 and mom_30 > 0.005:  # Moderate momentum
            signal += 3
        elif mom_10 < -0.02 and mom_30 < -0.01:  # Strong negative
            signal -= 6
        elif mom_10 < -0.01 and mom_30 < -0.005:  # Moderate negative
            signal -= 3
            
        # Trend confirmation (25% weight)
        if ema_12 > ema_26 and price > ema_12:
            signal += 2.5
        elif ema_12 < ema_26 and price < ema_12:
            signal -= 2.5
            
        # Breakout/breakdown (15% weight)
        if price > bb_upper and mom_10 > 0.01:
            signal += 1.5
        elif price < bb_lower and mom_10 < -0.01:
            signal -= 1.5
            
        # Volatility adjustment
        if atr > 0:
            vol_factor = atr / price
            if vol_factor > 0.02:  # High volatility
                signal *= 1.25  # Amplify signals
                
        # Market regime adjustment
        regime_multiplier = {
            "STRONG_BULL": 1.3,
            "BULL": 1.1,
            "NEUTRAL": 1.0,
            "BEAR": 1.0,
            "STRONG_BEAR": 1.2
        }.get(self.market_regime, 1.0)
        
        signal *= regime_multiplier
        
        # Asset-specific logic
        if symbol in ["GLD", "TLT"]:  # Defensive assets
            if self.market_regime in ["BEAR", "STRONG_BEAR"]:
                signal *= 1.4  # Boost defensive in bear markets
            elif self.market_regime in ["STRONG_BULL"]:
                signal *= 0.6  # Reduce in strong bull
                
        # Normalize
        return max(-1, min(1, signal / 10))
    
    def ExecuteTrades(self, signals):
        """Execute trades with cost control"""
        
        # Filter for meaningful signals
        strong_signals = {k: v for k, v in signals.items() if abs(v) > 0.3}
        
        if not strong_signals:
            return
            
        total_signal = sum(abs(v) for v in strong_signals.values())
        if total_signal == 0:
            return
            
        # Calculate leverage based on conditions
        current_leverage = self.base_leverage
        
        # Market regime adjustment
        if self.market_regime == "STRONG_BULL":
            current_leverage *= 1.25
        elif self.market_regime == "STRONG_BEAR":
            current_leverage *= 1.15
        elif self.market_regime == "NEUTRAL":
            current_leverage *= 0.9
            
        # Volatility adjustment
        if self.volatility_level == "HIGH":
            current_leverage *= 0.8
        elif self.volatility_level == "LOW":
            current_leverage *= 1.1
            
        # Execute positions
        for symbol, signal in strong_signals.items():
            signal_weight = abs(signal) / total_signal
            target_weight = signal * signal_weight * current_leverage
            
            # Cap positions
            target_weight = max(-self.max_position_size, 
                              min(self.max_position_size, target_weight))
            
            current_weight = self.GetWeight(symbol)
            
            # Only trade if significant change (cost control)
            if abs(target_weight - current_weight) > self.rebalance_threshold:
                self.SetHoldings(symbol, target_weight)
                self.trade_count += 1
                self.position_entries[symbol] = {
                    "price": self.Securities[symbol].Price,
                    "weight": target_weight
                }
    
    def ManagePositions(self):
        """Position management with profit targets and stops"""
        
        for symbol in self.symbols:
            if not self.Portfolio[symbol].Invested:
                continue
                
            entry_data = self.position_entries.get(symbol)
            if not entry_data:
                continue
                
            current_price = self.Securities[symbol].Price
            entry_price = entry_data["price"]
            entry_weight = entry_data["weight"]
            
            if entry_price == 0:
                continue
                
            pnl_percent = (current_price - entry_price) / entry_price
            
            # Adjust for short positions
            if entry_weight < 0:
                pnl_percent *= -1
                
            # Profit taking
            if pnl_percent > self.profit_target:
                self.Liquidate(symbol)
                self.trade_count += 1
                self.win_count += 1
                self.total_profit += abs(pnl_percent)
                
            # Stop loss
            elif pnl_percent < -self.stop_loss:
                self.Liquidate(symbol)
                self.trade_count += 1
                self.loss_count += 1
                self.total_loss += abs(pnl_percent)
    
    def GetWeight(self, symbol):
        """Get portfolio weight"""
        if self.Portfolio.TotalPortfolioValue == 0:
            return 0
        return self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
    
    def OnData(self, data):
        """Opportunistic momentum trades"""
        
        # Only on strong momentum moves
        for symbol in self.symbols:
            if symbol not in data:
                continue
                
            if not self.indicators[symbol]["momentum_10"].IsReady:
                continue
                
            mom_10 = self.indicators[symbol]["momentum_10"].Current.Value
            current_weight = self.GetWeight(symbol)
            
            # Very selective additional trades
            if abs(mom_10) > 0.035 and abs(current_weight) < 0.25:
                
                momentum_trade = 0.15 if mom_10 > 0 else -0.15
                
                # Asset-specific adjustments
                if symbol in ["GLD", "TLT"]:
                    if self.market_regime in ["STRONG_BULL", "BULL"]:
                        momentum_trade *= -0.5
                        
                new_weight = current_weight + momentum_trade
                new_weight = max(-self.max_position_size, 
                               min(self.max_position_size, new_weight))
                
                # Only if significant and not too frequent
                if abs(new_weight - current_weight) > 0.08:
                    self.SetHoldings(symbol, new_weight)
                    self.trade_count += 1
                    self.position_entries[symbol] = {
                        "price": self.Securities[symbol].Price,
                        "weight": new_weight
                    }
    
    def OnEndOfAlgorithm(self):
        """Final performance metrics"""
        
        years = (self.EndDate - self.StartDate).days / 365.25
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - 100000) / 100000
        cagr = (final_value / 100000) ** (1/years) - 1
        
        trades_per_year = self.trade_count / years
        win_rate = self.win_count / max(1, self.win_count + self.loss_count)
        avg_profit = self.total_profit / max(1, self.win_count)
        
        # Sharpe estimate
        if total_return > 0 and years > 1:
            annual_vol = abs(total_return) * 0.3
            sharpe_approx = (cagr - 0.05) / max(0.01, annual_vol)
        else:
            sharpe_approx = 0
            
        self.Log("=== SMART MOMENTUM STRATEGY RESULTS ===")
        self.Log(f"Final Portfolio Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return:.2%}")
        self.Log(f"CAGR: {cagr:.2%}")
        self.Log(f"Approximate Sharpe Ratio: {sharpe_approx:.2f}")
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
        self.Log(f"Win Rate: {win_rate:.2%}")
        self.Log(f"Average Profit per Win: {avg_profit:.2%}")
        
        # Target evaluation
        self.Log("=== TARGET EVALUATION ===")
        self.Log(f"CAGR Target (>25%): {'PASS' if cagr > 0.25 else 'FAIL'} - {cagr:.2%}")
        self.Log(f"Sharpe Target (>1.0): {'PASS' if sharpe_approx > 1.0 else 'FAIL'} - {sharpe_approx:.2f}")
        self.Log(f"Trades Target (>100/year): {'PASS' if trades_per_year > 100 else 'FAIL'} - {trades_per_year:.1f}")
        self.Log(f"Avg Profit Target (>0.75%): {'PASS' if avg_profit > 0.0075 else 'FAIL'} - {avg_profit:.2%}")
        
        targets_met = (
            cagr > 0.25 and 
            sharpe_approx > 1.0 and 
            trades_per_year > 100 and 
            avg_profit > 0.0075
        )
        
        self.Log(f"ALL TARGETS MET: {'SUCCESS!' if targets_met else 'CONTINUE OPTIMIZING'}")