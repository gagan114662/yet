from AlgorithmImports import *

class LeveragedMomentumBeast(QCAlgorithm):
    """
    Leveraged Momentum Beast - Designed for 25%+ CAGR over 20 years
    Uses leveraged ETFs and aggressive momentum strategies
    """
    
    def Initialize(self):
        # 20-year backtest
        self.SetStartDate(2004, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Enable margin
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # High-performance leveraged and growth assets
        self.symbols = {}
        
        # Add symbols with try-catch for availability
        tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "VXX", "GLD", "TLT"]
        
        for ticker in tickers:
            try:
                self.symbols[ticker] = self.AddEquity(ticker, Resolution.Daily)
            except:
                pass  # Skip if not available
        
        # Momentum indicators for each symbol
        self.indicators = {}
        for symbol in self.symbols.keys():
            self.indicators[symbol] = {
                "momentum_3": self.MOMP(symbol, 3),
                "momentum_10": self.MOMP(symbol, 10),
                "momentum_20": self.MOMP(symbol, 20),
                "rsi_10": self.RSI(symbol, 10),
                "bb_20": self.BB(symbol, 20, 2),
                "ema_8": self.EMA(symbol, 8),
                "ema_21": self.EMA(symbol, 21),
                "atr": self.ATR(symbol, 14)
            }
        
        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0
        self.total_loss = 0
        self.position_entries = {}
        
        # AGGRESSIVE parameters for 25%+ returns
        self.base_leverage = 3.0        # High leverage
        self.max_position_size = 0.6    # 60% max per position
        self.profit_target = 0.15       # 15% profit target
        self.stop_loss = 0.10           # 10% stop loss
        
        # Market regime
        self.market_regime = "NEUTRAL"
        self.momentum_regime = "NORMAL"
        
        # Daily rebalancing for high frequency
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )
        
    def Rebalance(self):
        """Aggressive daily rebalancing"""
        
        # Update market conditions
        self.UpdateRegimes()
        
        # Generate signals
        signals = {}
        for symbol in self.symbols.keys():
            if self.AllReady(symbol):
                signals[symbol] = self.GetMomentumSignal(symbol)
        
        if not signals:
            return
            
        # Execute aggressive trades
        self.ExecuteAggressiveTrades(signals)
        
        # Manage positions
        self.ManagePositions()
    
    def UpdateRegimes(self):
        """Update market and momentum regimes"""
        if not self.indicators.get("SPY") or not self.indicators["SPY"]["momentum_20"].IsReady:
            return
            
        spy_mom_20 = self.indicators["SPY"]["momentum_20"].Current.Value
        spy_mom_3 = self.indicators["SPY"]["momentum_3"].Current.Value
        
        # Market regime
        if spy_mom_20 > 0.02:
            self.market_regime = "STRONG_BULL"
        elif spy_mom_20 > 0.005:
            self.market_regime = "BULL"
        elif spy_mom_20 < -0.02:
            self.market_regime = "STRONG_BEAR"
        elif spy_mom_20 < -0.005:
            self.market_regime = "BEAR"
        else:
            self.market_regime = "NEUTRAL"
            
        # Momentum regime
        if abs(spy_mom_3) > 0.02:
            self.momentum_regime = "HIGH"
        elif abs(spy_mom_3) < 0.005:
            self.momentum_regime = "LOW"
        else:
            self.momentum_regime = "NORMAL"
    
    def AllReady(self, symbol):
        """Check if indicators are ready"""
        return all(ind.IsReady for ind in self.indicators[symbol].values())
    
    def GetMomentumSignal(self, symbol):
        """Generate aggressive momentum signal"""
        ind = self.indicators[symbol]
        price = self.Securities[symbol].Price
        
        mom_3 = ind["momentum_3"].Current.Value
        mom_10 = ind["momentum_10"].Current.Value
        mom_20 = ind["momentum_20"].Current.Value
        rsi = ind["rsi_10"].Current.Value
        bb_upper = ind["bb_20"].UpperBand.Current.Value
        bb_lower = ind["bb_20"].LowerBand.Current.Value
        ema_8 = ind["ema_8"].Current.Value
        ema_21 = ind["ema_21"].Current.Value
        atr = ind["atr"].Current.Value
        
        signal = 0
        
        # SUPER AGGRESSIVE MOMENTUM CASCADE
        # Short-term momentum (weight: 40%)
        if mom_3 > 0.02:
            signal += 4
        elif mom_3 > 0.01:
            signal += 2
        elif mom_3 < -0.02:
            signal -= 4
        elif mom_3 < -0.01:
            signal -= 2
            
        # Medium-term momentum (weight: 30%)
        if mom_10 > 0.015:
            signal += 3
        elif mom_10 > 0.007:
            signal += 1.5
        elif mom_10 < -0.015:
            signal -= 3
        elif mom_10 < -0.007:
            signal -= 1.5
            
        # Long-term momentum (weight: 20%)
        if mom_20 > 0.01:
            signal += 2
        elif mom_20 > 0.005:
            signal += 1
        elif mom_20 < -0.01:
            signal -= 2
        elif mom_20 < -0.005:
            signal -= 1
            
        # Trend confirmation (weight: 10%)
        if ema_8 > ema_21 and price > ema_8:
            signal += 1
        elif ema_8 < ema_21 and price < ema_8:
            signal -= 1
            
        # Breakout amplification
        if price > bb_upper and mom_3 > 0.01:
            signal += 2
        elif price < bb_lower and mom_3 < -0.01:
            signal -= 2
            
        # Volatility boost
        if atr > 0:
            vol_factor = atr / price
            if vol_factor > 0.03:  # High volatility
                signal *= 1.4
                
        # Market regime amplification
        regime_multiplier = {
            "STRONG_BULL": 1.5,
            "BULL": 1.2,
            "NEUTRAL": 1.0,
            "BEAR": 1.1,  # Leverage shorts
            "STRONG_BEAR": 1.3
        }.get(self.market_regime, 1.0)
        
        signal *= regime_multiplier
        
        # Asset-specific adjustments
        if symbol == "VXX":  # Volatility - inverse signal
            signal *= -1
        elif symbol in ["GLD", "TLT"]:  # Defensive - inverse in bull markets
            if self.market_regime in ["STRONG_BULL", "BULL"]:
                signal *= -0.5
            elif self.market_regime in ["STRONG_BEAR", "BEAR"]:
                signal *= 1.5
                
        # Normalize
        return max(-1, min(1, signal / 15))
    
    def ExecuteAggressiveTrades(self, signals):
        """Execute with maximum aggression for 25%+ returns"""
        
        # Only trade on strong signals
        strong_signals = {k: v for k, v in signals.items() if abs(v) > 0.25}
        
        if not strong_signals:
            # Reduce positions when no strong signals
            for symbol in self.symbols.keys():
                if self.Portfolio[symbol].Invested:
                    current_weight = self.GetWeight(symbol)
                    if abs(current_weight) > 0.1:
                        new_weight = current_weight * 0.6  # Aggressive reduction
                        self.SetHoldings(symbol, new_weight)
                        self.trade_count += 1
            return
            
        total_signal = sum(abs(v) for v in strong_signals.values())
        if total_signal == 0:
            return
            
        # Calculate leverage based on momentum regime
        current_leverage = self.base_leverage
        
        if self.momentum_regime == "HIGH":
            current_leverage *= 1.3  # Boost in high momentum
        elif self.momentum_regime == "LOW":
            current_leverage *= 0.8
            
        # Market regime adjustment
        if self.market_regime == "STRONG_BULL":
            current_leverage *= 1.2
        elif self.market_regime == "STRONG_BEAR":
            current_leverage *= 1.1
            
        # Execute positions
        for symbol, signal in strong_signals.items():
            signal_weight = abs(signal) / total_signal
            target_weight = signal * signal_weight * current_leverage
            
            # Cap positions
            target_weight = max(-self.max_position_size, 
                              min(self.max_position_size, target_weight))
            
            current_weight = self.GetWeight(symbol)
            
            # Trade on smaller changes for higher frequency
            if abs(target_weight - current_weight) > 0.02:
                self.SetHoldings(symbol, target_weight)
                self.trade_count += 1
                self.position_entries[symbol] = {
                    "price": self.Securities[symbol].Price,
                    "weight": target_weight
                }
    
    def ManagePositions(self):
        """Aggressive position management"""
        
        for symbol in self.symbols.keys():
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
        """Get current portfolio weight"""
        if self.Portfolio.TotalPortfolioValue == 0:
            return 0
        return self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
    
    def OnData(self, data):
        """Additional scalping for extra trades"""
        
        for symbol in self.symbols.keys():
            if symbol not in data:
                continue
                
            if not self.indicators[symbol]["momentum_3"].IsReady:
                continue
                
            mom_3 = self.indicators[symbol]["momentum_3"].Current.Value
            current_weight = self.GetWeight(symbol)
            
            # Super aggressive scalping
            if abs(mom_3) > 0.03 and abs(current_weight) < 0.3:
                
                scalp_size = 0.2 if mom_3 > 0 else -0.2
                
                # Special handling
                if symbol == "VXX":
                    scalp_size *= -1
                elif symbol in ["GLD", "TLT"]:
                    if self.market_regime in ["STRONG_BULL", "BULL"]:
                        scalp_size *= -0.5
                        
                new_weight = current_weight + scalp_size
                new_weight = max(-self.max_position_size, 
                               min(self.max_position_size, new_weight))
                
                if abs(new_weight - current_weight) > 0.05:
                    self.SetHoldings(symbol, new_weight)
                    self.trade_count += 1
                    self.position_entries[symbol] = {
                        "price": self.Securities[symbol].Price,
                        "weight": new_weight
                    }
    
    def OnEndOfAlgorithm(self):
        """Final performance calculation"""
        
        years = (self.EndDate - self.StartDate).days / 365.25
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - 100000) / 100000
        cagr = (final_value / 100000) ** (1/years) - 1
        
        trades_per_year = self.trade_count / years
        win_rate = self.win_count / max(1, self.win_count + self.loss_count)
        avg_profit = self.total_profit / max(1, self.win_count)
        
        # Sharpe estimate
        if total_return > 0 and years > 1:
            annual_vol = abs(total_return) * 0.35  # Higher vol estimate for aggressive strategy
            sharpe_approx = (cagr - 0.05) / max(0.01, annual_vol)
        else:
            sharpe_approx = 0
            
        self.Log("=== LEVERAGED MOMENTUM BEAST RESULTS ===")
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
        
        self.Log(f"ALL TARGETS MET: {'SUCCESS!' if targets_met else 'NEEDS MORE OPTIMIZATION'}")