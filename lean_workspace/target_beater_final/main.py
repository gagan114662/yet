# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class TargetBeaterFinal(QCAlgorithm):
    """
    TARGET BEATER FINAL - Simplified Extreme Strategy
    Uses maximum leverage with proven momentum and volatility edges
    Targets: CAGR > 25%, Sharpe > 1.0, 100+ trades/year, Profit > 0.75%, DD < 20%
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # EXTREME LEVERAGE UNIVERSE
        # Leveraged ETFs for maximum exposure
        self.add_equity("TQQQ", Resolution.DAILY).set_leverage(4.0)  # 3x NASDAQ * 4x = 12x effective
        self.add_equity("UPRO", Resolution.DAILY).set_leverage(4.0)  # 3x S&P * 4x = 12x effective
        self.add_equity("SOXL", Resolution.DAILY).set_leverage(4.0)  # 3x Semis * 4x = 12x effective
        self.add_equity("TECL", Resolution.DAILY).set_leverage(4.0)  # 3x Tech * 4x = 12x effective
        
        # Inverse ETFs for shorting
        self.add_equity("SQQQ", Resolution.DAILY).set_leverage(4.0)  # -3x NASDAQ * 4x = -12x effective
        self.add_equity("SPXU", Resolution.DAILY).set_leverage(4.0)  # -3x S&P * 4x = -12x effective
        
        # Volatility products for crisis alpha
        self.add_equity("VXX", Resolution.DAILY).set_leverage(6.0)   # 6x VIX exposure
        self.add_equity("UVXY", Resolution.DAILY).set_leverage(4.0)  # 2x VIX * 4x = 8x effective
        self.add_equity("SVXY", Resolution.DAILY).set_leverage(4.0)  # -0.5x VIX * 4x = -2x effective
        
        # Safe haven / alternatives
        self.add_equity("TLT", Resolution.DAILY).set_leverage(6.0)   # Long bonds
        self.add_equity("GLD", Resolution.DAILY).set_leverage(6.0)   # Gold
        
        # Core market
        self.add_equity("SPY", Resolution.DAILY).set_leverage(6.0)   # S&P 500
        self.add_equity("QQQ", Resolution.DAILY).set_leverage(6.0)   # NASDAQ
        
        # Multi-timeframe momentum indicators
        self.symbols = ["TQQQ", "UPRO", "SOXL", "TECL", "SQQQ", "SPXU", "VXX", "UVXY", "SVXY", "TLT", "GLD", "SPY", "QQQ"]
        
        self.momentum_short = {}  # 5-day momentum
        self.momentum_medium = {} # 20-day momentum
        self.momentum_long = {}   # 60-day momentum
        self.rsi_fast = {}        # 3-day RSI
        self.rsi_slow = {}        # 14-day RSI
        self.atr = {}             # Average True Range
        self.bb = {}              # Bollinger Bands
        
        for symbol in self.symbols:
            self.momentum_short[symbol] = self.momp(symbol, 5)
            self.momentum_medium[symbol] = self.momp(symbol, 20)
            self.momentum_long[symbol] = self.momp(symbol, 60)
            self.rsi_fast[symbol] = self.rsi(symbol, 3)
            self.rsi_slow[symbol] = self.rsi(symbol, 14)
            self.atr[symbol] = self.atr(symbol, 14)
            self.bb[symbol] = self.bb(symbol, 20, 2)
        
        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.peak = 100000
        self.max_dd = 0
        self.returns = []
        self.last_value = 100000
        self.positions = {}
        
        # Strategy parameters for target beating
        self.max_positions = 3            # Concentrated positions
        self.base_position_size = 1.0     # 100% positions with extreme leverage
        self.momentum_threshold = 0.02    # 2% momentum threshold
        self.volatility_threshold = 0.3   # 30% VIX threshold for crisis mode
        self.profit_target = 0.025        # 2.5% profit target
        self.stop_loss = 0.015            # 1.5% stop loss
        self.drawdown_limit = 0.18        # 18% max drawdown
        
        # Market regime
        self.market_regime = "NORMAL"     # NORMAL, BULL, BEAR, CRISIS
        self.regime_lookback = 20
        
        # HIGH FREQUENCY SCHEDULES for 100+ trades/year
        
        # Daily momentum scans
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("SPY", 30),
            self.momentum_scan_and_trade
        )
        
        # Midday regime check and rebalance
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("SPY", 120),
            self.regime_check_and_rebalance
        )
        
        # Afternoon momentum boost
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("SPY", 240),
            self.afternoon_momentum_boost
        )
        
        # End of day management
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.before_market_close("SPY", 30),
            self.end_of_day_management
        )
        
        # Weekly regime update
        self.schedule.on(
            self.date_rules.week_start(),
            self.time_rules.after_market_open("SPY", 60),
            self.weekly_regime_update
        )
    
    def momentum_scan_and_trade(self):
        """Core momentum scanning and trading logic"""
        
        # Performance tracking
        current_value = self.portfolio.total_portfolio_value
        if current_value > self.peak:
            self.peak = current_value
            
        drawdown = (self.peak - current_value) / self.peak
        if drawdown > self.max_dd:
            self.max_dd = drawdown
            
        # Emergency protection
        if drawdown > self.drawdown_limit:
            self.liquidate()
            self.trades += 1
            self.positions.clear()
            return
            
        # Track returns
        ret = (current_value - self.last_value) / self.last_value if self.last_value > 0 else 0
        self.returns.append(ret)
        self.last_value = current_value
        
        # Find opportunities based on regime
        if self.market_regime == "CRISIS":
            self.execute_crisis_strategy()
        elif self.market_regime == "BULL":
            self.execute_bull_strategy()
        elif self.market_regime == "BEAR":
            self.execute_bear_strategy()
        else:
            self.execute_normal_strategy()
    
    def execute_crisis_strategy(self):
        """Execute crisis strategy - massive volatility long"""
        
        crisis_instruments = {
            "VXX": 0.6,    # 60% VXX with 6x leverage = 360% exposure
            "UVXY": 0.4,   # 40% UVXY with 4x leverage = 160% exposure
            "TLT": 0.3,    # 30% TLT with 6x leverage = 180% exposure
            "GLD": 0.2     # 20% GLD with 6x leverage = 120% exposure
        }
        
        for symbol, weight in crisis_instruments.items():
            if symbol not in self.positions and self.indicators_ready(symbol):
                self.set_holdings(symbol, weight)
                self.trades += 1
                self.positions[symbol] = {
                    "entry_price": self.securities[symbol].price,
                    "strategy": "CRISIS",
                    "entry_time": self.time
                }
                self.debug(f"CRISIS TRADE: {symbol} at {weight:.1%}")
    
    def execute_bull_strategy(self):
        """Execute bull market strategy - leveraged momentum"""
        
        opportunities = []
        
        for symbol in ["TQQQ", "UPRO", "SOXL", "TECL"]:
            if not self.indicators_ready(symbol):
                continue
                
            momentum_short = self.momentum_short[symbol].current.value
            momentum_medium = self.momentum_medium[symbol].current.value
            momentum_long = self.momentum_long[symbol].current.value
            rsi_fast = self.rsi_fast[symbol].current.value
            rsi_slow = self.rsi_slow[symbol].current.value
            
            # Strong momentum setup
            if (momentum_short > self.momentum_threshold and
                momentum_medium > 0.05 and
                momentum_long > 0.1 and
                rsi_slow < 80):
                
                strength = momentum_short * 50 + momentum_medium * 25 + momentum_long * 10
                opportunities.append({
                    "symbol": symbol,
                    "strength": strength,
                    "momentum": momentum_short
                })\
        
        # Execute top opportunities
        if opportunities:
            opportunities.sort(key=lambda x: x["strength"], reverse=True)
            self.execute_momentum_trades(opportunities[:self.max_positions])
    
    def execute_bear_strategy(self):
        """Execute bear market strategy - short leveraged + safe havens"""
        
        # Short the market
        short_instruments = ["SQQQ", "SPXU"]
        for symbol in short_instruments:
            if symbol not in self.positions and self.indicators_ready(symbol):
                momentum = self.momentum_medium[symbol].current.value
                if momentum > 0.02:  # Inverse ETFs - positive momentum means market falling
                    self.set_holdings(symbol, 0.5)  # 50% with 4x leverage = 200% short exposure
                    self.trades += 1
                    self.positions[symbol] = {
                        "entry_price": self.securities[symbol].price,
                        "strategy": "BEAR_SHORT",
                        "entry_time": self.time
                    }
        
        # Safe havens
        safe_havens = {"TLT": 0.4, "GLD": 0.3}
        for symbol, weight in safe_havens.items():
            if symbol not in self.positions and self.indicators_ready(symbol):
                momentum = self.momentum_medium[symbol].current.value
                if momentum > 0:
                    self.set_holdings(symbol, weight)
                    self.trades += 1
                    self.positions[symbol] = {
                        "entry_price": self.securities[symbol].price,
                        "strategy": "SAFE_HAVEN",
                        "entry_time": self.time
                    }
    
    def execute_normal_strategy(self):
        """Execute normal market strategy - best momentum across all assets"""
        
        all_opportunities = []
        
        for symbol in self.symbols:
            if not self.indicators_ready(symbol):
                continue
                
            momentum_short = self.momentum_short[symbol].current.value
            momentum_medium = self.momentum_medium[symbol].current.value
            rsi_fast = self.rsi_fast[symbol].current.value
            
            if abs(momentum_short) > self.momentum_threshold and 20 < rsi_fast < 80:
                strength = abs(momentum_short) * 100 + abs(momentum_medium) * 50
                direction = 1 if momentum_short > 0 else -1
                
                all_opportunities.append({
                    "symbol": symbol,
                    "strength": strength,
                    "direction": direction,
                    "momentum": momentum_short
                })
        
        # Execute top opportunities
        if all_opportunities:
            all_opportunities.sort(key=lambda x: x["strength"], reverse=True)
            self.execute_directional_trades(all_opportunities[:self.max_positions])
    
    def execute_momentum_trades(self, opportunities):
        """Execute long momentum trades"""
        
        for opp in opportunities:
            symbol = opp["symbol"]
            if symbol not in self.positions:
                # Dynamic position sizing based on momentum strength
                momentum_multiplier = min(1.5, max(0.5, opp["momentum"] * 20))
                position_size = self.base_position_size * momentum_multiplier / len(opportunities)
                
                self.set_holdings(symbol, position_size)
                self.trades += 1
                self.positions[symbol] = {
                    "entry_price": self.securities[symbol].price,
                    "strategy": "MOMENTUM_LONG",
                    "momentum": opp["momentum"],
                    "entry_time": self.time
                }
                self.debug(f"MOMENTUM LONG: {symbol} at {position_size:.1%}")
    
    def execute_directional_trades(self, opportunities):
        """Execute directional trades based on momentum"""
        
        for opp in opportunities:
            symbol = opp["symbol"]
            if symbol not in self.positions:
                position_size = (self.base_position_size / len(opportunities)) * opp["direction"]
                
                self.set_holdings(symbol, position_size)
                self.trades += 1
                self.positions[symbol] = {
                    "entry_price": self.securities[symbol].price,
                    "strategy": "DIRECTIONAL",
                    "direction": opp["direction"],
                    "entry_time": self.time
                }
                direction_text = "LONG" if opp["direction"] > 0 else "SHORT"
                self.debug(f"DIRECTIONAL {direction_text}: {symbol} at {position_size:.1%}")
    
    def regime_check_and_rebalance(self):
        """Midday regime check and position rebalancing"""
        
        # Update market regime
        self.update_market_regime()
        
        # Rebalance if regime changed significantly
        current_positions = len(self.positions)
        if current_positions < self.max_positions:
            # Look for additional opportunities
            self.momentum_scan_and_trade()
    
    def afternoon_momentum_boost(self):
        """Afternoon momentum amplification"""
        
        # Look for late-day momentum acceleration
        for symbol in list(self.positions.keys()):
            if self.portfolio[symbol].invested and self.indicators_ready(symbol):
                position = self.positions[symbol]
                current_momentum = self.momentum_short[symbol].current.value
                
                # Amplify winning momentum positions
                if ("MOMENTUM" in position["strategy"] and 
                    current_momentum > position.get("momentum", 0) * 1.5):
                    
                    current_weight = self.portfolio[symbol].holdings_value / self.portfolio.total_portfolio_value
                    if abs(current_weight) < 0.8:  # Don't exceed 80% in any position
                        new_weight = current_weight * 1.2  # 20% increase
                        self.set_holdings(symbol, new_weight)
                        self.trades += 1
                        self.debug(f"MOMENTUM BOOST: {symbol} to {new_weight:.1%}")
    
    def end_of_day_management(self):
        """End of day profit taking and risk management"""
        
        for symbol in list(self.positions.keys()):
            if not self.portfolio[symbol].invested:
                if symbol in self.positions:
                    del self.positions[symbol]
                continue
                
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            current_price = self.securities[symbol].price
            
            if entry_price <= 0:
                continue
                
            pnl = (current_price - entry_price) / entry_price
            
            # Adjust for short positions
            if self.portfolio[symbol].is_short:
                pnl = -pnl
            
            # Strategy-specific exits
            strategy = position["strategy"]
            
            if strategy == "CRISIS":
                # Let crisis trades run for explosive profits
                if pnl > 0.3:  # 30% profit
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                    del self.positions[symbol]
                elif pnl < -0.12:  # 12% stop
                    self.liquidate(symbol)
                    self.trades += 1
                    self.losses += 1
                    del self.positions[symbol]
            
            else:
                # Standard exits for other strategies
                if pnl > self.profit_target:
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                    del self.positions[symbol]
                elif pnl < -self.stop_loss:
                    self.liquidate(symbol)
                    self.trades += 1
                    self.losses += 1
                    del self.positions[symbol]
    
    def weekly_regime_update(self):
        """Weekly comprehensive regime update"""
        
        self.update_market_regime()
        
        # Clear positions if regime changed dramatically
        if hasattr(self, 'last_regime'):
            if ((self.last_regime == "NORMAL" and self.market_regime == "CRISIS") or
                (self.last_regime == "BULL" and self.market_regime == "BEAR") or
                (self.last_regime == "CRISIS" and self.market_regime == "BULL")):
                
                # Major regime change - clear all positions
                for symbol in list(self.positions.keys()):
                    if self.portfolio[symbol].invested:
                        self.liquidate(symbol)
                        self.trades += 1
                self.positions.clear()
                self.debug(f"REGIME CHANGE: {self.last_regime} -> {self.market_regime}")
        
        self.last_regime = self.market_regime
    
    def update_market_regime(self):
        """Update market regime based on multiple indicators"""
        
        if not self.indicators_ready("SPY") or not self.indicators_ready("VXX"):
            return
            
        # VIX level for crisis detection
        vxx_price = self.securities["VXX"].price
        if vxx_price > 35:  # VIX equivalent above 35
            self.market_regime = "CRISIS"
            return
            
        # SPY momentum for bull/bear
        spy_momentum_short = self.momentum_short["SPY"].current.value
        spy_momentum_medium = self.momentum_medium["SPY"].current.value
        spy_momentum_long = self.momentum_long["SPY"].current.value
        
        # Bull market
        if (spy_momentum_short > 0.02 and 
            spy_momentum_medium > 0.05 and 
            spy_momentum_long > 0.08):
            self.market_regime = "BULL"
        
        # Bear market
        elif (spy_momentum_short < -0.02 and 
              spy_momentum_medium < -0.03 and 
              spy_momentum_long < -0.05):
            self.market_regime = "BEAR"
        
        # Normal market
        else:
            self.market_regime = "NORMAL"
    
    def indicators_ready(self, symbol):
        """Check if all indicators are ready for a symbol"""
        return (symbol in self.momentum_short and self.momentum_short[symbol].is_ready and
                symbol in self.momentum_medium and self.momentum_medium[symbol].is_ready and
                symbol in self.momentum_long and self.momentum_long[symbol].is_ready and
                symbol in self.rsi_fast and self.rsi_fast[symbol].is_ready and
                symbol in self.rsi_slow and self.rsi_slow[symbol].is_ready)
    
    def on_end_of_algorithm(self):
        """Final target beating validation"""
        
        years = (self.end_date - self.start_date).days / 365.25
        final_value = self.portfolio.total_portfolio_value
        total_return = (final_value - 100000) / 100000
        cagr = (final_value / 100000) ** (1/years) - 1
        trades_per_year = self.trades / years
        
        # Calculate metrics
        total_decided = self.wins + self.losses
        win_rate = self.wins / total_decided if total_decided > 0 else 0
        avg_profit_per_trade = total_return / self.trades if self.trades > 0 else 0
        
        # Sharpe ratio calculation
        if len(self.returns) > 100:
            returns_array = np.array(self.returns[-252*5:])  # Last 5 years
            if len(returns_array) > 50:
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                if std_return > 0:
                    sharpe = (mean_return / std_return) * np.sqrt(252)
                else:
                    sharpe = 0
            else:
                sharpe = 0
        else:
            sharpe = 0
            
        self.log("=== TARGET BEATER FINAL RESULTS ===")
        self.log(f"Final Value: ${final_value:,.2f}")
        self.log(f"Total Return: {total_return:.2%}")
        self.log(f"CAGR: {cagr:.2%}")
        self.log(f"Sharpe Ratio: {sharpe:.2f}")
        self.log(f"Total Trades: {self.trades}")
        self.log(f"Trades/Year: {trades_per_year:.1f}")
        self.log(f"Win Rate: {win_rate:.2%}")
        self.log(f"Avg Profit/Trade: {avg_profit_per_trade:.2%}")
        self.log(f"Max Drawdown: {self.max_dd:.2%}")
        
        # TARGET VALIDATION
        self.log("=== TARGET ACHIEVEMENT VALIDATION ===")
        target_1 = cagr > 0.25
        target_2 = sharpe > 1.0
        target_3 = trades_per_year > 100
        target_4 = avg_profit_per_trade > 0.0075
        target_5 = self.max_dd < 0.20
        
        self.log(f"TARGET 1 - CAGR > 25%: {'ACHIEVED' if target_1 else 'FAILED'} - {cagr:.2%}")
        self.log(f"TARGET 2 - Sharpe > 1.0: {'ACHIEVED' if target_2 else 'FAILED'} - {sharpe:.2f}")
        self.log(f"TARGET 3 - Trades > 100/yr: {'ACHIEVED' if target_3 else 'FAILED'} - {trades_per_year:.1f}")
        self.log(f"TARGET 4 - Profit > 0.75%: {'ACHIEVED' if target_4 else 'FAILED'} - {avg_profit_per_trade:.2%}")
        self.log(f"TARGET 5 - Drawdown < 20%: {'ACHIEVED' if target_5 else 'FAILED'} - {self.max_dd:.2%}")
        
        targets_achieved = sum([target_1, target_2, target_3, target_4, target_5])
        self.log(f"TOTAL TARGETS ACHIEVED: {targets_achieved}/5")
        
        if targets_achieved == 5:
            self.log("ALL 5 TARGETS ACHIEVED - MISSION ACCOMPLISHED!")
        elif targets_achieved >= 4:
            self.log("EXCELLENT PERFORMANCE - 4/5 TARGETS ACHIEVED!")
        elif targets_achieved >= 3:
            self.log("GOOD PERFORMANCE - 3/5 TARGETS ACHIEVED!")
        else:
            self.log("PERFORMANCE BELOW EXPECTATIONS")
            
        self.log("=== STRATEGY SUMMARY ===")
        self.log("Extreme leverage strategy with regime adaptation:")
        self.log("- Up to 12x effective leverage on 3x leveraged ETFs")
        self.log("- Crisis alpha with massive VIX exposure (360%+ during stress)")
        self.log("- Multi-regime positioning (Bull/Bear/Crisis/Normal)")
        self.log("- High-frequency trading with 4+ daily scans")
        self.log("- Dynamic position sizing based on momentum strength")
        self.log("- 18% drawdown protection with emergency liquidation")
        
        if targets_achieved >= 4:
            self.log("TARGET BEATER FINAL: SUCCESS!")