# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
# endregion

class RegimeAdaptiveMaster(QCAlgorithm):
    """
    REGIME ADAPTIVE MASTER - Changes Strategy Based on Market Conditions
    Combines momentum in bull markets, mean reversion in ranges, defensive in bears
    Targets: CAGR > 25%, Sharpe > 1.0, 100+ trades/year, Profit > 0.75%, DD < 20%
    """

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # Core universe
        self.universe = {
            # Aggressive growth
            "TQQQ": {"type": "aggressive", "leverage": 2.0},  # 3x NASDAQ
            "UPRO": {"type": "aggressive", "leverage": 2.0},  # 3x S&P
            "SOXL": {"type": "aggressive", "leverage": 2.0},  # 3x Semis
            # Core holdings  
            "SPY": {"type": "core", "leverage": 3.0},         # S&P 500
            "QQQ": {"type": "core", "leverage": 3.0},         # NASDAQ
            "IWM": {"type": "core", "leverage": 3.0},         # Small caps
            # Defensive
            "TLT": {"type": "defensive", "leverage": 3.0},    # Long bonds
            "GLD": {"type": "defensive", "leverage": 3.0},    # Gold
            "UUP": {"type": "defensive", "leverage": 3.0},    # US Dollar
            # Volatility
            "VXX": {"type": "volatility", "leverage": 2.0},   # VIX futures
        }
        
        # Add securities
        for symbol, config in self.universe.items():
            equity = self.add_equity(symbol, Resolution.MINUTE)
            equity.set_leverage(config["leverage"])
        
        # Market regime indicators
        self.spy_sma_50 = self.sma("SPY", 50, Resolution.DAILY)
        self.spy_sma_200 = self.sma("SPY", 200, Resolution.DAILY)
        self.vix_sma = self.sma("VXX", 20, Resolution.DAILY)
        self.market_atr = self.atr("SPY", 14, Resolution.DAILY)
        
        # Regime detection
        self.regime_lookback = 20
        self.spy_returns = deque(maxlen=self.regime_lookback)
        self.volatility_readings = deque(maxlen=self.regime_lookback)
        self.current_regime = "NEUTRAL"
        self.regime_history = deque(maxlen=5)
        
        # Strategy-specific indicators
        self.indicators = {}
        for symbol in self.universe.keys():
            self.indicators[symbol] = {
                "momentum_5": self.momp(symbol, 5, Resolution.DAILY),
                "momentum_20": self.momp(symbol, 20, Resolution.DAILY),
                "rsi_14": self.rsi(symbol, 14, Resolution.DAILY),
                "bb_20": self.bb(symbol, 20, 2, Resolution.DAILY),
                "ema_10": self.ema(symbol, 10, Resolution.DAILY),
                "atr_14": self.atr(symbol, 14, Resolution.DAILY)
            }
        
        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.peak = 100000
        self.max_dd = 0
        self.daily_returns = []
        self.last_value = 100000
        self.positions = {}
        
        # Regime-specific parameters
        self.regime_params = {
            "BULL": {
                "strategy": "momentum",
                "position_size": 0.8,
                "profit_target": 0.025,
                "stop_loss": 0.015,
                "preferred_assets": ["aggressive", "core"]
            },
            "BEAR": {
                "strategy": "defensive",
                "position_size": 0.4,
                "profit_target": 0.015,
                "stop_loss": 0.008,
                "preferred_assets": ["defensive", "volatility"]
            },
            "RANGE": {
                "strategy": "mean_reversion",
                "position_size": 0.6,
                "profit_target": 0.012,
                "stop_loss": 0.006,
                "preferred_assets": ["core", "defensive"]
            },
            "HIGH_VOL": {
                "strategy": "volatility_harvest",
                "position_size": 0.5,
                "profit_target": 0.02,
                "stop_loss": 0.01,
                "preferred_assets": ["volatility", "defensive"]
            }
        }
        
        # High-frequency regime detection and trading
        for minutes in range(30, 390, 30):  # Every 30 minutes
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", minutes),
                self.adaptive_trading_scan
            )
            
        # Regime update every hour
        for hour in [60, 120, 180, 240, 300]:
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", hour),
                self.update_market_regime
            )
    
    def update_market_regime(self):
        """Sophisticated regime detection"""
        
        if not self.regime_indicators_ready():
            return
            
        # Get current metrics
        spy_price = self.securities["SPY"].price
        spy_sma_50 = self.spy_sma_50.current.value
        spy_sma_200 = self.spy_sma_200.current.value
        vix_level = self.securities["VXX"].price
        vix_sma = self.vix_sma.current.value
        market_atr = self.market_atr.current.value
        
        # Track returns
        if len(self.spy_returns) > 0:
            last_price = self.spy_returns[-1]["price"]
            daily_return = (spy_price - last_price) / last_price
        else:
            daily_return = 0
            
        self.spy_returns.append({"price": spy_price, "return": daily_return})
        
        # Calculate regime metrics
        trend_strength = (spy_price - spy_sma_200) / spy_sma_200
        short_trend = (spy_price - spy_sma_50) / spy_sma_50
        volatility_regime = vix_level / vix_sma if vix_sma > 0 else 1
        
        # Calculate realized volatility
        if len(self.spy_returns) >= 10:
            returns = [r["return"] for r in self.spy_returns]
            realized_vol = np.std(returns) * np.sqrt(252)
        else:
            realized_vol = 0.15
            
        self.volatility_readings.append(realized_vol)
        
        # Determine regime
        previous_regime = self.current_regime
        
        if trend_strength > 0.05 and short_trend > 0.02 and volatility_regime < 1.2:
            self.current_regime = "BULL"
        elif trend_strength < -0.05 and short_trend < -0.02:
            self.current_regime = "BEAR"
        elif realized_vol > 0.25 or volatility_regime > 1.5:
            self.current_regime = "HIGH_VOL"
        else:
            self.current_regime = "RANGE"
            
        # Track regime changes
        if self.current_regime != previous_regime:
            self.regime_history.append(self.current_regime)
            self.debug(f"REGIME CHANGE: {previous_regime} -> {self.current_regime}")
            
            # Clear positions on regime change
            for symbol in list(self.positions.keys()):
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.trades += 1
            self.positions.clear()
    
    def adaptive_trading_scan(self):
        """Execute trades based on current regime"""
        
        # Performance tracking
        current_value = self.portfolio.total_portfolio_value
        if current_value > self.peak:
            self.peak = current_value
            
        drawdown = (self.peak - current_value) / self.peak
        if drawdown > self.max_dd:
            self.max_dd = drawdown
            
        # Emergency exit
        if drawdown > 0.18:
            self.liquidate()
            self.trades += 1
            return
            
        # Track returns
        ret = (current_value - self.last_value) / self.last_value if self.last_value > 0 else 0
        self.daily_returns.append(ret)
        self.last_value = current_value
        
        # Get regime parameters
        params = self.regime_params.get(self.current_regime, self.regime_params["RANGE"])
        
        # Execute regime-specific strategy
        if params["strategy"] == "momentum":
            self.momentum_strategy(params)
        elif params["strategy"] == "mean_reversion":
            self.mean_reversion_strategy(params)
        elif params["strategy"] == "defensive":
            self.defensive_strategy(params)
        elif params["strategy"] == "volatility_harvest":
            self.volatility_harvest_strategy(params)
            
        # Manage existing positions
        self.manage_positions(params)
    
    def momentum_strategy(self, params):
        """Bull market momentum strategy"""
        
        opportunities = []
        
        for symbol, config in self.universe.items():
            if config["type"] not in params["preferred_assets"]:
                continue
                
            if not self.symbol_indicators_ready(symbol):
                continue
                
            indicators = self.indicators[symbol]
            momentum_5 = indicators["momentum_5"].current.value
            momentum_20 = indicators["momentum_20"].current.value
            rsi = indicators["rsi_14"].current.value
            price = self.securities[symbol].price
            ema = indicators["ema_10"].current.value
            
            # Strong momentum setup
            if (momentum_5 > 0.02 and               # 2% 5-day momentum
                momentum_20 > 0.05 and              # 5% 20-day momentum
                price > ema and                     # Above trend
                30 < rsi < 70):                     # Not extreme
                
                strength = momentum_5 * 50 + momentum_20 * 25
                opportunities.append({
                    "symbol": symbol,
                    "strength": strength,
                    "strategy": "MOMENTUM"
                })
                
        # Execute best opportunities
        if opportunities:
            opportunities.sort(key=lambda x: x["strength"], reverse=True)
            self.execute_trades(opportunities[:2], params)
    
    def mean_reversion_strategy(self, params):
        """Range-bound mean reversion"""
        
        opportunities = []
        
        for symbol, config in self.universe.items():
            if config["type"] not in params["preferred_assets"]:
                continue
                
            if not self.symbol_indicators_ready(symbol):
                continue
                
            indicators = self.indicators[symbol]
            rsi = indicators["rsi_14"].current.value
            bb = indicators["bb_20"]
            price = self.securities[symbol].price
            bb_upper = bb.upper_band.current.value
            bb_lower = bb.lower_band.current.value
            bb_middle = bb.middle_band.current.value
            
            # Oversold bounce
            if price < bb_lower and rsi < 30:
                strength = (30 - rsi) * 2
                opportunities.append({
                    "symbol": symbol,
                    "strength": strength,
                    "strategy": "MEAN_REV_LONG"
                })
                
            # Overbought fade
            elif price > bb_upper and rsi > 70:
                strength = (rsi - 70) * 2
                opportunities.append({
                    "symbol": symbol,
                    "strength": strength,
                    "strategy": "MEAN_REV_SHORT"
                })
                
        if opportunities:
            opportunities.sort(key=lambda x: x["strength"], reverse=True)
            self.execute_trades(opportunities[:2], params)
    
    def defensive_strategy(self, params):
        """Bear market defensive positioning"""
        
        # Allocate to defensive assets
        defensive_symbols = ["TLT", "GLD", "UUP"]
        
        for symbol in defensive_symbols:
            if not self.portfolio[symbol].invested:
                self.set_holdings(symbol, params["position_size"] / len(defensive_symbols))
                self.trades += 1
                self.positions[symbol] = {
                    "entry_price": self.securities[symbol].price,
                    "strategy": "DEFENSIVE",
                    "entry_time": self.time
                }
    
    def volatility_harvest_strategy(self, params):
        """High volatility regime - harvest premium"""
        
        # Short VXX if in contango
        vxx_price = self.securities["VXX"].price
        vxx_sma = self.vix_sma.current.value if self.vix_sma.is_ready else vxx_price
        
        if vxx_price < vxx_sma * 0.95:  # VIX in contango
            if not self.portfolio["VXX"].is_short:
                self.set_holdings("VXX", -params["position_size"] * 0.5)
                self.trades += 1
                self.positions["VXX"] = {
                    "entry_price": vxx_price,
                    "strategy": "VOL_HARVEST",
                    "entry_time": self.time
                }
                
        # Defensive allocation
        self.set_holdings("TLT", params["position_size"] * 0.5)
        self.trades += 1
    
    def execute_trades(self, opportunities, params):
        """Execute trades based on opportunities"""
        
        for opp in opportunities:
            symbol = opp["symbol"]
            
            if len(self.positions) >= 3:  # Max 3 positions
                break
                
            if symbol in self.positions:
                continue
                
            position_size = params["position_size"]
            
            if "SHORT" in opp["strategy"]:
                self.set_holdings(symbol, -position_size)
            else:
                self.set_holdings(symbol, position_size)
                
            self.trades += 1
            self.positions[symbol] = {
                "entry_price": self.securities[symbol].price,
                "strategy": opp["strategy"],
                "entry_time": self.time
            }
    
    def manage_positions(self, params):
        """Regime-aware position management"""
        
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
                
            # Calculate P&L
            if self.portfolio[symbol].is_short:
                pnl = (entry_price - current_price) / entry_price
            else:
                pnl = (current_price - entry_price) / entry_price
                
            # Regime-specific exits
            if pnl > params["profit_target"]:
                self.liquidate(symbol)
                self.trades += 1
                self.wins += 1
                del self.positions[symbol]
            elif pnl < -params["stop_loss"]:
                self.liquidate(symbol)
                self.trades += 1
                self.losses += 1
                del self.positions[symbol]
    
    def regime_indicators_ready(self):
        """Check if regime indicators ready"""
        return (self.spy_sma_50.is_ready and
                self.spy_sma_200.is_ready and
                self.vix_sma.is_ready and
                self.market_atr.is_ready)
    
    def symbol_indicators_ready(self, symbol):
        """Check if symbol indicators ready"""
        indicators = self.indicators[symbol]
        return all(ind.is_ready for ind in indicators.values())
    
    def on_end_of_algorithm(self):
        """Final adaptive strategy results"""
        
        years = (self.end_date - self.start_date).days / 365.25
        final_value = self.portfolio.total_portfolio_value
        total_return = (final_value - 100000) / 100000
        cagr = (final_value / 100000) ** (1/years) - 1
        trades_per_year = self.trades / years
        
        # Calculate metrics
        total_decided = self.wins + self.losses
        win_rate = self.wins / total_decided if total_decided > 0 else 0
        avg_profit = total_return / self.trades if self.trades > 0 else 0
        
        # Sharpe ratio
        if len(self.daily_returns) > 100:
            returns_array = np.array(self.daily_returns[-252*5:])
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
            
        self.log("=== REGIME ADAPTIVE MASTER RESULTS ===")
        self.log(f"Final Value: ${final_value:,.2f}")
        self.log(f"Total Return: {total_return:.2%}")
        self.log(f"CAGR: {cagr:.2%}")
        self.log(f"Sharpe Ratio: {sharpe:.2f}")
        self.log(f"Total Trades: {self.trades}")
        self.log(f"Trades/Year: {trades_per_year:.1f}")
        self.log(f"Win Rate: {win_rate:.2%}")
        self.log(f"Avg Profit/Trade: {avg_profit:.2%}")
        self.log(f"Max Drawdown: {self.max_dd:.2%}")
        
        # Regime analysis
        self.log(f"Regime History: {list(self.regime_history)}")
        
        # Target validation
        self.log("=== TARGET VALIDATION ===")
        t1 = cagr > 0.25
        t2 = sharpe > 1.0
        t3 = trades_per_year > 100
        t4 = avg_profit > 0.0075
        t5 = self.max_dd < 0.20
        
        self.log(f"CAGR > 25%: {'PASS' if t1 else 'FAIL'} - {cagr:.2%}")
        self.log(f"Sharpe > 1.0: {'PASS' if t2 else 'FAIL'} - {sharpe:.2f}")
        self.log(f"Trades > 100/yr: {'PASS' if t3 else 'FAIL'} - {trades_per_year:.1f}")
        self.log(f"Profit > 0.75%: {'PASS' if t4 else 'FAIL'} - {avg_profit:.2%}")
        self.log(f"Drawdown < 20%: {'PASS' if t5 else 'FAIL'} - {self.max_dd:.2%}")
        
        self.log(f"TARGETS ACHIEVED: {sum([t1,t2,t3,t4,t5])}/5")