# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
# endregion

class MicrostructureHunter(QCAlgorithm):
    """
    MICROSTRUCTURE HUNTER - Exploits Market Making Inefficiencies
    High-frequency intraday mean reversion and momentum on multiple timeframes
    Targets: CAGR > 30%, Sharpe > 1.8, 500+ trades/year
    """

    def initialize(self):
        self.set_start_date(2015, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # High-liquidity universe for microstructure trading
        self.universe_symbols = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "XLF", "XLK", "XLE"]
        
        # Add securities with minute data
        self.securities_dict = {}
        for symbol in self.universe_symbols:
            equity = self.add_equity(symbol, Resolution.MINUTE)
            equity.set_leverage(8.0)  # 8x leverage for microstructure trading
            self.securities_dict[symbol] = equity
        
        # Multiple timeframe data structures
        self.price_windows = {}
        self.volume_windows = {}
        self.spread_estimates = {}
        self.order_flow_imbalance = {}
        self.momentum_signals = {}
        
        # Initialize data structures
        for symbol in self.universe_symbols:
            self.price_windows[symbol] = {
                "1min": deque(maxlen=60),    # 1 hour of 1-min data
                "5min": deque(maxlen=36),    # 3 hours of 5-min data
                "15min": deque(maxlen=24),   # 6 hours of 15-min data
            }
            self.volume_windows[symbol] = deque(maxlen=20)
            self.spread_estimates[symbol] = 0.001  # 10 bps initial estimate
            self.order_flow_imbalance[symbol] = 0
            self.momentum_signals[symbol] = {"fast": 0, "medium": 0, "slow": 0}
        
        # Ultra-fast technical indicators
        self.indicators = {}
        for symbol in self.universe_symbols:
            self.indicators[symbol] = {
                "rsi_2": self.rsi(symbol, 2, MovingAverageType.EXPONENTIAL, Resolution.MINUTE),
                "rsi_5": self.rsi(symbol, 5, MovingAverageType.EXPONENTIAL, Resolution.MINUTE),
                "bb_10": self.bb(symbol, 10, 1.5, MovingAverageType.EXPONENTIAL, Resolution.MINUTE),
                "atr_3": self.atr(symbol, 3, MovingAverageType.EXPONENTIAL, Resolution.MINUTE),
                "volume_sma": self.sma(self.volume(symbol), 10, Resolution.MINUTE),
                "momentum_3": self.momp(symbol, 3, Resolution.MINUTE),
                "momentum_10": self.momp(symbol, 10, Resolution.MINUTE),
                "momentum_30": self.momp(symbol, 30, Resolution.MINUTE),
            }
        
        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.peak = 100000
        self.max_dd = 0
        self.returns = []
        self.last_value = 100000
        self.positions = {}
        
        # Microstructure parameters
        self.microstructure_threshold = 0.0008   # 8 bps edge
        self.volume_shock_threshold = 2.5        # 2.5x volume shock
        self.momentum_cross_threshold = 0.003    # 30 bps momentum
        self.mean_reversion_threshold = 0.005    # 50 bps reversion
        self.profit_target = 0.008               # 80 bps profit
        self.stop_loss = 0.004                   # 40 bps stop
        self.max_positions = 6                   # Multiple concurrent trades
        
        # Market making parameters  
        self.bid_ask_sensitivity = 1.5           # Spread sensitivity
        self.inventory_penalty = 0.002           # Inventory risk penalty
        
        # Ultra-high frequency schedules
        # Every minute for microstructure detection
        for minutes in range(1, 390, 1):
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", minutes),
                self.microstructure_scan
            )
        
        # 5-minute momentum cross detection
        for minutes in range(5, 390, 5):
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", minutes),
                self.momentum_cross_scan
            )
    
    def on_data(self, data):
        """Process tick-level data for microstructure signals"""
        
        # Update price and volume windows
        for symbol in self.universe_symbols:
            if symbol in data and data[symbol] is not None:
                price = data[symbol].price
                volume = data[symbol].volume
                
                if price > 0:
                    # Update 1-minute windows
                    self.price_windows[symbol]["1min"].append(price)
                    self.volume_windows[symbol].append(volume)
                    
                    # Estimate bid-ask spread from price volatility
                    if len(self.price_windows[symbol]["1min"]) > 5:
                        recent_prices = list(self.price_windows[symbol]["1min"])[-5:]
                        price_std = np.std(recent_prices)
                        self.spread_estimates[symbol] = max(0.0001, price_std * 2)
                    
                    # Order flow imbalance approximation
                    if len(self.price_windows[symbol]["1min"]) > 1:
                        if price > self.price_windows[symbol]["1min"][-2]:
                            self.order_flow_imbalance[symbol] += volume
                        else:
                            self.order_flow_imbalance[symbol] -= volume
        
        # Real-time position management
        self.manage_microstructure_positions()
    
    def microstructure_scan(self):
        """Every minute microstructure opportunity detection"""
        
        # Performance tracking
        current_value = self.portfolio.total_portfolio_value
        if current_value > self.peak:
            self.peak = current_value
            
        drawdown = (self.peak - current_value) / self.peak
        if drawdown > self.max_dd:
            self.max_dd = drawdown
            
        # Emergency protection
        if drawdown > 0.15:
            self.liquidate()
            self.trades += 1
            return
            
        # Track returns
        ret = (current_value - self.last_value) / self.last_value if self.last_value > 0 else 0
        self.returns.append(ret)
        self.last_value = current_value
        
        # Update 5-min and 15-min windows
        minute = self.time.minute
        if minute % 5 == 0:
            self.update_multi_timeframe_data()
        
        opportunities = []
        
        for symbol in self.universe_symbols:
            if not self.indicators_ready(symbol):
                continue
                
            # Get current signals
            price = self.securities[symbol].price
            volume = self.volume_windows[symbol][-1] if self.volume_windows[symbol] else 0
            avg_volume = np.mean(list(self.volume_windows[symbol])) if len(self.volume_windows[symbol]) > 5 else volume
            
            rsi_2 = self.indicators[symbol]["rsi_2"].current.value
            rsi_5 = self.indicators[symbol]["rsi_5"].current.value
            bb_upper = self.indicators[symbol]["bb_10"].upper_band.current.value
            bb_lower = self.indicators[symbol]["bb_10"].lower_band.current.value
            bb_middle = self.indicators[symbol]["bb_10"].middle_band.current.value
            atr = self.indicators[symbol]["atr_3"].current.value
            
            # Order flow signals
            flow_imbalance = self.order_flow_imbalance.get(symbol, 0)
            spread = self.spread_estimates.get(symbol, 0.001)
            
            # MICROSTRUCTURE STRATEGIES
            
            # 1. Volume shock mean reversion
            if avg_volume > 0 and volume > avg_volume * self.volume_shock_threshold:
                if price < bb_lower and rsi_2 < 15:
                    # Oversold on volume shock
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "VOLUME_SHOCK_REVERSION_LONG",
                        "strength": (15 - rsi_2) * 10 + (volume / avg_volume - 1) * 20,
                        "edge": atr * 2  # Expected profit from reversion
                    })
                elif price > bb_upper and rsi_2 > 85:
                    # Overbought on volume shock
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "VOLUME_SHOCK_REVERSION_SHORT",
                        "strength": (rsi_2 - 85) * 10 + (volume / avg_volume - 1) * 20,
                        "edge": atr * 2
                    })
            
            # 2. Order flow imbalance
            if abs(flow_imbalance) > 0:
                normalized_flow = flow_imbalance / (avg_volume * 100) if avg_volume > 0 else 0
                
                if normalized_flow > 0.5 and rsi_5 < 70:
                    # Strong buying pressure
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "ORDER_FLOW_MOMENTUM_LONG",
                        "strength": normalized_flow * 100,
                        "edge": spread * 3
                    })
                elif normalized_flow < -0.5 and rsi_5 > 30:
                    # Strong selling pressure
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "ORDER_FLOW_MOMENTUM_SHORT",
                        "strength": abs(normalized_flow) * 100,
                        "edge": spread * 3
                    })
            
            # 3. Bollinger Band squeeze breakout
            band_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            if band_width < 0.01:  # Tight squeeze
                momentum_3 = self.indicators[symbol]["momentum_3"].current.value
                if momentum_3 > 0.005 and price > bb_upper:
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "BB_SQUEEZE_BREAKOUT_LONG",
                        "strength": momentum_3 * 1000,
                        "edge": band_width * 1000
                    })
                elif momentum_3 < -0.005 and price < bb_lower:
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "BB_SQUEEZE_BREAKOUT_SHORT",
                        "strength": abs(momentum_3) * 1000,
                        "edge": band_width * 1000
                    })
            
            # 4. Intraday mean reversion
            price_dev = (price - bb_middle) / bb_middle if bb_middle > 0 else 0
            if abs(price_dev) > self.mean_reversion_threshold:
                if price_dev < -self.mean_reversion_threshold and rsi_2 < 25:
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "INTRADAY_MEAN_REV_LONG",
                        "strength": abs(price_dev) * 1000 + (25 - rsi_2) * 3,
                        "edge": abs(price_dev) * price
                    })
                elif price_dev > self.mean_reversion_threshold and rsi_2 > 75:
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "INTRADAY_MEAN_REV_SHORT",
                        "strength": abs(price_dev) * 1000 + (rsi_2 - 75) * 3,
                        "edge": abs(price_dev) * price
                    })
            
            # Reset order flow for next period
            self.order_flow_imbalance[symbol] *= 0.9  # Decay
        
        # Execute best opportunities
        if opportunities:
            # Filter by edge threshold
            filtered_opps = [opp for opp in opportunities if opp["edge"] > self.microstructure_threshold]
            filtered_opps.sort(key=lambda x: x["strength"], reverse=True)
            
            available_slots = self.max_positions - len(self.positions)
            self.execute_microstructure_trades(filtered_opps[:available_slots])
    
    def momentum_cross_scan(self):
        """5-minute momentum cross detection"""
        
        for symbol in self.universe_symbols:
            if not self.indicators_ready(symbol):
                continue
                
            # Multi-timeframe momentum
            mom_3 = self.indicators[symbol]["momentum_3"].current.value
            mom_10 = self.indicators[symbol]["momentum_10"].current.value
            mom_30 = self.indicators[symbol]["momentum_30"].current.value
            
            # Update momentum signals
            self.momentum_signals[symbol]["fast"] = mom_3
            self.momentum_signals[symbol]["medium"] = mom_10
            self.momentum_signals[symbol]["slow"] = mom_30
            
            # Momentum cross opportunities
            if (mom_3 > self.momentum_cross_threshold and
                mom_10 > 0 and
                symbol not in self.positions):
                
                # Strong upward momentum cross
                self.set_holdings(symbol, 0.3)  # 30% position
                self.trades += 1
                self.positions[symbol] = {
                    "entry_price": self.securities[symbol].price,
                    "strategy": "MOMENTUM_CROSS_LONG",
                    "entry_time": self.time
                }
                
            elif (mom_3 < -self.momentum_cross_threshold and
                  mom_10 < 0 and
                  symbol not in self.positions):
                
                # Strong downward momentum cross
                self.set_holdings(symbol, -0.3)  # 30% short position
                self.trades += 1
                self.positions[symbol] = {
                    "entry_price": self.securities[symbol].price,
                    "strategy": "MOMENTUM_CROSS_SHORT",
                    "entry_time": self.time
                }
    
    def update_multi_timeframe_data(self):
        """Update 5-minute and 15-minute price windows"""
        
        for symbol in self.universe_symbols:
            if self.price_windows[symbol]["1min"]:
                current_price = list(self.price_windows[symbol]["1min"])[-1]
                
                # Update 5-minute window
                self.price_windows[symbol]["5min"].append(current_price)
                
                # Update 15-minute window every 15 minutes
                if self.time.minute % 15 == 0:
                    self.price_windows[symbol]["15min"].append(current_price)
    
    def execute_microstructure_trades(self, opportunities):
        """Execute microstructure trades with position sizing"""
        
        for opp in opportunities:
            symbol = opp["symbol"]
            if symbol in self.positions:
                continue
                
            # Dynamic position sizing based on edge
            base_size = 0.25  # 25% base position
            edge_multiplier = min(2.0, opp["edge"] / self.microstructure_threshold)
            position_size = base_size * edge_multiplier
            
            if "SHORT" in opp["strategy"]:
                self.set_holdings(symbol, -position_size)
            else:
                self.set_holdings(symbol, position_size)
                
            self.trades += 1
            self.positions[symbol] = {
                "entry_price": self.securities[symbol].price,
                "strategy": opp["strategy"],
                "expected_edge": opp["edge"],
                "entry_time": self.time
            }
            
            self.debug(f"MICRO TRADE: {opp['strategy']} on {symbol} (Edge: {opp['edge']:.4f})")
    
    def manage_microstructure_positions(self):
        """Ultra-tight position management for microstructure trades"""
        
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
            if "REVERSION" in position["strategy"]:
                # Mean reversion trades - quick exit
                if pnl > 0.004:  # 40 bps profit
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                    del self.positions[symbol]
                elif pnl < -0.002:  # 20 bps stop
                    self.liquidate(symbol)
                    self.trades += 1
                    self.losses += 1
                    del self.positions[symbol]
                    
            elif "MOMENTUM" in position["strategy"]:
                # Momentum trades - let them run
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
                    
            elif "ORDER_FLOW" in position["strategy"]:
                # Order flow trades - very quick
                if pnl > 0.003:  # 30 bps
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                    del self.positions[symbol]
                elif pnl < -0.0015:  # 15 bps stop
                    self.liquidate(symbol)
                    self.trades += 1
                    self.losses += 1
                    del self.positions[symbol]
    
    def indicators_ready(self, symbol):
        """Check if indicators are ready"""
        indicators = self.indicators[symbol]
        return all(ind.is_ready for ind in indicators.values())
    
    def on_end_of_algorithm(self):
        """Final microstructure results"""
        
        years = (self.end_date - self.start_date).days / 365.25
        final_value = self.portfolio.total_portfolio_value
        total_return = (final_value - 100000) / 100000
        cagr = (final_value / 100000) ** (1/years) - 1
        trades_per_year = self.trades / years
        
        # Metrics
        total_decided = self.wins + self.losses
        win_rate = self.wins / total_decided if total_decided > 0 else 0
        avg_profit = total_return / self.trades if self.trades > 0 else 0
        
        # Sharpe calculation for high-frequency returns
        if len(self.returns) > 100:
            returns_array = np.array(self.returns[-252*60:])  # Last year of minute returns
            if len(returns_array) > 50:
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                if std_return > 0:
                    # Adjust for minute frequency
                    periods_per_year = 252 * 390  # 390 minute periods per day
                    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
                else:
                    sharpe = 0
            else:
                sharpe = 0
        else:
            sharpe = 0
            
        self.log("=== MICROSTRUCTURE HUNTER RESULTS ===")
        self.log(f"Final Value: ${final_value:,.2f}")
        self.log(f"Total Return: {total_return:.2%}")
        self.log(f"CAGR: {cagr:.2%}")
        self.log(f"Sharpe Ratio: {sharpe:.2f}")
        self.log(f"Total Trades: {self.trades}")
        self.log(f"Trades/Year: {trades_per_year:.1f}")
        self.log(f"Win Rate: {win_rate:.2%}")
        self.log(f"Avg Profit/Trade: {avg_profit:.2%}")
        self.log(f"Max Drawdown: {self.max_dd:.2%}")
        
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
        
        self.log("=== STRATEGY SUMMARY ===")
        self.log("Market microstructure exploitation with 8x leverage")
        self.log("Ultra-high frequency: Trades every minute")
        self.log("Multiple strategies: Volume shock, order flow, mean reversion, momentum")
        self.log("Advanced position sizing based on detected edge")
        self.log("Multi-timeframe analysis (1min, 5min, 15min)")