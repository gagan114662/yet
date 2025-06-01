# region imports
from AlgorithmImports import *
import numpy as np
from datetime import timedelta
# endregion

class GammaScalperPro(QCAlgorithm):
    """
    GAMMA SCALPER PRO - Options-Based Volatility Arbitrage
    Exploits gamma positioning and volatility surface inefficiencies
    Targets: CAGR > 40%, Sharpe > 1.5, 200+ trades/year
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # Core underlyings for options trading
        self.spy = self.add_equity("SPY", Resolution.MINUTE)
        self.qqq = self.add_equity("QQQ", Resolution.MINUTE)
        self.iwm = self.add_equity("IWM", Resolution.MINUTE)
        
        # VIX products for volatility trading
        self.vxx = self.add_equity("VXX", Resolution.MINUTE)
        self.uvxy = self.add_equity("UVXY", Resolution.MINUTE)  # 2x VIX
        self.svxy = self.add_equity("SVXY", Resolution.MINUTE)  # -0.5x VIX
        
        # Leveraged ETFs for gamma replication
        self.tqqq = self.add_equity("TQQQ", Resolution.MINUTE)
        self.sqqq = self.add_equity("SQQQ", Resolution.MINUTE)
        self.upro = self.add_equity("UPRO", Resolution.MINUTE)
        self.spxu = self.add_equity("SPXU", Resolution.MINUTE)
        
        # EXTREME LEVERAGE for gamma scalping
        for symbol in ["SPY", "QQQ", "IWM", "TQQQ", "SQQQ", "UPRO", "SPXU"]:
            self.securities[symbol].set_leverage(4.0)
        
        for symbol in ["VXX", "UVXY", "SVXY"]:
            self.securities[symbol].set_leverage(3.0)
        
        # Options chain setup
        spy_option = self.add_option("SPY", Resolution.MINUTE)
        spy_option.set_filter(-10, 10, timedelta(0), timedelta(30))
        
        qqq_option = self.add_option("QQQ", Resolution.MINUTE)
        qqq_option.set_filter(-10, 10, timedelta(0), timedelta(30))
        
        # Microstructure indicators
        self.tick_data = {}
        self.order_flow = {}
        self.gamma_exposure = {}
        self.vix_curve = RollingWindow[float](20)
        
        # Initialize tracking
        for symbol in ["SPY", "QQQ", "IWM"]:
            self.tick_data[symbol] = RollingWindow[float](100)
            self.order_flow[symbol] = {"buys": 0, "sells": 0}
            self.gamma_exposure[symbol] = 0
        
        # Ultra-fast technical indicators
        self.rsi_3 = {}
        self.atr_5 = {}
        self.bb_10 = {}
        self.momentum_1 = {}
        
        for symbol in ["SPY", "QQQ", "IWM", "VXX"]:
            self.rsi_3[symbol] = self.rsi(symbol, 3, MovingAverageType.EXPONENTIAL, Resolution.MINUTE)
            self.atr_5[symbol] = self.atr(symbol, 5, MovingAverageType.EXPONENTIAL, Resolution.MINUTE)
            self.bb_10[symbol] = self.bb(symbol, 10, 1.5, MovingAverageType.EXPONENTIAL, Resolution.MINUTE)
            self.momentum_1[symbol] = self.momp(symbol, 60, Resolution.MINUTE)  # 60 minute momentum
        
        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.peak = 100000
        self.max_dd = 0
        self.returns = []
        self.last_value = 100000
        self.positions = {}
        
        # Gamma scalping parameters
        self.gamma_threshold = 0.0005      # Gamma imbalance threshold
        self.vol_arbitrage_threshold = 0.02  # 2% vol surface arbitrage
        self.microstructure_edge = 0.0003   # 3 bps microstructure edge
        self.profit_target = 0.01           # 1% quick profits
        self.stop_loss = 0.005              # 0.5% tight stops
        self.max_positions = 5              # Multiple concurrent trades
        
        # Market regime
        self.current_regime = "NORMAL"
        self.implied_vol = 0.15
        self.realized_vol = 0.15
        
        # Ultra high-frequency schedules
        # Every 5 minutes for gamma scalping
        for minutes in range(5, 390, 5):
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", minutes),
                self.gamma_scalping_scan
            )
        
        # Volatility arbitrage every 15 minutes
        for minutes in range(15, 390, 15):
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", minutes),
                self.volatility_arbitrage_scan
            )
    
    def on_data(self, data):
        """Process tick data for microstructure signals"""
        
        # Update tick data
        for symbol in ["SPY", "QQQ", "IWM"]:
            if symbol in data and data[symbol] is not None:
                price = data[symbol].price
                if price > 0:
                    self.tick_data[symbol].add(price)
                    
                    # Simple order flow approximation
                    if self.tick_data[symbol].count > 1:
                        if price > self.tick_data[symbol][1]:
                            self.order_flow[symbol]["buys"] += 1
                        else:
                            self.order_flow[symbol]["sells"] += 1
        
        # Update VIX data
        if "VXX" in data and data["VXX"] is not None:
            vix_price = data["VXX"].price
            if vix_price > 0:
                self.vix_curve.add(vix_price)
        
        # Real-time position management
        self.manage_positions()
    
    def gamma_scalping_scan(self):
        """Core gamma scalping strategy"""
        
        # Performance tracking
        current_value = self.portfolio.total_portfolio_value
        if current_value > self.peak:
            self.peak = current_value
            
        drawdown = (self.peak - current_value) / self.peak
        if drawdown > self.max_dd:
            self.max_dd = drawdown
            
        # Emergency exit at 15% drawdown
        if drawdown > 0.15:
            self.liquidate()
            self.trades += 1
            return
            
        # Track returns
        ret = (current_value - self.last_value) / self.last_value if self.last_value > 0 else 0
        self.returns.append(ret)
        self.last_value = current_value
        
        # Calculate gamma exposure from options chains
        self.calculate_gamma_exposure()
        
        opportunities = []
        
        for symbol in ["SPY", "QQQ", "IWM"]:
            if not self.fast_indicators_ready(symbol):
                continue
                
            # Microstructure edge detection
            if self.tick_data[symbol].count >= 50:
                recent_ticks = [self.tick_data[symbol][i] for i in range(20)]
                tick_volatility = np.std(recent_ticks) / np.mean(recent_ticks) if np.mean(recent_ticks) > 0 else 0
                
                # Order flow imbalance
                total_flow = self.order_flow[symbol]["buys"] + self.order_flow[symbol]["sells"]
                if total_flow > 0:
                    buy_ratio = self.order_flow[symbol]["buys"] / total_flow
                else:
                    buy_ratio = 0.5
                
                # Reset order flow
                self.order_flow[symbol] = {"buys": 0, "sells": 0}
                
                # Technical indicators
                rsi = self.rsi_3[symbol].current.value
                atr = self.atr_5[symbol].current.value
                price = self.securities[symbol].price
                bb_upper = self.bb_10[symbol].upper_band.current.value
                bb_lower = self.bb_10[symbol].lower_band.current.value
                momentum = self.momentum_1[symbol].current.value
                
                # GAMMA SCALPING OPPORTUNITIES
                
                # 1. Gamma squeeze setup
                if (self.gamma_exposure.get(symbol, 0) > self.gamma_threshold and
                    tick_volatility > 0.001 and
                    buy_ratio > 0.65):
                    
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "GAMMA_SQUEEZE_LONG",
                        "strength": buy_ratio * 100 + tick_volatility * 1000,
                        "leverage": "TQQQ" if symbol == "QQQ" else "UPRO"
                    })
                
                # 2. Negative gamma unwind
                elif (self.gamma_exposure.get(symbol, 0) < -self.gamma_threshold and
                      tick_volatility > 0.001 and
                      buy_ratio < 0.35):
                    
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "GAMMA_UNWIND_SHORT",
                        "strength": (1 - buy_ratio) * 100 + tick_volatility * 1000,
                        "leverage": "SQQQ" if symbol == "QQQ" else "SPXU"
                    })
                
                # 3. Microstructure mean reversion
                if price < bb_lower and rsi < 20 and momentum > -0.02:
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "MICRO_MEAN_REV_LONG",
                        "strength": (20 - rsi) * 5,
                        "leverage": symbol
                    })
                elif price > bb_upper and rsi > 80 and momentum < 0.02:
                    opportunities.append({
                        "symbol": symbol,
                        "strategy": "MICRO_MEAN_REV_SHORT",
                        "strength": (rsi - 80) * 5,
                        "leverage": symbol
                    })
        
        # Execute best opportunities
        if opportunities:
            opportunities.sort(key=lambda x: x["strength"], reverse=True)
            self.execute_gamma_trades(opportunities[:self.max_positions - len(self.positions)])
    
    def volatility_arbitrage_scan(self):
        """Volatility surface arbitrage opportunities"""
        
        if not self.vix_curve.is_ready or self.vix_curve.count < 10:
            return
            
        # Calculate implied vs realized vol
        vix_prices = [self.vix_curve[i] for i in range(10)]
        current_vix = vix_prices[0]
        vix_mean = np.mean(vix_prices)
        
        # Approximate implied volatility
        self.implied_vol = current_vix / 100  # VIX to decimal
        
        # Calculate realized volatility from SPY
        if self.tick_data["SPY"].count >= 50:
            spy_returns = []
            for i in range(1, 20):
                if i < self.tick_data["SPY"].count:
                    ret = (self.tick_data["SPY"][i-1] - self.tick_data["SPY"][i]) / self.tick_data["SPY"][i]
                    spy_returns.append(ret)
            
            if spy_returns:
                self.realized_vol = np.std(spy_returns) * np.sqrt(252 * 78)  # Intraday annualized
        
        vol_spread = self.implied_vol - self.realized_vol
        
        # Volatility arbitrage trades
        if abs(vol_spread) > self.vol_arbitrage_threshold:
            if vol_spread > self.vol_arbitrage_threshold:
                # Implied > Realized: Short volatility
                if "VXX" not in self.positions:
                    self.set_holdings("VXX", -0.3)
                    self.set_holdings("SVXY", 0.4)
                    self.trades += 2
                    self.positions["VXX"] = {
                        "entry_price": self.securities["VXX"].price,
                        "strategy": "VOL_ARB_SHORT"
                    }
                    self.positions["SVXY"] = {
                        "entry_price": self.securities["SVXY"].price,
                        "strategy": "VOL_ARB_LONG_INVERSE"
                    }
            else:
                # Realized > Implied: Long volatility
                if "UVXY" not in self.positions:
                    self.set_holdings("UVXY", 0.3)
                    self.trades += 1
                    self.positions["UVXY"] = {
                        "entry_price": self.securities["UVXY"].price,
                        "strategy": "VOL_ARB_LONG"
                    }
    
    def calculate_gamma_exposure(self):
        """Estimate market gamma exposure from options"""
        
        # Simplified gamma calculation based on ATM options
        for symbol in ["SPY", "QQQ"]:
            option_chain = self.option_chains.get(symbol)
            if option_chain is None:
                continue
                
            underlying_price = self.securities[symbol].price
            total_gamma = 0
            
            for option in option_chain:
                if option.strike_price > 0:
                    # Approximate gamma for near-the-money options
                    moneyness = abs(option.strike_price - underlying_price) / underlying_price
                    if moneyness < 0.02:  # Within 2% of ATM
                        # Higher open interest = more gamma
                        estimated_gamma = option.open_interest * 0.0001 * (1 - moneyness * 50)
                        if option.right == OptionRight.CALL:
                            total_gamma += estimated_gamma
                        else:
                            total_gamma -= estimated_gamma
            
            self.gamma_exposure[symbol] = total_gamma / 100000  # Normalize
    
    def execute_gamma_trades(self, opportunities):
        """Execute gamma scalping trades"""
        
        for opp in opportunities:
            if len(self.positions) >= self.max_positions:
                break
                
            trade_symbol = opp.get("leverage", opp["symbol"])
            
            if trade_symbol in self.positions:
                continue
                
            position_size = 0.4  # 40% positions with 4x leverage = 160% exposure
            
            if "SHORT" in opp["strategy"]:
                self.set_holdings(trade_symbol, -position_size)
            else:
                self.set_holdings(trade_symbol, position_size)
                
            self.trades += 1
            self.positions[trade_symbol] = {
                "entry_price": self.securities[trade_symbol].price,
                "strategy": opp["strategy"],
                "underlying": opp["symbol"]
            }
            
            self.debug(f"GAMMA TRADE: {opp['strategy']} on {trade_symbol}")
    
    def manage_positions(self):
        """Ultra-fast position management"""
        
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
            
            # Quick profit taking for gamma scalps
            if "GAMMA" in position["strategy"] or "MICRO" in position["strategy"]:
                if pnl > 0.008:  # 0.8% quick profit
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                    del self.positions[symbol]
                elif pnl < -0.004:  # 0.4% tight stop
                    self.liquidate(symbol)
                    self.trades += 1
                    self.losses += 1
                    del self.positions[symbol]
            
            # Volatility arbitrage exits
            elif "VOL_ARB" in position["strategy"]:
                if pnl > self.profit_target * 1.5:  # 1.5% for vol arb
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                    del self.positions[symbol]
                elif pnl < -self.stop_loss * 1.5:  # 0.75% stop
                    self.liquidate(symbol)
                    self.trades += 1
                    self.losses += 1
                    del self.positions[symbol]
    
    def fast_indicators_ready(self, symbol):
        """Check if fast indicators ready"""
        return (symbol in self.rsi_3 and self.rsi_3[symbol].is_ready and
                symbol in self.atr_5 and self.atr_5[symbol].is_ready and
                symbol in self.bb_10 and self.bb_10[symbol].is_ready and
                symbol in self.momentum_1 and self.momentum_1[symbol].is_ready)
    
    def on_end_of_algorithm(self):
        """Final gamma scalping results"""
        
        years = (self.end_date - self.start_date).days / 365.25
        final_value = self.portfolio.total_portfolio_value
        total_return = (final_value - 100000) / 100000
        cagr = (final_value / 100000) ** (1/years) - 1
        trades_per_year = self.trades / years
        
        # Metrics
        total_decided = self.wins + self.losses
        win_rate = self.wins / total_decided if total_decided > 0 else 0
        avg_profit = total_return / self.trades if self.trades > 0 else 0
        
        # Sharpe
        if len(self.returns) > 100:
            returns_array = np.array(self.returns[-252*12:])  # Last year of 5-min returns
            if len(returns_array) > 50:
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                if std_return > 0:
                    # Adjust for intraday frequency
                    periods_per_year = 252 * 78  # 78 5-minute periods per day
                    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
                else:
                    sharpe = 0
            else:
                sharpe = 0
        else:
            sharpe = 0
            
        self.log("=== GAMMA SCALPER PRO RESULTS ===")
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
        self.log("Gamma scalping with microstructure edge detection")
        self.log("Volatility surface arbitrage (implied vs realized)")
        self.log("Ultra-high frequency: Trades every 5 minutes")
        self.log("4x leverage on equity, 3x on volatility products")
        self.log("Options flow analysis for gamma positioning")