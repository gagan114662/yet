# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
# endregion

class MultiFactorAlpha(QCAlgorithm):
    """
    MULTI-FACTOR ALPHA - Combines Momentum, Value, Quality, Low Vol
    Professional factor-based approach with dynamic weighting
    Targets: CAGR > 25%, Sharpe > 1.0, 100+ trades/year, Profit > 0.75%, DD < 20%
    """

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # Sector ETF universe for factor investing
        self.universe = {
            "XLK": {"sector": "Technology", "leverage": 3.0},
            "XLF": {"sector": "Financials", "leverage": 3.0},
            "XLE": {"sector": "Energy", "leverage": 3.0},
            "XLV": {"sector": "Healthcare", "leverage": 3.0},
            "XLI": {"sector": "Industrials", "leverage": 3.0},
            "XLY": {"sector": "Consumer Disc", "leverage": 3.0},
            "XLP": {"sector": "Consumer Staples", "leverage": 3.0},
            "XLB": {"sector": "Materials", "leverage": 3.0},
            "XLRE": {"sector": "Real Estate", "leverage": 3.0},
            "XLU": {"sector": "Utilities", "leverage": 3.0},
            # Add leveraged ETFs for enhanced returns
            "TQQQ": {"sector": "Tech 3x", "leverage": 2.0},
            "UPRO": {"sector": "Market 3x", "leverage": 2.0},
            "SOXL": {"sector": "Semi 3x", "leverage": 2.0},
        }
        
        # Add securities and fundamentals
        self.factor_data = {}
        for symbol, config in self.universe.items():
            equity = self.add_equity(symbol, Resolution.DAILY)
            equity.set_leverage(config["leverage"])
            
            self.factor_data[symbol] = {
                "momentum_1m": self.momp(symbol, 21),           # 1-month momentum
                "momentum_3m": self.momp(symbol, 63),           # 3-month momentum  
                "momentum_6m": self.momp(symbol, 126),          # 6-month momentum
                "rsi": self.rsi(symbol, 14),                    # Relative strength
                "volatility": self.std(symbol, 20),             # 20-day volatility
                "volume_ratio": self.v(symbol, Resolution.DAILY), # Volume analysis
                "bb": self.bb(symbol, 20, 2),                   # Bollinger bands
                "macd": self.macd(symbol, 12, 26, 9),          # MACD
                "sma_50": self.sma(symbol, 50),                # 50-day SMA
                "sma_200": self.sma(symbol, 200),              # 200-day SMA
                "price_history": RollingWindow[float](252),     # 1-year prices
                "returns": RollingWindow[float](252),           # Daily returns
                "factor_scores": {}                             # Combined scores
            }
        
        # Factor weights (dynamic)
        self.factor_weights = {
            "momentum": 0.35,
            "value": 0.25,
            "quality": 0.20,
            "low_volatility": 0.20
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
        self.last_rebalance = self.time
        
        # Multi-factor parameters
        self.rebalance_days = 5          # Rebalance every 5 days
        self.top_positions = 5            # Hold top 5 factor scores
        self.position_size = 0.8          # 80% of capital with leverage
        self.factor_threshold = 0.5       # Minimum factor score
        
        # High-frequency factor updates
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("SPY", 30),
            self.update_factor_scores
        )
        
        # Intraday momentum checks
        for hour in [60, 120, 180, 240, 300]:
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", hour),
                self.intraday_factor_trading
            )
            
        # Weekly rebalancing
        self.schedule.on(
            self.date_rules.week_start(),
            self.time_rules.after_market_open("SPY", 45),
            self.rebalance_factor_portfolio
        )
    
    def update_factor_scores(self):
        """Calculate multi-factor scores for each asset"""
        
        # Performance tracking
        current_value = self.portfolio.total_portfolio_value
        if current_value > self.peak:
            self.peak = current_value
            
        drawdown = (self.peak - current_value) / self.peak
        if drawdown > self.max_dd:
            self.max_dd = drawdown
            
        # Track returns
        ret = (current_value - self.last_value) / self.last_value if self.last_value > 0 else 0
        self.daily_returns.append(ret)
        self.last_value = current_value
        
        # Update price data
        for symbol, data in self.factor_data.items():
            price = self.securities[symbol].price
            if price > 0:
                data["price_history"].add(price)
                if data["price_history"].count > 1:
                    prev_price = data["price_history"][1]
                    daily_return = (price - prev_price) / prev_price
                    data["returns"].add(daily_return)
        
        # Calculate factor scores
        factor_rankings = {
            "momentum": {},
            "value": {},
            "quality": {},
            "low_volatility": {}
        }
        
        valid_symbols = []
        
        for symbol, data in self.factor_data.items():
            if not self.indicators_ready(symbol):
                continue
                
            valid_symbols.append(symbol)
            
            # MOMENTUM FACTOR
            mom_1m = data["momentum_1m"].current.value
            mom_3m = data["momentum_3m"].current.value
            mom_6m = data["momentum_6m"].current.value
            momentum_score = (mom_1m * 0.5 + mom_3m * 0.3 + mom_6m * 0.2)
            factor_rankings["momentum"][symbol] = momentum_score
            
            # VALUE FACTOR (using price relative to moving averages)
            price = self.securities[symbol].price
            sma_50 = data["sma_50"].current.value
            sma_200 = data["sma_200"].current.value
            value_score = 2 - (price / sma_50 + price / sma_200) / 2  # Lower is better
            factor_rankings["value"][symbol] = value_score
            
            # QUALITY FACTOR (using MACD and RSI)
            macd_signal = data["macd"].signal.current.value
            macd_current = data["macd"].current.value
            rsi = data["rsi"].current.value
            quality_score = (macd_current - macd_signal) * 100 + (rsi - 50) / 50
            factor_rankings["quality"][symbol] = quality_score
            
            # LOW VOLATILITY FACTOR
            volatility = data["volatility"].current.value
            low_vol_score = 1 / (1 + volatility * 100)  # Lower vol is better
            factor_rankings["low_volatility"][symbol] = low_vol_score
        
        # Normalize and combine factors
        for symbol in valid_symbols:
            combined_score = 0
            
            for factor, weight in self.factor_weights.items():
                # Rank-based scoring
                factor_scores = factor_rankings[factor]
                sorted_symbols = sorted(factor_scores.keys(), 
                                      key=lambda x: factor_scores[x], 
                                      reverse=(factor != "value"))  # Value is inverse
                
                rank = sorted_symbols.index(symbol)
                normalized_score = 1 - (rank / len(sorted_symbols))
                combined_score += normalized_score * weight
                
            self.factor_data[symbol]["factor_scores"]["combined"] = combined_score
    
    def rebalance_factor_portfolio(self):
        """Weekly factor-based rebalancing"""
        
        # Emergency exit check
        if (self.peak - self.portfolio.total_portfolio_value) / self.peak > 0.18:
            self.liquidate()
            self.trades += 1
            return
            
        # Get factor scores
        scores = []
        for symbol, data in self.factor_data.items():
            if "combined" in data["factor_scores"]:
                scores.append((symbol, data["factor_scores"]["combined"]))
                
        if not scores:
            return
            
        # Sort by combined factor score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top positions
        selected = [s[0] for s in scores[:self.top_positions] if s[1] > self.factor_threshold]
        
        # Exit positions not in top selections
        for symbol in list(self.positions.keys()):
            if symbol not in selected:
                self.liquidate(symbol)
                self.trades += 1
                if symbol in self.positions:
                    del self.positions[symbol]
                    
        # Enter new positions
        if selected:
            weight_per_position = self.position_size / len(selected)
            
            for symbol in selected:
                if symbol not in self.positions:
                    self.set_holdings(symbol, weight_per_position)
                    self.trades += 1
                    self.positions[symbol] = {
                        "entry_price": self.securities[symbol].price,
                        "factor_score": self.factor_data[symbol]["factor_scores"]["combined"],
                        "entry_time": self.time
                    }
                    
        self.last_rebalance = self.time
        self.debug(f"REBALANCED: Holding {len(self.positions)} positions")
    
    def intraday_factor_trading(self):
        """High-frequency factor momentum trades"""
        
        # Look for extreme factor movements
        for symbol, data in self.factor_data.items():
            if not self.indicators_ready(symbol):
                continue
                
            # Skip if already positioned
            if symbol in self.positions:
                continue
                
            # Check for factor momentum
            momentum_1m = data["momentum_1m"].current.value
            rsi = data["rsi"].current.value
            bb = data["bb"]
            price = self.securities[symbol].price
            lower_band = bb.lower_band.current.value
            upper_band = bb.upper_band.current.value
            
            # Strong factor momentum setup
            if (momentum_1m > 0.05 and              # 5% monthly momentum
                rsi > 60 and rsi < 80 and           # Strong but not overbought
                price > lower_band and              # Not oversold
                price < upper_band * 1.02):         # Not extended
                
                # Quick momentum trade
                if len(self.positions) < self.top_positions + 2:  # Allow 2 extra
                    self.set_holdings(symbol, 0.2)  # 20% position
                    self.trades += 1
                    self.positions[symbol] = {
                        "entry_price": price,
                        "factor_score": 0,  # Momentum trade
                        "entry_time": self.time,
                        "trade_type": "MOMENTUM"
                    }
        
        # Manage momentum trades
        for symbol in list(self.positions.keys()):
            if symbol not in self.positions:
                continue
                
            position = self.positions[symbol]
            if position.get("trade_type") == "MOMENTUM":
                current_price = self.securities[symbol].price
                entry_price = position["entry_price"]
                
                if entry_price > 0:
                    pnl = (current_price - entry_price) / entry_price
                    
                    # Quick profit taking
                    if pnl > 0.015:  # 1.5% profit
                        self.liquidate(symbol)
                        self.trades += 1
                        self.wins += 1
                        del self.positions[symbol]
                    elif pnl < -0.008:  # 0.8% stop
                        self.liquidate(symbol)
                        self.trades += 1
                        self.losses += 1
                        del self.positions[symbol]
    
    def indicators_ready(self, symbol):
        """Check if all indicators ready"""
        data = self.factor_data[symbol]
        return (data["momentum_1m"].is_ready and
                data["momentum_3m"].is_ready and
                data["momentum_6m"].is_ready and
                data["rsi"].is_ready and
                data["volatility"].is_ready and
                data["bb"].is_ready and
                data["macd"].is_ready and
                data["sma_50"].is_ready and
                data["sma_200"].is_ready)
    
    def on_end_of_algorithm(self):
        """Final multi-factor results"""
        
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
            
        self.log("=== MULTI-FACTOR ALPHA RESULTS ===")
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