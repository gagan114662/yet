# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
# endregion

class RiskParityLever(QCAlgorithm):
    """
    RISK PARITY WITH LEVERAGE - Professional Asset Allocation
    Balances risk contribution across assets, uses leverage for returns
    Targets: CAGR > 25%, Sharpe > 1.0, 100+ trades/year, Profit > 0.75%, DD < 20%
    """

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # Diversified universe for risk parity
        self.asset_universe = {
            # Equities
            "SPY": {"leverage": 3.0, "target_vol": 0.15},    # S&P 500
            "QQQ": {"leverage": 3.0, "target_vol": 0.20},    # NASDAQ
            "IWM": {"leverage": 3.0, "target_vol": 0.20},    # Russell 2000
            "EFA": {"leverage": 3.0, "target_vol": 0.15},    # International
            "EEM": {"leverage": 3.0, "target_vol": 0.25},    # Emerging Markets
            # Bonds
            "TLT": {"leverage": 3.0, "target_vol": 0.15},    # Long-term Treasury
            "IEF": {"leverage": 3.0, "target_vol": 0.08},    # Intermediate Treasury
            "HYG": {"leverage": 3.0, "target_vol": 0.10},    # High Yield
            # Commodities
            "GLD": {"leverage": 3.0, "target_vol": 0.18},    # Gold
            "DBC": {"leverage": 3.0, "target_vol": 0.20},    # Commodities
            # Real Estate
            "VNQ": {"leverage": 3.0, "target_vol": 0.20},    # REITs
        }
        
        # Add securities and set leverage
        self.securities_data = {}
        for symbol, config in self.asset_universe.items():
            equity = self.add_equity(symbol, Resolution.DAILY)
            equity.set_leverage(config["leverage"])
            
            self.securities_data[symbol] = {
                "returns": RollingWindow[float](252),
                "volatility": 0.15,  # Initial guess
                "target_vol": config["target_vol"],
                "weight": 0,
                "momentum": self.momp(symbol, 60),  # 60-day momentum
                "ema_fast": self.ema(symbol, 20),
                "ema_slow": self.ema(symbol, 60),
                "rsi": self.rsi(symbol, 14)
            }
        
        # Risk parity parameters
        self.lookback_vol = 60          # Days for volatility calculation
        self.lookback_corr = 252        # Days for correlation
        self.target_portfolio_vol = 0.25 # 25% annual volatility target
        self.leverage_cap = 3.0          # Maximum total leverage
        self.rebalance_threshold = 0.05  # 5% weight deviation triggers rebalance
        
        # Momentum overlay parameters
        self.momentum_factor = 0.3       # 30% weight to momentum
        self.min_momentum = -0.10        # Exclude assets with < -10% momentum
        
        # Performance tracking
        self.trades = 0
        self.rebalances = 0
        self.peak = 100000
        self.max_dd = 0
        self.daily_returns = []
        self.last_value = 100000
        self.last_weights = {}
        
        # Daily volatility update
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("SPY", 30),
            self.update_volatilities
        )
        
        # Intraday momentum checks for high frequency
        for hour in [60, 120, 180, 240, 300]:
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", hour),
                self.momentum_overlay_check
            )
        
        # Weekly rebalancing
        self.schedule.on(
            self.date_rules.week_start(),
            self.time_rules.after_market_open("SPY", 60),
            self.rebalance_portfolio
        )
    
    def update_volatilities(self):
        """Update rolling volatilities and returns"""
        
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
        
        # Update returns for each asset
        for symbol, data in self.securities_data.items():
            price = self.securities[symbol].price
            if price > 0 and data["returns"].count > 0:
                last_price = data["returns"][0] if data["returns"][0] > 0 else price
                daily_return = (price - last_price) / last_price
                data["returns"].add(daily_return)
                
                # Calculate volatility
                if data["returns"].count >= 20:
                    returns = [data["returns"][i] for i in range(min(self.lookback_vol, data["returns"].count))]
                    data["volatility"] = np.std(returns) * np.sqrt(252)
    
    def calculate_risk_parity_weights(self):
        """Calculate risk parity weights with momentum overlay"""
        
        # Get assets with sufficient data and positive momentum
        valid_assets = []
        volatilities = []
        momentum_scores = []
        
        for symbol, data in self.securities_data.items():
            if (data["returns"].count >= self.lookback_vol and
                data["momentum"].is_ready and
                data["volatility"] > 0):
                
                # Check momentum filter
                momentum = data["momentum"].current.value
                if momentum > self.min_momentum:
                    valid_assets.append(symbol)
                    volatilities.append(data["volatility"])
                    momentum_scores.append(max(0, momentum))
        
        if not valid_assets:
            return {}
            
        # Convert to arrays
        vols = np.array(volatilities)
        mom_scores = np.array(momentum_scores)
        
        # Risk parity weights (inverse volatility)
        rp_weights = (1 / vols) / np.sum(1 / vols)
        
        # Momentum weights
        if np.sum(mom_scores) > 0:
            mom_weights = mom_scores / np.sum(mom_scores)
        else:
            mom_weights = np.ones(len(valid_assets)) / len(valid_assets)
            
        # Combine risk parity and momentum
        combined_weights = ((1 - self.momentum_factor) * rp_weights + 
                          self.momentum_factor * mom_weights)
        
        # Normalize
        combined_weights = combined_weights / np.sum(combined_weights)
        
        # Apply leverage to achieve target volatility
        portfolio_vol = self.calculate_portfolio_volatility(valid_assets, combined_weights)
        
        if portfolio_vol > 0:
            leverage = min(self.target_portfolio_vol / portfolio_vol, self.leverage_cap)
            final_weights = combined_weights * leverage
        else:
            final_weights = combined_weights
            
        # Create weight dictionary
        weights = {}
        for i, symbol in enumerate(valid_assets):
            weights[symbol] = final_weights[i]
            
        return weights
    
    def calculate_portfolio_volatility(self, assets, weights):
        """Estimate portfolio volatility (simplified)"""
        
        # Simple weighted average (ignores correlation)
        weighted_vol = 0
        for i, symbol in enumerate(assets):
            weighted_vol += weights[i] * self.securities_data[symbol]["volatility"]
            
        return weighted_vol * 0.7  # Diversification benefit assumption
    
    def rebalance_portfolio(self):
        """Rebalance to risk parity weights"""
        
        if drawdown := (self.peak - self.portfolio.total_portfolio_value) / self.peak > 0.18:
            # Emergency deleveraging
            for symbol in self.asset_universe.keys():
                if self.portfolio[symbol].invested:
                    self.set_holdings(symbol, self.portfolio[symbol].holdings_value / self.portfolio.total_portfolio_value * 0.5)
                    self.trades += 1
            return
            
        # Calculate new weights
        target_weights = self.calculate_risk_parity_weights()
        
        if not target_weights:
            return
            
        # Check if rebalancing needed
        needs_rebalance = False
        for symbol, target_weight in target_weights.items():
            current_weight = self.portfolio[symbol].holdings_value / self.portfolio.total_portfolio_value if self.portfolio.total_portfolio_value > 0 else 0
            if abs(current_weight - target_weight) > self.rebalance_threshold:
                needs_rebalance = True
                break
                
        if needs_rebalance:
            # Execute rebalance
            for symbol in self.asset_universe.keys():
                target = target_weights.get(symbol, 0)
                if target > 0.01:  # Minimum 1% position
                    self.set_holdings(symbol, target)
                    self.trades += 1
                elif self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.trades += 1
                    
            self.rebalances += 1
            self.last_weights = target_weights.copy()
            self.debug(f"REBALANCED: {self.rebalances} times")
    
    def momentum_overlay_check(self):
        """High-frequency momentum overlay for additional trades"""
        
        # Look for extreme momentum opportunities
        for symbol, data in self.securities_data.items():
            if not self.indicators_ready(symbol):
                continue
                
            momentum = data["momentum"].current.value
            rsi = data["rsi"].current.value
            ema_fast = data["ema_fast"].current.value
            ema_slow = data["ema_slow"].current.value
            price = self.securities[symbol].price
            
            current_weight = self.portfolio[symbol].holdings_value / self.portfolio.total_portfolio_value if self.portfolio.total_portfolio_value > 0 else 0
            target_weight = self.last_weights.get(symbol, 0)
            
            # Momentum surge - increase position
            if (momentum > 0.10 and                    # Strong momentum
                ema_fast > ema_slow and                # Uptrend
                rsi < 70 and                           # Not overbought
                current_weight < target_weight * 1.5): # Room to increase
                
                new_weight = min(target_weight * 1.5, current_weight + 0.05)
                self.set_holdings(symbol, new_weight)
                self.trades += 1
                
            # Momentum crash - reduce position  
            elif (momentum < -0.05 and                  # Negative momentum
                  ema_fast < ema_slow and              # Downtrend
                  current_weight > target_weight * 0.5): # Room to decrease
                
                new_weight = max(target_weight * 0.5, current_weight - 0.05)
                self.set_holdings(symbol, new_weight)
                self.trades += 1
    
    def indicators_ready(self, symbol):
        """Check if indicators ready"""
        data = self.securities_data[symbol]
        return (data["momentum"].is_ready and
                data["ema_fast"].is_ready and
                data["ema_slow"].is_ready and
                data["rsi"].is_ready)
    
    def on_end_of_algorithm(self):
        """Final performance analysis"""
        
        years = (self.end_date - self.start_date).days / 365.25
        final_value = self.portfolio.total_portfolio_value
        total_return = (final_value - 100000) / 100000
        cagr = (final_value / 100000) ** (1/years) - 1
        trades_per_year = self.trades / years
        
        # Calculate metrics
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
            
        self.log("=== RISK PARITY LEVERAGED RESULTS ===")
        self.log(f"Final Value: ${final_value:,.2f}")
        self.log(f"Total Return: {total_return:.2%}")
        self.log(f"CAGR: {cagr:.2%}")
        self.log(f"Sharpe Ratio: {sharpe:.2f}")
        self.log(f"Total Trades: {self.trades}")
        self.log(f"Trades/Year: {trades_per_year:.1f}")
        self.log(f"Rebalances: {self.rebalances}")
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