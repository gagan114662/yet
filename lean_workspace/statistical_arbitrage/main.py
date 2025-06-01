# region imports
from AlgorithmImports import *
import numpy as np
from scipy import stats
# endregion

class StatisticalArbitrage(QCAlgorithm):
    """
    STATISTICAL ARBITRAGE - Quant Pairs Trading
    Exploits mean reversion in correlated asset pairs with mathematical precision
    Targets: CAGR > 25%, Sharpe > 1.0, 100+ trades/year, Profit > 0.75%, DD < 20%
    """

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # High correlation pairs for stat arb
        self.pairs = [
            ("XLF", "BAC"),  # Financials ETF vs Bank of America
            ("XLE", "XOM"),  # Energy ETF vs Exxon
            ("QQQ", "MSFT"), # Tech ETF vs Microsoft
            ("SPY", "IWM"),  # Large cap vs Small cap
            ("GLD", "GDX"),  # Gold vs Gold miners
            ("USO", "XLE"),  # Oil vs Energy sector
        ]
        
        # Add all securities
        self.securities_data = {}
        for pair in self.pairs:
            for symbol in pair:
                if symbol not in self.securities_data:
                    equity = self.add_equity(symbol, Resolution.MINUTE)
                    equity.set_leverage(3.0)  # 3x leverage for arb
                    self.securities_data[symbol] = {
                        "security": equity,
                        "prices": RollingWindow[float](252),
                        "returns": RollingWindow[float](20)
                    }
        
        # Z-score indicators for each pair
        self.zscore_window = 20
        self.pair_data = {}
        for pair in self.pairs:
            pair_id = f"{pair[0]}_{pair[1]}"
            self.pair_data[pair_id] = {
                "spread": RollingWindow[float](self.zscore_window),
                "hedge_ratio": 1.0,
                "correlation": 0.0,
                "last_update": self.time
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
        
        # Stat arb parameters
        self.entry_zscore = 2.0      # Enter when spread > 2 std devs
        self.exit_zscore = 0.5       # Exit when spread returns to 0.5 std devs
        self.stop_zscore = 3.0       # Stop loss at 3 std devs
        self.min_correlation = 0.6   # Minimum correlation to trade
        self.lookback = 60           # Days for correlation calculation
        self.max_positions = 3       # Maximum simultaneous pairs
        
        # High frequency scanning
        for minutes in range(15, 390, 15):  # Every 15 minutes
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", minutes),
                self.scan_arbitrage_opportunities
            )
            
        # Recalculate correlations daily
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("SPY", 5),
            self.update_pair_statistics
        )
    
    def update_pair_statistics(self):
        """Update correlation and hedge ratios for pairs"""
        
        for pair in self.pairs:
            pair_id = f"{pair[0]}_{pair[1]}"
            data1 = self.securities_data[pair[0]]
            data2 = self.securities_data[pair[1]]
            
            # Need enough data
            if data1["prices"].count < self.lookback or data2["prices"].count < self.lookback:
                continue
                
            # Get price arrays
            prices1 = np.array([data1["prices"][i] for i in range(self.lookback)])
            prices2 = np.array([data2["prices"][i] for i in range(self.lookback)])
            
            # Calculate returns
            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]
            
            # Calculate correlation
            if len(returns1) > 2 and len(returns2) > 2:
                correlation = np.corrcoef(returns1, returns2)[0, 1]
                self.pair_data[pair_id]["correlation"] = correlation
                
                # Calculate hedge ratio using OLS regression
                if correlation > self.min_correlation:
                    slope, intercept = np.polyfit(prices2, prices1, 1)
                    self.pair_data[pair_id]["hedge_ratio"] = slope
                    
            self.pair_data[pair_id]["last_update"] = self.time
    
    def scan_arbitrage_opportunities(self):
        """Scan for statistical arbitrage opportunities"""
        
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
        
        # Update price data
        for symbol, data in self.securities_data.items():
            price = self.securities[symbol].price
            if price > 0:
                data["prices"].add(price)
                
        # Check each pair for opportunities
        opportunities = []
        
        for pair in self.pairs:
            pair_id = f"{pair[0]}_{pair[1]}"
            pair_info = self.pair_data[pair_id]
            
            # Skip if low correlation
            if pair_info["correlation"] < self.min_correlation:
                continue
                
            # Calculate current spread
            price1 = self.securities[pair[0]].price
            price2 = self.securities[pair[1]].price
            
            if price1 <= 0 or price2 <= 0:
                continue
                
            # Calculate spread using hedge ratio
            spread = price1 - pair_info["hedge_ratio"] * price2
            pair_info["spread"].add(spread)
            
            # Need enough spread data
            if pair_info["spread"].count < self.zscore_window:
                continue
                
            # Calculate z-score
            spread_array = np.array([pair_info["spread"][i] for i in range(pair_info["spread"].count)])
            mean_spread = np.mean(spread_array)
            std_spread = np.std(spread_array)
            
            if std_spread <= 0:
                continue
                
            zscore = (spread - mean_spread) / std_spread
            
            # Check for entry signals
            if pair_id not in self.positions:
                if abs(zscore) > self.entry_zscore:
                    opportunities.append({
                        "pair_id": pair_id,
                        "pair": pair,
                        "zscore": zscore,
                        "correlation": pair_info["correlation"],
                        "hedge_ratio": pair_info["hedge_ratio"]
                    })
            else:
                # Manage existing position
                self.manage_pair_position(pair_id, pair, zscore)
        
        # Execute best opportunities
        if opportunities and len(self.positions) < self.max_positions:
            # Sort by absolute z-score (strongest divergence)
            opportunities.sort(key=lambda x: abs(x["zscore"]), reverse=True)
            
            for opp in opportunities[:self.max_positions - len(self.positions)]:
                self.execute_pair_trade(opp)
    
    def execute_pair_trade(self, opportunity):
        """Execute a pair trade"""
        
        pair_id = opportunity["pair_id"]
        pair = opportunity["pair"]
        zscore = opportunity["zscore"]
        hedge_ratio = opportunity["hedge_ratio"]
        
        # Position sizing - scale by correlation strength
        base_size = 0.3  # 30% per leg
        position_size = base_size * min(1.0, opportunity["correlation"])
        
        if zscore > self.entry_zscore:
            # Spread too high - short first, long second
            self.set_holdings(pair[0], -position_size)
            self.set_holdings(pair[1], position_size * hedge_ratio)
            position_type = "SHORT_SPREAD"
        else:
            # Spread too low - long first, short second
            self.set_holdings(pair[0], position_size)
            self.set_holdings(pair[1], -position_size * hedge_ratio)
            position_type = "LONG_SPREAD"
            
        self.trades += 2
        
        self.positions[pair_id] = {
            "pair": pair,
            "type": position_type,
            "entry_zscore": zscore,
            "hedge_ratio": hedge_ratio,
            "entry_time": self.time
        }
        
        self.debug(f"PAIR TRADE: {pair_id} Z={zscore:.2f} Type={position_type}")
    
    def manage_pair_position(self, pair_id, pair, current_zscore):
        """Manage existing pair position"""
        
        position = self.positions[pair_id]
        
        # Exit conditions
        exit_trade = False
        
        # Mean reversion - take profit
        if abs(current_zscore) < self.exit_zscore:
            exit_trade = True
            self.wins += 1
            
        # Stop loss - divergence increased
        elif abs(current_zscore) > self.stop_zscore:
            exit_trade = True
            self.losses += 1
            
        # Flipped sides - strong reversal signal
        elif (position["type"] == "SHORT_SPREAD" and current_zscore < -self.entry_zscore):
            exit_trade = True
            self.wins += 1
        elif (position["type"] == "LONG_SPREAD" and current_zscore > self.entry_zscore):
            exit_trade = True
            self.wins += 1
            
        if exit_trade:
            # Close both legs
            self.liquidate(pair[0])
            self.liquidate(pair[1])
            self.trades += 2
            del self.positions[pair_id]
            self.debug(f"EXIT PAIR: {pair_id} Z={current_zscore:.2f}")
    
    def on_end_of_algorithm(self):
        """Final performance analysis"""
        
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
            
        self.log("=== STATISTICAL ARBITRAGE RESULTS ===")
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