# region imports
from AlgorithmImports import *
import numpy as np
from datetime import timedelta
# endregion

class TargetExceeder(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2015, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)

        # Add the highest performing leveraged ETFs
        self.tqqq = self.add_equity("TQQQ", Resolution.MINUTE)
        self.upro = self.add_equity("UPRO", Resolution.MINUTE)
        self.soxl = self.add_equity("SOXL", Resolution.MINUTE)

        # EXTREME leverage for 25%+ CAGR
        self.tqqq.set_leverage(6.0)  # 6x on 3x = 18x effective
        self.upro.set_leverage(6.0)  # 6x on 3x = 18x effective
        self.soxl.set_leverage(6.0)  # 6x on 3x = 18x effective

        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.peak = 100000
        self.max_dd = 0
        self.returns = []
        self.last_value = 100000

        # EXTREME parameters
        self.profit_target = 0.008   # 0.8% profit (for 0.75%+ avg)
        self.stop_loss = 0.004      # 0.4% stop
        self.entry_threshold = 0.0001  # 0.01% for max frequency

        # Store symbols for easier iteration
        self.symbols = ["TQQQ", "UPRO", "SOXL"]
        
        # Historical data for momentum calculation
        self.history_window = 60  # minutes to look back
        
        # Track position allocation to avoid over-leveraging
        self.max_total_allocation = 1.8  # Maximum 180% allocation
        self.position_size = 0.6  # 60% per position

        # Trade every 15 minutes for 200+ trades/year
        self.schedule.on(self.date_rules.every_day(),
                        self.time_rules.every(timedelta(minutes=15)),
                        self.extreme_trading)

    def extreme_trading(self):
        current_value = self.portfolio.total_portfolio_value

        # Track returns for Sharpe
        if current_value > 0 and self.last_value > 0:
            ret = (current_value - self.last_value) / self.last_value
            self.returns.append(ret)
            self.last_value = current_value

        # Drawdown protection
        if current_value > self.peak:
            self.peak = current_value
        dd = (self.peak - current_value) / self.peak
        if dd > self.max_dd:
            self.max_dd = dd
        if dd > 0.18:  # Emergency exit
            self.liquidate()
            self.trades += 1
            return

        # Check current total allocation
        current_allocation = sum(abs(self.portfolio[symbol].holdings_value) 
                               for symbol in self.symbols) / current_value

        # Extreme momentum trading
        for symbol in self.symbols:
            if not self.securities[symbol].has_data:
                continue
                
            current_price = self.securities[symbol].price
            if current_price == 0:
                continue

            # Get historical data for proper momentum calculation
            try:
                # Get history and convert to DataFrame
                history = self.history([symbol], 2, Resolution.MINUTE)
                
                # Check if we have any data
                if history is None:
                    continue
                    
                # Convert to DataFrame and check length
                history_df = pd.DataFrame(history)
                if history_df.empty or len(history_df) < 2:
                    continue
                    
                # Remove symbol level from MultiIndex if present
                if isinstance(history_df.index, pd.MultiIndex):
                    history_df = history_df.droplevel(0, axis=0)
                
                if len(history_df) < 2:
                    continue
                    
                # Calculate momentum using actual historical data
                previous_price = history_df['close'].iloc[-2]  # Previous bar close
                momentum = (current_price - previous_price) / previous_price
                
            except Exception as e:
                self.debug(f"Error getting history for {symbol}: {e}")
                continue

            # Enter on positive momentum (but check allocation limits)
            if momentum > self.entry_threshold:
                if not self.portfolio[symbol].invested:
                    # Check if we can add another position without exceeding limits
                    projected_allocation = current_allocation + self.position_size
                    if projected_allocation <= self.max_total_allocation:
                        self.set_holdings(symbol, self.position_size)
                        self.trades += 1

            # Profit/loss management
            if self.portfolio[symbol].invested:
                unrealized_pct = self.portfolio[symbol].unrealized_profit_percent
                if unrealized_pct > self.profit_target:
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                elif unrealized_pct < -self.stop_loss:
                    self.liquidate(symbol)
                    self.trades += 1

    def on_end_of_algorithm(self):
        years = (self.end_date - self.start_date).days / 365.25
        final_value = self.portfolio.total_portfolio_value
        total_return = (final_value - 100000) / 100000
        cagr = (final_value / 100000) ** (1/years) - 1
        trades_per_year = self.trades / years
        avg_profit = total_return / self.trades if self.trades > 0 else 0

        # Fixed Sharpe calculation for 15-minute intervals
        if len(self.returns) > 100:
            ret_array = np.array(self.returns)
            mean_ret = np.mean(ret_array)
            std_ret = np.std(ret_array)
            # Annualize: 252 trading days * 24 hours * 4 (15-min intervals)
            periods_per_year = 252 * 6.5 * 4  # 6.5 trading hours per day
            sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Calculate win rate
        win_rate = self.wins / self.trades if self.trades > 0 else 0

        self.log(f"CAGR: {cagr:.2%} (Target: >25%)")
        self.log(f"Sharpe: {sharpe:.2f} (Target: >1.0)")
        self.log(f"Trades/Year: {trades_per_year:.0f} (Target: >100)")
        self.log(f"Avg Profit: {avg_profit:.2%} (Target: >0.75%)")
        self.log(f"Max DD: {self.max_dd:.1%} (Target: <20%)")
        self.log(f"Win Rate: {win_rate:.1%}")
        self.log(f"Total Trades: {self.trades}")

        exceeded = sum([cagr > 0.25, sharpe > 1.0, trades_per_year > 100,
                       avg_profit > 0.0075, self.max_dd < 0.20])
        self.log(f"TARGETS EXCEEDED: {exceeded}/5")
