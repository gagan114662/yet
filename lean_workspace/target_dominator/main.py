# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class TargetDominator(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Add high-performance leveraged ETFs
        self.tqqq = self.add_equity("TQQQ", Resolution.DAILY)
        self.upro = self.add_equity("UPRO", Resolution.DAILY) 
        self.soxl = self.add_equity("SOXL", Resolution.DAILY)
        
        # Maximum leverage for target domination
        self.tqqq.set_leverage(4.0)  # 4x on 3x = 12x effective
        self.upro.set_leverage(4.0)  # 4x on 3x = 12x effective  
        self.soxl.set_leverage(4.0)  # 4x on 3x = 12x effective
        
        # Ultra-fast indicators for high frequency
        self.tqqq_mom = self.momp("TQQQ", 1)
        self.upro_mom = self.momp("UPRO", 1) 
        self.soxl_mom = self.momp("SOXL", 1)
        
        self.tqqq_rsi = self.rsi("TQQQ", 2)
        self.upro_rsi = self.rsi("UPRO", 2)
        self.soxl_rsi = self.rsi("SOXL", 2)
        
        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.peak = 100000
        self.max_dd = 0
        self.returns = []
        self.last_val = 100000
        
        # EXTREME parameters for target domination
        self.profit_target = 0.02   # 2% profit target
        self.stop_loss = 0.01      # 1% stop loss
        self.position_size = 0.8   # 80% position sizes
        
        # Schedule ultra-high frequency trading
        for minutes in range(30, 360, 15):  # Every 15 minutes
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("TQQQ", minutes),
                self.trade_momentum
            )
    
    def trade_momentum(self):
        """Ultra-high frequency momentum trading"""
        
        # Track performance
        current_val = self.portfolio.total_portfolio_value
        if current_val > self.peak:
            self.peak = current_val
            
        dd = (self.peak - current_val) / self.peak
        if dd > self.max_dd:
            self.max_dd = dd
            
        # Emergency exit if drawdown > 18%
        if dd > 0.18:
            self.liquidate()
            self.trades += 1
            return
            
        # Track returns
        ret = (current_val - self.last_val) / self.last_val if self.last_val > 0 else 0
        self.returns.append(ret)
        self.last_val = current_val
        
        # Ultra-aggressive momentum trading
        symbols = ["TQQQ", "UPRO", "SOXL"]
        
        for symbol in symbols:
            if not self.securities[symbol].has_data:
                continue
                
            # Get momentum and RSI
            if symbol == "TQQQ":
                mom = self.tqqq_mom.current.value if self.tqqq_mom.is_ready else 0
                rsi = self.tqqq_rsi.current.value if self.tqqq_rsi.is_ready else 50
            elif symbol == "UPRO":
                mom = self.upro_mom.current.value if self.upro_mom.is_ready else 0
                rsi = self.upro_rsi.current.value if self.upro_rsi.is_ready else 50
            else:
                mom = self.soxl_mom.current.value if self.soxl_mom.is_ready else 0
                rsi = self.soxl_rsi.current.value if self.soxl_rsi.is_ready else 50
            
            # EXTREME momentum entry (any positive momentum)
            if mom > 0.0001 and rsi < 95:  # Ultra-low threshold
                if not self.portfolio[symbol].invested:
                    self.set_holdings(symbol, self.position_size)
                    self.trades += 1
                    
            # Quick profit taking
            elif self.portfolio[symbol].invested:
                if self.portfolio[symbol].unrealized_profit_percent > self.profit_target:
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                elif self.portfolio[symbol].unrealized_profit_percent < -self.stop_loss:
                    self.liquidate(symbol)
                    self.trades += 1
                    self.losses += 1
    
    def on_end_of_algorithm(self):
        """Final target domination results"""
        
        years = (self.end_date - self.start_date).days / 365.25
        final_val = self.portfolio.total_portfolio_value
        total_ret = (final_val - 100000) / 100000
        cagr = (final_val / 100000) ** (1/years) - 1
        trades_per_year = self.trades / years
        
        # Calculate metrics
        total_decided = self.wins + self.losses
        win_rate = self.wins / total_decided if total_decided > 0 else 0
        avg_profit = total_ret / self.trades if self.trades > 0 else 0
        
        # Sharpe calculation
        if len(self.returns) > 100:
            ret_array = np.array(self.returns[-1000:])  # Last 1000 periods
            if len(ret_array) > 50:
                mean_ret = np.mean(ret_array)
                std_ret = np.std(ret_array)
                if std_ret > 0:
                    sharpe = (mean_ret / std_ret) * np.sqrt(252)
                else:
                    sharpe = 0
            else:
                sharpe = 0
        else:
            sharpe = 0
            
        self.log("=== TARGET DOMINATION RESULTS ===")
        self.log(f"Final Value: ${final_val:,.2f}")
        self.log(f"CAGR: {cagr:.2%}")
        self.log(f"Sharpe: {sharpe:.2f}")
        self.log(f"Trades: {self.trades}")
        self.log(f"Trades/Year: {trades_per_year:.1f}")
        self.log(f"Avg Profit/Trade: {avg_profit:.2%}")
        self.log(f"Max Drawdown: {self.max_dd:.2%}")
        self.log(f"Win Rate: {win_rate:.2%}")
        
        # TARGET VALIDATION
        t1 = cagr > 0.25
        t2 = sharpe > 1.0
        t3 = trades_per_year > 100
        t4 = avg_profit > 0.0075
        t5 = self.max_dd < 0.20
        
        self.log("=== TARGET DOMINATION STATUS ===")
        self.log(f"CAGR > 25%: {'✅' if t1 else '❌'} - {cagr:.2%}")
        self.log(f"Sharpe > 1.0: {'✅' if t2 else '❌'} - {sharpe:.2f}")
        self.log(f"Trades > 100/yr: {'✅' if t3 else '❌'} - {trades_per_year:.1f}")
        self.log(f"Profit > 0.75%: {'✅' if t4 else '❌'} - {avg_profit:.2%}")
        self.log(f"Drawdown < 20%: {'✅' if t5 else '❌'} - {self.max_dd:.2%}")
        
        targets_hit = sum([t1, t2, t3, t4, t5])
        self.log(f"TARGETS DOMINATED: {targets_hit}/5")
        
        if targets_hit == 5:
            self.log("🏆 COMPLETE TARGET DOMINATION! 🏆")
        elif targets_hit >= 4:
            self.log("🥇 EXCELLENT TARGET PERFORMANCE!")
        else:
            self.log("⚡ NEEDS MORE AGGRESSION")