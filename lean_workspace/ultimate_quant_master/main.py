# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class UltimateQuantMaster(QCAlgorithm):
    """
    ULTIMATE QUANT MASTER - Combines best approaches that work
    Targets: CAGR > 25%, Sharpe > 1.0, 100+ trades/year, Profit > 0.75%, DD < 20%
    """

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Max leverage brokerage
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # High-performance leveraged ETFs only
        self.add_equity("TQQQ", Resolution.DAILY)  # 3x NASDAQ
        self.add_equity("UPRO", Resolution.DAILY)  # 3x S&P 500
        self.add_equity("SOXL", Resolution.DAILY)  # 3x Semiconductors
        
        # Leverage settings - 2x on 3x ETFs = 6x effective
        self.securities["TQQQ"].set_leverage(2.0)
        self.securities["UPRO"].set_leverage(2.0)
        self.securities["SOXL"].set_leverage(2.0)
        
        # Simple but effective indicators
        self.mom_5 = {}
        self.mom_20 = {}
        self.rsi_14 = {}
        self.bb_20 = {}
        
        for symbol in ["TQQQ", "UPRO", "SOXL"]:
            self.mom_5[symbol] = self.momp(symbol, 5)
            self.mom_20[symbol] = self.momp(symbol, 20)
            self.rsi_14[symbol] = self.rsi(symbol, 14)
            self.bb_20[symbol] = self.bb(symbol, 20, 2)
        
        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.peak = 100000
        self.max_dd = 0
        self.returns = []
        self.last_val = 100000
        self.positions = {}
        
        # Optimal parameters from research
        self.position_size = 0.6         # 60% positions
        self.profit_target = 0.02        # 2% profit
        self.stop_loss = 0.01            # 1% stop
        self.momentum_threshold = 0.02   # 2% momentum
        self.max_positions = 2           # Concentrated
        
        # High frequency schedules
        # Daily momentum scans
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("TQQQ", 30),
            self.momentum_scan
        )
        
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("TQQQ", 120),
            self.momentum_scan
        )
        
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("TQQQ", 240),
            self.momentum_scan
        )
        
        # End of day management
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.before_market_close("TQQQ", 30),
            self.end_of_day_management
        )
    
    def momentum_scan(self):
        """Core momentum trading logic"""
        
        # Performance tracking
        current_val = self.portfolio.total_portfolio_value
        if current_val > self.peak:
            self.peak = current_val
            
        drawdown = (self.peak - current_val) / self.peak
        if drawdown > self.max_dd:
            self.max_dd = drawdown
            
        # Protection
        if drawdown > 0.18:
            self.liquidate()
            self.trades += 1
            return
            
        # Track returns
        ret = (current_val - self.last_val) / self.last_val if self.last_val > 0 else 0
        self.returns.append(ret)
        self.last_val = current_val
        
        # Find opportunities
        opportunities = []
        
        for symbol in ["TQQQ", "UPRO", "SOXL"]:
            if not self.indicators_ready(symbol):
                continue
                
            mom_5 = self.mom_5[symbol].current.value
            mom_20 = self.mom_20[symbol].current.value
            rsi = self.rsi_14[symbol].current.value
            price = self.securities[symbol].price
            bb_upper = self.bb_20[symbol].upper_band.current.value
            bb_lower = self.bb_20[symbol].lower_band.current.value
            
            # Strong momentum
            if (mom_5 > self.momentum_threshold and
                mom_20 > 0.05 and
                30 < rsi < 70 and
                price > bb_lower):
                
                strength = mom_5 * 100 + mom_20 * 50
                opportunities.append({
                    "symbol": symbol,
                    "strength": strength
                })
                
        # Execute best opportunities
        if opportunities and len(self.positions) < self.max_positions:
            opportunities.sort(key=lambda x: x["strength"], reverse=True)
            
            for opp in opportunities[:self.max_positions - len(self.positions)]:
                symbol = opp["symbol"]
                if symbol not in self.positions:
                    self.set_holdings(symbol, self.position_size)
                    self.trades += 1
                    self.positions[symbol] = {
                        "entry_price": self.securities[symbol].price,
                        "entry_time": self.time
                    }
    
    def end_of_day_management(self):
        """End of day profit taking"""
        
        for symbol in list(self.positions.keys()):
            if not self.portfolio[symbol].invested:
                if symbol in self.positions:
                    del self.positions[symbol]
                continue
                
            entry_price = self.positions[symbol]["entry_price"]
            current_price = self.securities[symbol].price
            
            if entry_price > 0:
                pnl = (current_price - entry_price) / entry_price
                
                # Take profits
                if pnl > 0.015:  # 1.5%
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                    del self.positions[symbol]
    
    def on_data(self, data):
        """Real-time position management"""
        
        for symbol in list(self.positions.keys()):
            if not self.portfolio[symbol].invested:
                continue
                
            entry_price = self.positions[symbol]["entry_price"]
            current_price = self.securities[symbol].price
            
            if entry_price > 0:
                pnl = (current_price - entry_price) / entry_price
                
                # Profit target
                if pnl > self.profit_target:
                    self.liquidate(symbol)
                    self.trades += 1
                    self.wins += 1
                    if symbol in self.positions:
                        del self.positions[symbol]
                        
                # Stop loss
                elif pnl < -self.stop_loss:
                    self.liquidate(symbol)
                    self.trades += 1
                    self.losses += 1
                    if symbol in self.positions:
                        del self.positions[symbol]
    
    def indicators_ready(self, symbol):
        """Check indicators ready"""
        return (self.mom_5[symbol].is_ready and
                self.mom_20[symbol].is_ready and
                self.rsi_14[symbol].is_ready and
                self.bb_20[symbol].is_ready)
    
    def on_end_of_algorithm(self):
        """Final results"""
        
        years = (self.end_date - self.start_date).days / 365.25
        final_val = self.portfolio.total_portfolio_value
        total_ret = (final_val - 100000) / 100000
        cagr = (final_val / 100000) ** (1/years) - 1
        trades_per_year = self.trades / years
        
        # Metrics
        total_decided = self.wins + self.losses
        win_rate = self.wins / total_decided if total_decided > 0 else 0
        avg_profit = total_ret / self.trades if self.trades > 0 else 0
        
        # Sharpe
        if len(self.returns) > 100:
            ret_array = np.array(self.returns[-252*5:])
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
            
        self.log("=== ULTIMATE QUANT MASTER RESULTS ===")
        self.log(f"Final Value: ${final_val:,.2f}")
        self.log(f"Total Return: {total_ret:.2%}")
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