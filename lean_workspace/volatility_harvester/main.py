# region imports
from AlgorithmImports import *
import numpy as np
from datetime import timedelta
# endregion

class VolatilityHarvester(QCAlgorithm):
    """
    VOLATILITY HARVESTING STRATEGY - Professional Quant Approach
    Harvests volatility risk premium through strategic VIX trading
    Targets: CAGR > 25%, Sharpe > 1.0, 100+ trades/year, Profit > 0.75%, DD < 20%
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)  # VXX inception
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Professional brokerage model
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # Core volatility instruments
        self.vxx = self.add_equity("VXX", Resolution.MINUTE)   # Short-term VIX futures
        self.svxy = self.add_equity("SVXY", Resolution.MINUTE) # Inverse VIX
        self.spy = self.add_equity("SPY", Resolution.MINUTE)   # Market benchmark
        self.tlt = self.add_equity("TLT", Resolution.MINUTE)   # Bonds for hedging
        
        # Leverage for harvesting premium
        self.vxx.set_leverage(3.0)
        self.svxy.set_leverage(2.0)
        self.spy.set_leverage(2.0)
        self.tlt.set_leverage(2.0)
        
        # Volatility indicators
        self.vix_ema_fast = self.ema("VXX", 5, Resolution.DAILY)
        self.vix_ema_slow = self.ema("VXX", 20, Resolution.DAILY)
        self.vix_bb = self.bb("VXX", 20, 2, Resolution.DAILY)
        self.spy_atr = self.atr("SPY", 14, Resolution.DAILY)
        self.market_regime = self.sma("SPY", 200, Resolution.DAILY)
        
        # Historical volatility calculation
        self.vix_history = RollingWindow[float](252)
        self.spy_returns = RollingWindow[float](20)
        self.last_spy_price = None
        
        # Performance tracking
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.peak = 100000
        self.max_dd = 0
        self.daily_returns = []
        self.last_value = 100000
        self.positions = {}
        
        # Professional parameters
        self.vix_threshold_high = 25      # VIX above this = high fear
        self.vix_threshold_low = 15       # VIX below this = complacency  
        self.contango_threshold = 0.02    # 2% contango for carry trades
        self.profit_target = 0.02         # 2% profit target
        self.stop_loss = 0.01             # 1% stop loss
        self.max_drawdown = 0.18          # 18% drawdown protection
        
        # Strategy state
        self.current_regime = "NEUTRAL"
        self.vol_regime = "NORMAL"
        
        # High-frequency schedules for 100+ trades
        # Scan every 30 minutes during market hours
        for minutes in range(30, 390, 30):
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.after_market_open("SPY", minutes),
                self.volatility_harvest_scan
            )
        
        # End of day rebalancing
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.before_market_close("SPY", 15),
            self.end_of_day_rebalance
        )
    
    def volatility_harvest_scan(self):
        """Core volatility harvesting logic"""
        
        # Performance tracking
        current_value = self.portfolio.total_portfolio_value
        if current_value > self.peak:
            self.peak = current_value
            
        drawdown = (self.peak - current_value) / self.peak
        if drawdown > self.max_dd:
            self.max_dd = drawdown
            
        # Emergency protection
        if drawdown > self.max_drawdown:
            self.liquidate()
            self.trades += 1
            return
            
        # Track returns
        ret = (current_value - self.last_value) / self.last_value if self.last_value > 0 else 0
        self.daily_returns.append(ret)
        self.last_value = current_value
        
        # Update volatility metrics
        if self.securities["VXX"].price > 0:
            self.vix_history.add(self.securities["VXX"].price)
            
        if self.last_spy_price and self.securities["SPY"].price > 0:
            spy_return = (self.securities["SPY"].price - self.last_spy_price) / self.last_spy_price
            self.spy_returns.add(spy_return)
        self.last_spy_price = self.securities["SPY"].price
        
        # Determine volatility regime
        if not self.indicators_ready():
            return
            
        vix_price = self.securities["VXX"].price
        vix_ema_fast = self.vix_ema_fast.current.value
        vix_ema_slow = self.vix_ema_slow.current.value
        vix_bb_upper = self.vix_bb.upper_band.current.value
        vix_bb_lower = self.vix_bb.lower_band.current.value
        spy_atr = self.spy_atr.current.value
        market_trend = self.securities["SPY"].price > self.market_regime.current.value
        
        # Calculate realized volatility
        realized_vol = self.calculate_realized_volatility()
        
        # Determine volatility regime
        if vix_price > self.vix_threshold_high:
            self.vol_regime = "HIGH_VOL"
        elif vix_price < self.vix_threshold_low:
            self.vol_regime = "LOW_VOL"
        else:
            self.vol_regime = "NORMAL"
            
        # STRATEGY 1: Short volatility in contango (most profitable)
        if (self.vol_regime == "NORMAL" and 
            vix_ema_fast < vix_ema_slow and  # Downtrending volatility
            vix_price < vix_bb_upper and      # Not overbought
            market_trend):                    # Bull market
            
            # Short VXX (profit from contango decay)
            if not self.portfolio["VXX"].is_short:
                self.set_holdings("VXX", -0.3)  # 30% short position
                self.trades += 1
                self.positions["VXX"] = {
                    "entry_price": vix_price,
                    "type": "SHORT_VOL",
                    "entry_time": self.time
                }
                
            # Long SVXY (inverse VIX)
            if not self.portfolio["SVXY"].is_long:
                self.set_holdings("SVXY", 0.4)  # 40% long position
                self.trades += 1
                self.positions["SVXY"] = {
                    "entry_price": self.securities["SVXY"].price,
                    "type": "INVERSE_VOL",
                    "entry_time": self.time
                }
                
        # STRATEGY 2: Long volatility at extremes
        elif (self.vol_regime == "LOW_VOL" and
              vix_price < vix_bb_lower and    # Oversold volatility
              realized_vol < 0.10):           # Low realized vol
            
            # Long VXX for mean reversion
            if not self.portfolio["VXX"].is_long:
                self.set_holdings("VXX", 0.25)  # 25% position
                self.trades += 1
                self.positions["VXX"] = {
                    "entry_price": vix_price,
                    "type": "LONG_VOL",
                    "entry_time": self.time
                }
                
        # STRATEGY 3: Crisis alpha - long vol in high regime
        elif (self.vol_regime == "HIGH_VOL" and
              vix_ema_fast > vix_ema_slow and  # Rising volatility
              not market_trend):                # Bear market
            
            # Defensive positioning
            if not self.portfolio["TLT"].is_long:
                self.set_holdings("TLT", 0.5)   # 50% bonds
                self.set_holdings("VXX", 0.2)   # 20% long vol
                self.trades += 2
                
        # Position management
        self.manage_positions()
    
    def manage_positions(self):
        """Professional position management with targets and stops"""
        
        for symbol in list(self.positions.keys()):
            if symbol not in self.portfolio or not self.portfolio[symbol].invested:
                if symbol in self.positions:
                    del self.positions[symbol]
                continue
                
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            current_price = self.securities[symbol].price
            
            if entry_price <= 0 or current_price <= 0:
                continue
                
            # Calculate P&L based on position type
            if position["type"] in ["SHORT_VOL"] and self.portfolio[symbol].is_short:
                pnl = (entry_price - current_price) / entry_price
            else:
                pnl = (current_price - entry_price) / entry_price
                
            # Dynamic exit based on volatility regime
            if position["type"] in ["SHORT_VOL", "INVERSE_VOL"]:
                # Tighter stops for short vol
                profit_target = 0.015  # 1.5%
                stop_loss = 0.008      # 0.8%
            else:
                profit_target = self.profit_target
                stop_loss = self.stop_loss
                
            # Exit logic
            if pnl > profit_target:
                self.liquidate(symbol)
                self.trades += 1
                self.wins += 1
                del self.positions[symbol]
            elif pnl < -stop_loss:
                self.liquidate(symbol)
                self.trades += 1
                self.losses += 1
                del self.positions[symbol]
    
    def end_of_day_rebalance(self):
        """End of day risk management and rebalancing"""
        
        # Close positions before weekend if Friday
        if self.time.weekday() == 4:  # Friday
            for symbol in ["VXX", "SVXY"]:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.trades += 1
                    if symbol in self.positions:
                        del self.positions[symbol]
    
    def calculate_realized_volatility(self):
        """Calculate 20-day realized volatility"""
        if not self.spy_returns.is_ready:
            return 0.15  # Default 15% vol
            
        returns = [self.spy_returns[i] for i in range(min(20, self.spy_returns.count))]
        if len(returns) < 2:
            return 0.15
            
        return np.std(returns) * np.sqrt(252)
    
    def indicators_ready(self):
        """Check if all indicators ready"""
        return (self.vix_ema_fast.is_ready and
                self.vix_ema_slow.is_ready and
                self.vix_bb.is_ready and
                self.spy_atr.is_ready and
                self.market_regime.is_ready)
    
    def on_end_of_algorithm(self):
        """Final performance reporting"""
        
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
            
        self.log("=== VOLATILITY HARVESTER RESULTS ===")
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