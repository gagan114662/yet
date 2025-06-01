# region imports
from AlgorithmImports import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Indicators import *
from QuantConnect.Orders import *
import json
# endregion

class ConfigurableMultiTimeframeStrategy(QCAlgorithm):
    """
    A configurable multi-timeframe trading strategy that uses parameters from config.json
    to implement daily, weekly, and monthly trading signals with advanced risk management.
    """

    def initialize(self):
        self.set_start_date(2020, 1, 1)  # Set Start Date
        self.set_end_date(2025, 1, 1)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash
        self.set_brokerage_model(BrokerageName.DEFAULT)
        
        # Load configuration parameters
        self.load_config()
        
        # Add assets with different resolutions for multi-timeframe analysis
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.qqq = self.add_equity("QQQ", Resolution.DAILY).symbol
        self.tlt = self.add_equity("TLT", Resolution.DAILY).symbol  # Treasury bonds
        self.gld = self.add_equity("GLD", Resolution.DAILY).symbol  # Gold
        self.efa = self.add_equity("EFA", Resolution.DAILY).symbol  # International equities
        
        # Set benchmark
        self.set_benchmark("SPY")
        
        # Initialize indicators for different timeframes
        # Daily indicators
        self.rsi_spy = self.rsi(self.spy, int(self.config["rsi_period"]), MovingAverageType.SIMPLE, Resolution.DAILY)
        self.bb_spy = self.bb(self.spy, int(self.config["bb_period"]), int(self.config["bb_std_dev"]), MovingAverageType.SIMPLE, Resolution.DAILY)
        
        # Weekly indicators
        self.macd_spy = self.macd(self.spy, int(self.config["macd_fast"]), int(self.config["macd_slow"]), 
                             int(self.config["macd_signal"]), MovingAverageType.EXPONENTIAL, Resolution.DAILY)
        
        # Monthly indicators
        self.fast_ema = self.ema(self.spy, int(self.config["ema_fast"]), Resolution.DAILY)
        self.slow_ema = self.ema(self.spy, int(self.config["ema_slow"]), Resolution.DAILY)
        
        # Risk management
        self.atr_spy = self.atr(self.spy, 14, MovingAverageType.SIMPLE, Resolution.DAILY)
        
        # Schedule multi-timeframe strategies
        # Daily strategy
        self.schedule.on(self.date_rules.every_day(),
                        self.time_rules.after_market_open(self.spy, 10),
                        self.daily_strategy)
                        
        # Weekly strategy
        self.schedule.on(self.date_rules.week_start(),
                        self.time_rules.after_market_open(self.spy, 20),
                        self.weekly_strategy)
                        
        # Monthly strategy
        self.schedule.on(self.date_rules.month_start(),
                        self.time_rules.after_market_open(self.spy, 30),
                        self.monthly_strategy)
        
        # Track performance metrics
        self.daily_returns = []
        
        self.log("Multi-timeframe strategy initialized with configuration parameters")
        self.log(f"Daily allocation: {self.config['daily_allocation']}, Weekly: {self.config['weekly_allocation']}, Monthly: {self.config['monthly_allocation']}")

    def load_config(self):
        """Load configuration parameters from config.json"""
        try:
            # Default configuration
            self.config = {
                "daily_allocation": 0.2,
                "weekly_allocation": 0.3,
                "monthly_allocation": 0.5,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "bb_period": 20,
                "bb_std_dev": 2,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "ema_fast": 20,
                "ema_slow": 50,
                "stop_loss_pct": 5,
                "take_profit_pct": 20,
                "risk_per_trade_pct": 1
            }
            
            # Override with parameters from the algorithm
            params = self.get_parameters()
            if params:
                for key in params:
                    value = params[key]
                    if key in self.config:
                        # Convert string parameters to appropriate types
                        if isinstance(self.config[key], int):
                            self.config[key] = int(value)
                        elif isinstance(self.config[key], float):
                            self.config[key] = float(value)
                        else:
                            self.config[key] = value
                        
            self.log("Configuration loaded successfully")
        except Exception as e:
            self.log(f"Error loading configuration: {str(e)}")

    def daily_strategy(self):
        """Daily trading strategy using RSI and Bollinger Bands"""
        if not self.rsi_spy.is_ready or not self.bb_spy.is_ready:
            return
            
        # RSI oversold/overbought strategy
        rsi_value = self.rsi_spy.current.value
        
        # Calculate position size based on ATR for risk management
        if self.atr_spy.is_ready:
            risk_pct = self.config["risk_per_trade_pct"] / 100  # Risk 1% of portfolio per trade
            stop_loss_atr_multiplier = 2
            stop_loss_distance = self.atr_spy.current.value * stop_loss_atr_multiplier
            price = self.securities[self.spy].price
            position_size = (self.portfolio.cash * risk_pct) / stop_loss_distance
            shares = int(position_size / price)
        else:
            shares = int(self.portfolio.cash * self.config["daily_allocation"] / self.securities[self.spy].price)
        
        # RSI strategy
        if rsi_value < self.config["rsi_oversold"]:  # Oversold
            if not self.portfolio[self.spy].invested:
                self.log(f"Daily Strategy: RSI oversold ({rsi_value}), buying {shares} shares of SPY")
                self.market_order(self.spy, shares)
        elif rsi_value > self.config["rsi_overbought"]:  # Overbought
            if self.portfolio[self.spy].invested:
                self.log(f"Daily Strategy: RSI overbought ({rsi_value}), selling SPY position")
                self.market_order(self.spy, -shares)
                
        # Bollinger Bands strategy for GLD
        if self.bb_spy.is_ready:
            price = self.securities[self.gld].price
            lower = self.bb_spy.lower_band.current.value
            upper = self.bb_spy.upper_band.current.value
            
            if price < lower:  # Price below lower band - potential buy
                gld_shares = int(self.portfolio.cash * 0.1 / price)
                if gld_shares > 0 and not self.portfolio[self.gld].invested:
                    self.log(f"Daily Strategy: GLD below lower BB, buying {gld_shares} shares")
                    self.market_order(self.gld, gld_shares)
            elif price > upper:  # Price above upper band - potential sell
                if self.portfolio[self.gld].invested:
                    self.log("Daily Strategy: GLD above upper BB, selling position")
                    self.liquidate(self.gld)

    def weekly_strategy(self):
        """Weekly trading strategy using MACD and trend following"""
        if not self.macd_spy.is_ready:
            return
            
        # MACD crossover strategy
        macd_value = self.macd_spy.current.value
        signal_value = self.macd_spy.signal.current.value
        
        # Calculate position size - use a larger allocation for weekly trades
        qqq_price = self.securities[self.qqq].price
        efa_price = self.securities[self.efa].price
        qqq_shares = int(self.portfolio.cash * self.config["weekly_allocation"] * 0.6 / qqq_price)
        efa_shares = int(self.portfolio.cash * self.config["weekly_allocation"] * 0.4 / efa_price)
        
        # MACD signal for QQQ
        if macd_value > signal_value and macd_value > 0:  # Bullish signal
            if not self.portfolio[self.qqq].invested and qqq_shares > 0:
                self.log(f"Weekly Strategy: MACD bullish crossover, buying {qqq_shares} shares of QQQ")
                self.market_order(self.qqq, qqq_shares)
                
                # Also buy international equities for diversification
                if efa_shares > 0:
                    self.log(f"Weekly Strategy: Adding {efa_shares} shares of EFA for diversification")
                    self.market_order(self.efa, efa_shares)
        elif macd_value < signal_value and macd_value < 0:  # Bearish signal
            if self.portfolio[self.qqq].invested:
                self.log("Weekly Strategy: MACD bearish crossover, selling QQQ position")
                self.liquidate(self.qqq)
                
            # Rotate to bonds in bearish environment
            tlt_price = self.securities[self.tlt].price
            tlt_shares = int(self.portfolio.cash * self.config["weekly_allocation"] * 0.7 / tlt_price)
            if tlt_shares > 0 and not self.portfolio[self.tlt].invested:
                self.log(f"Weekly Strategy: Rotating to bonds, buying {tlt_shares} shares of TLT")
                self.market_order(self.tlt, tlt_shares)

    def monthly_strategy(self):
        """Monthly rebalancing logic using moving averages"""
        if not self.fast_ema.is_ready or not self.slow_ema.is_ready:
            return
            
        # Moving average crossover strategy with larger allocation
        if self.fast_ema.current.value > self.slow_ema.current.value:
            if not self.portfolio[self.spy].invested:
                self.log("Monthly Strategy: EMA bullish crossover, allocating to SPY")
                self.set_holdings(self.spy, self.config["monthly_allocation"])
                self.liquidate(self.qqq)
        else:
            if not self.portfolio[self.qqq].invested:
                self.log("Monthly Strategy: EMA bearish crossover, allocating to QQQ")
                self.set_holdings(self.qqq, self.config["monthly_allocation"])
                self.liquidate(self.spy)
                
        # Monthly portfolio rebalancing
        self.rebalance_portfolio()

    def rebalance_portfolio(self):
        """Rebalance the entire portfolio to maintain target allocations"""
        # Calculate total equity
        total_equity = self.portfolio.total_portfolio_value
        
        # Log current allocations
        self.log(f"Monthly Rebalance: Portfolio value: ${total_equity}")
        for symbol in [self.spy, self.qqq, self.tlt, self.gld, self.efa]:
            if self.portfolio[symbol].invested:
                market_value = self.portfolio[symbol].holdings_value
                current_pct = (market_value / total_equity) * 100
                self.log(f"  {symbol.value}: ${market_value} ({current_pct:.1f}%)")

    def on_data(self, data: Slice):
        """Track portfolio performance and implement stop-loss"""
        # Log portfolio value
        self.plot("Portfolio", "Value", self.portfolio.total_portfolio_value)
        
        # Track daily returns
        self.daily_returns.append(self.portfolio.total_portfolio_value)
        
        # Implement trailing stop-loss for risk management
        self.check_stop_loss(data)
        
    def check_stop_loss(self, data):
        """Check if any positions need to be stopped out"""
        # Only check once per day to avoid overtrading
        # We'll use a simpler approach - check at the end of each day
        hour = self.time.hour
        if hour != 15:  # 3 PM, near market close
            return
            
        for symbol in [self.spy, self.qqq, self.tlt, self.gld, self.efa]:
            if self.portfolio[symbol].invested:
                position = self.portfolio[symbol]
                current_price = self.securities[symbol].price
                entry_price = position.average_price
                
                # Calculate unrealized profit/loss percentage
                pnl_pct = (current_price / entry_price - 1) * 100
                
                # Stop loss at configured percentage or if profit is more than take profit percentage
                if pnl_pct < -self.config["stop_loss_pct"]:
                    self.log(f"Stop Loss: Closing {symbol.value} position at {pnl_pct:.1f}% loss")
                    self.liquidate(symbol)
                elif pnl_pct > self.config["take_profit_pct"]:
                    self.log(f"Take Profit: Closing {symbol.value} position at {pnl_pct:.1f}% gain")
                    self.liquidate(symbol)
