# region imports
from AlgorithmImports import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Indicators import *
from QuantConnect.Orders import *
# endregion

class EnhancedMultiTimeframeStrategy(QCAlgorithm):
    """
    Enhanced multi-timeframe trading strategy with advanced risk management,
    additional trading signals, and adaptive position sizing.
    
    Key improvements:
    1. Adaptive position sizing based on volatility and trend strength
    2. Market regime detection for dynamic asset allocation
    3. Correlation-based portfolio construction
    4. Advanced risk management with trailing stops and time-based exits
    5. Multiple technical indicators across different timeframes
    6. Volatility-based hedging strategies
    """

    def initialize(self):
        self.set_start_date(2020, 1, 1)  # Set Start Date
        self.set_end_date(2025, 1, 1)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash
        self.set_brokerage_model(BrokerageName.DEFAULT)
        
        # Add assets with different resolutions for multi-timeframe analysis
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.qqq = self.add_equity("QQQ", Resolution.DAILY).symbol
        self.tlt = self.add_equity("TLT", Resolution.DAILY).symbol  # Treasury bonds
        self.gld = self.add_equity("GLD", Resolution.DAILY).symbol  # Gold
        self.efa = self.add_equity("EFA", Resolution.DAILY).symbol  # International equities
        
        # Add sector ETFs for rotation strategy
        self.xlf = self.add_equity("XLF", Resolution.DAILY).symbol  # Financials
        self.xlk = self.add_equity("XLK", Resolution.DAILY).symbol  # Technology
        self.xle = self.add_equity("XLE", Resolution.DAILY).symbol  # Energy
        self.xlv = self.add_equity("XLV", Resolution.DAILY).symbol  # Healthcare
        self.xlu = self.add_equity("XLU", Resolution.DAILY).symbol  # Utilities
        
        # Add volatility ETF for hedging
        self.vxx = self.add_equity("VXX", Resolution.DAILY).symbol  # Volatility ETF
        
        # Set benchmark
        self.set_benchmark("SPY")
        
        # Initialize parameters
        self.initialize_parameters()
        
        # Initialize indicators for different timeframes
        self.initialize_indicators()
        
        # Schedule multi-timeframe strategies
        self.schedule_strategies()
        
        # Track performance metrics
        self.daily_returns = []
        self.equity_curve = []
        self.drawdowns = []
        self.max_drawdown = 0
        self.peak_value = self.portfolio.total_portfolio_value
        
        # Risk management
        self.risk_environment = "normal"  # Can be "normal", "high_risk", "low_risk"
        self.market_regime = "bull"  # Can be "bull", "bear", "neutral"
        
        # Correlation matrix for position sizing
        self.correlation_matrix = {}
        
        # Sector rotation data
        self.sector_performance = {}
        self.sector_ranks = {}
        
        # Stop loss tracking
        self.stop_levels = {}
        
        self.log("Enhanced multi-timeframe strategy initialized")

    def initialize_parameters(self):
        """Initialize strategy parameters"""
        # RSI parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Bollinger Bands parameters
        self.bb_period = 20
        self.bb_std_dev = 2
        
        # MACD parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Moving Average parameters
        self.ema_fast = 20
        self.ema_slow = 50
        
        # Risk management parameters
        self.stop_loss_pct = 5
        self.take_profit_pct = 20
        self.risk_per_trade_pct = 1
        
        # Allocation parameters
        self.daily_allocation = 0.2
        self.weekly_allocation = 0.3
        self.monthly_allocation = 0.5
        
        # Advanced parameters
        self.volatility_lookback = 20
        self.correlation_lookback = 60
        self.trend_strength_threshold = 25
        self.sector_lookback = 90  # Days to look back for sector performance

    def initialize_indicators(self):
        """Initialize all technical indicators used in the strategy"""
        # Daily indicators
        self.rsi_spy = self.rsi(self.spy, self.rsi_period, MovingAverageType.SIMPLE, Resolution.DAILY)
        self.bb_spy = self.bb(self.spy, self.bb_period, self.bb_std_dev, MovingAverageType.SIMPLE, Resolution.DAILY)
        
        # Weekly indicators
        self.macd_spy = self.macd(self.spy, self.macd_fast, self.macd_slow, 
                             self.macd_signal, MovingAverageType.EXPONENTIAL, Resolution.DAILY)
        
        # Monthly indicators
        self.fast_ema = self.ema(self.spy, self.ema_fast, Resolution.DAILY)
        self.slow_ema = self.ema(self.spy, self.ema_slow, Resolution.DAILY)
        
        # Risk management indicators
        self.atr_spy = self.atr(self.spy, 14, MovingAverageType.SIMPLE, Resolution.DAILY)
        
        # Advanced indicators
        self.adx = self.adx(self.spy, 14, Resolution.DAILY)  # Average Directional Index for trend strength
        self.obv = self.obv(self.spy, Resolution.DAILY)  # On-Balance Volume
        self.volatility = self.std(self.spy, self.volatility_lookback, Resolution.DAILY)  # Standard deviation for volatility
        
        # Sector rotation indicators
        for sector in [self.xlf, self.xlk, self.xle, self.xlv, self.xlu]:
            # Momentum indicators for each sector
            self.register_indicator(f"roc_{sector.value}", self.roc(sector, 20, Resolution.DAILY))
            # Relative strength vs SPY
            self.register_indicator(f"rs_{sector.value}", self.relative_strength(sector, self.spy, 60, Resolution.DAILY))
        
        # Market regime indicators
        self.spy_sma200 = self.sma(self.spy, 200, Resolution.DAILY)
        self.vix = self.add_equity("VIX", Resolution.DAILY).symbol  # VIX for market fear
        
        # Correlation tracking
        self.price_dict = {
            self.spy: RollingWindow[float](self.correlation_lookback),
            self.qqq: RollingWindow[float](self.correlation_lookback),
            self.tlt: RollingWindow[float](self.correlation_lookback),
            self.gld: RollingWindow[float](self.correlation_lookback),
            self.efa: RollingWindow[float](self.correlation_lookback)
        }

    def relative_strength(self, symbol, benchmark, period, resolution):
        """Custom indicator for relative strength"""
        return CustomIndicator(f"RS_{symbol.value}", 
                              lambda x: x[symbol] / x[benchmark] if x[benchmark] != 0 else 0,
                              [symbol, benchmark], 
                              period, 
                              resolution)

    def register_indicator(self, name, indicator):
        """Register an indicator for easy access"""
        setattr(self, name, indicator)

    def schedule_strategies(self):
        """Schedule the multi-timeframe strategies"""
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
        
        # Risk assessment - run daily
        self.schedule.on(self.date_rules.every_day(),
                        self.time_rules.after_market_open(self.spy, 5),
                        self.assess_risk_environment)
        
        # Update correlation matrix - run weekly
        self.schedule.on(self.date_rules.week_start(),
                        self.time_rules.after_market_open(self.spy, 15),
                        self.update_correlation_matrix)
        
        # Sector rotation analysis - run monthly
        self.schedule.on(self.date_rules.month_start(),
                        self.time_rules.after_market_open(self.spy, 25),
                        self.analyze_sector_rotation)

    def assess_risk_environment(self):
        """Assess the current risk environment and market regime"""
        if not self.spy_sma200.is_ready or not self.volatility.is_ready:
            return
            
        # Determine market regime based on price relative to 200-day SMA
        current_price = self.securities[self.spy].price
        sma200 = self.spy_sma200.current.value
        
        if current_price > sma200 * 1.05:
            self.market_regime = "bull"
        elif current_price < sma200 * 0.95:
            self.market_regime = "bear"
        else:
            self.market_regime = "neutral"
            
        # Determine risk environment based on volatility
        current_volatility = self.volatility.current.value
        historical_volatility = self.volatility.current.value
        
        if current_volatility > historical_volatility * 1.5:
            self.risk_environment = "high_risk"
        elif current_volatility < historical_volatility * 0.75:
            self.risk_environment = "low_risk"
        else:
            self.risk_environment = "normal"
            
        # Log the current environment
        self.log(f"Market Regime: {self.market_regime}, Risk Environment: {self.risk_environment}")
        
        # Adjust position sizes based on risk environment
        self.adjust_risk_parameters()

    def adjust_risk_parameters(self):
        """Adjust risk parameters based on the current environment"""
        base_risk = self.risk_per_trade_pct
        
        if self.risk_environment == "high_risk":
            # Reduce risk in high volatility environments
            self.risk_per_trade_pct = base_risk * 0.5
            self.stop_loss_pct = self.stop_loss_pct * 1.5  # Wider stops in volatile markets
        elif self.risk_environment == "low_risk":
            # Increase risk in low volatility environments
            self.risk_per_trade_pct = base_risk * 1.5
            self.stop_loss_pct = self.stop_loss_pct * 0.8  # Tighter stops in calm markets
        else:
            # Normal risk environment
            self.risk_per_trade_pct = base_risk
            # Keep stop loss at default

    def update_correlation_matrix(self):
        """Update the correlation matrix for position sizing"""
        # Check if we have enough data
        for symbol, window in self.price_dict.items():
            if not window.is_ready:
                return
                
        # Calculate correlations between assets
        self.correlation_matrix = {}
        symbols = list(self.price_dict.keys())
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                sym1 = symbols[i]
                sym2 = symbols[j]
                key = f"{sym1.value}_{sym2.value}"
                
                # Calculate correlation
                corr = self.calculate_correlation(self.price_dict[sym1], self.price_dict[sym2])
                self.correlation_matrix[key] = corr
                
        # Log key correlations
        self.log(f"SPY-QQQ Correlation: {self.correlation_matrix.get('SPY_QQQ', 0):.2f}")
        self.log(f"SPY-TLT Correlation: {self.correlation_matrix.get('SPY_TLT', 0):.2f}")
        self.log(f"SPY-GLD Correlation: {self.correlation_matrix.get('SPY_GLD', 0):.2f}")

    def calculate_correlation(self, series1, series2):
        """Calculate correlation between two series"""
        if not series1.is_ready or not series2.is_ready:
            return 0
            
        # Extract values from rolling windows
        values1 = [series1[i] for i in range(series1.count)]
        values2 = [series2[i] for i in range(series2.count)]
        
        # Calculate means
        mean1 = sum(values1) / len(values1)
        mean2 = sum(values2) / len(values2)
        
        # Calculate covariance and variances
        covariance = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(len(values1)))
        variance1 = sum((x - mean1) ** 2 for x in values1)
        variance2 = sum((x - mean2) ** 2 for x in values2)
        
        # Calculate correlation
        if variance1 == 0 or variance2 == 0:
            return 0
        return covariance / (variance1 ** 0.5 * variance2 ** 0.5)

    def analyze_sector_rotation(self):
        """Analyze sector performance for rotation strategy"""
        sectors = [self.xlf, self.xlk, self.xle, self.xlv, self.xlu]
        
        # Calculate performance for each sector
        self.sector_performance = {}
        for sector in sectors:
            # Get relative strength indicator
            rs_indicator = getattr(self, f"rs_{sector.value}")
            if rs_indicator.is_ready:
                # Use relative strength as performance metric
                self.sector_performance[sector] = rs_indicator.current.value
                
        # Rank sectors by performance
        if self.sector_performance:
            ranked_sectors = sorted(self.sector_performance.items(), key=lambda x: x[1], reverse=True)
            self.sector_ranks = {sector: rank for rank, (sector, _) in enumerate(ranked_sectors)}
            
            # Log sector rankings
            self.log("Sector Rankings:")
            for sector, rank in self.sector_ranks.items():
                self.log(f"  {sector.value}: Rank {rank+1}, RS: {self.sector_performance[sector]:.2f}")

    def daily_strategy(self):
        """Daily trading strategy using RSI and Bollinger Bands with advanced risk management"""
        if not self.rsi_spy.is_ready or not self.bb_spy.is_ready:
            return
            
        # RSI oversold/overbought strategy
        rsi_value = self.rsi_spy.current.value
        
        # Calculate position size based on ATR for risk management
        if self.atr_spy.is_ready:
            risk_pct = self.risk_per_trade_pct / 100  # Risk percentage of portfolio per trade
            stop_loss_atr_multiplier = 2
            stop_loss_distance = self.atr_spy.current.value * stop_loss_atr_multiplier
            price = self.securities[self.spy].price
            position_size = (self.portfolio.cash * risk_pct) / stop_loss_distance
            shares = int(position_size / price)
            
            # Adjust position size based on trend strength
            if self.adx.is_ready:
                adx_value = self.adx.current.value
                if adx_value > self.trend_strength_threshold:
                    # Strong trend - increase position size
                    shares = int(shares * 1.2)
                elif adx_value < self.trend_strength_threshold / 2:
                    # Weak trend - decrease position size
                    shares = int(shares * 0.8)
        else:
            shares = int(self.portfolio.cash * self.daily_allocation / self.securities[self.spy].price)
        
        # RSI strategy with volume confirmation
        if rsi_value < self.rsi_oversold:  # Oversold
            if not self.portfolio[self.spy].invested and self.obv.is_ready:
                # Check for volume confirmation
                if self.obv.current.value > self.obv.current.value:
                    self.log(f"Daily Strategy: RSI oversold ({rsi_value}) with volume confirmation, buying {shares} shares of SPY")
                    self.market_order(self.spy, shares)
                    
                    # Set adaptive stop loss
                    self.set_adaptive_stop_loss(self.spy, "long", stop_loss_atr_multiplier)
        elif rsi_value > self.rsi_overbought:  # Overbought
            if self.portfolio[self.spy].invested:
                self.log(f"Daily Strategy: RSI overbought ({rsi_value}), selling SPY position")
                self.market_order(self.spy, -shares)
                
        # Bollinger Bands strategy for GLD with trend confirmation
        if self.bb_spy.is_ready and self.adx.is_ready:
            price = self.securities[self.gld].price
            lower = self.bb_spy.lower_band.current.value
            upper = self.bb_spy.upper_band.current.value
            
            # Only take BB trades if trend is not too strong (mean reversion works better in ranging markets)
            if self.adx.current.value < self.trend_strength_threshold:
                if price < lower:  # Price below lower band - potential buy
                    gld_shares = int(self.portfolio.cash * 0.1 / price)
                    if gld_shares > 0 and not self.portfolio[self.gld].invested:
                        self.log(f"Daily Strategy: GLD below lower BB in ranging market, buying {gld_shares} shares")
                        self.market_order(self.gld, gld_shares)
                        
                        # Set adaptive stop loss
                        self.set_adaptive_stop_loss(self.gld, "long", stop_loss_atr_multiplier)
                elif price > upper:  # Price above upper band - potential sell
                    if self.portfolio[self.gld].invested:
                        self.log("Daily Strategy: GLD above upper BB, selling position")
                        self.liquidate(self.gld)
        
        # Add hedging in high risk environments
        if self.risk_environment == "high_risk" and self.market_regime == "bear":
            vxx_price = self.securities[self.vxx].price
            vxx_shares = int(self.portfolio.cash * 0.05 / vxx_price)  # Small allocation to volatility
            
            if vxx_shares > 0 and not self.portfolio[self.vxx].invested:
                self.log(f"Adding {vxx_shares} shares of VXX as hedge in high risk environment")
                self.market_order(self.vxx, vxx_shares)

    def weekly_strategy(self):
        """Weekly trading strategy using MACD and trend following with correlation-based position sizing"""
        if not self.macd_spy.is_ready:
            return
            
        # MACD crossover strategy
        macd_value = self.macd_spy.current.value
        signal_value = self.macd_spy.signal.current.value
        
        # Calculate position size with correlation adjustment
        qqq_price = self.securities[self.qqq].price
        efa_price = self.securities[self.efa].price
        
        # Base position sizes
        qqq_allocation = self.weekly_allocation * 0.6
        efa_allocation = self.weekly_allocation * 0.4
        
        # Adjust for correlation if available
        if 'SPY_QQQ' in self.correlation_matrix:
            spy_qqq_corr = self.correlation_matrix['SPY_QQQ']
            # Reduce allocation for highly correlated assets
            if spy_qqq_corr > 0.8:
                qqq_allocation *= 0.8
                self.log(f"Reducing QQQ allocation due to high correlation with SPY ({spy_qqq_corr:.2f})")
        
        qqq_shares = int(self.portfolio.cash * qqq_allocation / qqq_price)
        efa_shares = int(self.portfolio.cash * efa_allocation / efa_price)
        
        # MACD signal for QQQ with trend strength confirmation
        if self.adx.is_ready:
            adx_value = self.adx.current.value
            trend_is_strong = adx_value > self.trend_strength_threshold
            
            if macd_value > signal_value and macd_value > 0:  # Bullish signal
                # Only take trend following trades in strong trends
                if trend_is_strong and not self.portfolio[self.qqq].invested and qqq_shares > 0:
                    self.log(f"Weekly Strategy: MACD bullish crossover with strong trend (ADX: {adx_value}), buying {qqq_shares} shares of QQQ")
                    self.market_order(self.qqq, qqq_shares)
                    
                    # Set adaptive stop loss
                    self.set_adaptive_stop_loss(self.qqq, "long", 2.5)
                    
                    # Also buy international equities for diversification if correlation is not too high
                    if 'QQQ_EFA' in self.correlation_matrix and self.correlation_matrix['QQQ_EFA'] < 0.7:
                        if efa_shares > 0:
                            self.log(f"Weekly Strategy: Adding {efa_shares} shares of EFA for diversification")
                            self.market_order(self.efa, efa_shares)
                            self.set_adaptive_stop_loss(self.efa, "long", 2.5)
            elif macd_value < signal_value and macd_value < 0:  # Bearish signal
                if self.portfolio[self.qqq].invested:
                    self.log("Weekly Strategy: MACD bearish crossover, selling QQQ position")
                    self.liquidate(self.qqq)
                    
                # Rotate to bonds in bearish environment
                if self.market_regime == "bear":
                    tlt_price = self.securities[self.tlt].price
                    tlt_shares = int(self.portfolio.cash * self.weekly_allocation * 0.7 / tlt_price)
                    if tlt_shares > 0 and not self.portfolio[self.tlt].invested:
                        self.log(f"Weekly Strategy: Rotating to bonds in bear market, buying {tlt_shares} shares of TLT")
                        self.market_order(self.tlt, tlt_shares)
                        self.set_adaptive_stop_loss(self.tlt, "long", 2)
        
        # Sector rotation strategy
        if self.sector_ranks:
            # Invest in top 2 sectors if we're in a bull market
            if self.market_regime == "bull":
                top_sectors = [sector for sector, rank in self.sector_ranks.items() if rank < 2]
                
                for sector in top_sectors:
                    if not self.portfolio[sector].invested:
                        sector_price = self.securities[sector].price
                        sector_shares = int(self.portfolio.cash * 0.1 / sector_price)  # 10% allocation per top sector
                        
                        if sector_shares > 0:
                            self.log(f"Weekly Strategy: Sector rotation - buying {sector_shares} shares of {sector.value} (Rank: {self.sector_ranks[sector]+1})")
                            self.market_order(sector, sector_shares)
                            self.set_adaptive_stop_loss(sector, "long", 2)

    def monthly_strategy(self):
        """Monthly rebalancing logic using moving averages with adaptive allocation"""
        if not self.fast_ema.is_ready or not self.slow_ema.is_ready:
            return
            
        # Moving average crossover strategy with allocation based on market regime
        if self.fast_ema.current.value > self.slow_ema.current.value:
            # Bullish signal - allocate based on market regime
            if self.market_regime == "bull":
                allocation = self.monthly_allocation
            elif self.market_regime == "neutral":
                allocation = self.monthly_allocation * 0.7
            else:  # Bear market
                allocation = self.monthly_allocation * 0.4
                
            if not self.portfolio[self.spy].invested:
                self.log(f"Monthly Strategy: EMA bullish crossover, allocating {allocation:.1%} to SPY in {self.market_regime} market")
                self.set_holdings(self.spy, allocation)
                self.liquidate(self.qqq)
        else:
            # Bearish signal - allocate to QQQ (more defensive) or reduce exposure
            if self.market_regime == "bull":
                allocation = self.monthly_allocation * 0.8
            elif self.market_regime == "neutral":
                allocation = self.monthly_allocation * 0.5
            else:  # Bear market
                allocation = self.monthly_allocation * 0.2
                
            if not self.portfolio[self.qqq].invested:
                self.log(f"Monthly Strategy: EMA bearish crossover, allocating {allocation:.1%} to QQQ in {self.market_regime} market")
                self.set_holdings(self.qqq, allocation)
                self.liquidate(self.spy)
                
        # Monthly portfolio rebalancing with risk-based adjustments
        self.rebalance_portfolio()

    def rebalance_portfolio(self):
        """Rebalance the entire portfolio to maintain target allocations with risk adjustments"""
        # Calculate total equity
        total_equity = self.portfolio.total_portfolio_value
        
        # Log current allocations
        self.log(f"Monthly Rebalance: Portfolio value: ${total_equity} in {self.market_regime} market regime")
        
        # Calculate target allocations based on market regime
        if self.market_regime == "bull":
            target_allocations = {
                self.spy: 0.4,
                self.qqq: 0.3,
                self.efa: 0.15,
                self.tlt: 0.05,
                self.gld: 0.1
            }
        elif self.market_regime == "neutral":
            target_allocations = {
                self.spy: 0.3,
                self.qqq: 0.2,
                self.efa: 0.1,
                self.tlt: 0.25,
                self.gld: 0.15
            }
        else:  # Bear market
            target_allocations = {
                self.spy: 0.15,
                self.qqq: 0.1,
                self.efa: 0.05,
                self.tlt: 0.5,
                self.gld: 0.2
            }
            
        # Log target allocations
        self.log(f"Target allocations for {self.market_regime} market:")
        for symbol, target in target_allocations.items():
            self.log(f"  {symbol.value}: {target:.1%}")
            
        # Log current allocations
        for symbol in [self.spy, self.qqq, self.tlt, self.gld, self.efa]:
            if self.portfolio[symbol].invested:
                market_value = self.portfolio[symbol].holdings_value
                current_pct = (market_value / total_equity) * 100
                self.log(f"  Current {symbol.value}: ${market_value} ({current_pct:.1f}%)")
                
        # Implement rebalancing if needed
        # This is a simplified approach - in practice, you might want to only rebalance
        # if the current allocation deviates significantly from the target
        if self.time.month % 3 == 0:  # Full rebalance quarterly
            self.log("Performing quarterly full rebalance")
            for symbol, target in target_allocations.items():
                self.set_holdings(symbol, target)

    def set_adaptive_stop_loss(self, symbol, direction, atr_multiplier):
        """Set an adaptive stop loss based on ATR"""
        if not self.atr_spy.is_ready:
            return
            
        price = self.securities[symbol].price
        atr_value = self.atr_spy.current.value
        
        if direction == "long":
            stop_price = price - (atr_value * atr_multiplier)
            self.log(f"Setting adaptive stop loss for {symbol.value} at ${stop_price:.2f} (ATR: ${atr_value:.2f})")
            # Store stop level for checking in on_data
            self.stop_levels[symbol] = stop_price
        else:  # short
            stop_price = price + (atr_value * atr_multiplier)
            self.log(f"Setting adaptive stop loss for {symbol.value} short at ${stop_price:.2f} (ATR: ${atr_value:.2f})")
            self.stop_levels[symbol] = stop_price

    def on_data(self, data: Slice):
        """Track portfolio performance and implement stop-loss"""
        # Update equity curve and calculate drawdown
        current_value = self.portfolio.total_portfolio_value
        self.equity_curve.append(current_value)
        
        # Update peak value and calculate drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        current_drawdown = (self.peak_value - current_value) / self.peak_value * 100
        self.drawdowns.append(current_drawdown)
        
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            
        # Plot portfolio metrics
        self.plot("Portfolio", "Value", current_value)
        self.plot("Portfolio", "Drawdown", current_drawdown)
        
        # Update price data for correlation calculation
        for symbol in self.price_dict:
            if data.contains_key(symbol):
                self.price_dict[symbol].add(data[symbol].close)
        
        # Implement trailing stop-loss for risk management
        self.check_stop_loss(data)
        
    def check_stop_loss(self, data):
        """Check if any positions need to be stopped out using adaptive stops"""
        # Only check once per day to avoid overtrading
        hour = self.time.hour
        if hour != 15:  # 3 PM, near market close
            return
            
        for symbol in [self.spy, self.qqq, self.tlt, self.gld, self.efa, self.xlf, self.xlk, self.xle, self.xlv, self.xlu]:
            if self.portfolio[symbol].invested:
                position = self.portfolio[symbol]
                current_price = self.securities[symbol].price
                entry_price = position.average_price
                
                # Calculate unrealized profit/loss percentage
                pnl_pct = (current_price / entry_price - 1) * 100
                
                # Check if we have a stop level for this symbol
                if symbol in self.stop_levels:
                    stop_level = self.stop_levels[symbol]
                    
                    # For long positions
                    if position.quantity > 0 and current_price < stop_level:
                        self.log(f"Adaptive Stop Loss: Closing {symbol.value} position at {pnl_pct:.1f}% P&L")
                        self.liquidate(symbol)
                        # Remove stop level
                        del self.stop_levels[symbol]
                    # For short positions
                    elif position.quantity < 0 and current_price > stop_level:
                        self.log(f"Adaptive Stop Loss: Closing {symbol.value} short position at {pnl_pct:.1f}% P&L")
                        self.liquidate(symbol)
                        # Remove stop level
                        del self.stop_levels[symbol]
                
                # Take profit at configured percentage
                elif pnl_pct > self.take_profit_pct:
                    self.log(f"Take Profit: Closing {symbol.value} position at {pnl_pct:.1f}% gain")
                    self.liquidate(symbol)
                    
                # Time-based exit for positions not performing
                days_held = (self.time - position.entry_time).days
                if days_held > 30 and pnl_pct < 2:
                    self.log(f"Time Stop: Closing {symbol.value} position after {days_held} days with minimal gain ({pnl_pct:.1f}%)")
                    self.liquidate(symbol)
                    
                # Update trailing stops for profitable positions
                if symbol in self.stop_levels and pnl_pct > 10:
                    # Move stop up to lock in some profits
                    new_stop = max(self.stop_levels[symbol], entry_price * 1.05)  # At least 5% profit
                    self.stop_levels[symbol] = new_stop
                    self.log(f"Raising trailing stop for {symbol.value} to ${new_stop:.2f} to lock in profits")

    def on_end_of_algorithm(self):
        """Calculate final performance metrics"""
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = [(self.equity_curve[i] / self.equity_curve[i-1]) - 1 for i in range(1, len(self.equity_curve))]
            avg_return = sum(returns) / len(returns)
            std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            
            if std_dev > 0:
                sharpe_ratio = avg_return / std_dev * (252 ** 0.5)  # Annualized
                self.log(f"Final Sharpe Ratio: {sharpe_ratio:.3f}")
            
        self.log(f"Maximum Drawdown: {self.max_drawdown:.2f}%")
        self.log(f"Final Portfolio Value: ${self.portfolio.total_portfolio_value:.2f}")
        self.log(f"Total Return: {(self.portfolio.total_portfolio_value / 100000 - 1) * 100:.2f}%")
        
        # Log total number of trades
        self.log(f"Total number of trades: {self.transactions.count()}")
        
        # Calculate win rate
        wins = 0
        losses = 0
        for transaction in self.transactions.get_orders():
            if transaction.profit_loss > 0:
                wins += 1
            elif transaction.profit_loss < 0:
                losses += 1
                
        total_trades = wins + losses
        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
            self.log(f"Win Rate: {win_rate:.1f}% ({wins}/{total_trades})")
            
        # Log performance by timeframe
        self.log("Performance by timeframe:")
        self.log(f"  Daily strategy: {self.daily_allocation * 100:.1f}% allocation")
        self.log(f"  Weekly strategy: {self.weekly_allocation * 100:.1f}% allocation")
        self.log(f"  Monthly strategy: {self.monthly_allocation * 100:.1f}% allocation")
