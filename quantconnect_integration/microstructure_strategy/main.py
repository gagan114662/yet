# STRATEGY 5: MICROSTRUCTURE & MEAN REVERSION MASTER
# Target: 45%+ CAGR, Sharpe > 2.5 via intraday inefficiencies and high-frequency mean reversion

from AlgorithmImports import *
import numpy as np
from datetime import timedelta
from collections import deque

class MicrostructureStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # High-frequency setup with maximum leverage
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Liquid instruments for microstructure trading
        self.spy = self.AddEquity("SPY", Resolution.Second).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Second).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Second).Symbol
        
        # High-volume individual stocks for scalping
        self.aapl = self.AddEquity("AAPL", Resolution.Second).Symbol
        self.msft = self.AddEquity("MSFT", Resolution.Second).Symbol
        self.amzn = self.AddEquity("AMZN", Resolution.Second).Symbol
        
        # Alternative high-volume instruments
        self.gld = self.AddEquity("GLD", Resolution.Second).Symbol  # Gold for volatility
        self.tlt = self.AddEquity("TLT", Resolution.Second).Symbol  # Bonds for reversion
        
        # Universe for microstructure analysis
        self.universe = [self.spy, self.qqq, self.iwm, self.aapl, self.msft, self.amzn, self.gld, self.tlt]
        
        # Microstructure indicators
        self.bid_ask_spreads = {}
        self.order_flow_imbalance = {}
        self.volume_profile = {}
        self.tick_data = {}
        
        # Multi-timeframe mean reversion
        self.mean_reversion_fast = {}   # 30-second reversion
        self.mean_reversion_medium = {} # 5-minute reversion  
        self.mean_reversion_slow = {}   # 30-minute reversion
        
        # Market microstructure tracking
        self.tick_pressure = {}
        self.momentum_bursts = {}
        self.liquidity_events = {}
        
        # High-frequency indicators
        for asset in self.universe:
            # Very short-term mean reversion
            self.mean_reversion_fast[asset] = self.RSI(asset, 10, Resolution.Second)
            self.mean_reversion_medium[asset] = self.RSI(asset, 300, Resolution.Second)  # 5 min
            self.mean_reversion_slow[asset] = self.RSI(asset, 1800, Resolution.Second)   # 30 min
            
            # Initialize tracking structures
            self.tick_data[asset] = deque(maxlen=1000)  # Last 1000 ticks
            self.order_flow_imbalance[asset] = 0.0
            self.volume_profile[asset] = {}
            
        # Microstructure parameters
        self.max_leverage = 15.0        # Extreme leverage for HF trading
        self.position_hold_time = 300   # 5 minute max hold time
        self.scalp_threshold = 0.001    # 0.1% scalping threshold
        
        # Risk management for high-frequency
        self.max_positions = 8          # Max concurrent positions
        self.position_heat = 0.12       # 12% risk per position
        self.daily_trade_limit = 500    # Max trades per day
        self.current_trade_count = 0
        
        # Intraday regime detection
        self.market_regime = "UNKNOWN"
        self.volatility_regime = "NORMAL"
        self.liquidity_regime = "NORMAL"
        
        # Market making parameters
        self.market_make_spread = 0.0002  # 2 bps spread for market making
        self.inventory_target = {}
        
        # Schedule high-frequency analysis
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.Every(TimeSpan.FromSeconds(30)), 
                        self.AnalyzeMicrostructure)
        
        # Track performance metrics
        self.scalp_trades = []
        self.mean_reversion_trades = []
        
        # Volatility surface for regime detection
        self.realized_volatility = {}
        self.volume_weighted_price = {}
        
    def OnData(self, data):
        # High-frequency data processing
        self.ProcessTickData(data)
        
        # Detect microstructure inefficiencies
        self.DetectMicrostructureSignals(data)
        
        # Execute mean reversion strategies
        self.ExecuteMeanReversionStrategy(data)
        
        # Market making opportunities
        self.ExecuteMarketMaking(data)
        
        # Momentum burst scalping
        self.ScalpMomentumBursts(data)
        
        # Limit trade frequency
        if self.current_trade_count >= self.daily_trade_limit:
            return
    
    def ProcessTickData(self, data):
        """Process second-level tick data for microstructure analysis"""
        for asset in self.universe:
            if asset in data and data[asset] is not None:
                tick = {
                    'time': self.Time,
                    'price': data[asset].Close,
                    'volume': data[asset].Volume if hasattr(data[asset], 'Volume') else 0,
                    'spread': self.EstimateBidAskSpread(data[asset])
                }
                
                self.tick_data[asset].append(tick)
                
                # Update real-time indicators
                self.UpdateOrderFlowImbalance(asset, tick)
                self.UpdateVolumeProfile(asset, tick)
                self.DetectTickPressure(asset, tick)
    
    def EstimateBidAskSpread(self, bar):
        """Estimate bid-ask spread from bar data"""
        # Use high-low as spread proxy
        if hasattr(bar, 'High') and hasattr(bar, 'Low'):
            return (bar.High - bar.Low) / bar.Close if bar.Close > 0 else 0.001
        return 0.001  # Default 10 bps
    
    def UpdateOrderFlowImbalance(self, asset, tick):
        """Calculate order flow imbalance"""
        if len(self.tick_data[asset]) < 2:
            return
            
        current_tick = self.tick_data[asset][-1]
        previous_tick = self.tick_data[asset][-2]
        
        price_change = current_tick['price'] - previous_tick['price']
        volume = current_tick['volume']
        
        # Estimate buy/sell pressure based on price movement and volume
        if price_change > 0:
            buy_volume = volume
            sell_volume = 0
        elif price_change < 0:
            buy_volume = 0
            sell_volume = volume
        else:
            buy_volume = volume / 2
            sell_volume = volume / 2
            
        # Update running imbalance
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
        
        # Exponential moving average of imbalance
        alpha = 0.1
        self.order_flow_imbalance[asset] = (alpha * imbalance + 
                                          (1 - alpha) * self.order_flow_imbalance[asset])
    
    def UpdateVolumeProfile(self, asset, tick):
        """Update volume profile for support/resistance"""
        price_level = round(tick['price'], 2)  # Round to nearest cent
        
        if price_level not in self.volume_profile[asset]:
            self.volume_profile[asset][price_level] = 0
            
        self.volume_profile[asset][price_level] += tick['volume']
        
        # Keep only recent volume profile (last 1000 levels)
        if len(self.volume_profile[asset]) > 1000:
            # Remove oldest entries
            sorted_levels = sorted(self.volume_profile[asset].items(), key=lambda x: x[1])
            to_remove = sorted_levels[:100]  # Remove 100 lowest volume levels
            
            for level, _ in to_remove:
                del self.volume_profile[asset][level]
    
    def DetectTickPressure(self, asset, tick):
        """Detect unusual tick pressure and momentum bursts"""
        if len(self.tick_data[asset]) < 20:
            return
            
        recent_ticks = list(self.tick_data[asset])[-20:]
        
        # Calculate tick momentum
        prices = [t['price'] for t in recent_ticks]
        volumes = [t['volume'] for t in recent_ticks]
        
        price_momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        volume_ratio = np.mean(volumes[-5:]) / np.mean(volumes[:-5]) if np.mean(volumes[:-5]) > 0 else 1.0
        
        # Detect momentum burst
        momentum_threshold = 0.002  # 0.2% in 20 seconds
        volume_threshold = 2.0      # 2x volume spike
        
        if abs(price_momentum) > momentum_threshold and volume_ratio > volume_threshold:
            self.momentum_bursts[asset] = {
                'direction': np.sign(price_momentum),
                'strength': abs(price_momentum) * volume_ratio,
                'time': self.Time
            }
    
    def DetectMicrostructureSignals(self, data):
        """Detect microstructure trading signals"""
        for asset in self.universe:
            if asset not in data or data[asset] is None:
                continue
                
            # 1. Order flow imbalance signal
            imbalance = self.order_flow_imbalance.get(asset, 0)
            
            # 2. Mean reversion setup
            if (self.mean_reversion_fast[asset].IsReady and 
                self.mean_reversion_medium[asset].IsReady):
                
                fast_rsi = self.mean_reversion_fast[asset].Current.Value
                medium_rsi = self.mean_reversion_medium[asset].Current.Value
                
                # Extreme levels for mean reversion
                if fast_rsi > 80 and medium_rsi > 60:
                    self.ExecuteShortMeanReversion(asset, fast_rsi)
                elif fast_rsi < 20 and medium_rsi < 40:
                    self.ExecuteLongMeanReversion(asset, fast_rsi)
            
            # 3. Volume profile support/resistance
            self.CheckVolumeProfileLevels(asset, data[asset].Close)
    
    def ExecuteLongMeanReversion(self, asset, signal_strength):
        """Execute long mean reversion trade"""
        if self.Portfolio[asset].Invested or len(self.Portfolio.Values) >= self.max_positions:
            return
            
        # Calculate position size based on signal strength
        oversold_degree = (20 - signal_strength) / 20.0  # How oversold
        position_size = self.position_heat * oversold_degree
        
        # Apply leverage
        leverage_multiplier = min(self.max_leverage, 5.0 + oversold_degree * 5.0)
        final_position = position_size * leverage_multiplier
        
        self.SetHoldings(asset, final_position)
        
        # Set stop and target for mean reversion
        current_price = self.Securities[asset].Price
        self.StopMarketOrder(asset, -int(self.Portfolio[asset].Quantity), current_price * 0.995)  # 0.5% stop
        
        self.current_trade_count += 1
        
        # Schedule position close
        self.Schedule.On(self.DateRules.Today, 
                        self.TimeRules.AfterMarketOpen(asset, self.position_hold_time),
                        lambda: self.ClosePosition(asset))
    
    def ExecuteShortMeanReversion(self, asset, signal_strength):
        """Execute short mean reversion trade"""
        if self.Portfolio[asset].Invested or len(self.Portfolio.Values) >= self.max_positions:
            return
            
        # Calculate position size based on signal strength
        overbought_degree = (signal_strength - 80) / 20.0  # How overbought
        position_size = -self.position_heat * overbought_degree
        
        # Apply leverage
        leverage_multiplier = min(self.max_leverage, 5.0 + overbought_degree * 5.0)
        final_position = position_size * leverage_multiplier
        
        self.SetHoldings(asset, final_position)
        
        # Set stop and target for mean reversion
        current_price = self.Securities[asset].Price
        self.StopMarketOrder(asset, -int(self.Portfolio[asset].Quantity), current_price * 1.005)  # 0.5% stop
        
        self.current_trade_count += 1
        
        # Schedule position close
        self.Schedule.On(self.DateRules.Today, 
                        self.TimeRules.AfterMarketOpen(asset, self.position_hold_time),
                        lambda: self.ClosePosition(asset))
    
    def CheckVolumeProfileLevels(self, asset, current_price):
        """Check volume profile for support/resistance scalping"""
        if asset not in self.volume_profile or len(self.volume_profile[asset]) < 10:
            return
            
        # Find nearby high volume levels
        price_levels = list(self.volume_profile[asset].keys())
        volumes = list(self.volume_profile[asset].values())
        
        # Sort by volume to find key levels
        volume_sorted = sorted(zip(price_levels, volumes), key=lambda x: x[1], reverse=True)
        key_levels = [level for level, volume in volume_sorted[:5]]  # Top 5 volume levels
        
        # Check if price is near key level
        for level in key_levels:
            distance = abs(current_price - level) / current_price
            
            if distance < 0.001:  # Within 0.1% of key level
                # Scalp around volume-based support/resistance
                if current_price < level:  # Below resistance, expect bounce
                    self.ScalpLongVolumeLevel(asset, level)
                else:  # Above support, expect pullback
                    self.ScalpShortVolumeLevel(asset, level)
    
    def ScalpLongVolumeLevel(self, asset, level):
        """Scalp long around volume support level"""
        if self.Portfolio[asset].Invested:
            return
            
        scalp_size = 0.05 * 3.0  # Small scalp with 3x leverage
        self.SetHoldings(asset, scalp_size)
        
        # Quick scalp - close after small profit or loss
        target_price = level * 1.001  # 0.1% target
        stop_price = level * 0.999    # 0.1% stop
        
        self.current_trade_count += 1
    
    def ScalpShortVolumeLevel(self, asset, level):
        """Scalp short around volume resistance level"""
        if self.Portfolio[asset].Invested:
            return
            
        scalp_size = -0.05 * 3.0  # Small scalp with 3x leverage
        self.SetHoldings(asset, scalp_size)
        
        # Quick scalp - close after small profit or loss
        target_price = level * 0.999  # 0.1% target
        stop_price = level * 1.001    # 0.1% stop
        
        self.current_trade_count += 1
    
    def ExecuteMarketMaking(self, data):
        """Execute market making strategy during low volatility"""
        if self.volatility_regime != "LOW":
            return
            
        for asset in [self.spy, self.qqq]:  # Only most liquid assets
            if asset in data and data[asset] is not None:
                current_price = data[asset].Close
                spread = self.EstimateBidAskSpread(data[asset])
                
                # Only market make when spread is wide enough
                if spread > self.market_make_spread * 2:
                    self.ExecuteMarketMakeOrders(asset, current_price, spread)
    
    def ExecuteMarketMakeOrders(self, asset, price, spread):
        """Place market making orders"""
        if self.Portfolio[asset].Invested:
            return
            
        # Place small buy and sell orders around current price
        order_size = 0.02  # 2% position size
        
        # Buy slightly below market
        buy_price = price * (1 - spread/2)
        self.LimitOrder(asset, int(self.Portfolio.TotalPortfolioValue * order_size / buy_price), buy_price)
        
        # Sell slightly above market
        sell_price = price * (1 + spread/2)
        self.LimitOrder(asset, -int(self.Portfolio.TotalPortfolioValue * order_size / sell_price), sell_price)
    
    def ScalpMomentumBursts(self, data):
        """Scalp momentum bursts for quick profits"""
        for asset, burst_info in self.momentum_bursts.items():
            if asset in data and data[asset] is not None:
                # Check if burst is recent (within 30 seconds)
                if (self.Time - burst_info['time']).total_seconds() < 30:
                    direction = burst_info['direction']
                    strength = burst_info['strength']
                    
                    # Scalp in direction of burst
                    scalp_size = direction * min(0.1, strength * 10) * 5.0  # Up to 50% with 5x leverage
                    
                    if not self.Portfolio[asset].Invested and abs(scalp_size) > 0.02:
                        self.SetHoldings(asset, scalp_size)
                        self.current_trade_count += 1
                        
                        # Quick exit after 60 seconds
                        self.Schedule.On(self.DateRules.Today, 
                                       self.TimeRules.AfterMarketOpen(asset, 60),
                                       lambda: self.ClosePosition(asset))
    
    def ClosePosition(self, asset):
        """Close position for mean reversion/scalping strategies"""
        if self.Portfolio[asset].Invested:
            self.Liquidate(asset)
    
    def AnalyzeMicrostructure(self):
        """30-second microstructure analysis"""
        # Detect current market regime
        self.DetectMarketRegime()
        
        # Update volatility regime
        self.DetectVolatilityRegime()
        
        # Risk management
        self.ManageHighFrequencyRisk()
        
        # Performance tracking
        self.TrackScalpingPerformance()
    
    def DetectMarketRegime(self):
        """Detect intraday market regime"""
        if not self.mean_reversion_medium[self.spy].IsReady:
            return
            
        spy_rsi = self.mean_reversion_medium[self.spy].Current.Value
        order_flow = self.order_flow_imbalance.get(self.spy, 0)
        
        if spy_rsi > 70 and order_flow > 0.3:
            self.market_regime = "TRENDING_UP"
        elif spy_rsi < 30 and order_flow < -0.3:
            self.market_regime = "TRENDING_DOWN"
        elif 40 < spy_rsi < 60 and abs(order_flow) < 0.2:
            self.market_regime = "RANGING"
        else:
            self.market_regime = "TRANSITIONAL"
    
    def DetectVolatilityRegime(self):
        """Detect volatility regime for strategy selection"""
        if len(self.tick_data[self.spy]) < 100:
            return
            
        recent_prices = [tick['price'] for tick in list(self.tick_data[self.spy])[-100:]]
        price_changes = np.diff(recent_prices)
        
        if len(price_changes) > 0:
            volatility = np.std(price_changes) / np.mean(recent_prices) * np.sqrt(252 * 24 * 60 * 60)  # Annualized
            
            if volatility < 0.15:
                self.volatility_regime = "LOW"
            elif volatility > 0.30:
                self.volatility_regime = "HIGH"
            else:
                self.volatility_regime = "NORMAL"
    
    def ManageHighFrequencyRisk(self):
        """Manage risk for high-frequency trading"""
        # Check total leverage
        total_leverage = sum([abs(x.HoldingsValue) for x in self.Portfolio.Values]) / self.Portfolio.TotalPortfolioValue
        
        if total_leverage > self.max_leverage:
            # Reduce all positions proportionally
            scale_factor = self.max_leverage / total_leverage * 0.9  # 10% buffer
            
            for holding in self.Portfolio.Values:
                if holding.Invested:
                    current_weight = holding.HoldingsValue / self.Portfolio.TotalPortfolioValue
                    new_weight = current_weight * scale_factor
                    self.SetHoldings(holding.Symbol, new_weight)
        
        # Check trade count limit
        if self.current_trade_count >= self.daily_trade_limit:
            # Stop new trades, only manage existing positions
            self.Debug(f"Daily trade limit reached: {self.current_trade_count}")
    
    def TrackScalpingPerformance(self):
        """Track scalping and mean reversion performance"""
        if self.Portfolio.TotalPortfolioValue > 0:
            current_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            
            # Log performance every minute
            if self.Time.second == 0:
                active_positions = len([x for x in self.Portfolio.Values if x.Invested])
                self.Debug(f"Microstructure: Return {current_return:.3%}, Positions: {active_positions}, Trades: {self.current_trade_count}")
    
    def OnEndOfDay(self, symbol):
        """Daily reset and performance analysis"""
        # Reset daily trade count
        self.current_trade_count = 0
        
        # Clear momentum bursts
        self.momentum_bursts.clear()
        
        # Performance analysis
        if self.Portfolio.TotalPortfolioValue > 0:
            daily_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            
            # Adjust parameters based on performance
            if daily_return > 0.03:  # Great day
                self.position_heat = min(0.15, self.position_heat * 1.02)
                self.daily_trade_limit = min(750, self.daily_trade_limit + 25)
            elif daily_return < -0.02:  # Bad day
                self.position_heat = max(0.08, self.position_heat * 0.98)
                self.daily_trade_limit = max(250, self.daily_trade_limit - 25)
                
        # Clear old volume profile data
        for asset in self.volume_profile:
            if len(self.volume_profile[asset]) > 500:
                # Keep only most recent high-volume levels
                sorted_levels = sorted(self.volume_profile[asset].items(), 
                                     key=lambda x: x[1], reverse=True)
                self.volume_profile[asset] = dict(sorted_levels[:250])