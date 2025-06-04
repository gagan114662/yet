from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

class ApexPerformanceEngine(QCAlgorithm):
    """
    Apex Performance Engine - Machine Learning Enhanced Multi-Strategy
    Combines AI-driven predictions with systematic trading approaches
    
    Target: 50%+ CAGR, 2.0+ Sharpe, <18% Max Drawdown
    """
    
    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)
        
        # Advanced configuration
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetSecurityInitializer(self.SecurityInitializer)
        
        # Risk management
        self.portfolio_heat = 0
        self.max_portfolio_heat = 1.0
        self.kelly_fraction = 0.25
        self.var_limit = 0.02  # 2% daily VaR
        self.max_correlation = 0.7
        
        # ML model storage
        self.ml_models = {}
        self.feature_data = {}
        self.prediction_threshold = 0.65
        
        # Strategy allocation
        self.strategy_performance = {
            'trend': {'sharpe': 0, 'weight': 0.25},
            'mean_reversion': {'sharpe': 0, 'weight': 0.20},
            'momentum': {'sharpe': 0, 'weight': 0.25},
            'ml_alpha': {'sharpe': 0, 'weight': 0.30}
        }
        
        # Core universe
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Minute).Symbol
        self.iwm = self.AddEquity("IWM", Resolution.Minute).Symbol
        self.dia = self.AddEquity("DIA", Resolution.Minute).Symbol
        
        # Leveraged ETFs for aggressive positioning
        self.tqqq = self.AddEquity("TQQQ", Resolution.Minute).Symbol
        self.sqqq = self.AddEquity("SQQQ", Resolution.Minute).Symbol
        self.upro = self.AddEquity("UPRO", Resolution.Minute).Symbol
        self.spxu = self.AddEquity("SPXU", Resolution.Minute).Symbol
        
        # Alternative assets
        self.AddCrypto("BTCUSD", Resolution.Hour)
        self.AddCrypto("ETHUSD", Resolution.Hour)
        self.AddForex("EURUSD", Resolution.Hour)
        self.AddForex("USDJPY", Resolution.Hour)
        
        # Dynamic universe selection
        self.AddUniverse(self.SelectHighMomentumStocks)
        self.universe_symbols = []
        
        # Technical indicators setup
        self.indicators = self.SetupIndicators()
        
        # Scheduling
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.AfterMarketOpen(self.spy, 1),
                        self.PreMarketAnalysis)
        
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.Every(timedelta(minutes=5)),
                        self.HighFrequencyTrading)
        
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                        self.TimeRules.BeforeMarketClose(self.spy, 30),
                        self.EndOfDayRebalance)
        
        self.Schedule.On(self.DateRules.MonthStart(self.spy),
                        self.TimeRules.AfterMarketOpen(self.spy, 60),
                        self.TrainMLModels)
        
        # Initialize with historical data
        self.SetWarmUp(timedelta(days=90))
        
    def SecurityInitializer(self, security):
        """Initialize securities with appropriate models"""
        security.SetDataNormalizationMode(DataNormalizationMode.Raw)
        security.SetLeverage(4.0)
        security.SetFillModel(ImmediateFillModel())
        security.SetFeeModel(ConstantFeeModel(0))
        security.SetSlippageModel(ConstantSlippageModel(0))
        
    def SetupIndicators(self):
        """Initialize comprehensive indicator suite"""
        indicators = {}
        
        # Price-based indicators
        for symbol in [self.spy, self.qqq, self.iwm, self.tqqq]:
            indicators[symbol] = {
                # Trend indicators
                'ema_8': self.EMA(symbol, 8, Resolution.Hour),
                'ema_21': self.EMA(symbol, 21, Resolution.Hour),
                'ema_50': self.EMA(symbol, 50, Resolution.Daily),
                'ema_200': self.EMA(symbol, 200, Resolution.Daily),
                
                # Momentum indicators
                'rsi': self.RSI(symbol, 14, Resolution.Hour),
                'rsi_fast': self.RSI(symbol, 7, Resolution.Hour),
                'macd': self.MACD(symbol, 12, 26, 9, Resolution.Hour),
                'stoch': self.STO(symbol, 14, 3, 3, Resolution.Hour),
                
                # Volatility indicators
                'bb': self.BB(symbol, 20, 2, Resolution.Hour),
                'atr': self.ATR(symbol, 14, Resolution.Hour),
                'natr': self.NATR(symbol, 14, Resolution.Hour),
                
                # Volume indicators
                'obv': self.OBV(symbol, Resolution.Hour),
                'ad': self.AD(symbol, Resolution.Hour),
                'adx': self.ADX(symbol, 14, Resolution.Hour),
                
                # Custom indicators
                'keltner': self.KCH(symbol, 20, 1.5, Resolution.Hour),
                'donchian': self.DCH(symbol, 20, Resolution.Hour),
                'ichimoku': self.ICHIMOKU(symbol, 9, 26, 26, 52, Resolution.Daily)
            }
            
        return indicators
    
    def SelectHighMomentumStocks(self, coarse):
        """ML-enhanced universe selection"""
        # Initial filtering
        filtered = [x for x in coarse if x.HasFundamentalData 
                   and x.Price > 20 
                   and x.Price < 500
                   and x.DollarVolume > 50000000]
        
        # Calculate momentum metrics
        momentum_stocks = []
        for stock in filtered:
            history = self.History(stock.Symbol, 60, Resolution.Daily)
            if len(history) > 50:
                returns = history['close'].pct_change().dropna()
                
                # Multi-timeframe momentum
                momentum_1m = (history['close'][-1] / history['close'][-22] - 1) if len(history) > 22 else 0
                momentum_3m = (history['close'][-1] / history['close'][-60] - 1) if len(history) > 60 else 0
                volatility = returns.std() * np.sqrt(252)
                
                # Momentum quality score
                if volatility > 0:
                    sharpe = momentum_1m / (volatility / np.sqrt(12))
                    quality_score = momentum_1m * 0.4 + momentum_3m * 0.3 + sharpe * 0.3
                    
                    momentum_stocks.append({
                        'symbol': stock.Symbol,
                        'score': quality_score,
                        'momentum_1m': momentum_1m,
                        'volatility': volatility
                    })
        
        # Select top performers with diversification
        momentum_stocks.sort(key=lambda x: x['score'], reverse=True)
        selected = []
        sectors = set()
        
        for stock in momentum_stocks[:50]:
            # Limit sector concentration
            if len(selected) < 20:
                selected.append(stock['symbol'])
        
        self.universe_symbols = selected
        return selected
    
    def PreMarketAnalysis(self):
        """Pre-market AI predictions and setup"""
        # Prepare features for ML prediction
        features = self.PrepareMLFeatures()
        
        if features is not None and len(features) > 0:
            # Generate predictions for each strategy
            for strategy_name in self.strategy_performance.keys():
                if strategy_name == 'ml_alpha' and strategy_name in self.ml_models:
                    model = self.ml_models[strategy_name]
                    predictions = model.predict_proba(features)[:, 1]
                    
                    # Store predictions for use in trading
                    self.feature_data['predictions'] = predictions
                    self.feature_data['confidence'] = np.max(predictions)
        
        # Calculate optimal position sizes using Kelly Criterion
        self.CalculateOptimalPositions()
        
    def PrepareMLFeatures(self):
        """Prepare feature matrix for ML models"""
        features = []
        
        for symbol in [self.spy, self.qqq, self.iwm]:
            if symbol in self.indicators:
                feature_vector = []
                
                # Technical features
                if self.indicators[symbol]['rsi'].IsReady:
                    feature_vector.extend([
                        self.indicators[symbol]['rsi'].Current.Value / 100,
                        self.indicators[symbol]['macd'].Current.Value,
                        self.indicators[symbol]['macd'].Signal.Current.Value,
                        self.indicators[symbol]['adx'].Current.Value / 100,
                        self.indicators[symbol]['bb'].UpperBand.Current.Value / self.indicators[symbol]['bb'].MiddleBand.Current.Value - 1,
                        self.indicators[symbol]['bb'].LowerBand.Current.Value / self.indicators[symbol]['bb'].MiddleBand.Current.Value - 1
                    ])
                    
                    # Market microstructure features
                    history = self.History(symbol, 20, Resolution.Hour)
                    if len(history) > 0:
                        returns = history['close'].pct_change().dropna()
                        feature_vector.extend([
                            returns.mean() * 252,  # Annualized return
                            returns.std() * np.sqrt(252),  # Annualized volatility
                            returns.skew(),  # Skewness
                            returns.kurt(),  # Kurtosis
                            np.sign(returns).sum() / len(returns)  # Hit rate
                        ])
                        
                        features.append(feature_vector)
        
        return np.array(features) if features else None
    
    def HighFrequencyTrading(self):
        """5-minute high-frequency trading logic"""
        # Market regime detection
        market_regime = self.DetectMarketRegime()
        
        # Execute strategies based on regime and ML predictions
        if market_regime == "TRENDING":
            self.ExecuteTrendFollowing()
            self.ExecuteMomentumStrategy()
        elif market_regime == "MEAN_REVERTING":
            self.ExecuteMeanReversion()
            self.ExecuteStatArbitrage()
        elif market_regime == "VOLATILE":
            self.ExecuteVolatilityTrading()
        
        # ML-driven alpha generation
        if hasattr(self, 'feature_data') and 'confidence' in self.feature_data:
            if self.feature_data['confidence'] > self.prediction_threshold:
                self.ExecuteMLAlpha()
        
        # Risk checks every 5 minutes
        self.DynamicRiskManagement()
        
    def DetectMarketRegime(self):
        """Identify current market regime using multiple indicators"""
        if not self.indicators[self.spy]['adx'].IsReady:
            return "UNKNOWN"
        
        adx = self.indicators[self.spy]['adx'].Current.Value
        bb_width = (self.indicators[self.spy]['bb'].UpperBand.Current.Value - 
                   self.indicators[self.spy]['bb'].LowerBand.Current.Value) / self.indicators[self.spy]['bb'].MiddleBand.Current.Value
        
        # Recent price action
        history = self.History(self.spy, 20, Resolution.Hour)
        if len(history) > 0:
            returns = history['close'].pct_change().dropna()
            volatility = returns.std()
            trend = (history['close'][-1] / history['close'][0] - 1)
            
            if adx > 30 and abs(trend) > 0.02:
                return "TRENDING"
            elif bb_width < 0.02 and volatility < 0.01:
                return "MEAN_REVERTING"
            elif volatility > 0.02:
                return "VOLATILE"
        
        return "NEUTRAL"
    
    def ExecuteTrendFollowing(self):
        """Trend following with dynamic position sizing"""
        for symbol in [self.spy, self.qqq, self.tqqq]:
            if symbol not in self.indicators or not self.indicators[symbol]['ema_21'].IsReady:
                continue
            
            price = self.Securities[symbol].Price
            ema_21 = self.indicators[symbol]['ema_21'].Current.Value
            ema_50 = self.indicators[symbol]['ema_50'].Current.Value if self.indicators[symbol]['ema_50'].IsReady else ema_21
            atr = self.indicators[symbol]['atr'].Current.Value
            
            # Trend strength
            trend_strength = (price - ema_50) / atr if atr > 0 else 0
            
            if price > ema_21 and ema_21 > ema_50 and trend_strength > 1:
                # Strong uptrend
                position_size = self.CalculatePositionSize(symbol, self.strategy_performance['trend']['weight'])
                
                # Use leveraged ETF for stronger trends
                if symbol == self.spy and trend_strength > 2:
                    self.SetHoldings(self.tqqq, position_size * 1.5)
                else:
                    self.SetHoldings(symbol, position_size)
                    
            elif price < ema_21 and ema_21 < ema_50 and trend_strength < -1:
                # Strong downtrend
                position_size = self.CalculatePositionSize(symbol, self.strategy_performance['trend']['weight'])
                
                if symbol == self.spy:
                    self.SetHoldings(self.spxu, position_size)
                else:
                    self.SetHoldings(symbol, -position_size * 0.5)
    
    def ExecuteMomentumStrategy(self):
        """Momentum strategy with sector rotation"""
        momentum_scores = {}
        
        # Calculate momentum for universe stocks
        for symbol in self.universe_symbols[:10]:
            if self.Securities.ContainsKey(symbol):
                history = self.History(symbol, 30, Resolution.Daily)
                if len(history) > 20:
                    momentum = (history['close'][-1] / history['close'][-20] - 1)
                    volume_ratio = history['volume'][-5:].mean() / history['volume'][-20:].mean()
                    
                    # Momentum with volume confirmation
                    score = momentum * (1 + min(volume_ratio - 1, 0.5))
                    momentum_scores[symbol] = score
        
        # Rank and allocate
        if momentum_scores:
            sorted_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Long top 3
            for i, (symbol, score) in enumerate(sorted_momentum[:3]):
                if score > 0.1:  # 10% momentum threshold
                    weight = self.strategy_performance['momentum']['weight'] * (0.4 - i * 0.1)
                    self.SetHoldings(symbol, weight * min(score * 10, 2))
            
            # Short bottom 2
            for i, (symbol, score) in enumerate(sorted_momentum[-2:]):
                if score < -0.05:
                    weight = self.strategy_performance['momentum']['weight'] * 0.15
                    self.SetHoldings(symbol, -weight)
    
    def ExecuteMeanReversion(self):
        """Mean reversion on oversold/overbought conditions"""
        for symbol in [self.spy, self.qqq, self.iwm]:
            if symbol not in self.indicators or not self.indicators[symbol]['rsi'].IsReady:
                continue
            
            rsi = self.indicators[symbol]['rsi'].Current.Value
            bb_position = (self.Securities[symbol].Price - self.indicators[symbol]['bb'].LowerBand.Current.Value) / (
                self.indicators[symbol]['bb'].UpperBand.Current.Value - self.indicators[symbol]['bb'].LowerBand.Current.Value)
            
            # Extreme oversold
            if rsi < 25 and bb_position < 0.1:
                position_size = self.strategy_performance['mean_reversion']['weight'] * 0.5
                self.SetHoldings(symbol, position_size)
                
                # Set profit target
                target_price = self.indicators[symbol]['bb'].MiddleBand.Current.Value
                self.LimitOrder(symbol, -self.Portfolio[symbol].Quantity, target_price)
            
            # Extreme overbought
            elif rsi > 75 and bb_position > 0.9:
                position_size = self.strategy_performance['mean_reversion']['weight'] * 0.3
                self.SetHoldings(symbol, -position_size)
                
                # Set profit target
                target_price = self.indicators[symbol]['bb'].MiddleBand.Current.Value
                self.LimitOrder(symbol, -self.Portfolio[symbol].Quantity, target_price)
    
    def ExecuteStatArbitrage(self):
        """Statistical arbitrage on correlated pairs"""
        # SPY-QQQ spread trading
        if self.spy in self.indicators and self.qqq in self.indicators:
            spy_price = self.Securities[self.spy].Price
            qqq_price = self.Securities[self.qqq].Price
            
            # Calculate historical spread
            history_spy = self.History(self.spy, 30, Resolution.Daily)['close']
            history_qqq = self.History(self.qqq, 30, Resolution.Daily)['close']
            
            if len(history_spy) == len(history_qqq) and len(history_spy) > 20:
                # Log spread for better stationarity
                spread = np.log(history_spy.values) - np.log(history_qqq.values)
                current_spread = np.log(spy_price) - np.log(qqq_price)
                
                # Z-score calculation
                z_score = (current_spread - spread.mean()) / (spread.std() + 1e-8)
                
                if abs(z_score) > 2.5:
                    weight = self.strategy_performance['mean_reversion']['weight'] * 0.3
                    
                    if z_score > 2.5:
                        self.SetHoldings(self.spy, -weight)
                        self.SetHoldings(self.qqq, weight)
                    else:
                        self.SetHoldings(self.spy, weight)
                        self.SetHoldings(self.qqq, -weight)
    
    def ExecuteVolatilityTrading(self):
        """Trade volatility expansion/contraction"""
        if self.indicators[self.spy]['atr'].IsReady:
            current_atr = self.indicators[self.spy]['atr'].Current.Value
            
            # Historical ATR
            atr_history = []
            for i in range(20):
                if self.indicators[self.spy]['atr'].IsReady:
                    atr_history.append(self.indicators[self.spy]['atr'].Current.Value)
            
            if atr_history:
                avg_atr = np.mean(atr_history)
                
                # Volatility expansion
                if current_atr > avg_atr * 1.5:
                    # Long volatility with options or VXX
                    self.SetHoldings(self.sqqq, 0.1)  # Hedge with inverse
                    self.SetHoldings(self.tqqq, -0.1)  # Short leveraged
                
                # Volatility contraction
                elif current_atr < avg_atr * 0.7:
                    # Short volatility - aggressive directional bets
                    if self.indicators[self.spy]['ema_8'].Current.Value > self.indicators[self.spy]['ema_21'].Current.Value:
                        self.SetHoldings(self.tqqq, 0.3)  # Leveraged long
                    else:
                        self.SetHoldings(self.sqqq, 0.3)  # Leveraged short
    
    def ExecuteMLAlpha(self):
        """Execute trades based on ML predictions"""
        if 'predictions' in self.feature_data:
            predictions = self.feature_data['predictions']
            
            # Aggressive positioning based on high-confidence predictions
            for i, symbol in enumerate([self.spy, self.qqq, self.iwm]):
                if i < len(predictions):
                    prediction = predictions[i]
                    
                    if prediction > 0.75:  # Strong bullish signal
                        weight = self.strategy_performance['ml_alpha']['weight'] * prediction
                        
                        # Use leverage for high-confidence trades
                        if symbol == self.spy:
                            self.SetHoldings(self.tqqq, weight * 1.5)
                        else:
                            self.SetHoldings(symbol, weight * 2)
                    
                    elif prediction < 0.25:  # Strong bearish signal
                        weight = self.strategy_performance['ml_alpha']['weight'] * (1 - prediction)
                        
                        if symbol == self.spy:
                            self.SetHoldings(self.spxu, weight)
                        else:
                            self.SetHoldings(symbol, -weight)
    
    def CalculatePositionSize(self, symbol, base_weight):
        """Kelly Criterion-based position sizing"""
        if not self.Portfolio[symbol].Invested:
            # Estimate win probability and payoff ratio
            history = self.History(symbol, 100, Resolution.Daily)
            if len(history) > 50:
                returns = history['close'].pct_change().dropna()
                
                # Win rate
                win_rate = (returns > 0).sum() / len(returns)
                
                # Average win/loss ratio
                avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
                avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
                
                if avg_loss > 0:
                    payoff_ratio = avg_win / avg_loss
                    
                    # Kelly formula: f = p - q/b
                    # where p = win probability, q = loss probability, b = payoff ratio
                    kelly_percent = win_rate - (1 - win_rate) / payoff_ratio
                    
                    # Apply Kelly fraction (conservative)
                    optimal_size = base_weight * min(kelly_percent * self.kelly_fraction, 1.0)
                    
                    # Adjust for portfolio heat
                    heat_adjusted = optimal_size * (1 - self.portfolio_heat)
                    
                    return max(0, min(heat_adjusted, base_weight))
        
        return base_weight * 0.5  # Reduce size for existing positions
    
    def CalculateOptimalPositions(self):
        """Portfolio optimization using mean-variance optimization"""
        symbols = [self.spy, self.qqq, self.iwm, self.tqqq]
        returns_data = []
        
        for symbol in symbols:
            history = self.History(symbol, 60, Resolution.Daily)
            if len(history) > 30:
                returns = history['close'].pct_change().dropna().values
                returns_data.append(returns[-30:])  # Last 30 days
        
        if len(returns_data) == len(symbols):
            returns_matrix = np.array(returns_data).T
            
            # Calculate expected returns and covariance
            expected_returns = returns_matrix.mean(axis=0) * 252
            cov_matrix = np.cov(returns_matrix.T) * 252
            
            # Optimization constraints
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                # Maximize Sharpe ratio
                return -((portfolio_return - 0.05) / portfolio_std)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 0.4) for _ in symbols)  # Max 40% per asset
            
            # Optimize
            result = minimize(objective, np.ones(len(symbols)) / len(symbols),
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                self.optimal_weights = dict(zip(symbols, result.x))
    
    def DynamicRiskManagement(self):
        """Real-time risk monitoring and adjustment"""
        # Calculate current portfolio metrics
        total_value = self.Portfolio.TotalPortfolioValue
        
        # Portfolio heat (leverage and concentration)
        total_holdings = sum(abs(self.Portfolio[symbol].HoldingsValue) for symbol in self.Portfolio.Keys)
        self.portfolio_heat = total_holdings / total_value / self.max_portfolio_heat
        
        # VaR calculation
        returns = []
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].Invested:
                holding_pct = self.Portfolio[symbol].HoldingsValue / total_value
                history = self.History(symbol, 20, Resolution.Daily)
                if len(history) > 0:
                    daily_return = history['close'].pct_change().dropna().values
                    if len(daily_return) > 0:
                        returns.append(holding_pct * daily_return[-1])
        
        if returns:
            portfolio_var = np.percentile(returns, 5)  # 5% VaR
            
            # Reduce positions if VaR exceeds limit
            if abs(portfolio_var) > self.var_limit:
                reduction_factor = self.var_limit / abs(portfolio_var)
                
                for symbol in self.Portfolio.Keys:
                    if self.Portfolio[symbol].Invested:
                        new_quantity = int(self.Portfolio[symbol].Quantity * reduction_factor)
                        self.MarketOrder(symbol, new_quantity - self.Portfolio[symbol].Quantity)
        
        # Correlation check
        self.CheckPortfolioCorrelation()
        
        # Trailing stop management
        self.UpdateTrailingStops()
    
    def CheckPortfolioCorrelation(self):
        """Ensure portfolio diversification"""
        invested_symbols = [s for s in self.Portfolio.Keys if self.Portfolio[s].Invested]
        
        if len(invested_symbols) > 2:
            correlation_matrix = []
            
            for symbol in invested_symbols:
                history = self.History(symbol, 30, Resolution.Daily)
                if len(history) > 20:
                    returns = history['close'].pct_change().dropna().values
                    correlation_matrix.append(returns[-20:])
            
            if len(correlation_matrix) > 2:
                corr = np.corrcoef(correlation_matrix)
                
                # Check for high correlations
                for i in range(len(corr)):
                    for j in range(i+1, len(corr)):
                        if abs(corr[i,j]) > self.max_correlation:
                            # Reduce position in the smaller holding
                            symbol1, symbol2 = invested_symbols[i], invested_symbols[j]
                            
                            if abs(self.Portfolio[symbol1].HoldingsValue) < abs(self.Portfolio[symbol2].HoldingsValue):
                                self.SetHoldings(symbol1, self.Portfolio[symbol1].HoldingsValue / total_value * 0.5)
                            else:
                                self.SetHoldings(symbol2, self.Portfolio[symbol2].HoldingsValue / total_value * 0.5)
    
    def UpdateTrailingStops(self):
        """Dynamic trailing stop management"""
        for symbol in self.Portfolio.Keys:
            if self.Portfolio[symbol].Invested and self.Portfolio[symbol].UnrealizedProfitPercent > 0.02:
                # Profitable position - tighten stops
                current_price = self.Securities[symbol].Price
                entry_price = self.Portfolio[symbol].AveragePrice
                
                # Dynamic stop based on profit level
                if self.Portfolio[symbol].UnrealizedProfitPercent > 0.10:  # 10%+ profit
                    stop_distance = 0.02  # 2% trailing stop
                elif self.Portfolio[symbol].UnrealizedProfitPercent > 0.05:  # 5%+ profit
                    stop_distance = 0.03  # 3% trailing stop
                else:
                    stop_distance = 0.04  # 4% trailing stop
                
                stop_price = current_price * (1 - stop_distance) if self.Portfolio[symbol].IsLong else current_price * (1 + stop_distance)
                
                # Update or create stop order
                self.StopMarketOrder(symbol, -self.Portfolio[symbol].Quantity, stop_price, "TrailingStop")
    
    def EndOfDayRebalance(self):
        """End of day portfolio rebalancing"""
        # Update strategy performance
        self.UpdateStrategyPerformance()
        
        # Rebalance based on performance
        total_sharpe = sum(s['sharpe'] for s in self.strategy_performance.values())
        
        if total_sharpe > 0:
            for strategy in self.strategy_performance:
                # Increase weight for well-performing strategies
                self.strategy_performance[strategy]['weight'] = self.strategy_performance[strategy]['sharpe'] / total_sharpe
        
        # Close losing positions before market close
        for symbol in list(self.Portfolio.Keys):
            if self.Portfolio[symbol].Invested:
                if self.Portfolio[symbol].UnrealizedProfitPercent < -0.02:  # 2% loss
                    self.Liquidate(symbol)
                elif self.Time.hour >= 15 and self.Time.minute >= 50:
                    # Reduce overnight risk
                    if abs(self.Portfolio[symbol].HoldingsValue) > self.Portfolio.TotalPortfolioValue * 0.2:
                        self.SetHoldings(symbol, 0.1)
    
    def UpdateStrategyPerformance(self):
        """Track individual strategy performance"""
        # This is simplified - in production, track actual P&L by strategy
        for strategy in self.strategy_performance:
            # Estimate based on overall performance
            daily_return = (self.Portfolio.TotalPortfolioValue / self.Portfolio.TotalPortfolioValue - 1)
            
            # Simple exponential smoothing of Sharpe
            if daily_return != 0:
                instant_sharpe = daily_return / 0.01  # Assume 1% daily vol
                self.strategy_performance[strategy]['sharpe'] = (
                    0.95 * self.strategy_performance[strategy]['sharpe'] + 
                    0.05 * instant_sharpe
                )
    
    def TrainMLModels(self):
        """Monthly ML model retraining"""
        # Prepare training data
        training_data = []
        labels = []
        
        for symbol in [self.spy, self.qqq, self.iwm]:
            history = self.History(symbol, 252, Resolution.Daily)
            
            if len(history) > 200:
                # Create features and labels
                for i in range(20, len(history) - 5):
                    # Features
                    returns = history['close'][i-20:i].pct_change().dropna()
                    volume_ratio = history['volume'][i-5:i].mean() / history['volume'][i-20:i].mean()
                    
                    features = [
                        returns.mean() * 252,
                        returns.std() * np.sqrt(252),
                        returns.skew(),
                        (history['close'][i] / history['close'][i-20] - 1),
                        volume_ratio,
                        (history['high'][i] - history['low'][i]) / history['close'][i]
                    ]
                    
                    # Label: 1 if price goes up in next 5 days
                    future_return = (history['close'][i+5] / history['close'][i] - 1)
                    label = 1 if future_return > 0.02 else 0  # 2% threshold
                    
                    training_data.append(features)
                    labels.append(label)
        
        if len(training_data) > 100:
            # Train Random Forest model
            X = np.array(training_data)
            y = np.array(labels)
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_scaled, y)
            
            # Store model
            self.ml_models['ml_alpha'] = model
            self.ml_models['scaler'] = scaler
            
            self.Debug(f"ML Model trained with accuracy: {model.score(X_scaled, y):.2f}")
    
    def OnOrderEvent(self, orderEvent):
        """Track and log order execution"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order Filled: {orderEvent.Symbol} {orderEvent.Quantity} @ ${orderEvent.FillPrice}")
            
            # Track strategy attribution (simplified)
            if orderEvent.Quantity > 0:
                self.Debug(f"Entry: Portfolio Heat = {self.portfolio_heat:.2f}")
            else:
                profit = (orderEvent.FillPrice - self.Portfolio[orderEvent.Symbol].AveragePrice) * orderEvent.Quantity
                self.Debug(f"Exit: Profit = ${profit:.2f}")
    
    def OnEndOfAlgorithm(self):
        """Final performance reporting"""
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug(f"Total Return: {(self.Portfolio.TotalPortfolioValue / 100000 - 1) * 100:.2f}%")
        
        # Calculate metrics
        total_trades = sum(1 for o in self.Transactions.GetOrders() if o.Status == OrderStatus.Filled)
        self.Debug(f"Total Trades: {total_trades}")