# region imports
from AlgorithmImports import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Indicators import *
from QuantConnect.Orders import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from research_engine import ResearchEngine
# endregion

class MarketRegime:
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"

class RiskLevel:
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class FinancialAnalyzer:
    """
    Advanced financial analyzer that provides enhanced market intelligence
    for trading strategies.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the financial analyzer.
        
        Args:
            algorithm: The QCAlgorithm instance
        """
        self.algorithm = algorithm
        self.market_regime = MarketRegime.NEUTRAL
        self.risk_level = RiskLevel.MEDIUM
        self.sentiment_score = 0.0
        
    def analyze_market_regime(self, fast_ma, slow_ma, volatility_indicator):
        """
        Determine the current market regime based on moving averages and volatility.
        
        Args:
            fast_ma: Fast moving average indicator
            slow_ma: Slow moving average indicator
            volatility_indicator: Volatility indicator (e.g., ATR)
            
        Returns:
            str: The identified market regime
        """
        if not fast_ma.IsReady or not slow_ma.IsReady or not volatility_indicator.IsReady:
            return MarketRegime.NEUTRAL
            
        # Trend analysis
        current_fast = fast_ma.Current.Value
        current_slow = slow_ma.Current.Value
        
        # Volatility analysis
        current_volatility = volatility_indicator.Current.Value
        
        # Determine market regime
        if current_fast > current_slow:
            if current_volatility < volatility_indicator.Current.Value * 0.8:
                self.market_regime = MarketRegime.BULL
            else:
                self.market_regime = MarketRegime.TRENDING
        elif current_fast < current_slow:
            if current_volatility > volatility_indicator.Current.Value * 1.2:
                self.market_regime = MarketRegime.BEAR
            else:
                self.market_regime = MarketRegime.TRENDING
        else:
            if current_volatility > volatility_indicator.Current.Value * 1.5:
                self.market_regime = MarketRegime.VOLATILE
            else:
                self.market_regime = MarketRegime.RANGING
                
        return self.market_regime
    
    def assess_risk_level(self, volatility_indicator):
        """
        Assess the current market risk level based on volatility.
        
        Args:
            volatility_indicator: Volatility indicator (e.g., ATR)
            
        Returns:
            str: The assessed risk level
        """
        if not volatility_indicator.IsReady:
            return RiskLevel.MEDIUM
            
        # Get current volatility
        current_volatility = volatility_indicator.Current.Value
        
        # Base risk assessment on volatility
        if current_volatility < 0.01:  # Low volatility
            base_risk = RiskLevel.LOW
        elif current_volatility < 0.02:  # Medium volatility
            base_risk = RiskLevel.MEDIUM
        elif current_volatility < 0.03:  # High volatility
            base_risk = RiskLevel.HIGH
        else:  # Extreme volatility
            base_risk = RiskLevel.EXTREME
            
        self.risk_level = base_risk
        return self.risk_level
    
    def get_optimal_position_size(self, symbol, portfolio_value, risk_per_trade=0.01):
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            symbol: The symbol to trade
            portfolio_value: Total portfolio value
            risk_per_trade: Maximum risk per trade as a fraction of portfolio
            
        Returns:
            float: Optimal position size as a fraction of portfolio
        """
        # Adjust risk based on market regime and risk level
        adjusted_risk = risk_per_trade
        
        if self.risk_level == RiskLevel.HIGH:
            adjusted_risk *= 0.7
        elif self.risk_level == RiskLevel.EXTREME:
            adjusted_risk *= 0.5
            
        if self.market_regime == MarketRegime.VOLATILE:
            adjusted_risk *= 0.8
        elif self.market_regime == MarketRegime.BEAR:
            adjusted_risk *= 0.7
            
        return adjusted_risk

class EnhancedTradingStrategy(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2010, 1, 1)  # Set Start Date to 15 years ago
        self.set_end_date(2025, 1, 1)    # Set End Date
        self.set_cash(100000)            # Set Strategy Cash
        self.set_brokerage_model(BrokerageName.DEFAULT)
        
        # Add assets - including leveraged ETFs for enhanced performance
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.qqq = self.add_equity("QQQ", Resolution.DAILY).symbol
        self.tqqq = self.add_equity("TQQQ", Resolution.DAILY).symbol  # 3x leveraged QQQ
        self.spxl = self.add_equity("SPXL", Resolution.DAILY).symbol  # 3x leveraged S&P 500
        self.sqqq = self.add_equity("SQQQ", Resolution.DAILY).symbol  # Inverse 3x QQQ for hedging
        
        # Set benchmark
        self.set_benchmark("SPY")
        
        # Initialize indicators
        self.fast_ema = self.ema(self.spy, 20, Resolution.DAILY)
        self.slow_ema = self.ema(self.spy, 50, Resolution.DAILY)
        self.very_slow_ema = self.ema(self.spy, 200, Resolution.DAILY)
        
        # Volatility indicator
        self.volatility_atr = self.atr(self.spy, 14)
        
        # RSI for momentum
        self.momentum_rsi = self.rsi(self.spy, 14)
        
        # Initialize financial analyzer
        self.analyzer = FinancialAnalyzer(self)
        
        # Initialize research engine
        self.research = ResearchEngine(self)
        
        # Performance tracking
        self.previous_portfolio_value = self.portfolio.total_portfolio_value
        self.highest_portfolio_value = self.portfolio.total_portfolio_value
        self.drawdown = 0
        
        # Schedule functions at different timeframes
        # Daily analysis and potential adjustments
        self.schedule.on(self.date_rules.every_day(),
                        self.time_rules.before_market_close(self.spy, 10),
                        self.daily_analysis)
        
        # Weekly rebalancing - Every Monday
        self.schedule.on(self.date_rules.every(DayOfWeek.MONDAY),
                        self.time_rules.after_market_open(self.spy, 30),
                        self.weekly_rebalance)
        
        # Monthly strategy review and major adjustments
        self.schedule.on(self.date_rules.month_start(),
                        self.time_rules.after_market_open(self.spy, 30),
                        self.monthly_strategy_review)
        
        # Initialize tracking variables
        self.last_regime = MarketRegime.NEUTRAL
        self.current_regime = MarketRegime.NEUTRAL
        self.risk_level = RiskLevel.MEDIUM
        self.current_positions = {}
        self.stop_loss_levels = {}
        self.take_profit_levels = {}

    def daily_analysis(self):
        """Daily market analysis and risk management"""
        if not self.volatility_atr.IsReady or not self.fast_ema.IsReady or not self.slow_ema.IsReady:
            return
            
        # Update market regime
        self.current_regime = self.analyzer.analyze_market_regime(
            self.fast_ema, 
            self.slow_ema, 
            self.volatility_atr
        )
        
        # Update risk level
        self.risk_level = self.analyzer.assess_risk_level(self.volatility_atr)
        
        # Track performance
        current_value = self.portfolio.total_portfolio_value
        daily_return = (current_value / self.previous_portfolio_value) - 1
        self.previous_portfolio_value = current_value
        
        # Update highest portfolio value and drawdown
        if current_value > self.highest_portfolio_value:
            self.highest_portfolio_value = current_value
        
        self.drawdown = 1 - (current_value / self.highest_portfolio_value)
        
        # Log daily performance
        self.log(f"Daily Analysis - Regime: {self.current_regime}, Risk: {self.risk_level}")
        self.log(f"Daily Return: {daily_return:.2%}, Drawdown: {self.drawdown:.2%}")
        
        # Check stop-loss and take-profit levels
        self.check_risk_management()
        
        # Plot performance metrics
        self.plot("Performance", "Portfolio Value", current_value)
        self.plot("Performance", "Drawdown", self.drawdown)
        self.plot("Market Regime", "Regime", 1 if self.current_regime == MarketRegime.BULL else 
                                           -1 if self.current_regime == MarketRegime.BEAR else 0)

    def weekly_rebalance(self):
        """Weekly portfolio rebalancing"""
        if not self.volatility_atr.IsReady or not self.fast_ema.IsReady or not self.slow_ema.IsReady:
            return
            
        # Get current market conditions
        regime = self.current_regime
        risk = self.risk_level
        
        # Determine allocation based on market regime and risk level
        if regime == MarketRegime.BULL:
            # In bull markets, use leveraged ETFs for enhanced returns
            if risk == RiskLevel.LOW or risk == RiskLevel.MEDIUM:
                self.adjust_position(self.tqqq, 0.4)  # 40% in 3x leveraged QQQ
                self.adjust_position(self.spxl, 0.4)  # 40% in 3x leveraged S&P 500
                self.adjust_position(self.spy, 0.2)   # 20% in regular S&P 500
                self.liquidate(self.sqqq)             # No inverse ETFs in bull market
            else:
                # More conservative in higher risk environments
                self.adjust_position(self.tqqq, 0.3)
                self.adjust_position(self.spxl, 0.3)
                self.adjust_position(self.spy, 0.4)
                self.liquidate(self.sqqq)
        
        elif regime == MarketRegime.BEAR:
            # In bear markets, use inverse ETFs and reduce exposure
            if risk == RiskLevel.HIGH or risk == RiskLevel.EXTREME:
                self.adjust_position(self.sqqq, 0.3)  # 30% in inverse 3x QQQ
                self.adjust_position(self.spy, 0.2)   # 20% in S&P 500
                self.liquidate(self.tqqq)             # Liquidate leveraged long ETFs
                self.liquidate(self.spxl)
            else:
                self.adjust_position(self.sqqq, 0.2)
                self.adjust_position(self.spy, 0.3)
                self.liquidate(self.tqqq)
                self.liquidate(self.spxl)
        
        else:  # NEUTRAL, VOLATILE, TRENDING, or RANGING
            # Balanced approach in mixed or uncertain markets
            if risk == RiskLevel.LOW:
                self.adjust_position(self.spy, 0.4)
                self.adjust_position(self.qqq, 0.4)
                self.adjust_position(self.tqqq, 0.2)
                self.liquidate(self.sqqq)
                self.liquidate(self.spxl)
            elif risk == RiskLevel.MEDIUM:
                self.adjust_position(self.spy, 0.5)
                self.adjust_position(self.qqq, 0.5)
                self.liquidate(self.tqqq)
                self.liquidate(self.sqqq)
                self.liquidate(self.spxl)
            else:  # HIGH or EXTREME risk
                self.adjust_position(self.spy, 0.7)
                self.adjust_position(self.sqqq, 0.1)  # Small hedge
                self.liquidate(self.qqq)
                self.liquidate(self.tqqq)
                self.liquidate(self.spxl)
        
        # Set stop-loss and take-profit levels
        self.set_risk_management_levels()
        
        self.log(f"Weekly Rebalance - Regime: {regime}, Risk: {risk}")

    def monthly_strategy_review(self):
        """Monthly comprehensive strategy review"""
        if not self.very_slow_ema.IsReady:
            return
            
        # Analyze long-term trend
        long_term_trend = "bullish" if self.fast_ema.Current.Value > self.very_slow_ema.Current.Value else "bearish"
        
        # Calculate monthly performance
        monthly_return = (self.portfolio.total_portfolio_value / self.previous_portfolio_value) - 1
        
        # Log monthly performance
        self.log(f"Monthly Strategy Review - Long-term trend: {long_term_trend}")
        self.log(f"Monthly Return: {monthly_return:.2%}, Total Drawdown: {self.drawdown:.2%}")
        
        # Adjust strategy based on long-term trend
        if long_term_trend == "bullish":
            # In long-term bull markets, we can be more aggressive
            if self.current_regime == MarketRegime.BULL:
                # Increase allocation to leveraged ETFs
                self.adjust_position(self.tqqq, 0.5)
                self.adjust_position(self.spxl, 0.3)
                self.adjust_position(self.spy, 0.2)
                self.liquidate(self.sqqq)
            elif self.current_regime == MarketRegime.NEUTRAL:
                # Balanced but still bullish
                self.adjust_position(self.tqqq, 0.3)
                self.adjust_position(self.spy, 0.4)
                self.adjust_position(self.qqq, 0.3)
                self.liquidate(self.sqqq)
                self.liquidate(self.spxl)
        else:  # bearish long-term trend
            # In long-term bear markets, be more defensive
            if self.current_regime == MarketRegime.BEAR:
                # Increase hedging
                self.adjust_position(self.sqqq, 0.4)
                self.adjust_position(self.spy, 0.3)
                self.liquidate(self.tqqq)
                self.liquidate(self.spxl)
                self.liquidate(self.qqq)
            elif self.current_regime == MarketRegime.NEUTRAL:
                # Defensive positioning
                self.adjust_position(self.spy, 0.6)
                self.adjust_position(self.sqqq, 0.2)
                self.liquidate(self.tqqq)
                self.liquidate(self.spxl)
                self.liquidate(self.qqq)
        
        # Reset performance tracking for the new month
        self.previous_portfolio_value = self.portfolio.total_portfolio_value

    def adjust_position(self, symbol, target_pct):
        """
        Adjust position to target percentage of portfolio.
        
        Args:
            symbol: The symbol to adjust
            target_pct: Target percentage of portfolio (0.0 to 1.0)
        """
        self.set_holdings(symbol, target_pct)
        self.current_positions[symbol] = target_pct
        
        # Log the position adjustment
        self.log(f"Adjusted position for {symbol}: {target_pct:.1%} of portfolio")

    def set_risk_management_levels(self):
        """Set stop-loss and take-profit levels for current positions"""
        for symbol, allocation in self.current_positions.items():
            if allocation > 0 and self.securities.contains_key(symbol):
                current_price = self.securities[symbol].price
                
                # Set stop-loss at 5-10% below current price, depending on risk level
                stop_loss_pct = 0.05  # Default 5%
                
                if self.risk_level == RiskLevel.HIGH:
                    stop_loss_pct = 0.07  # 7%
                elif self.risk_level == RiskLevel.EXTREME:
                    stop_loss_pct = 0.10  # 10%
                
                self.stop_loss_levels[symbol] = current_price * (1 - stop_loss_pct)
                
                # Set take-profit at 10-20% above current price
                take_profit_pct = 0.15  # Default 15%
                
                if self.current_regime == MarketRegime.BULL:
                    take_profit_pct = 0.20  # 20% in bull markets
                elif self.current_regime == MarketRegime.BEAR:
                    take_profit_pct = 0.10  # 10% in bear markets
                
                self.take_profit_levels[symbol] = current_price * (1 + take_profit_pct)
                
                self.log(f"Risk management for {symbol}: Stop-loss at {self.stop_loss_levels[symbol]:.2f}, Take-profit at {self.take_profit_levels[symbol]:.2f}")

    def check_risk_management(self):
        """Check if stop-loss or take-profit levels have been hit"""
        for symbol in list(self.current_positions.keys()):
            if not self.securities.contains_key(symbol):
                continue
                
            current_price = self.securities[symbol].price
            
            # Check stop-loss
            if symbol in self.stop_loss_levels and current_price <= self.stop_loss_levels[symbol]:
                self.liquidate(symbol)
                self.log(f"Stop-loss triggered for {symbol} at {current_price:.2f}")
                del self.current_positions[symbol]
                del self.stop_loss_levels[symbol]
                if symbol in self.take_profit_levels:
                    del self.take_profit_levels[symbol]
            
            # Check take-profit
            elif symbol in self.take_profit_levels and current_price >= self.take_profit_levels[symbol]:
                # Take partial profits (sell half)
                current_allocation = self.current_positions[symbol]
                new_allocation = current_allocation / 2
                self.adjust_position(symbol, new_allocation)
                self.log(f"Take-profit triggered for {symbol} at {current_price:.2f}, reducing position to {new_allocation:.1%}")
                
                # Update stop-loss to break-even
                if symbol in self.stop_loss_levels:
                    # Use current price as new stop-loss (break-even)
                    self.stop_loss_levels[symbol] = current_price

    def on_data(self, data: Slice):
        """Process market data"""
        # Update portfolio value chart
        self.plot("Portfolio", "Value", self.portfolio.total_portfolio_value)
        
        # Check if regime has changed
        if self.current_regime != self.last_regime:
            self.log(f"Market regime changed from {self.last_regime} to {self.current_regime}")
            self.last_regime = self.current_regime
