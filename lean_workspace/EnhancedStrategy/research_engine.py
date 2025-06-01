"""
Research Engine Module

This module provides advanced research capabilities for the trading strategy,
integrating with the MCP Financial Analyzer to enhance decision-making.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from AlgorithmImports import *

class ResearchEngine:
    """
    Advanced research engine that provides enhanced market intelligence
    and research capabilities for trading strategies.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the research engine.
        
        Args:
            algorithm: The QCAlgorithm instance
        """
        self.algorithm = algorithm
        self.research_cache = {}
        self.last_update = {}
        
    def analyze_sector_performance(self, sectors):
        """
        Analyze performance of market sectors.
        
        Args:
            sectors: Dictionary of sector ETFs
            
        Returns:
            dict: Sector performance data
        """
        sector_performance = {}
        
        for sector_name, symbol in sectors.items():
            if not self.algorithm.securities.contains_key(symbol):
                continue
                
            # Get historical data
            history = self.algorithm.history(symbol, 30, Resolution.DAILY)
            
            if history.empty:
                continue
                
            # Calculate performance metrics
            returns = history["close"].pct_change().dropna()
            performance = returns.sum()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate momentum (last 5 days vs previous 25 days)
            recent_perf = history["close"].iloc[-5:].pct_change().sum()
            earlier_perf = history["close"].iloc[-30:-5].pct_change().sum()
            momentum = recent_perf - earlier_perf
            
            sector_performance[sector_name] = {
                "performance": performance,
                "volatility": volatility,
                "momentum": momentum,
                "sharpe": performance / volatility if volatility > 0 else 0
            }
            
        # Sort sectors by performance
        sorted_sectors = sorted(
            sector_performance.items(),
            key=lambda x: x[1]["performance"],
            reverse=True
        )
        
        result = {
            "sectors": sector_performance,
            "top_sectors": [s[0] for s in sorted_sectors[:3]],
            "bottom_sectors": [s[0] for s in sorted_sectors[-3:]]
        }
        
        return result
    
    def analyze_correlation_matrix(self, symbols):
        """
        Calculate correlation matrix between assets.
        
        Args:
            symbols: List of symbols
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Get historical data for all symbols
        history_dict = {}
        
        for symbol in symbols:
            if not self.algorithm.securities.contains_key(symbol):
                continue
                
            history = self.algorithm.history(symbol, 60, Resolution.DAILY)
            
            if not history.empty:
                history_dict[str(symbol)] = history["close"]
        
        if not history_dict:
            return pd.DataFrame()
            
        # Create price DataFrame
        prices = pd.DataFrame(history_dict)
        
        # Calculate correlation matrix
        correlation_matrix = prices.pct_change().dropna().corr()
        
        return correlation_matrix
    
    def identify_market_anomalies(self, symbols):
        """
        Identify market anomalies and potential trading opportunities.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            dict: Identified anomalies and opportunities
        """
        anomalies = {}
        
        for symbol in symbols:
            if not self.algorithm.securities.contains_key(symbol):
                continue
                
            # Get historical data
            history = self.algorithm.history(symbol, 90, Resolution.DAILY)
            
            if history.empty:
                continue
                
            # Calculate metrics
            returns = history["close"].pct_change().dropna()
            current_price = history["close"].iloc[-1]
            avg_20d = history["close"].rolling(20).mean().iloc[-1]
            avg_50d = history["close"].rolling(50).mean().iloc[-1]
            avg_90d = history["close"].rolling(90).mean().iloc[-1]
            
            # Calculate volatility
            volatility_20d = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            volatility_90d = returns.rolling(90).std().iloc[-1] * np.sqrt(252)
            
            # Identify anomalies
            symbol_anomalies = []
            
            # Price significantly below moving averages
            if current_price < avg_20d * 0.95 and current_price < avg_50d * 0.95:
                symbol_anomalies.append({
                    "type": "price_below_ma",
                    "description": f"Price significantly below 20-day and 50-day moving averages",
                    "severity": "high"
                })
                
            # Price significantly above moving averages
            if current_price > avg_20d * 1.05 and current_price > avg_50d * 1.05:
                symbol_anomalies.append({
                    "type": "price_above_ma",
                    "description": f"Price significantly above 20-day and 50-day moving averages",
                    "severity": "medium"
                })
                
            # Volatility spike
            if volatility_20d > volatility_90d * 1.5:
                symbol_anomalies.append({
                    "type": "volatility_spike",
                    "description": f"Recent volatility significantly higher than historical",
                    "severity": "high"
                })
                
            # Volatility contraction
            if volatility_20d < volatility_90d * 0.5:
                symbol_anomalies.append({
                    "type": "volatility_contraction",
                    "description": f"Recent volatility significantly lower than historical",
                    "severity": "medium"
                })
                
            # Moving average crossover
            if avg_20d > avg_50d and history["close"].rolling(20).mean().shift(1).iloc[-1] <= history["close"].rolling(50).mean().shift(1).iloc[-1]:
                symbol_anomalies.append({
                    "type": "ma_crossover_bullish",
                    "description": f"20-day MA crossed above 50-day MA",
                    "severity": "medium"
                })
                
            # Moving average crossunder
            if avg_20d < avg_50d and history["close"].rolling(20).mean().shift(1).iloc[-1] >= history["close"].rolling(50).mean().shift(1).iloc[-1]:
                symbol_anomalies.append({
                    "type": "ma_crossover_bearish",
                    "description": f"20-day MA crossed below 50-day MA",
                    "severity": "medium"
                })
                
            if symbol_anomalies:
                anomalies[str(symbol)] = symbol_anomalies
        
        return anomalies
    
    def generate_trading_signals(self, symbols):
        """
        Generate trading signals for the given symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            dict: Trading signals for each symbol
        """
        signals = {}
        
        for symbol in symbols:
            if not self.algorithm.securities.contains_key(symbol):
                continue
                
            # Get historical data
            history = self.algorithm.history(symbol, 100, Resolution.DAILY)
            
            if history.empty:
                continue
                
            # Calculate indicators
            close_prices = history["close"]
            
            # Simple moving averages
            sma20 = close_prices.rolling(20).mean()
            sma50 = close_prices.rolling(50).mean()
            sma200 = close_prices.rolling(200).mean()
            
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = close_prices.ewm(span=12).mean()
            ema26 = close_prices.ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.rolling(9).mean()
            
            # Current values
            current_price = close_prices.iloc[-1]
            current_sma20 = sma20.iloc[-1]
            current_sma50 = sma50.iloc[-1]
            current_sma200 = sma200.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_macd = macd.iloc[-1]
            current_signal = signal_line.iloc[-1]
            
            # Generate signal
            signal = "neutral"
            confidence = 0.5
            reasons = []
            
            # Trend analysis
            if current_price > current_sma20 > current_sma50 > current_sma200:
                signal = "buy"
                confidence += 0.1
                reasons.append("Strong uptrend (price > SMA20 > SMA50 > SMA200)")
            elif current_price < current_sma20 < current_sma50 < current_sma200:
                signal = "sell"
                confidence += 0.1
                reasons.append("Strong downtrend (price < SMA20 < SMA50 < SMA200)")
            elif current_price > current_sma20 and current_price > current_sma50:
                signal = "buy"
                confidence += 0.05
                reasons.append("Price above short-term moving averages")
            elif current_price < current_sma20 and current_price < current_sma50:
                signal = "sell"
                confidence += 0.05
                reasons.append("Price below short-term moving averages")
                
            # RSI analysis
            if current_rsi < 30:
                if signal == "sell":
                    signal = "neutral"
                    confidence = 0.5
                else:
                    signal = "buy"
                    confidence += 0.1
                reasons.append(f"RSI oversold ({current_rsi:.1f})")
            elif current_rsi > 70:
                if signal == "buy":
                    signal = "neutral"
                    confidence = 0.5
                else:
                    signal = "sell"
                    confidence += 0.1
                reasons.append(f"RSI overbought ({current_rsi:.1f})")
                
            # MACD analysis
            if current_macd > current_signal and current_macd > 0:
                if signal != "sell":
                    signal = "buy"
                    confidence += 0.1
                    reasons.append("MACD bullish crossover above zero")
            elif current_macd < current_signal and current_macd < 0:
                if signal != "buy":
                    signal = "sell"
                    confidence += 0.1
                    reasons.append("MACD bearish crossover below zero")
                    
            # Cap confidence at 0.95
            confidence = min(confidence, 0.95)
            
            signals[str(symbol)] = {
                "signal": signal,
                "confidence": confidence,
                "reasons": reasons
            }
        
        return signals
    
    def analyze_economic_indicators(self):
        """
        Analyze current economic indicators.
        
        Returns:
            dict: Economic indicator analysis
        """
        # In a real implementation, this would use actual economic data
        # For this example, we'll use simplified logic based on market data
        
        # Get SPY data as a proxy for market
        spy_history = self.algorithm.history("SPY", 252, Resolution.DAILY)
        
        if spy_history.empty:
            return {"status": "insufficient_data"}
            
        # Calculate market metrics
        returns = spy_history["close"].pct_change().dropna()
        
        # Annualized return
        annual_return = returns.mean() * 252
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Trend
        current_price = spy_history["close"].iloc[-1]
        sma200 = spy_history["close"].rolling(200).mean().iloc[-1]
        
        # Determine economic environment
        if annual_return > 0.15:
            growth = "strong"
        elif annual_return > 0.05:
            growth = "moderate"
        elif annual_return > 0:
            growth = "weak"
        else:
            growth = "negative"
            
        if volatility > 0.25:
            risk = "high"
        elif volatility > 0.15:
            risk = "moderate"
        else:
            risk = "low"
            
        if current_price > sma200:
            trend = "bullish"
        else:
            trend = "bearish"
            
        return {
            "growth": growth,
            "risk": risk,
            "trend": trend,
            "metrics": {
                "annual_return": annual_return,
                "volatility": volatility,
                "price_to_sma200": current_price / sma200
            }
        }
    
    def get_market_insights(self, symbols):
        """
        Generate comprehensive market insights.
        
        Args:
            symbols: List of symbols
            
        Returns:
            dict: Comprehensive market insights
        """
        # Analyze correlations
        correlations = self.analyze_correlation_matrix(symbols)
        
        # Generate trading signals
        signals = self.generate_trading_signals(symbols)
        
        # Identify anomalies
        anomalies = self.identify_market_anomalies(symbols)
        
        # Analyze economic indicators
        economic = self.analyze_economic_indicators()
        
        # Sector analysis
        sectors = {
            "Technology": self.algorithm.symbol("XLK"),
            "Healthcare": self.algorithm.symbol("XLV"),
            "Financials": self.algorithm.symbol("XLF"),
            "Consumer Discretionary": self.algorithm.symbol("XLY"),
            "Consumer Staples": self.algorithm.symbol("XLP"),
            "Energy": self.algorithm.symbol("XLE"),
            "Utilities": self.algorithm.symbol("XLU"),
            "Materials": self.algorithm.symbol("XLB"),
            "Industrials": self.algorithm.symbol("XLI"),
            "Real Estate": self.algorithm.symbol("XLRE"),
            "Communication Services": self.algorithm.symbol("XLC")
        }
        
        sector_performance = self.analyze_sector_performance(sectors)
        
        # Generate market summary
        if economic["trend"] == "bullish" and economic["growth"] in ["moderate", "strong"]:
            market_summary = "The market is in a bullish regime with strong upward momentum."
        elif economic["trend"] == "bearish" and economic["growth"] == "negative":
            market_summary = "The market is in a bearish regime with downward pressure."
        elif economic["risk"] == "high":
            market_summary = "The market is highly volatile with increased uncertainty."
        else:
            market_summary = "The market is showing mixed signals with no clear direction."
            
        # Generate ticker-specific insights
        ticker_insights = {}
        for symbol, signal_data in signals.items():
            signal = signal_data["signal"]
            confidence = signal_data["confidence"]
            
            if signal == "buy":
                action = "Consider buying"
                if confidence > 0.7:
                    confidence_text = "high confidence"
                else:
                    confidence_text = "moderate confidence"
            elif signal == "sell":
                action = "Consider selling"
                if confidence > 0.7:
                    confidence_text = "high confidence"
                else:
                    confidence_text = "moderate confidence"
            else:
                action = "Hold/monitor"
                confidence_text = "neutral stance"
                
            ticker_insights[symbol] = f"{action} with {confidence_text}."
            
        return {
            "market_summary": market_summary,
            "economic_outlook": economic,
            "top_sectors": sector_performance["top_sectors"],
            "ticker_insights": ticker_insights,
            "signals": signals,
            "anomalies": anomalies
        }
