"""
Market Regime Detection System
Priority 2: Implement market regime detection for adaptive strategy selection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    WEAK_BULL = "weak_bull"
    SIDEWAYS = "sideways"
    WEAK_BEAR = "weak_bear"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    CRASH = "crash"
    RECOVERY = "recovery"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class MarketRegimeDetector:
    """
    Sophisticated market regime detection using multiple indicators:
    - Price trends (SMA crossovers)
    - Volatility (VIX, ATR)
    - Market breadth
    - Economic indicators
    - Sentiment analysis
    """
    
    def __init__(self):
        self.regime_history = []
        self.regime_strategies = self._initialize_regime_strategies()
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        
        # Indicator thresholds
        self.vix_thresholds = {
            'low': 12,
            'normal': 20,
            'high': 30,
            'extreme': 40
        }
        
        self.trend_thresholds = {
            'strong_up': 0.02,    # 2% above SMA
            'up': 0.005,          # 0.5% above SMA
            'neutral': -0.005,    # -0.5% to 0.5%
            'down': -0.02,        # -2% below SMA
            'strong_down': -0.05  # -5% below SMA
        }
    
    def _initialize_regime_strategies(self) -> Dict[MarketRegime, Dict]:
        """Initialize optimal strategies for each market regime"""
        return {
            MarketRegime.STRONG_BULL: {
                'preferred_strategies': ['momentum', 'trend_following', 'breakout'],
                'asset_allocation': {'equities': 0.7, 'crypto': 0.2, 'commodities': 0.1},
                'leverage': 2.5,
                'risk_level': 'aggressive',
                'indicators': ['RSI', 'MACD', 'ADX'],
                'stop_loss': 0.15
            },
            MarketRegime.BULL: {
                'preferred_strategies': ['momentum', 'mean_reversion', 'factor_based'],
                'asset_allocation': {'equities': 0.6, 'crypto': 0.2, 'forex': 0.1, 'commodities': 0.1},
                'leverage': 2.0,
                'risk_level': 'moderate',
                'indicators': ['RSI', 'MACD', 'BB'],
                'stop_loss': 0.12
            },
            MarketRegime.WEAK_BULL: {
                'preferred_strategies': ['mean_reversion', 'pairs_trading', 'arbitrage'],
                'asset_allocation': {'equities': 0.5, 'forex': 0.3, 'commodities': 0.2},
                'leverage': 1.5,
                'risk_level': 'conservative',
                'indicators': ['RSI', 'BB', 'ATR'],
                'stop_loss': 0.10
            },
            MarketRegime.SIDEWAYS: {
                'preferred_strategies': ['mean_reversion', 'range_trading', 'theta_strategies'],
                'asset_allocation': {'equities': 0.4, 'forex': 0.3, 'commodities': 0.2, 'cash': 0.1},
                'leverage': 1.0,
                'risk_level': 'conservative',
                'indicators': ['RSI', 'BB', 'Stochastic'],
                'stop_loss': 0.08
            },
            MarketRegime.WEAK_BEAR: {
                'preferred_strategies': ['short_momentum', 'defensive', 'hedged_positions'],
                'asset_allocation': {'cash': 0.3, 'forex': 0.3, 'commodities': 0.3, 'equities': 0.1},
                'leverage': 0.5,
                'risk_level': 'defensive',
                'indicators': ['RSI', 'MACD', 'VIX'],
                'stop_loss': 0.06
            },
            MarketRegime.BEAR: {
                'preferred_strategies': ['short_positions', 'inverse_etfs', 'volatility_trading'],
                'asset_allocation': {'cash': 0.4, 'inverse_etfs': 0.3, 'commodities': 0.2, 'forex': 0.1},
                'leverage': 0.3,
                'risk_level': 'very_defensive',
                'indicators': ['VIX', 'Put/Call', 'MACD'],
                'stop_loss': 0.05
            },
            MarketRegime.STRONG_BEAR: {
                'preferred_strategies': ['pure_short', 'volatility_long', 'crisis_alpha'],
                'asset_allocation': {'cash': 0.5, 'inverse_etfs': 0.3, 'volatility': 0.2},
                'leverage': 0.2,
                'risk_level': 'ultra_defensive',
                'indicators': ['VIX', 'SKEW', 'Term_Structure'],
                'stop_loss': 0.04
            },
            MarketRegime.CRASH: {
                'preferred_strategies': ['cash', 'ultra_short_term', 'tail_hedging'],
                'asset_allocation': {'cash': 0.8, 'treasuries': 0.2},
                'leverage': 0.0,
                'risk_level': 'cash_preservation',
                'indicators': ['VIX', 'VVIX', 'Credit_Spreads'],
                'stop_loss': 0.02
            },
            MarketRegime.RECOVERY: {
                'preferred_strategies': ['early_cycle', 'quality_momentum', 'selective_longs'],
                'asset_allocation': {'equities': 0.4, 'crypto': 0.2, 'commodities': 0.2, 'cash': 0.2},
                'leverage': 1.2,
                'risk_level': 'cautious_optimistic',
                'indicators': ['RSI', 'MACD', 'Breadth'],
                'stop_loss': 0.10
            },
            MarketRegime.HIGH_VOLATILITY: {
                'preferred_strategies': ['volatility_arbitrage', 'options_strategies', 'short_term'],
                'asset_allocation': {'options': 0.4, 'forex': 0.3, 'cash': 0.3},
                'leverage': 0.8,
                'risk_level': 'adaptive',
                'indicators': ['VIX', 'ATR', 'Bollinger_Width'],
                'stop_loss': 0.06
            },
            MarketRegime.LOW_VOLATILITY: {
                'preferred_strategies': ['carry_trades', 'dividend_capture', 'theta_decay'],
                'asset_allocation': {'equities': 0.6, 'forex': 0.2, 'crypto': 0.2},
                'leverage': 2.5,
                'risk_level': 'moderate_aggressive',
                'indicators': ['VIX', 'HV', 'IV'],
                'stop_loss': 0.15
            }
        }
    
    def detect_regime(self, market_data: Dict) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime using multiple indicators
        
        Args:
            market_data: Dictionary containing market indicators
            
        Returns:
            Tuple of (regime, confidence_score)
        """
        # Calculate individual regime scores
        trend_regime = self._analyze_trend(market_data)
        volatility_regime = self._analyze_volatility(market_data)
        breadth_regime = self._analyze_breadth(market_data)
        sentiment_regime = self._analyze_sentiment(market_data)
        economic_regime = self._analyze_economic_indicators(market_data)
        
        # Combine regime signals with weights
        regime_scores = {}
        weights = {
            'trend': 0.3,
            'volatility': 0.25,
            'breadth': 0.20,
            'sentiment': 0.15,
            'economic': 0.10
        }
        
        # Aggregate scores for each regime
        all_regimes = [trend_regime, volatility_regime, breadth_regime, sentiment_regime, economic_regime]
        
        for regime_dict, weight_key in zip(all_regimes, weights.keys()):
            if regime_dict:
                for regime, score in regime_dict.items():
                    if regime not in regime_scores:
                        regime_scores[regime] = 0
                    regime_scores[regime] += score * weights[weight_key]
        
        # Find dominant regime
        if regime_scores:
            dominant_regime = max(regime_scores.items(), key=lambda x: x[1])
            regime = dominant_regime[0]
            confidence = min(dominant_regime[1], 1.0)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.5
        
        # Update regime history
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'confidence': confidence,
            'indicators': market_data
        })
        
        self.current_regime = regime
        self.regime_confidence = confidence
        
        logger.info(f"Detected market regime: {regime.value} (confidence: {confidence:.2f})")
        
        return regime, confidence
    
    def _analyze_trend(self, market_data: Dict) -> Dict[MarketRegime, float]:
        """Analyze price trends to determine regime"""
        scores = {}
        
        # Get price relative to moving averages
        price = market_data.get('spy_price', 100)
        sma_50 = market_data.get('spy_sma_50', 100)
        sma_200 = market_data.get('spy_sma_200', 100)
        
        # Calculate trend strength
        short_trend = (price - sma_50) / sma_50 if sma_50 > 0 else 0
        long_trend = (price - sma_200) / sma_200 if sma_200 > 0 else 0
        
        # Determine regime based on trends
        if short_trend > self.trend_thresholds['strong_up'] and long_trend > self.trend_thresholds['up']:
            scores[MarketRegime.STRONG_BULL] = 0.9
            scores[MarketRegime.BULL] = 0.7
        elif short_trend > self.trend_thresholds['up'] and long_trend > 0:
            scores[MarketRegime.BULL] = 0.8
            scores[MarketRegime.WEAK_BULL] = 0.5
        elif abs(short_trend) < self.trend_thresholds['up'] and abs(long_trend) < self.trend_thresholds['up']:
            scores[MarketRegime.SIDEWAYS] = 0.8
            scores[MarketRegime.WEAK_BULL] = 0.3
            scores[MarketRegime.WEAK_BEAR] = 0.3
        elif short_trend < self.trend_thresholds['down'] and long_trend < 0:
            scores[MarketRegime.BEAR] = 0.8
            scores[MarketRegime.WEAK_BEAR] = 0.5
        elif short_trend < self.trend_thresholds['strong_down']:
            scores[MarketRegime.STRONG_BEAR] = 0.9
            scores[MarketRegime.CRASH] = 0.6
        
        # Check for recovery patterns
        momentum = market_data.get('momentum_20d', 0)
        if long_trend < -0.1 and short_trend > 0 and momentum > 0:
            scores[MarketRegime.RECOVERY] = 0.7
        
        return scores
    
    def _analyze_volatility(self, market_data: Dict) -> Dict[MarketRegime, float]:
        """Analyze volatility indicators"""
        scores = {}
        
        vix = market_data.get('vix', 20)
        vix_sma = market_data.get('vix_sma_20', 20)
        historical_vol = market_data.get('historical_volatility', 0.15)
        
        # VIX-based regime detection
        if vix < self.vix_thresholds['low']:
            scores[MarketRegime.LOW_VOLATILITY] = 0.9
            scores[MarketRegime.BULL] = 0.6
        elif vix < self.vix_thresholds['normal']:
            scores[MarketRegime.SIDEWAYS] = 0.5
            scores[MarketRegime.BULL] = 0.4
        elif vix < self.vix_thresholds['high']:
            scores[MarketRegime.HIGH_VOLATILITY] = 0.6
            scores[MarketRegime.WEAK_BEAR] = 0.5
        elif vix < self.vix_thresholds['extreme']:
            scores[MarketRegime.HIGH_VOLATILITY] = 0.8
            scores[MarketRegime.BEAR] = 0.7
        else:
            scores[MarketRegime.CRASH] = 0.9
            scores[MarketRegime.STRONG_BEAR] = 0.8
        
        # Check volatility trend
        vix_trend = (vix - vix_sma) / vix_sma if vix_sma > 0 else 0
        if vix_trend > 0.2:  # Rising volatility
            scores[MarketRegime.HIGH_VOLATILITY] = scores.get(MarketRegime.HIGH_VOLATILITY, 0) + 0.2
        elif vix_trend < -0.2:  # Falling volatility
            scores[MarketRegime.LOW_VOLATILITY] = scores.get(MarketRegime.LOW_VOLATILITY, 0) + 0.2
        
        return scores
    
    def _analyze_breadth(self, market_data: Dict) -> Dict[MarketRegime, float]:
        """Analyze market breadth indicators"""
        scores = {}
        
        advance_decline = market_data.get('advance_decline_ratio', 1.0)
        new_highs_lows = market_data.get('new_highs_lows_ratio', 1.0)
        percent_above_200ma = market_data.get('percent_above_200ma', 50)
        
        # Breadth-based regime detection
        if advance_decline > 2.0 and new_highs_lows > 3.0:
            scores[MarketRegime.STRONG_BULL] = 0.8
            scores[MarketRegime.BULL] = 0.9
        elif advance_decline > 1.5 and percent_above_200ma > 70:
            scores[MarketRegime.BULL] = 0.8
            scores[MarketRegime.WEAK_BULL] = 0.5
        elif 0.7 < advance_decline < 1.3 and 40 < percent_above_200ma < 60:
            scores[MarketRegime.SIDEWAYS] = 0.8
        elif advance_decline < 0.5 and percent_above_200ma < 30:
            scores[MarketRegime.BEAR] = 0.8
            scores[MarketRegime.STRONG_BEAR] = 0.6
        elif advance_decline < 0.3 and new_highs_lows < 0.1:
            scores[MarketRegime.STRONG_BEAR] = 0.9
            scores[MarketRegime.CRASH] = 0.7
        
        return scores
    
    def _analyze_sentiment(self, market_data: Dict) -> Dict[MarketRegime, float]:
        """Analyze market sentiment indicators"""
        scores = {}
        
        put_call_ratio = market_data.get('put_call_ratio', 1.0)
        bull_bear_spread = market_data.get('bull_bear_spread', 0)
        fear_greed_index = market_data.get('fear_greed_index', 50)
        
        # Sentiment-based regime detection
        if fear_greed_index > 80:
            scores[MarketRegime.STRONG_BULL] = 0.7
            scores[MarketRegime.HIGH_VOLATILITY] = 0.3  # Extreme greed often precedes volatility
        elif fear_greed_index > 60:
            scores[MarketRegime.BULL] = 0.8
        elif 40 <= fear_greed_index <= 60:
            scores[MarketRegime.SIDEWAYS] = 0.7
        elif fear_greed_index < 20:
            scores[MarketRegime.STRONG_BEAR] = 0.7
            scores[MarketRegime.CRASH] = 0.5
        else:
            scores[MarketRegime.BEAR] = 0.7
            scores[MarketRegime.WEAK_BEAR] = 0.5
        
        # Put/Call ratio analysis
        if put_call_ratio > 1.5:  # High fear
            scores[MarketRegime.BEAR] = scores.get(MarketRegime.BEAR, 0) + 0.3
        elif put_call_ratio < 0.7:  # High complacency
            scores[MarketRegime.BULL] = scores.get(MarketRegime.BULL, 0) + 0.2
        
        return scores
    
    def _analyze_economic_indicators(self, market_data: Dict) -> Dict[MarketRegime, float]:
        """Analyze economic indicators"""
        scores = {}
        
        yield_curve = market_data.get('yield_curve_10y2y', 1.0)
        unemployment_trend = market_data.get('unemployment_trend', 0)
        gdp_growth = market_data.get('gdp_growth', 2.0)
        inflation_rate = market_data.get('inflation_rate', 2.0)
        
        # Economic indicator-based regime detection
        if yield_curve > 1.5 and gdp_growth > 3.0 and unemployment_trend < 0:
            scores[MarketRegime.STRONG_BULL] = 0.7
            scores[MarketRegime.BULL] = 0.8
        elif yield_curve > 0 and gdp_growth > 2.0:
            scores[MarketRegime.BULL] = 0.7
            scores[MarketRegime.WEAK_BULL] = 0.5
        elif yield_curve < 0:  # Inverted yield curve
            scores[MarketRegime.BEAR] = 0.6
            scores[MarketRegime.WEAK_BEAR] = 0.7
        
        # High inflation regime
        if inflation_rate > 5.0:
            scores[MarketRegime.HIGH_VOLATILITY] = scores.get(MarketRegime.HIGH_VOLATILITY, 0) + 0.3
            scores[MarketRegime.SIDEWAYS] = scores.get(MarketRegime.SIDEWAYS, 0) + 0.2
        
        return scores
    
    def get_regime_strategy_params(self, regime: Optional[MarketRegime] = None) -> Dict:
        """
        Get optimal strategy parameters for current or specified regime
        
        Args:
            regime: Optional regime to get parameters for
            
        Returns:
            Dictionary of strategy parameters
        """
        if regime is None:
            regime = self.current_regime
        
        return self.regime_strategies.get(regime, self.regime_strategies[MarketRegime.SIDEWAYS])
    
    def get_regime_transition_probability(self) -> Dict[MarketRegime, float]:
        """
        Calculate probability of transitioning to different regimes
        
        Returns:
            Dictionary of transition probabilities
        """
        if len(self.regime_history) < 2:
            return {regime: 1.0 / len(MarketRegime) for regime in MarketRegime}
        
        # Simple transition probability based on recent history
        transitions = {}
        for regime in MarketRegime:
            transitions[regime] = 0.1  # Base probability
        
        # Increase probability for current regime (persistence)
        transitions[self.current_regime] = 0.3
        
        # Adjust based on confidence
        if self.regime_confidence < 0.5:
            # Low confidence means higher transition probability
            for regime in MarketRegime:
                if regime != self.current_regime:
                    transitions[regime] += 0.05
        
        # Normalize probabilities
        total = sum(transitions.values())
        for regime in transitions:
            transitions[regime] /= total
        
        return transitions
    
    def suggest_portfolio_adjustment(self, current_positions: Dict) -> Dict:
        """
        Suggest portfolio adjustments based on regime change
        
        Args:
            current_positions: Current portfolio positions
            
        Returns:
            Suggested adjustments
        """
        regime_params = self.get_regime_strategy_params()
        
        adjustments = {
            'action': 'rebalance',
            'urgency': 'normal',
            'changes': {},
            'risk_adjustment': 1.0,
            'reasoning': []
        }
        
        # Check if regime changed significantly
        if len(self.regime_history) >= 2:
            prev_regime = self.regime_history[-2]['regime']
            if prev_regime != self.current_regime:
                adjustments['urgency'] = 'high'
                adjustments['reasoning'].append(f"Regime changed from {prev_regime.value} to {self.current_regime.value}")
        
        # Suggest asset allocation changes
        target_allocation = regime_params['asset_allocation']
        for asset, target_weight in target_allocation.items():
            current_weight = current_positions.get(asset, 0)
            if abs(current_weight - target_weight) > 0.1:  # 10% threshold
                adjustments['changes'][asset] = {
                    'current': current_weight,
                    'target': target_weight,
                    'action': 'increase' if target_weight > current_weight else 'decrease'
                }
        
        # Adjust risk based on regime
        if self.current_regime in [MarketRegime.CRASH, MarketRegime.STRONG_BEAR]:
            adjustments['risk_adjustment'] = 0.3
            adjustments['action'] = 'de-risk'
            adjustments['urgency'] = 'immediate'
        elif self.current_regime in [MarketRegime.HIGH_VOLATILITY]:
            adjustments['risk_adjustment'] = 0.5
            adjustments['action'] = 'reduce_exposure'
        elif self.current_regime in [MarketRegime.STRONG_BULL, MarketRegime.LOW_VOLATILITY]:
            adjustments['risk_adjustment'] = 1.5
            adjustments['action'] = 'increase_exposure'
        
        # Add strategy recommendations
        adjustments['recommended_strategies'] = regime_params['preferred_strategies']
        adjustments['avoid_strategies'] = self._get_strategies_to_avoid()
        
        return adjustments
    
    def _get_strategies_to_avoid(self) -> List[str]:
        """Get strategies to avoid in current regime"""
        avoid_map = {
            MarketRegime.CRASH: ['momentum', 'trend_following', 'leveraged_long'],
            MarketRegime.STRONG_BEAR: ['momentum', 'buy_dips', 'growth'],
            MarketRegime.HIGH_VOLATILITY: ['carry_trades', 'short_volatility'],
            MarketRegime.LOW_VOLATILITY: ['volatility_trading', 'tail_hedging'],
            MarketRegime.SIDEWAYS: ['trend_following', 'momentum']
        }
        
        return avoid_map.get(self.current_regime, [])
    
    def generate_regime_specific_code(self, base_strategy: str, regime: Optional[MarketRegime] = None) -> str:
        """
        Modify strategy code to be regime-aware
        
        Args:
            base_strategy: Base strategy code
            regime: Target regime (uses current if None)
            
        Returns:
            Modified strategy code with regime adaptations
        """
        if regime is None:
            regime = self.current_regime
        
        params = self.get_regime_strategy_params(regime)
        
        # Insert regime-specific modifications
        regime_code = f"""
# Regime-specific adaptations for {regime.value}
self.current_regime = "{regime.value}"
self.regime_params = {{
    'leverage': {params['leverage']},
    'stop_loss': {params['stop_loss']},
    'risk_level': '{params['risk_level']}',
    'preferred_indicators': {params['indicators']}
}}

# Adjust position sizing based on regime
self.position_size = self.position_size * {params['leverage'] / 2.0}  # Regime adjustment
self.stop_loss = {params['stop_loss']}

# Add regime-specific entry filters
def IsRegimeAppropriate(self, symbol):
    \"\"\"Check if trade is appropriate for current regime\"\"\"
    if self.current_regime in ['crash', 'strong_bear']:
        # Only short positions in crash/strong bear
        return self.signal < 0
    elif self.current_regime in ['strong_bull']:
        # Only long positions in strong bull
        return self.signal > 0
    elif self.current_regime == 'high_volatility':
        # Require stronger signals in high volatility
        return abs(self.signal) > 0.7
    else:
        return True
"""
        
        # Insert regime code into base strategy
        if "def OnData" in base_strategy:
            insertion_point = base_strategy.find("def OnData")
            modified_strategy = (
                base_strategy[:insertion_point] + 
                regime_code + "\n\n    " + 
                base_strategy[insertion_point:]
            )
        else:
            modified_strategy = base_strategy + "\n\n" + regime_code
        
        return modified_strategy
    
    def get_historical_regime_performance(self) -> pd.DataFrame:
        """Get historical performance by regime"""
        if not self.regime_history:
            return pd.DataFrame()
        
        data = []
        for entry in self.regime_history:
            data.append({
                'timestamp': entry['timestamp'],
                'regime': entry['regime'].value,
                'confidence': entry['confidence']
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df


# Example usage
if __name__ == '__main__':
    # Initialize detector
    detector = MarketRegimeDetector()
    
    # Example market data
    market_data = {
        'spy_price': 450,
        'spy_sma_50': 445,
        'spy_sma_200': 430,
        'vix': 18,
        'vix_sma_20': 20,
        'momentum_20d': 0.02,
        'advance_decline_ratio': 1.3,
        'new_highs_lows_ratio': 2.1,
        'percent_above_200ma': 65,
        'put_call_ratio': 0.9,
        'fear_greed_index': 55,
        'yield_curve_10y2y': 1.2,
        'gdp_growth': 2.5
    }
    
    # Detect regime
    regime, confidence = detector.detect_regime(market_data)
    print(f"Current Market Regime: {regime.value} (Confidence: {confidence:.2%})")
    
    # Get strategy parameters
    params = detector.get_regime_strategy_params()
    print(f"\nRecommended Strategy Parameters:")
    print(f"- Preferred Strategies: {params['preferred_strategies']}")
    print(f"- Asset Allocation: {params['asset_allocation']}")
    print(f"- Leverage: {params['leverage']}")
    print(f"- Risk Level: {params['risk_level']}")
    
    # Get portfolio adjustments
    current_positions = {'equities': 0.8, 'cash': 0.2}
    adjustments = detector.suggest_portfolio_adjustment(current_positions)
    print(f"\nSuggested Portfolio Adjustments:")
    print(f"- Action: {adjustments['action']} ({adjustments['urgency']} urgency)")
    print(f"- Risk Adjustment: {adjustments['risk_adjustment']}")
    print(f"- Changes: {adjustments['changes']}")