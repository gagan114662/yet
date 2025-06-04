"""
Ensemble Strategy Generator
Priority 3: Combine multiple uncorrelated strategies into robust portfolios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import logging
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EnsembleStrategyGenerator:
    """
    Generates ensemble strategies by combining multiple uncorrelated strategies
    using various methods:
    - Equal weighting
    - Risk parity
    - Maximum Sharpe optimization
    - Machine learning-based weighting
    - Regime-adaptive weighting
    """
    
    def __init__(self):
        self.strategy_pool = []
        self.correlation_threshold = 0.6
        self.min_strategies = 3
        self.max_strategies = 10
        self.ensemble_methods = ['equal', 'risk_parity', 'max_sharpe', 'ml_weighted', 'regime_adaptive']
        self.performance_history = []
        
    def add_strategy_to_pool(self, strategy: Dict, performance_metrics: Dict):
        """
        Add a strategy to the pool for ensemble consideration
        
        Args:
            strategy: Strategy definition
            performance_metrics: Historical performance metrics
        """
        self.strategy_pool.append({
            'strategy': strategy,
            'metrics': performance_metrics,
            'id': f"{strategy['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'added': datetime.now()
        })
        logger.info(f"Added strategy {strategy['name']} to ensemble pool. Total strategies: {len(self.strategy_pool)}")
    
    def generate_ensemble(self, 
                         method: str = 'max_sharpe',
                         target_size: int = 5,
                         regime: Optional[str] = None) -> Dict:
        """
        Generate an ensemble strategy using specified method
        
        Args:
            method: Ensemble generation method
            target_size: Target number of strategies in ensemble
            regime: Current market regime for adaptive weighting
            
        Returns:
            Ensemble strategy definition
        """
        if len(self.strategy_pool) < self.min_strategies:
            logger.warning(f"Insufficient strategies in pool ({len(self.strategy_pool)} < {self.min_strategies})")
            return None
        
        # Select uncorrelated strategies
        selected_strategies = self._select_uncorrelated_strategies(target_size)
        
        if len(selected_strategies) < self.min_strategies:
            logger.warning(f"Could not find enough uncorrelated strategies")
            return None
        
        # Calculate weights based on method
        if method == 'equal':
            weights = self._calculate_equal_weights(selected_strategies)
        elif method == 'risk_parity':
            weights = self._calculate_risk_parity_weights(selected_strategies)
        elif method == 'max_sharpe':
            weights = self._calculate_max_sharpe_weights(selected_strategies)
        elif method == 'ml_weighted':
            weights = self._calculate_ml_weights(selected_strategies)
        elif method == 'regime_adaptive':
            weights = self._calculate_regime_adaptive_weights(selected_strategies, regime)
        else:
            weights = self._calculate_equal_weights(selected_strategies)
        
        # Create ensemble definition
        ensemble = self._create_ensemble_definition(selected_strategies, weights, method)
        
        # Generate ensemble code
        ensemble['code'] = self._generate_ensemble_code(ensemble)
        
        logger.info(f"Generated {method} ensemble with {len(selected_strategies)} strategies")
        
        return ensemble
    
    def _select_uncorrelated_strategies(self, target_size: int) -> List[Dict]:
        """Select strategies with low correlation"""
        if len(self.strategy_pool) <= target_size:
            return self.strategy_pool
        
        # Sort strategies by Sharpe ratio
        sorted_pool = sorted(self.strategy_pool, 
                           key=lambda x: x['metrics'].get('sharpe_ratio', 0), 
                           reverse=True)
        
        selected = [sorted_pool[0]]  # Start with best Sharpe
        
        for candidate in sorted_pool[1:]:
            if len(selected) >= target_size:
                break
            
            # Check correlation with existing selections
            is_uncorrelated = True
            for selected_strategy in selected:
                correlation = self._estimate_correlation(candidate, selected_strategy)
                if abs(correlation) > self.correlation_threshold:
                    is_uncorrelated = False
                    break
            
            if is_uncorrelated:
                selected.append(candidate)
        
        return selected
    
    def _estimate_correlation(self, strategy1: Dict, strategy2: Dict) -> float:
        """
        Estimate correlation between two strategies
        
        In production, this would use actual return series.
        Here we use a heuristic based on strategy characteristics.
        """
        s1 = strategy1['strategy']
        s2 = strategy2['strategy']
        
        correlation = 0.0
        
        # Same strategy type = high correlation
        if s1.get('type') == s2.get('type'):
            correlation += 0.5
        
        # Similar indicators = moderate correlation
        indicators1 = set(s1.get('indicators', []))
        indicators2 = set(s2.get('indicators', []))
        indicator_overlap = len(indicators1.intersection(indicators2)) / max(len(indicators1), len(indicators2), 1)
        correlation += indicator_overlap * 0.3
        
        # Similar asset classes = moderate correlation
        assets1 = set(s1.get('asset_classes', {}).keys())
        assets2 = set(s2.get('asset_classes', {}).keys())
        asset_overlap = len(assets1.intersection(assets2)) / max(len(assets1), len(assets2), 1)
        correlation += asset_overlap * 0.2
        
        # Add some randomness for realism
        correlation += np.random.normal(0, 0.1)
        
        return np.clip(correlation, -1, 1)
    
    def _calculate_equal_weights(self, strategies: List[Dict]) -> np.array:
        """Calculate equal weights for all strategies"""
        n = len(strategies)
        return np.ones(n) / n
    
    def _calculate_risk_parity_weights(self, strategies: List[Dict]) -> np.array:
        """
        Calculate risk parity weights where each strategy contributes
        equally to portfolio risk
        """
        n = len(strategies)
        
        # Extract volatilities (using max_drawdown as proxy)
        volatilities = []
        for s in strategies:
            vol = s['metrics'].get('max_drawdown', 0.15)
            volatilities.append(max(vol, 0.01))  # Avoid division by zero
        
        volatilities = np.array(volatilities)
        
        # Risk parity: weight inversely proportional to volatility
        inverse_vols = 1.0 / volatilities
        weights = inverse_vols / inverse_vols.sum()
        
        return weights
    
    def _calculate_max_sharpe_weights(self, strategies: List[Dict]) -> np.array:
        """
        Calculate weights that maximize ensemble Sharpe ratio
        """
        n = len(strategies)
        
        # Extract returns and risks
        returns = np.array([s['metrics'].get('cagr', 0) for s in strategies])
        risks = np.array([s['metrics'].get('max_drawdown', 0.15) for s in strategies])
        
        # Correlation matrix (simplified - using estimates)
        corr_matrix = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                corr = self._estimate_correlation(strategies[i], strategies[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        # Covariance matrix (simplified)
        cov_matrix = np.outer(risks, risks) * corr_matrix
        
        # Optimization objective: maximize Sharpe ratio
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_risk = np.sqrt(portfolio_variance)
            
            if portfolio_risk == 0:
                return 999999
            
            sharpe = portfolio_return / portfolio_risk
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
        ]
        
        # Bounds (0 to 1 for each weight)
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(negative_sharpe, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning("Optimization failed, using equal weights")
            return self._calculate_equal_weights(strategies)
    
    def _calculate_ml_weights(self, strategies: List[Dict]) -> np.array:
        """
        Use machine learning to predict optimal weights based on
        strategy characteristics and market conditions
        """
        n = len(strategies)
        
        # Extract features for each strategy
        features = []
        for s in strategies:
            metrics = s['metrics']
            strategy_features = [
                metrics.get('sharpe_ratio', 0),
                metrics.get('cagr', 0),
                metrics.get('max_drawdown', 0.15),
                metrics.get('total_trades', 0) / 1000,  # Normalize
                len(s['strategy'].get('indicators', [])) / 10,  # Normalize
                s['strategy'].get('leverage', 1.0) / 3,  # Normalize
            ]
            features.append(strategy_features)
        
        features = np.array(features)
        
        # Simple ML model (in production, this would be pre-trained)
        # For now, we'll use a heuristic based on features
        
        # Score each strategy
        scores = []
        for i, feat in enumerate(features):
            # Higher score for better Sharpe, CAGR, lower drawdown
            score = (
                feat[0] * 0.4 +  # Sharpe ratio weight
                feat[1] * 0.3 +  # CAGR weight
                (1 - feat[2]) * 0.3  # Inverse drawdown weight
            )
            scores.append(max(score, 0.1))  # Minimum score
        
        scores = np.array(scores)
        
        # Convert scores to weights
        weights = scores / scores.sum()
        
        # Apply diversification constraint (no single strategy > 40%)
        weights = np.minimum(weights, 0.4)
        weights = weights / weights.sum()
        
        return weights
    
    def _calculate_regime_adaptive_weights(self, strategies: List[Dict], regime: str) -> np.array:
        """
        Calculate weights adapted to current market regime
        """
        n = len(strategies)
        base_weights = self._calculate_max_sharpe_weights(strategies)
        
        # Regime-specific adjustments
        regime_multipliers = {
            'bull': {'momentum': 1.5, 'trend_following': 1.3, 'mean_reversion': 0.7},
            'bear': {'momentum': 0.5, 'short': 2.0, 'defensive': 1.5, 'mean_reversion': 1.2},
            'sideways': {'mean_reversion': 1.5, 'range_trading': 1.5, 'momentum': 0.5},
            'high_volatility': {'volatility': 1.5, 'options': 1.5, 'trend_following': 0.7},
            'low_volatility': {'carry': 1.5, 'momentum': 1.2, 'volatility': 0.5}
        }
        
        # Apply regime adjustments
        adjusted_weights = base_weights.copy()
        
        if regime in regime_multipliers:
            multipliers = regime_multipliers[regime]
            
            for i, strategy_data in enumerate(strategies):
                strategy_type = strategy_data['strategy'].get('type', 'unknown')
                
                for type_key, multiplier in multipliers.items():
                    if type_key in strategy_type.lower():
                        adjusted_weights[i] *= multiplier
                        break
        
        # Renormalize
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        # Apply constraints
        adjusted_weights = np.minimum(adjusted_weights, 0.4)  # Max 40% per strategy
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        return adjusted_weights
    
    def _create_ensemble_definition(self, strategies: List[Dict], weights: np.array, method: str) -> Dict:
        """Create ensemble strategy definition"""
        ensemble = {
            'name': f'Ensemble_{method}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'type': 'ensemble',
            'method': method,
            'components': [],
            'weights': weights.tolist(),
            'expected_metrics': {},
            'risk_controls': {
                'max_correlation': self.correlation_threshold,
                'rebalance_frequency': 'daily',
                'weight_constraints': {'min': 0.05, 'max': 0.4},
                'drawdown_limit': 0.20
            }
        }
        
        # Add component strategies
        for i, strategy_data in enumerate(strategies):
            component = {
                'id': strategy_data['id'],
                'name': strategy_data['strategy']['name'],
                'weight': weights[i],
                'metrics': strategy_data['metrics']
            }
            ensemble['components'].append(component)
        
        # Calculate expected ensemble metrics
        ensemble['expected_metrics'] = self._calculate_expected_metrics(strategies, weights)
        
        return ensemble
    
    def _calculate_expected_metrics(self, strategies: List[Dict], weights: np.array) -> Dict:
        """Calculate expected metrics for the ensemble"""
        # Weighted average of returns
        returns = np.array([s['metrics'].get('cagr', 0) for s in strategies])
        expected_return = np.dot(weights, returns)
        
        # Risk calculation (simplified)
        risks = np.array([s['metrics'].get('max_drawdown', 0.15) for s in strategies])
        
        # Assuming some diversification benefit
        diversification_factor = 0.7  # 30% risk reduction from diversification
        expected_risk = np.sqrt(np.dot(weights**2, risks**2)) * diversification_factor
        
        # Expected Sharpe (simplified)
        expected_sharpe = expected_return / expected_risk if expected_risk > 0 else 0
        
        # Weighted average of other metrics
        total_trades = sum(w * s['metrics'].get('total_trades', 0) for w, s in zip(weights, strategies))
        
        return {
            'expected_cagr': expected_return,
            'expected_max_drawdown': expected_risk,
            'expected_sharpe_ratio': expected_sharpe,
            'expected_total_trades': int(total_trades),
            'diversification_ratio': 1 / np.sum(weights**2),  # Effective number of strategies
            'confidence_interval': {
                'cagr_lower': expected_return * 0.7,
                'cagr_upper': expected_return * 1.3
            }
        }
    
    def _generate_ensemble_code(self, ensemble: Dict) -> str:
        """Generate QuantConnect code for ensemble strategy"""
        
        code = f'''
from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta

class {ensemble['name']}(QCAlgorithm):
    """
    Ensemble strategy combining {len(ensemble['components'])} strategies
    Method: {ensemble['method']}
    Expected CAGR: {ensemble['expected_metrics']['expected_cagr']:.2%}
    Expected Sharpe: {ensemble['expected_metrics']['expected_sharpe_ratio']:.2f}
    """
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Ensemble configuration
        self.ensemble_method = "{ensemble['method']}"
        self.component_weights = {ensemble['weights']}
        self.rebalance_frequency = "{ensemble['risk_controls']['rebalance_frequency']}"
        self.max_drawdown_limit = {ensemble['risk_controls']['drawdown_limit']}
        
        # Component strategies
        self.components = []
        self.component_signals = {{}}
        self.component_positions = {{}}
        
        # Initialize each component
        self.InitializeComponents()
        
        # Schedule rebalancing
        if self.rebalance_frequency == "daily":
            self.Schedule.On(self.DateRules.EveryDay(), 
                           self.TimeRules.At(10, 0), 
                           self.RebalanceEnsemble)
        
        # Risk management
        self.last_rebalance = datetime.min
        self.peak_value = self.Portfolio.TotalPortfolioValue
        
    def InitializeComponents(self):
        """Initialize component strategies"""
        '''
        
        # Add component initialization
        for i, component in enumerate(ensemble['components']):
            code += f'''
        
        # Component {i}: {component['name']} (Weight: {component['weight']:.2%})
        component_{i} = {{
            'name': '{component['name']}',
            'weight': {component['weight']},
            'signals': {{}},
            'positions': {{}},
            'indicators': {{}}
        }}
        self.components.append(component_{i})
        '''
        
        code += '''
        
        # Initialize universe
        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse(self.CoarseSelectionFunction)
        
    def CoarseSelectionFunction(self, coarse):
        """Select universe for all components"""
        # Get liquid stocks
        filtered = [x for x in coarse if x.Price > 10 and x.DollarVolume > 10000000]
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        
        # Return top stocks for ensemble
        return [x.Symbol for x in sorted_by_volume[:100]]
    
    def OnSecuritiesChanged(self, changes):
        """Initialize indicators for new securities"""
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            
            # Initialize indicators for each component
            for i, component in enumerate(self.components):
                if symbol not in component['indicators']:
                    component['indicators'][symbol] = {
                        'rsi': self.RSI(symbol, 14),
                        'macd': self.MACD(symbol, 12, 26, 9),
                        'bb': self.BB(symbol, 20, 2),
                        'atr': self.ATR(symbol, 14)
                    }
    
    def RebalanceEnsemble(self):
        """Main ensemble rebalancing logic"""
        
        # Check drawdown limit
        current_value = self.Portfolio.TotalPortfolioValue
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        if drawdown > self.max_drawdown_limit:
            self.Liquidate("Ensemble drawdown limit exceeded")
            return
        
        # Generate signals for each component
        ensemble_signals = {}
        
        for i, component in enumerate(self.components):
            component_signals = self.GenerateComponentSignals(i)
            
            # Weight the signals
            for symbol, signal in component_signals.items():
                if symbol not in ensemble_signals:
                    ensemble_signals[symbol] = 0
                ensemble_signals[symbol] += signal * component['weight']
        
        # Execute ensemble signals
        self.ExecuteEnsembleSignals(ensemble_signals)
        
        self.last_rebalance = self.Time
    
    def GenerateComponentSignals(self, component_index):
        """Generate signals for a specific component"""
        component = self.components[component_index]
        signals = {}
        
        # Simple signal generation (would be component-specific in production)
        for symbol, indicators in component['indicators'].items():
            if not all(ind.IsReady for ind in indicators.values()):
                continue
            
            rsi = indicators['rsi'].Current.Value
            macd = indicators['macd'].Current.Value
            signal = indicators['macd'].Signal.Current.Value
            
            # Component-specific logic (simplified)
            if component_index == 0:  # Momentum component
                if rsi > 70 and macd > signal:
                    signals[symbol] = 1
                elif rsi < 30 and macd < signal:
                    signals[symbol] = -1
                else:
                    signals[symbol] = 0
            elif component_index == 1:  # Mean reversion component
                if rsi < 30:
                    signals[symbol] = 1
                elif rsi > 70:
                    signals[symbol] = -1
                else:
                    signals[symbol] = 0
            else:  # Trend following component
                if macd > signal:
                    signals[symbol] = 1
                elif macd < signal:
                    signals[symbol] = -1
                else:
                    signals[symbol] = 0
        
        return signals
    
    def ExecuteEnsembleSignals(self, ensemble_signals):
        """Execute the combined ensemble signals"""
        
        # Filter significant signals
        significant_signals = {s: v for s, v in ensemble_signals.items() if abs(v) > 0.3}
        
        # Calculate position sizes
        if significant_signals:
            total_allocation = 0.95  # Use 95% of capital
            num_positions = len(significant_signals)
            position_size = total_allocation / num_positions if num_positions > 0 else 0
            
            # Execute trades
            for symbol, signal in significant_signals.items():
                target_position = position_size * signal
                
                current_holding = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                
                if abs(target_position - current_holding) > 0.05:  # 5% threshold
                    self.SetHoldings(symbol, target_position)
        
        # Liquidate positions with no signal
        for symbol in list(self.Portfolio.Keys):
            if symbol not in significant_signals and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)
    
    def OnEndOfAlgorithm(self):
        """Log ensemble performance"""
        self.Log(f"Ensemble {self.ensemble_method} completed")
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        '''
        
        return code
    
    def analyze_ensemble_performance(self, ensemble: Dict, backtest_results: Dict) -> Dict:
        """
        Analyze ensemble performance vs individual components
        
        Args:
            ensemble: Ensemble definition
            backtest_results: Results from backtesting the ensemble
            
        Returns:
            Performance analysis
        """
        analysis = {
            'ensemble_metrics': backtest_results,
            'vs_components': {},
            'diversification_benefit': 0,
            'consistency_score': 0,
            'risk_reduction': 0
        }
        
        # Compare to individual components
        component_returns = []
        component_risks = []
        
        for component in ensemble['components']:
            comp_return = component['metrics'].get('cagr', 0)
            comp_risk = component['metrics'].get('max_drawdown', 0.15)
            component_returns.append(comp_return)
            component_risks.append(comp_risk)
            
            analysis['vs_components'][component['name']] = {
                'component_return': comp_return,
                'component_risk': comp_risk,
                'weight': component['weight'],
                'contribution': comp_return * component['weight']
            }
        
        # Calculate diversification benefit
        weighted_avg_return = np.dot(ensemble['weights'], component_returns)
        ensemble_return = backtest_results.get('cagr', 0)
        analysis['diversification_benefit'] = ensemble_return - weighted_avg_return
        
        # Calculate risk reduction
        weighted_avg_risk = np.dot(ensemble['weights'], component_risks)
        ensemble_risk = backtest_results.get('max_drawdown', 0.15)
        analysis['risk_reduction'] = (weighted_avg_risk - ensemble_risk) / weighted_avg_risk
        
        # Consistency score (how stable is the ensemble vs components)
        if backtest_results.get('total_trades', 0) > 0:
            analysis['consistency_score'] = min(
                backtest_results.get('sharpe_ratio', 0) / max([c['metrics'].get('sharpe_ratio', 1) for c in ensemble['components']]),
                1.0
            )
        
        return analysis
    
    def optimize_ensemble_live(self, current_performance: Dict) -> Dict:
        """
        Dynamically optimize ensemble weights based on recent performance
        
        Args:
            current_performance: Recent performance metrics for each component
            
        Returns:
            Updated ensemble configuration
        """
        # This would be called periodically to adjust weights
        # based on changing market conditions and component performance
        
        updated_weights = self.component_weights.copy()
        
        # Simple adaptive logic (would be more sophisticated in production)
        for i, perf in enumerate(current_performance):
            if perf['recent_sharpe'] > 1.5:
                updated_weights[i] *= 1.1  # Increase weight
            elif perf['recent_sharpe'] < 0.5:
                updated_weights[i] *= 0.9  # Decrease weight
        
        # Renormalize
        updated_weights = updated_weights / updated_weights.sum()
        
        # Apply constraints
        updated_weights = np.clip(updated_weights, 0.05, 0.4)
        updated_weights = updated_weights / updated_weights.sum()
        
        return {
            'updated_weights': updated_weights.tolist(),
            'reason': 'Performance-based adjustment',
            'timestamp': datetime.now()
        }


# Example usage
if __name__ == '__main__':
    # Initialize generator
    generator = EnsembleStrategyGenerator()
    
    # Add some example strategies to the pool
    example_strategies = [
        {
            'name': 'MomentumAlpha',
            'type': 'momentum',
            'indicators': ['RSI', 'MACD', 'ADX'],
            'asset_classes': {'equities': 0.8, 'crypto': 0.2}
        },
        {
            'name': 'MeanReversionBeta',
            'type': 'mean_reversion',
            'indicators': ['RSI', 'BB', 'ATR'],
            'asset_classes': {'equities': 0.6, 'forex': 0.4}
        },
        {
            'name': 'TrendFollowingGamma',
            'type': 'trend_following',
            'indicators': ['EMA', 'MACD', 'ADX'],
            'asset_classes': {'equities': 0.5, 'commodities': 0.5}
        },
        {
            'name': 'VolatilityHarvester',
            'type': 'volatility',
            'indicators': ['ATR', 'BB', 'KELT'],
            'asset_classes': {'options': 0.6, 'forex': 0.4}
        },
        {
            'name': 'ArbitrageSeeker',
            'type': 'arbitrage',
            'indicators': ['SPREAD', 'CORR', 'ZSCORE'],
            'asset_classes': {'equities': 0.5, 'forex': 0.5}
        }
    ]
    
    # Add strategies with example metrics
    for strategy in example_strategies:
        metrics = {
            'cagr': np.random.uniform(0.15, 0.35),
            'sharpe_ratio': np.random.uniform(0.8, 2.0),
            'max_drawdown': np.random.uniform(0.08, 0.20),
            'total_trades': np.random.randint(100, 1000)
        }
        generator.add_strategy_to_pool(strategy, metrics)
    
    # Generate different ensemble types
    print("=== Generating Ensemble Strategies ===\n")
    
    for method in ['equal', 'risk_parity', 'max_sharpe', 'ml_weighted', 'regime_adaptive']:
        ensemble = generator.generate_ensemble(method=method, target_size=3, regime='bull')
        
        if ensemble:
            print(f"\n{method.upper()} Ensemble:")
            print(f"Components: {len(ensemble['components'])}")
            print(f"Expected CAGR: {ensemble['expected_metrics']['expected_cagr']:.2%}")
            print(f"Expected Sharpe: {ensemble['expected_metrics']['expected_sharpe_ratio']:.2f}")
            print(f"Diversification Ratio: {ensemble['expected_metrics']['diversification_ratio']:.2f}")
            
            print("\nComponent Weights:")
            for comp in ensemble['components']:
                print(f"  - {comp['name']}: {comp['weight']:.2%}")
    
    # Test performance analysis
    print("\n=== Ensemble Performance Analysis ===")
    
    if ensemble:
        mock_results = {
            'cagr': 0.28,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.10,
            'total_trades': 500
        }
        
        analysis = generator.analyze_ensemble_performance(ensemble, mock_results)
        print(f"Diversification Benefit: {analysis['diversification_benefit']:.2%}")
        print(f"Risk Reduction: {analysis['risk_reduction']:.2%}")
        print(f"Consistency Score: {analysis['consistency_score']:.2f}")