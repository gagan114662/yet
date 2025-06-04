"""
Market Regime-Aware Evolution System
Evolves different strategy populations for different market conditions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import logging

from market_regime_detector import MarketRegimeDetector, MarketRegime
from self_improving_evolution import SelfImprovingEvolution, EvolutionMetrics

logger = logging.getLogger(__name__)


@dataclass
class RegimeStrategy:
    """Strategy with regime-specific adaptations"""
    strategy_id: str
    base_strategy: Dict
    regime_adaptations: Dict[str, Dict] = field(default_factory=dict)
    performance_by_regime: Dict[str, float] = field(default_factory=dict)
    creation_regime: str = "unknown"
    is_hybrid: bool = False
    regime_transitions: List[Tuple[str, str, float]] = field(default_factory=list)


class RegimePopulation:
    """Population specialized for a specific market regime"""
    
    def __init__(self, regime: MarketRegime, size: int = 50):
        self.regime = regime
        self.size = size
        self.population = []
        self.generation = 0
        self.performance_history = deque(maxlen=50)
        self.best_strategies = deque(maxlen=10)
        
        # Regime-specific parameters
        self.mutation_rate = self._get_regime_mutation_rate()
        self.crossover_rate = self._get_regime_crossover_rate()
        self.selection_pressure = self._get_regime_selection_pressure()
        
    def _get_regime_mutation_rate(self) -> float:
        """Get regime-specific mutation rate"""
        rates = {
            MarketRegime.CRASH: 0.4,  # High mutation in crash
            MarketRegime.STRONG_BEAR: 0.35,
            MarketRegime.BEAR: 0.3,
            MarketRegime.WEAK_BEAR: 0.28,
            MarketRegime.SIDEWAYS: 0.25,
            MarketRegime.WEAK_BULL: 0.22,
            MarketRegime.BULL: 0.2,
            MarketRegime.STRONG_BULL: 0.18,  # Low mutation in strong bull
            MarketRegime.HIGH_VOLATILITY: 0.35,
            MarketRegime.LOW_VOLATILITY: 0.15,
            MarketRegime.RECOVERY: 0.3
        }
        return rates.get(self.regime, 0.25)
    
    def _get_regime_crossover_rate(self) -> float:
        """Get regime-specific crossover rate"""
        rates = {
            MarketRegime.CRASH: 0.5,  # More exploration needed
            MarketRegime.STRONG_BEAR: 0.6,
            MarketRegime.BEAR: 0.65,
            MarketRegime.SIDEWAYS: 0.7,
            MarketRegime.BULL: 0.75,
            MarketRegime.STRONG_BULL: 0.8,  # High crossover in bull
            MarketRegime.HIGH_VOLATILITY: 0.6,
            MarketRegime.LOW_VOLATILITY: 0.8
        }
        return rates.get(self.regime, 0.7)
    
    def _get_regime_selection_pressure(self) -> float:
        """Get regime-specific selection pressure"""
        pressure = {
            MarketRegime.CRASH: 0.5,  # Low pressure, need diversity
            MarketRegime.STRONG_BEAR: 0.6,
            MarketRegime.BEAR: 0.7,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.BULL: 0.9,
            MarketRegime.STRONG_BULL: 1.0,  # High pressure in bull
            MarketRegime.HIGH_VOLATILITY: 0.6,
            MarketRegime.LOW_VOLATILITY: 0.9
        }
        return pressure.get(self.regime, 0.8)
    
    def initialize_population(self):
        """Initialize population with regime-appropriate strategies"""
        self.population = []
        
        # Get regime-specific strategy preferences
        preferred_types = self._get_preferred_strategy_types()
        
        for i in range(self.size):
            # Bias toward preferred types
            if np.random.random() < 0.7:
                strategy_type = np.random.choice(preferred_types)
            else:
                strategy_type = np.random.choice(['momentum', 'mean_reversion', 'trend_following', 'volatility'])
            
            strategy = self._create_regime_strategy(strategy_type)
            self.population.append(RegimeStrategy(
                strategy_id=f"{self.regime.value}_gen{self.generation}_{i}",
                base_strategy=strategy,
                creation_regime=self.regime.value
            ))
    
    def _get_preferred_strategy_types(self) -> List[str]:
        """Get preferred strategy types for regime"""
        preferences = {
            MarketRegime.STRONG_BULL: ['momentum', 'trend_following', 'breakout'],
            MarketRegime.BULL: ['momentum', 'trend_following'],
            MarketRegime.SIDEWAYS: ['mean_reversion', 'range_trading'],
            MarketRegime.BEAR: ['short_momentum', 'defensive'],
            MarketRegime.STRONG_BEAR: ['inverse', 'volatility'],
            MarketRegime.CRASH: ['defensive', 'cash'],
            MarketRegime.HIGH_VOLATILITY: ['volatility', 'options'],
            MarketRegime.LOW_VOLATILITY: ['carry', 'mean_reversion']
        }
        return preferences.get(self.regime, ['balanced'])
    
    def _create_regime_strategy(self, strategy_type: str) -> Dict:
        """Create strategy adapted for regime"""
        base_strategy = {
            'type': strategy_type,
            'regime_optimized': self.regime.value,
            'creation_time': datetime.now().isoformat()
        }
        
        # Regime-specific parameters
        if self.regime in [MarketRegime.CRASH, MarketRegime.STRONG_BEAR]:
            base_strategy.update({
                'leverage': np.random.uniform(0.0, 0.5),
                'position_size': np.random.uniform(0.02, 0.1),
                'stop_loss': np.random.uniform(0.02, 0.05),
                'max_positions': np.random.randint(1, 3),
                'defensive_mode': True
            })
        elif self.regime in [MarketRegime.STRONG_BULL]:
            base_strategy.update({
                'leverage': np.random.uniform(1.5, 3.0),
                'position_size': np.random.uniform(0.15, 0.3),
                'stop_loss': np.random.uniform(0.08, 0.15),
                'max_positions': np.random.randint(5, 10),
                'aggressive_mode': True
            })
        else:
            base_strategy.update({
                'leverage': np.random.uniform(0.8, 2.0),
                'position_size': np.random.uniform(0.1, 0.2),
                'stop_loss': np.random.uniform(0.05, 0.12),
                'max_positions': np.random.randint(3, 7)
            })
        
        # Add regime-specific indicators
        base_strategy['indicators'] = self._get_regime_indicators(strategy_type)
        
        return base_strategy
    
    def _get_regime_indicators(self, strategy_type: str) -> List[str]:
        """Get appropriate indicators for regime and strategy type"""
        regime_indicators = {
            MarketRegime.HIGH_VOLATILITY: ['ATR', 'BB', 'VIX'],
            MarketRegime.CRASH: ['VIX', 'PUT_CALL', 'SAFE_HAVEN'],
            MarketRegime.STRONG_BULL: ['RSI', 'MACD', 'ADX'],
            MarketRegime.SIDEWAYS: ['BB', 'RSI', 'STOCH']
        }
        
        base_indicators = regime_indicators.get(self.regime, ['RSI', 'MACD'])
        
        # Add strategy-specific indicators
        if strategy_type == 'momentum':
            base_indicators.extend(['ROC', 'MOM'])
        elif strategy_type == 'mean_reversion':
            base_indicators.extend(['BB', 'RSI'])
        
        return list(set(base_indicators))[:5]  # Limit to 5 indicators
    
    def evolve_generation(self):
        """Evolve one generation with regime awareness"""
        self.generation += 1
        
        # Evaluate current population
        fitness_scores = [self._evaluate_strategy(s) for s in self.population]
        
        # Track best strategies
        best_idx = np.argmax(fitness_scores)
        self.best_strategies.append((
            self.population[best_idx],
            fitness_scores[best_idx]
        ))
        
        # Select parents
        parents = self._select_parents(fitness_scores)
        
        # Create new generation
        new_population = []
        
        # Elite preservation
        elite_count = int(self.size * 0.1)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Generate offspring
        while len(new_population) < self.size:
            if np.random.random() < self.crossover_rate:
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)
                offspring = self._crossover(parent1, parent2)
            else:
                parent = np.random.choice(parents)
                offspring = self._clone_strategy(parent)
            
            # Mutate
            if np.random.random() < self.mutation_rate:
                offspring = self._mutate(offspring)
            
            new_population.append(offspring)
        
        self.population = new_population
        
        # Track performance
        avg_fitness = np.mean(fitness_scores)
        self.performance_history.append({
            'generation': self.generation,
            'avg_fitness': avg_fitness,
            'best_fitness': fitness_scores[best_idx],
            'diversity': self._calculate_diversity()
        })
    
    def _evaluate_strategy(self, strategy: RegimeStrategy) -> float:
        """Evaluate strategy fitness for current regime"""
        # Base fitness from strategy performance
        base_fitness = np.random.random()  # Placeholder - would use actual backtesting
        
        # Regime alignment bonus
        if strategy.creation_regime == self.regime.value:
            regime_bonus = 0.2
        elif strategy.is_hybrid:
            regime_bonus = 0.1
        else:
            regime_bonus = 0
        
        # Type alignment bonus
        preferred_types = self._get_preferred_strategy_types()
        if strategy.base_strategy.get('type') in preferred_types:
            type_bonus = 0.1
        else:
            type_bonus = 0
        
        return base_fitness + regime_bonus + type_bonus
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        # Type diversity
        types = [s.base_strategy.get('type', 'unknown') for s in self.population]
        type_diversity = len(set(types)) / len(types)
        
        # Parameter diversity
        leverages = [s.base_strategy.get('leverage', 1) for s in self.population]
        param_diversity = np.std(leverages) / (np.mean(leverages) + 1e-6)
        
        return 0.5 * type_diversity + 0.5 * min(param_diversity, 1.0)


class RegimeAwareEvolution:
    """
    Complete regime-aware evolution system with automatic switching
    """
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = MarketRegime.SIDEWAYS
        
        # Maintain populations for each regime
        self.regime_populations = {}
        self.initialize_all_populations()
        
        # Hybrid strategies that work across regimes
        self.hybrid_strategies = []
        
        # Performance tracking
        self.regime_history = deque(maxlen=100)
        self.transition_history = deque(maxlen=50)
        
        # Real-time adaptation
        self.adaptation_speed = 0.1
        self.regime_confidence_threshold = 0.7
        
    def initialize_all_populations(self):
        """Initialize populations for all regimes"""
        # Focus on main regimes
        main_regimes = [
            MarketRegime.STRONG_BULL,
            MarketRegime.BULL,
            MarketRegime.SIDEWAYS,
            MarketRegime.BEAR,
            MarketRegime.STRONG_BEAR,
            MarketRegime.HIGH_VOLATILITY
        ]
        
        for regime in main_regimes:
            pop = RegimePopulation(regime, size=30)
            pop.initialize_population()
            self.regime_populations[regime] = pop
            
        logger.info(f"Initialized {len(self.regime_populations)} regime populations")
    
    def detect_and_adapt(self, market_data: Dict) -> MarketRegime:
        """Detect regime and adapt strategies"""
        # Detect new regime
        new_regime, confidence = self.regime_detector.detect_regime(market_data)
        
        # Check if regime changed
        if new_regime != self.current_regime and confidence >= self.regime_confidence_threshold:
            self._handle_regime_transition(self.current_regime, new_regime, confidence)
            self.current_regime = new_regime
        
        # Record regime
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': new_regime,
            'confidence': confidence
        })
        
        return new_regime
    
    def _handle_regime_transition(self, old_regime: MarketRegime, new_regime: MarketRegime, confidence: float):
        """Handle transition between regimes"""
        logger.info(f"Regime transition: {old_regime.value} → {new_regime.value} (confidence: {confidence:.2f})")
        
        # Record transition
        self.transition_history.append({
            'timestamp': datetime.now(),
            'from_regime': old_regime,
            'to_regime': new_regime,
            'confidence': confidence
        })
        
        # Migrate successful strategies
        if old_regime in self.regime_populations and new_regime in self.regime_populations:
            self._migrate_strategies(old_regime, new_regime)
        
        # Create hybrid strategies
        self._create_hybrid_strategies(old_regime, new_regime)
    
    def _migrate_strategies(self, from_regime: MarketRegime, to_regime: MarketRegime):
        """Migrate successful strategies between regime populations"""
        from_pop = self.regime_populations[from_regime]
        to_pop = self.regime_populations[to_regime]
        
        # Get top strategies from old regime
        if from_pop.best_strategies:
            top_strategies = sorted(from_pop.best_strategies, key=lambda x: x[1], reverse=True)[:3]
            
            for strategy, fitness in top_strategies:
                # Adapt strategy for new regime
                adapted = self._adapt_strategy_for_regime(strategy, to_regime)
                
                # Add to new population
                to_pop.population.append(adapted)
                
                logger.info(f"Migrated strategy {strategy.strategy_id} from {from_regime.value} to {to_regime.value}")
    
    def _adapt_strategy_for_regime(self, strategy: RegimeStrategy, new_regime: MarketRegime) -> RegimeStrategy:
        """Adapt strategy for new regime"""
        adapted = RegimeStrategy(
            strategy_id=f"{strategy.strategy_id}_adapted_{new_regime.value}",
            base_strategy=strategy.base_strategy.copy(),
            regime_adaptations=strategy.regime_adaptations.copy(),
            performance_by_regime=strategy.performance_by_regime.copy(),
            creation_regime=strategy.creation_regime,
            is_hybrid=True
        )
        
        # Add regime-specific adaptations
        adaptations = self._get_regime_adaptations(new_regime)
        adapted.regime_adaptations[new_regime.value] = adaptations
        
        # Apply adaptations to base strategy
        if new_regime in [MarketRegime.CRASH, MarketRegime.STRONG_BEAR]:
            adapted.base_strategy['leverage'] *= 0.3
            adapted.base_strategy['position_size'] *= 0.5
            adapted.base_strategy['defensive_mode'] = True
        elif new_regime == MarketRegime.STRONG_BULL:
            adapted.base_strategy['leverage'] = min(adapted.base_strategy.get('leverage', 1) * 1.5, 3.0)
            adapted.base_strategy['aggressive_mode'] = True
        
        return adapted
    
    def _get_regime_adaptations(self, regime: MarketRegime) -> Dict:
        """Get specific adaptations for a regime"""
        adaptations = {
            MarketRegime.CRASH: {
                'max_positions': 1,
                'stop_loss_multiplier': 0.5,
                'position_size_multiplier': 0.3,
                'preferred_assets': ['bonds', 'gold', 'cash']
            },
            MarketRegime.STRONG_BEAR: {
                'max_positions': 3,
                'stop_loss_multiplier': 0.7,
                'position_size_multiplier': 0.5,
                'preferred_assets': ['inverse_etfs', 'commodities']
            },
            MarketRegime.SIDEWAYS: {
                'max_positions': 5,
                'stop_loss_multiplier': 1.0,
                'position_size_multiplier': 0.8,
                'preferred_assets': ['range_bound_stocks']
            },
            MarketRegime.STRONG_BULL: {
                'max_positions': 10,
                'stop_loss_multiplier': 1.5,
                'position_size_multiplier': 1.2,
                'preferred_assets': ['growth_stocks', 'leveraged_etfs']
            }
        }
        
        return adaptations.get(regime, {})
    
    def _create_hybrid_strategies(self, regime1: MarketRegime, regime2: MarketRegime):
        """Create strategies that work in multiple regimes"""
        if regime1 not in self.regime_populations or regime2 not in self.regime_populations:
            return
        
        pop1 = self.regime_populations[regime1]
        pop2 = self.regime_populations[regime2]
        
        # Get best from each regime
        if pop1.best_strategies and pop2.best_strategies:
            best1 = pop1.best_strategies[-1][0]
            best2 = pop2.best_strategies[-1][0]
            
            # Create hybrid
            hybrid = self._combine_strategies(best1, best2)
            hybrid.is_hybrid = True
            hybrid.regime_transitions.append((regime1.value, regime2.value, datetime.now().timestamp()))
            
            self.hybrid_strategies.append(hybrid)
            
            logger.info(f"Created hybrid strategy for {regime1.value} ↔ {regime2.value}")
    
    def _combine_strategies(self, strategy1: RegimeStrategy, strategy2: RegimeStrategy) -> RegimeStrategy:
        """Combine two strategies into a hybrid"""
        # Average parameters
        combined_params = {}
        for key in ['leverage', 'position_size', 'stop_loss']:
            val1 = strategy1.base_strategy.get(key, 1)
            val2 = strategy2.base_strategy.get(key, 1)
            combined_params[key] = (val1 + val2) / 2
        
        # Combine indicators
        indicators1 = set(strategy1.base_strategy.get('indicators', []))
        indicators2 = set(strategy2.base_strategy.get('indicators', []))
        combined_indicators = list(indicators1.union(indicators2))[:7]  # Limit total
        
        hybrid = RegimeStrategy(
            strategy_id=f"hybrid_{strategy1.strategy_id}_{strategy2.strategy_id}",
            base_strategy={
                'type': 'hybrid',
                'parent_strategies': [strategy1.strategy_id, strategy2.strategy_id],
                'indicators': combined_indicators,
                **combined_params
            },
            creation_regime='hybrid',
            is_hybrid=True
        )
        
        # Inherit adaptations
        hybrid.regime_adaptations.update(strategy1.regime_adaptations)
        hybrid.regime_adaptations.update(strategy2.regime_adaptations)
        
        return hybrid
    
    def evolve_current_population(self):
        """Evolve the current regime's population"""
        if self.current_regime in self.regime_populations:
            pop = self.regime_populations[self.current_regime]
            pop.evolve_generation()
            
            # Periodically test hybrid strategies
            if pop.generation % 5 == 0:
                self._evaluate_hybrid_strategies()
    
    def _evaluate_hybrid_strategies(self):
        """Evaluate hybrid strategies in current regime"""
        current_pop = self.regime_populations.get(self.current_regime)
        if not current_pop or not self.hybrid_strategies:
            return
        
        # Test each hybrid
        for hybrid in self.hybrid_strategies[-10:]:  # Test recent hybrids
            fitness = current_pop._evaluate_strategy(hybrid)
            hybrid.performance_by_regime[self.current_regime.value] = fitness
            
            # If performs well, add to population
            if fitness > 0.7:
                current_pop.population.append(hybrid)
                logger.info(f"Hybrid strategy {hybrid.strategy_id} added to {self.current_regime.value} population")
    
    def get_best_strategies_for_regime(self, regime: Optional[MarketRegime] = None) -> List[Tuple[RegimeStrategy, float]]:
        """Get best strategies for a regime"""
        if regime is None:
            regime = self.current_regime
        
        if regime not in self.regime_populations:
            return []
        
        pop = self.regime_populations[regime]
        return list(pop.best_strategies)
    
    def get_adaptive_portfolio(self) -> List[RegimeStrategy]:
        """Get portfolio of strategies adapted to current regime"""
        portfolio = []
        
        # Get best from current regime
        current_best = self.get_best_strategies_for_regime()
        portfolio.extend([s[0] for s in current_best[:3]])
        
        # Add best hybrid strategies
        if self.hybrid_strategies:
            # Sort by performance in current regime
            hybrids_sorted = sorted(
                self.hybrid_strategies,
                key=lambda s: s.performance_by_regime.get(self.current_regime.value, 0),
                reverse=True
            )
            portfolio.extend(hybrids_sorted[:2])
        
        return portfolio
    
    def generate_status_report(self) -> Dict:
        """Generate comprehensive status report"""
        report = {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_history[-1]['confidence'] if self.regime_history else 0,
            'populations': {},
            'hybrid_strategies': len(self.hybrid_strategies),
            'recent_transitions': [],
            'recommendations': []
        }
        
        # Population status
        for regime, pop in self.regime_populations.items():
            report['populations'][regime.value] = {
                'generation': pop.generation,
                'size': len(pop.population),
                'best_fitness': pop.best_strategies[-1][1] if pop.best_strategies else 0,
                'diversity': pop._calculate_diversity()
            }
        
        # Recent transitions
        for transition in list(self.transition_history)[-5:]:
            report['recent_transitions'].append({
                'from': transition['from_regime'].value,
                'to': transition['to_regime'].value,
                'time': transition['timestamp'].isoformat()
            })
        
        # Recommendations
        if self.current_regime in [MarketRegime.CRASH, MarketRegime.STRONG_BEAR]:
            report['recommendations'].append("Focus on defensive strategies and capital preservation")
        elif self.current_regime == MarketRegime.HIGH_VOLATILITY:
            report['recommendations'].append("Use volatility-based strategies and smaller positions")
        elif self.current_regime == MarketRegime.STRONG_BULL:
            report['recommendations'].append("Increase leverage and position sizes cautiously")
        
        return report


def demonstrate_regime_aware_evolution():
    """Demonstrate regime-aware evolution system"""
    logger.info("=== Regime-Aware Evolution Demo ===")
    
    # Create system
    system = RegimeAwareEvolution()
    
    # Simulate different market conditions
    market_scenarios = [
        {
            'name': 'Bull Market',
            'data': {
                'spy_price': 450,
                'spy_sma_50': 440,
                'spy_sma_200': 420,
                'vix': 15,
                'momentum_20d': 0.05,
                'advance_decline_ratio': 1.8
            }
        },
        {
            'name': 'Market Crash',
            'data': {
                'spy_price': 350,
                'spy_sma_50': 400,
                'spy_sma_200': 420,
                'vix': 45,
                'momentum_20d': -0.15,
                'advance_decline_ratio': 0.3
            }
        },
        {
            'name': 'Recovery',
            'data': {
                'spy_price': 380,
                'spy_sma_50': 370,
                'spy_sma_200': 390,
                'vix': 25,
                'momentum_20d': 0.03,
                'advance_decline_ratio': 1.2
            }
        }
    ]
    
    for scenario in market_scenarios:
        logger.info(f"\n--- {scenario['name']} Scenario ---")
        
        # Detect regime
        regime = system.detect_and_adapt(scenario['data'])
        logger.info(f"Detected regime: {regime.value}")
        
        # Evolve for a few generations
        for _ in range(3):
            system.evolve_current_population()
        
        # Get adaptive portfolio
        portfolio = system.get_adaptive_portfolio()
        logger.info(f"Adaptive portfolio size: {len(portfolio)}")
        
        if portfolio:
            logger.info("Portfolio strategies:")
            for strategy in portfolio[:3]:
                logger.info(f"  - {strategy.strategy_id} (type: {strategy.base_strategy.get('type')})")
    
    # Generate report
    report = system.generate_status_report()
    logger.info("\n=== Status Report ===")
    logger.info(f"Current regime: {report['current_regime']}")
    logger.info(f"Hybrid strategies created: {report['hybrid_strategies']}")
    logger.info("Population generations:")
    for regime, stats in report['populations'].items():
        logger.info(f"  {regime}: Gen {stats['generation']}, Best: {stats['best_fitness']:.3f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    demonstrate_regime_aware_evolution()