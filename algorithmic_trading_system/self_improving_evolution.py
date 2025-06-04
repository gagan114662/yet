"""
Self-Improving Evolutionary System
Implements meta-learning and adaptive evolution parameters
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class EvolutionMetrics:
    """Track evolution performance metrics"""
    generation: int
    best_fitness: float
    avg_fitness: float
    diversity_score: float
    mutation_success_rate: float
    crossover_success_rate: float
    convergence_rate: float
    time_to_improvement: float


@dataclass
class MutationParameters:
    """Adaptive mutation parameters"""
    rate: float = 0.3
    strength: float = 0.1
    type_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.type_weights is None:
            self.type_weights = {
                'parameter_tweak': 0.4,
                'indicator_change': 0.3,
                'strategy_shift': 0.2,
                'risk_adjustment': 0.1
            }


class MetaLearner:
    """
    Meta-learning system that optimizes the optimization process itself
    """
    
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.parameter_history = deque(maxlen=50)
        self.regime_performance = defaultdict(list)
        self.meta_parameters = {
            'learning_rate': 0.01,
            'adaptation_speed': 0.1,
            'exploration_bonus': 0.2
        }
        
    def update_meta_parameters(self, recent_performance: List[EvolutionMetrics]):
        """Update meta-parameters based on recent performance"""
        if len(recent_performance) < 5:
            return
        
        # Analyze performance trends
        fitness_trend = np.polyfit(
            range(len(recent_performance)),
            [m.best_fitness for m in recent_performance],
            1
        )[0]
        
        diversity_trend = np.mean([m.diversity_score for m in recent_performance])
        
        # Adjust meta-parameters
        if fitness_trend < 0.001:  # Stagnation
            self.meta_parameters['exploration_bonus'] *= 1.2
            self.meta_parameters['adaptation_speed'] *= 1.1
            logger.info("Detected stagnation - increasing exploration")
        elif diversity_trend < 0.5:  # Low diversity
            self.meta_parameters['exploration_bonus'] *= 1.3
            logger.info("Low diversity - boosting exploration bonus")
        else:  # Good progress
            self.meta_parameters['exploration_bonus'] *= 0.95
            
        # Keep parameters in reasonable bounds
        self.meta_parameters['exploration_bonus'] = np.clip(
            self.meta_parameters['exploration_bonus'], 0.1, 0.5
        )
        
    def suggest_evolution_adjustments(self) -> Dict:
        """Suggest adjustments to evolution parameters"""
        suggestions = {
            'mutation_rate_adjustment': 1.0,
            'crossover_rate_adjustment': 1.0,
            'population_size_adjustment': 1.0,
            'selection_pressure_adjustment': 1.0
        }
        
        # Analyze recent performance
        if len(self.performance_history) >= 10:
            recent = list(self.performance_history)[-10:]
            
            # Check mutation effectiveness
            avg_mutation_success = np.mean([m.mutation_success_rate for m in recent])
            if avg_mutation_success < 0.2:
                suggestions['mutation_rate_adjustment'] = 0.8  # Reduce mutations
            elif avg_mutation_success > 0.4:
                suggestions['mutation_rate_adjustment'] = 1.2  # Increase mutations
            
            # Check convergence
            avg_convergence = np.mean([m.convergence_rate for m in recent])
            if avg_convergence > 0.8:
                suggestions['population_size_adjustment'] = 1.5  # Increase diversity
                suggestions['selection_pressure_adjustment'] = 0.7  # Reduce pressure
        
        return suggestions


class SelfImprovingEvolution:
    """
    Self-improving evolutionary system with meta-learning capabilities
    """
    
    def __init__(self, initial_population_size: int = 50):
        self.population_size = initial_population_size
        self.generation = 0
        self.meta_learner = MetaLearner()
        
        # Adaptive parameters
        self.mutation_params = MutationParameters()
        self.crossover_rate = 0.7
        self.selection_params = {
            'method': 'tournament',
            'tournament_size': 3,
            'elite_ratio': 0.1
        }
        
        # Evolution of evolution parameters
        self.param_population = self._initialize_param_population()
        self.fitness_function_weights = {
            'return': 0.4,
            'sharpe': 0.3,
            'drawdown': 0.3
        }
        
        # Performance tracking
        self.evolution_history = []
        self.successful_mutations = defaultdict(int)
        self.failed_mutations = defaultdict(int)
        
    def _initialize_param_population(self) -> List[Dict]:
        """Initialize population of evolution parameters"""
        param_pop = []
        for _ in range(10):  # Keep smaller param population
            params = {
                'mutation_rate': np.random.uniform(0.1, 0.5),
                'crossover_rate': np.random.uniform(0.5, 0.9),
                'mutation_strength': np.random.uniform(0.05, 0.3),
                'selection_pressure': np.random.uniform(0.5, 2.0)
            }
            param_pop.append(params)
        return param_pop
    
    def evolve_parameters(self):
        """Evolve the evolution parameters themselves"""
        # Evaluate parameter sets based on recent performance
        param_fitness = []
        for params in self.param_population:
            # Simulate or use historical data to evaluate
            fitness = self._evaluate_parameter_fitness(params)
            param_fitness.append(fitness)
        
        # Select best parameters
        best_idx = np.argmax(param_fitness)
        best_params = self.param_population[best_idx]
        
        # Update current parameters with some inertia
        inertia = 0.7
        self.mutation_params.rate = (
            inertia * self.mutation_params.rate +
            (1 - inertia) * best_params['mutation_rate']
        )
        self.crossover_rate = (
            inertia * self.crossover_rate +
            (1 - inertia) * best_params['crossover_rate']
        )
        
        # Mutate parameter population
        self._mutate_param_population()
        
        logger.info(f"Evolved parameters - Mutation rate: {self.mutation_params.rate:.3f}, "
                   f"Crossover rate: {self.crossover_rate:.3f}")
    
    def _evaluate_parameter_fitness(self, params: Dict) -> float:
        """Evaluate fitness of a parameter set"""
        # Use recent history to evaluate parameters
        if len(self.evolution_history) < 5:
            return np.random.random()
        
        # Score based on improvement rate with these parameters
        # In practice, this would use actual historical data
        fitness = 0.0
        
        # Prefer moderate mutation rates
        if 0.2 <= params['mutation_rate'] <= 0.4:
            fitness += 0.3
        
        # Prefer high crossover rates
        if params['crossover_rate'] > 0.7:
            fitness += 0.2
        
        # Add noise for exploration
        fitness += np.random.normal(0, 0.1)
        
        return max(0, fitness)
    
    def _mutate_param_population(self):
        """Mutate the parameter population"""
        for params in self.param_population[1:]:  # Keep best unchanged
            if np.random.random() < 0.3:
                # Mutate mutation rate
                params['mutation_rate'] *= np.random.uniform(0.8, 1.2)
                params['mutation_rate'] = np.clip(params['mutation_rate'], 0.05, 0.8)
            
            if np.random.random() < 0.3:
                # Mutate crossover rate
                params['crossover_rate'] *= np.random.uniform(0.9, 1.1)
                params['crossover_rate'] = np.clip(params['crossover_rate'], 0.3, 0.95)
    
    def adapt_mutation_strategy(self, recent_success_rate: float):
        """Adapt mutation strategy based on success rate"""
        if recent_success_rate < 0.1:
            # Very low success - reduce mutation strength
            self.mutation_params.strength *= 0.9
            # Shift to safer mutations
            self.mutation_params.type_weights['parameter_tweak'] += 0.1
            self.mutation_params.type_weights['strategy_shift'] -= 0.1
        elif recent_success_rate > 0.3:
            # High success - can be more aggressive
            self.mutation_params.strength *= 1.1
            # Try more diverse mutations
            self.mutation_params.type_weights['strategy_shift'] += 0.05
        
        # Normalize weights
        total = sum(self.mutation_params.type_weights.values())
        for key in self.mutation_params.type_weights:
            self.mutation_params.type_weights[key] /= total
    
    def evolve_fitness_function(self, performance_data: List[Dict]):
        """Evolve the fitness function weights based on what works"""
        if len(performance_data) < 20:
            return
        
        # Analyze which strategies succeeded
        successful = [p for p in performance_data if p['met_targets']]
        failed = [p for p in performance_data if not p['met_targets']]
        
        if not successful:
            return
        
        # Find patterns in successful strategies
        avg_success_metrics = {
            'return': np.mean([s['metrics']['return'] for s in successful]),
            'sharpe': np.mean([s['metrics']['sharpe'] for s in successful]),
            'drawdown': np.mean([s['metrics']['drawdown'] for s in successful])
        }
        
        avg_failed_metrics = {
            'return': np.mean([f['metrics']['return'] for f in failed]) if failed else 0,
            'sharpe': np.mean([f['metrics']['sharpe'] for f in failed]) if failed else 0,
            'drawdown': np.mean([f['metrics']['drawdown'] for f in failed]) if failed else 1
        }
        
        # Adjust weights based on discriminative power
        for metric in ['return', 'sharpe', 'drawdown']:
            if metric == 'drawdown':
                # Lower is better for drawdown
                diff = avg_failed_metrics[metric] - avg_success_metrics[metric]
            else:
                diff = avg_success_metrics[metric] - avg_failed_metrics[metric]
            
            if diff > 0:
                # This metric discriminates well
                self.fitness_function_weights[metric] *= 1.05
        
        # Normalize weights
        total = sum(self.fitness_function_weights.values())
        for key in self.fitness_function_weights:
            self.fitness_function_weights[key] /= total
        
        logger.info(f"Evolved fitness weights: {self.fitness_function_weights}")
    
    def select_parents_adaptively(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Adaptively select parents using evolved selection strategy"""
        # Evolve selection parameters based on diversity
        diversity = self._calculate_diversity(population)
        
        if diversity < 0.3:
            # Low diversity - reduce selection pressure
            self.selection_params['tournament_size'] = max(2, self.selection_params['tournament_size'] - 1)
            self.selection_params['elite_ratio'] *= 0.9
            logger.info("Low diversity - reducing selection pressure")
        elif diversity > 0.8:
            # High diversity - increase selection pressure
            self.selection_params['tournament_size'] = min(5, self.selection_params['tournament_size'] + 1)
            self.selection_params['elite_ratio'] *= 1.1
        
        # Keep elite ratio reasonable
        self.selection_params['elite_ratio'] = np.clip(self.selection_params['elite_ratio'], 0.05, 0.3)
        
        # Perform selection
        num_parents = int(len(population) * 0.5)
        parents = []
        
        # Elite selection
        num_elite = int(len(population) * self.selection_params['elite_ratio'])
        elite_indices = np.argsort(fitness_scores)[-num_elite:]
        parents.extend([population[i] for i in elite_indices])
        
        # Tournament selection for the rest
        while len(parents) < num_parents:
            tournament_indices = np.random.choice(
                len(population),
                size=self.selection_params['tournament_size'],
                replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def _calculate_diversity(self, population: List[Dict]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 1.0
        
        # Simple diversity based on strategy types and parameters
        strategy_types = [p.get('type', 'unknown') for p in population]
        type_diversity = len(set(strategy_types)) / len(strategy_types)
        
        # Parameter diversity (simplified)
        param_diversity = 0
        param_keys = ['leverage', 'position_size', 'stop_loss']
        
        for key in param_keys:
            values = [p.get(key, 0) for p in population]
            if values:
                param_diversity += np.std(values) / (np.mean(values) + 1e-6)
        
        param_diversity /= len(param_keys)
        
        # Combined diversity
        diversity = 0.5 * type_diversity + 0.5 * min(param_diversity, 1.0)
        return diversity
    
    def apply_meta_learning(self, strategy_generation_history: List[Dict]):
        """Apply meta-learning to improve strategy generation"""
        if len(strategy_generation_history) < 50:
            return
        
        # Identify patterns in successful strategies
        successful_patterns = defaultdict(int)
        failed_patterns = defaultdict(int)
        
        for strategy in strategy_generation_history:
            patterns = self._extract_patterns(strategy)
            if strategy['successful']:
                for pattern in patterns:
                    successful_patterns[pattern] += 1
            else:
                for pattern in patterns:
                    failed_patterns[pattern] += 1
        
        # Create bias towards successful patterns
        self.generation_biases = {}
        for pattern, count in successful_patterns.items():
            success_rate = count / (count + failed_patterns.get(pattern, 0))
            if success_rate > 0.6:
                self.generation_biases[pattern] = success_rate
        
        logger.info(f"Learned {len(self.generation_biases)} successful patterns")
    
    def _extract_patterns(self, strategy: Dict) -> List[str]:
        """Extract patterns from a strategy"""
        patterns = []
        
        # Type patterns
        patterns.append(f"type:{strategy.get('type', 'unknown')}")
        
        # Parameter range patterns
        if strategy.get('leverage', 0) > 2:
            patterns.append("high_leverage")
        if strategy.get('stop_loss', 1) < 0.1:
            patterns.append("tight_stops")
        
        # Indicator patterns
        indicators = strategy.get('indicators', [])
        for ind in indicators:
            patterns.append(f"uses:{ind}")
        
        # Combination patterns
        if 'momentum' in strategy.get('type', '') and 'RSI' in indicators:
            patterns.append("momentum_rsi_combo")
        
        return patterns
    
    def generate_report(self) -> Dict:
        """Generate self-improvement report"""
        recent_metrics = self.evolution_history[-10:] if len(self.evolution_history) >= 10 else self.evolution_history
        
        report = {
            'generation': self.generation,
            'current_parameters': {
                'mutation_rate': self.mutation_params.rate,
                'crossover_rate': self.crossover_rate,
                'mutation_weights': self.mutation_params.type_weights,
                'selection_params': self.selection_params,
                'fitness_weights': self.fitness_function_weights
            },
            'performance_trend': {
                'fitness_improvement': self._calculate_improvement_rate(recent_metrics),
                'diversity_trend': np.mean([m.diversity_score for m in recent_metrics]) if recent_metrics else 0,
                'convergence_rate': recent_metrics[-1].convergence_rate if recent_metrics else 0
            },
            'meta_learning': {
                'parameters_evolved': self.generation > 0,
                'successful_patterns': len(getattr(self, 'generation_biases', {})),
                'adaptation_speed': self.meta_learner.meta_parameters['adaptation_speed']
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_improvement_rate(self, metrics: List[EvolutionMetrics]) -> float:
        """Calculate rate of improvement"""
        if len(metrics) < 2:
            return 0.0
        
        early = np.mean([m.best_fitness for m in metrics[:len(metrics)//2]])
        late = np.mean([m.best_fitness for m in metrics[len(metrics)//2:]])
        
        return (late - early) / (early + 1e-6)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if hasattr(self, 'evolution_history') and self.evolution_history:
            recent = self.evolution_history[-5:]
            
            # Check stagnation
            if all(m.best_fitness < 0.7 for m in recent):
                recommendations.append("Consider more aggressive mutations or parameter ranges")
            
            # Check diversity
            if any(m.diversity_score < 0.3 for m in recent):
                recommendations.append("Inject random strategies to increase diversity")
            
            # Check mutation success
            if any(m.mutation_success_rate < 0.1 for m in recent):
                recommendations.append("Mutation strategy may be too aggressive")
        
        return recommendations


class AdaptiveFitnessFunction:
    """
    Fitness function that evolves based on market conditions and success patterns
    """
    
    def __init__(self):
        self.base_weights = {
            'return': 0.33,
            'sharpe': 0.33,
            'drawdown': 0.34
        }
        self.regime_adjustments = {
            'bull': {'return': 1.2, 'sharpe': 0.9, 'drawdown': 0.9},
            'bear': {'return': 0.8, 'sharpe': 1.0, 'drawdown': 1.3},
            'sideways': {'return': 0.9, 'sharpe': 1.2, 'drawdown': 1.0},
            'high_volatility': {'return': 0.8, 'sharpe': 1.3, 'drawdown': 1.2}
        }
        self.success_history = deque(maxlen=100)
        
    def calculate_fitness(self, strategy_metrics: Dict, market_regime: str = 'neutral') -> float:
        """Calculate adaptive fitness score"""
        # Get base weights
        weights = self.base_weights.copy()
        
        # Apply regime adjustments
        if market_regime in self.regime_adjustments:
            for metric, adjustment in self.regime_adjustments[market_regime].items():
                weights[metric] *= adjustment
        
        # Normalize weights
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
        
        # Calculate fitness
        fitness = 0
        fitness += weights['return'] * self._score_return(strategy_metrics.get('cagr', 0))
        fitness += weights['sharpe'] * self._score_sharpe(strategy_metrics.get('sharpe_ratio', 0))
        fitness += weights['drawdown'] * self._score_drawdown(strategy_metrics.get('max_drawdown', 1))
        
        return fitness
    
    def _score_return(self, cagr: float) -> float:
        """Score return metric with adaptive targets"""
        # Adaptive targeting based on recent success
        if self.success_history:
            recent_returns = [s['cagr'] for s in list(self.success_history)[-20:]]
            adaptive_target = np.percentile(recent_returns, 75)
        else:
            adaptive_target = 0.20  # Default target
        
        if cagr >= adaptive_target:
            return 1.0
        elif cagr >= adaptive_target * 0.8:
            return 0.8 + 0.2 * (cagr - adaptive_target * 0.8) / (adaptive_target * 0.2)
        else:
            return max(0, cagr / adaptive_target)
    
    def _score_sharpe(self, sharpe: float) -> float:
        """Score Sharpe ratio"""
        if sharpe >= 1.0:
            return 1.0
        elif sharpe >= 0.5:
            return 0.5 + 0.5 * (sharpe - 0.5) / 0.5
        else:
            return max(0, sharpe)
    
    def _score_drawdown(self, drawdown: float) -> float:
        """Score drawdown (lower is better)"""
        if drawdown <= 0.15:
            return 1.0
        elif drawdown <= 0.25:
            return 0.5 + 0.5 * (0.25 - drawdown) / 0.10
        else:
            return max(0, 1 - drawdown)
    
    def update_success_history(self, strategy_metrics: Dict):
        """Update history with successful strategies"""
        self.success_history.append(strategy_metrics)
        
        # Periodically adjust base weights based on what works
        if len(self.success_history) % 20 == 0:
            self._adjust_base_weights()
    
    def _adjust_base_weights(self):
        """Adjust base weights based on success patterns"""
        if len(self.success_history) < 20:
            return
        
        recent = list(self.success_history)[-20:]
        
        # Analyze which metrics correlate with success
        # (Simplified - in practice would use more sophisticated analysis)
        avg_metrics = {
            'return': np.mean([s['cagr'] for s in recent]),
            'sharpe': np.mean([s['sharpe_ratio'] for s in recent]),
            'drawdown': np.mean([s['max_drawdown'] for s in recent])
        }
        
        # Slightly increase weight on metrics that are consistently good
        if avg_metrics['sharpe'] > 1.2:
            self.base_weights['sharpe'] *= 1.05
        if avg_metrics['return'] > 0.25:
            self.base_weights['return'] *= 1.05
            
        # Normalize
        total = sum(self.base_weights.values())
        for key in self.base_weights:
            self.base_weights[key] /= total


def create_self_improving_system():
    """Create and configure self-improving evolution system"""
    system = SelfImprovingEvolution(initial_population_size=50)
    
    # Configure initial parameters
    system.mutation_params = MutationParameters(
        rate=0.3,
        strength=0.15,
        type_weights={
            'parameter_tweak': 0.4,
            'indicator_change': 0.35,
            'strategy_shift': 0.15,
            'risk_adjustment': 0.1
        }
    )
    
    # Create adaptive fitness function
    system.fitness_function = AdaptiveFitnessFunction()
    
    logger.info("Created self-improving evolution system")
    logger.info(f"Initial mutation rate: {system.mutation_params.rate}")
    logger.info(f"Initial crossover rate: {system.crossover_rate}")
    
    return system


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Create self-improving system
    system = create_self_improving_system()
    
    # Simulate evolution with self-improvement
    logger.info("\n=== Self-Improving Evolution Demo ===")
    
    for generation in range(10):
        logger.info(f"\nGeneration {generation + 1}")
        
        # Simulate evolution metrics
        metrics = EvolutionMetrics(
            generation=generation,
            best_fitness=0.5 + generation * 0.05 + np.random.normal(0, 0.02),
            avg_fitness=0.3 + generation * 0.03,
            diversity_score=0.7 - generation * 0.03,
            mutation_success_rate=0.2 + np.random.normal(0, 0.05),
            crossover_success_rate=0.3 + np.random.normal(0, 0.05),
            convergence_rate=min(0.9, generation * 0.1),
            time_to_improvement=max(1, 5 - generation * 0.3)
        )
        
        system.evolution_history.append(metrics)
        
        # Every 3 generations, evolve parameters
        if generation % 3 == 0 and generation > 0:
            system.evolve_parameters()
            system.meta_learner.update_meta_parameters(system.evolution_history[-5:])
        
        # Adapt mutation strategy
        system.adapt_mutation_strategy(metrics.mutation_success_rate)
        
        # Show current state
        logger.info(f"  Best fitness: {metrics.best_fitness:.3f}")
        logger.info(f"  Diversity: {metrics.diversity_score:.3f}")
        logger.info(f"  Mutation rate: {system.mutation_params.rate:.3f}")
        logger.info(f"  Exploration bonus: {system.meta_learner.meta_parameters['exploration_bonus']:.3f}")
    
    # Generate report
    report = system.generate_report()
    logger.info("\n=== Self-Improvement Report ===")
    logger.info(f"Performance improvement: {report['performance_trend']['fitness_improvement']:.1%}")
    logger.info(f"Current diversity: {report['performance_trend']['diversity_trend']:.3f}")
    logger.info(f"Evolved parameters: {report['current_parameters']}")
    
    if report['recommendations']:
        logger.info("\nRecommendations:")
        for rec in report['recommendations']:
            logger.info(f"  - {rec}")