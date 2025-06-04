"""
Adaptive Trading System
Integrates self-improving evolution with regime awareness and meta-learning
"""

import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import time

# Import all our advanced components
from self_improving_evolution import SelfImprovingEvolution, EvolutionMetrics, AdaptiveFitnessFunction
from regime_aware_evolution import RegimeAwareEvolution, RegimeStrategy
from meta_learning_optimizer import MetaLearningOptimizer
from market_regime_detector import MarketRegimeDetector, MarketRegime
from enhanced_backtester import EnhancedBacktester
from performance_attribution_dashboard import PerformanceAttributionDashboard

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Current state of the adaptive system"""
    current_regime: MarketRegime
    regime_confidence: float
    best_strategies: List[Dict]
    generation: int
    success_rate: float
    avg_fitness: float
    learning_progress: float
    adaptation_rate: float
    computational_efficiency: float


class AdaptiveTradingSystem:
    """
    Complete adaptive trading system that:
    1. Self-improves its evolution parameters
    2. Adapts to market regimes automatically
    3. Uses meta-learning to generate better strategies faster
    4. Maintains real-time adaptation capabilities
    """
    
    def __init__(self, initial_population_size: int = 60):
        # Core components
        self.regime_detector = MarketRegimeDetector()
        self.regime_evolution = RegimeAwareEvolution()
        self.self_improving = SelfImprovingEvolution(initial_population_size)
        self.meta_learner = MetaLearningOptimizer()
        self.backtester = EnhancedBacktester(force_cloud=True)
        self.dashboard = PerformanceAttributionDashboard()
        
        # Adaptive fitness function
        self.adaptive_fitness = AdaptiveFitnessFunction()
        
        # System state
        self.current_state = SystemState(
            current_regime=MarketRegime.SIDEWAYS,
            regime_confidence=0.5,
            best_strategies=[],
            generation=0,
            success_rate=0.0,
            avg_fitness=0.0,
            learning_progress=0.0,
            adaptation_rate=0.0,
            computational_efficiency=0.0
        )
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_history = []
        self.computational_metrics = []
        
        # System parameters (self-tuning)
        self.system_params = {
            'regime_switch_threshold': 0.7,
            'learning_rate': 0.01,
            'adaptation_speed': 0.1,
            'exploration_rate': 0.2,
            'meta_learning_frequency': 10,
            'regime_evolution_frequency': 5
        }
        
        logger.info("🧠 Adaptive Trading System initialized")
        logger.info("   ✓ Self-improving evolution")
        logger.info("   ✓ Regime-aware populations")
        logger.info("   ✓ Meta-learning optimization")
        logger.info("   ✓ Real-time adaptation")
    
    def run_adaptive_evolution(self, max_generations: int = 100, 
                             target_strategies: int = 10) -> Dict:
        """
        Run the complete adaptive evolution system
        """
        logger.info(f"🚀 Starting adaptive evolution for {max_generations} generations")
        logger.info(f"Target: Find {target_strategies} successful strategies")
        
        start_time = time.time()
        successful_strategies = []
        
        for generation in range(max_generations):
            gen_start_time = time.time()
            logger.info(f"\n--- Generation {generation + 1} ---")
            
            # 1. Detect current market regime
            market_data = self._get_market_data()
            old_regime = self.current_state.current_regime
            new_regime = self.regime_evolution.detect_and_adapt(market_data)
            self.current_state.current_regime = new_regime
            
            if old_regime != new_regime:
                logger.info(f"📊 Regime change: {old_regime.value} → {new_regime.value}")
                self._handle_regime_transition(old_regime, new_regime)
            
            # 2. Generate strategies using all systems
            generation_strategies = self._generate_adaptive_strategies(
                generation=generation,
                regime=new_regime,
                num_strategies=20
            )
            
            # 3. Evaluate strategies with comprehensive testing
            evaluation_results = self._evaluate_strategies_comprehensive(
                generation_strategies, market_data
            )
            
            # 4. Update all learning systems
            self._update_learning_systems(generation_strategies, evaluation_results)
            
            # 5. Extract successful strategies
            gen_successful = [
                result for result in evaluation_results 
                if self._meets_success_criteria(result)
            ]
            successful_strategies.extend(gen_successful)
            
            # 6. Self-improve system parameters
            if generation % self.system_params['meta_learning_frequency'] == 0:
                self._self_improve_system()
            
            # 7. Update system state
            self._update_system_state(generation, evaluation_results, 
                                    time.time() - gen_start_time)
            
            # 8. Early termination if target reached
            if len(successful_strategies) >= target_strategies:
                logger.info(f"🎯 Target reached! Found {len(successful_strategies)} successful strategies")
                break
            
            # 9. Progress report
            self._log_generation_progress(generation, evaluation_results, 
                                        len(successful_strategies))
        
        total_time = time.time() - start_time
        
        # Generate final comprehensive report
        final_report = self._generate_final_report(
            successful_strategies, total_time, generation + 1
        )
        
        return final_report
    
    def _generate_adaptive_strategies(self, generation: int, regime: MarketRegime, 
                                    num_strategies: int) -> List[Dict]:
        """Generate strategies using all adaptive systems"""
        strategies = []
        
        # 1. Meta-learning optimized strategies (40%)
        meta_count = int(num_strategies * 0.4)
        for _ in range(meta_count):
            if self.meta_learner.models_trained:
                strategy = self.meta_learner.generate_optimized_strategy(
                    target_regime=regime.value
                )
                strategy['generation_method'] = 'meta_learned'
            else:
                strategy = self._generate_basic_strategy()
                strategy['generation_method'] = 'random'
            strategies.append(strategy)
        
        # 2. Regime-specific evolved strategies (30%)
        regime_count = int(num_strategies * 0.3)
        regime_portfolio = self.regime_evolution.get_adaptive_portfolio()
        for i in range(min(regime_count, len(regime_portfolio))):
            strategy = regime_portfolio[i].base_strategy.copy()
            strategy['generation_method'] = 'regime_evolved'
            strategies.append(strategy)
        
        # 3. Self-improving evolution strategies (20%)
        self_improve_count = int(num_strategies * 0.2)
        for _ in range(self_improve_count):
            strategy = self._generate_self_improved_strategy()
            strategy['generation_method'] = 'self_improved'
            strategies.append(strategy)
        
        # 4. Hybrid and crossover strategies (10%)
        hybrid_count = num_strategies - len(strategies)
        for _ in range(hybrid_count):
            if len(strategies) >= 2:
                strategy = self._create_hybrid_strategy(
                    np.random.choice(strategies, 2, replace=False)
                )
                strategy['generation_method'] = 'hybrid'
            else:
                strategy = self._generate_basic_strategy()
                strategy['generation_method'] = 'random'
            strategies.append(strategy)
        
        logger.info(f"Generated {len(strategies)} adaptive strategies")
        return strategies
    
    def _evaluate_strategies_comprehensive(self, strategies: List[Dict], 
                                         market_data: Dict) -> List[Dict]:
        """Evaluate strategies with comprehensive testing"""
        results = []
        
        for i, strategy in enumerate(strategies):
            eval_start = time.time()
            
            # 1. Meta-learning pre-screening
            if self.meta_learner.models_trained:
                should_skip, reason = self.meta_learner.should_skip_strategy(strategy)
                if should_skip:
                    logger.info(f"Skipping strategy {i+1}: {reason}")
                    continue
            
            # 2. Backtesting with walk-forward validation
            try:
                backtest_results = self.backtester.backtest_strategy(
                    strategy, use_walk_forward=True
                )
                
                if "error" in backtest_results:
                    logger.warning(f"Strategy {i+1} backtest failed: {backtest_results['error']}")
                    continue
                
                # 3. Performance attribution analysis
                trade_data = self._extract_trade_data(backtest_results)
                attribution = self.dashboard.analyze_strategy_performance(
                    backtest_results, trade_data, market_data
                )
                
                # 4. Adaptive fitness calculation
                fitness_score = self.adaptive_fitness.calculate_fitness(
                    backtest_results, market_regime=self.current_state.current_regime.value
                )
                
                eval_time = time.time() - eval_start
                
                result = {
                    'strategy': strategy,
                    'backtest_results': backtest_results,
                    'attribution': attribution,
                    'fitness_score': fitness_score,
                    'evaluation_time': eval_time,
                    'success': self._meets_success_criteria(backtest_results)
                }
                
                results.append(result)
                
                # Record for meta-learning
                self.meta_learner.record_generation_outcome(
                    strategy=strategy,
                    success=result['success'],
                    fitness_score=fitness_score,
                    time_to_evaluate=eval_time,
                    generation_method=strategy.get('generation_method', 'unknown')
                )
                
            except Exception as e:
                logger.error(f"Error evaluating strategy {i+1}: {e}")
                continue
        
        logger.info(f"Evaluated {len(results)} strategies successfully")
        return results
    
    def _update_learning_systems(self, strategies: List[Dict], 
                               evaluation_results: List[Dict]):
        """Update all learning systems with new data"""
        
        # 1. Update adaptive fitness function
        successful_results = [r for r in evaluation_results if r['success']]
        for result in successful_results:
            self.adaptive_fitness.update_success_history(result['backtest_results'])
        
        # 2. Update self-improving evolution
        fitness_scores = [r['fitness_score'] for r in evaluation_results]
        if fitness_scores:
            avg_fitness = np.mean(fitness_scores)
            best_fitness = np.max(fitness_scores)
            diversity = self._calculate_strategy_diversity(strategies)
            
            metrics = EvolutionMetrics(
                generation=self.current_state.generation,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                diversity_score=diversity,
                mutation_success_rate=len(successful_results) / len(evaluation_results),
                crossover_success_rate=0.7,  # Placeholder
                convergence_rate=max(0, 1 - diversity),
                time_to_improvement=np.mean([r['evaluation_time'] for r in evaluation_results])
            )
            
            self.self_improving.evolution_history.append(metrics)
            
            # Adapt mutation strategy
            success_rate = len(successful_results) / len(evaluation_results)
            self.self_improving.adapt_mutation_strategy(success_rate)
        
        # 3. Update regime evolution
        self.regime_evolution.evolve_current_population()
        
        # 4. Update meta-learner (already done in evaluation)
        pass
    
    def _meets_success_criteria(self, results: Dict) -> bool:
        """Check if strategy meets success criteria"""
        if isinstance(results, dict) and 'backtest_results' in results:
            results = results['backtest_results']
        
        # Use staged targets based on generation
        generation = self.current_state.generation
        
        if generation < 20:
            # Stage 1: More lenient
            return (results.get('cagr', 0) >= 0.15 and 
                    results.get('sharpe_ratio', 0) >= 0.8 and
                    results.get('max_drawdown', 1) <= 0.20)
        elif generation < 50:
            # Stage 2: Intermediate
            return (results.get('cagr', 0) >= 0.20 and 
                    results.get('sharpe_ratio', 0) >= 1.0 and
                    results.get('max_drawdown', 1) <= 0.18)
        else:
            # Stage 3: Full targets
            return (results.get('cagr', 0) >= 0.25 and 
                    results.get('sharpe_ratio', 0) >= 1.0 and
                    results.get('max_drawdown', 1) <= 0.15)
    
    def _self_improve_system(self):
        """Self-improve system parameters based on performance"""
        logger.info("🔧 Self-improving system parameters...")
        
        # Analyze recent performance
        if len(self.performance_history) >= 10:
            recent = self.performance_history[-10:]
            
            # Check if performance is improving
            early_success = np.mean([p['success_rate'] for p in recent[:5]])
            late_success = np.mean([p['success_rate'] for p in recent[5:]])
            
            if late_success > early_success:
                # Good progress - can be more aggressive
                self.system_params['exploration_rate'] *= 0.95
                self.system_params['adaptation_speed'] *= 1.05
                logger.info("   ↗️ Increasing adaptation speed")
            else:
                # Poor progress - need more exploration
                self.system_params['exploration_rate'] *= 1.05
                self.system_params['learning_rate'] *= 1.1
                logger.info("   🔍 Increasing exploration")
            
            # Keep parameters in bounds
            self.system_params['exploration_rate'] = np.clip(
                self.system_params['exploration_rate'], 0.1, 0.5
            )
            self.system_params['adaptation_speed'] = np.clip(
                self.system_params['adaptation_speed'], 0.05, 0.3
            )
        
        # Evolve evolution parameters
        self.self_improving.evolve_parameters()
        
        # Update meta-learner recommendations
        recommendations = self.meta_learner.get_generation_recommendations()
        logger.info(f"   📊 Meta-learner status: {recommendations['success_rate_trend']}")
    
    def _handle_regime_transition(self, old_regime: MarketRegime, new_regime: MarketRegime):
        """Handle transition between market regimes"""
        logger.info(f"🔄 Handling regime transition: {old_regime.value} → {new_regime.value}")
        
        # Adjust system parameters for new regime
        if new_regime in [MarketRegime.CRASH, MarketRegime.STRONG_BEAR]:
            self.system_params['exploration_rate'] *= 1.3  # More exploration in crisis
            self.system_params['adaptation_speed'] *= 1.2  # Faster adaptation
        elif new_regime == MarketRegime.STRONG_BULL:
            self.system_params['exploration_rate'] *= 0.8  # Less exploration in bull
            self.system_params['adaptation_speed'] *= 0.9  # Slower adaptation
        
        # Update fitness function for new regime
        self.adaptive_fitness.regime_adjustments[new_regime.value] = \
            self.adaptive_fitness.regime_adjustments.get(new_regime.value, {
                'return': 1.0, 'sharpe': 1.0, 'drawdown': 1.0
            })
        
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'from_regime': old_regime.value,
            'to_regime': new_regime.value,
            'adjustments_made': True
        })
    
    def _update_system_state(self, generation: int, evaluation_results: List[Dict], 
                           generation_time: float):
        """Update overall system state"""
        successful = [r for r in evaluation_results if r['success']]
        
        self.current_state.generation = generation
        self.current_state.success_rate = len(successful) / max(len(evaluation_results), 1)
        self.current_state.avg_fitness = np.mean([r['fitness_score'] for r in evaluation_results])
        
        # Learning progress (based on meta-learner improvement)
        if self.meta_learner.models_trained:
            self.current_state.learning_progress = min(1.0, len(self.meta_learner.generation_history) / 100)
        
        # Computational efficiency
        self.current_state.computational_efficiency = 1.0 / max(generation_time, 1.0)
        
        # Track performance history
        self.performance_history.append({
            'generation': generation,
            'success_rate': self.current_state.success_rate,
            'avg_fitness': self.current_state.avg_fitness,
            'time': generation_time,
            'regime': self.current_state.current_regime.value
        })
    
    def _log_generation_progress(self, generation: int, evaluation_results: List[Dict], 
                               total_successful: int):
        """Log detailed generation progress"""
        successful = [r for r in evaluation_results if r['success']]
        
        logger.info(f"Generation {generation + 1} Results:")
        logger.info(f"  📊 Evaluated: {len(evaluation_results)} strategies")
        logger.info(f"  ✅ Successful: {len(successful)} ({len(successful)/len(evaluation_results):.1%})")
        logger.info(f"  🎯 Total found: {total_successful}")
        logger.info(f"  📈 Avg fitness: {np.mean([r['fitness_score'] for r in evaluation_results]):.3f}")
        logger.info(f"  🧠 Learning progress: {self.current_state.learning_progress:.1%}")
        logger.info(f"  🏃 Efficiency: {self.current_state.computational_efficiency:.2f}")
        
        if successful:
            best = max(successful, key=lambda x: x['fitness_score'])
            logger.info(f"  🏆 Best strategy: {best['strategy']['name']} (fitness: {best['fitness_score']:.3f})")
    
    def _generate_final_report(self, successful_strategies: List[Dict], 
                             total_time: float, generations: int) -> Dict:
        """Generate comprehensive final report"""
        
        report = {
            'execution_summary': {
                'total_time_minutes': total_time / 60,
                'generations_completed': generations,
                'strategies_found': len(successful_strategies),
                'final_success_rate': self.current_state.success_rate,
                'final_regime': self.current_state.current_regime.value
            },
            'system_evolution': {
                'learning_progress': self.current_state.learning_progress,
                'parameter_adaptations': len(self.adaptation_history),
                'regime_transitions': len([h for h in self.adaptation_history if h.get('adjustments_made')]),
                'computational_efficiency_gain': self._calculate_efficiency_gain()
            },
            'successful_strategies': [],
            'performance_analytics': self._generate_performance_analytics(),
            'system_recommendations': self._generate_system_recommendations(),
            'meta_learning_insights': self.meta_learner.generate_learning_report()
        }
        
        # Add successful strategies
        for result in successful_strategies:
            strategy_summary = {
                'name': result['strategy']['name'],
                'type': result['strategy'].get('type', 'unknown'),
                'generation_method': result['strategy'].get('generation_method', 'unknown'),
                'metrics': {
                    'cagr': result['backtest_results'].get('cagr', 0),
                    'sharpe_ratio': result['backtest_results'].get('sharpe_ratio', 0),
                    'max_drawdown': result['backtest_results'].get('max_drawdown', 1),
                    'fitness_score': result['fitness_score']
                },
                'robustness_rating': result['backtest_results'].get('robustness_rating', 'Unknown')
            }
            report['successful_strategies'].append(strategy_summary)
        
        return report
    
    def _get_market_data(self) -> Dict:
        """Get current market data for regime detection"""
        # In production, this would fetch real market data
        return {
            'spy_price': 450 + np.random.normal(0, 10),
            'spy_sma_50': 445 + np.random.normal(0, 5),
            'spy_sma_200': 440 + np.random.normal(0, 3),
            'vix': 18 + np.random.normal(0, 5),
            'vix_sma_20': 20 + np.random.normal(0, 2),
            'momentum_20d': np.random.normal(0.02, 0.05),
            'advance_decline_ratio': 1.2 + np.random.normal(0, 0.3),
            'new_highs_lows_ratio': 1.8 + np.random.normal(0, 0.5),
            'percent_above_200ma': 62 + np.random.normal(0, 10),
            'put_call_ratio': 0.95 + np.random.normal(0, 0.2),
            'fear_greed_index': 58 + np.random.normal(0, 15)
        }
    
    def _generate_basic_strategy(self) -> Dict:
        """Generate basic strategy as fallback"""
        return {
            'name': f'Basic_{datetime.now().strftime("%H%M%S")}',
            'type': np.random.choice(['momentum', 'mean_reversion', 'trend_following']),
            'leverage': np.random.uniform(1.0, 2.5),
            'position_size': np.random.uniform(0.1, 0.25),
            'stop_loss': np.random.uniform(0.05, 0.15),
            'indicators': ['RSI', 'MACD', 'BB']
        }
    
    def _generate_self_improved_strategy(self) -> Dict:
        """Generate strategy using self-improving parameters"""
        strategy = self._generate_basic_strategy()
        
        # Apply self-improving biases
        if hasattr(self.self_improving, 'generation_biases'):
            for pattern, bias in self.self_improving.generation_biases.items():
                if 'high_leverage' in pattern and bias > 0.6:
                    strategy['leverage'] *= 1.2
                elif 'tight_stops' in pattern and bias > 0.6:
                    strategy['stop_loss'] *= 0.8
        
        return strategy
    
    def _create_hybrid_strategy(self, parent_strategies: List[Dict]) -> Dict:
        """Create hybrid strategy from parents"""
        parent1, parent2 = parent_strategies
        
        hybrid = {
            'name': f'Hybrid_{datetime.now().strftime("%H%M%S")}',
            'type': 'hybrid',
            'leverage': (parent1.get('leverage', 1) + parent2.get('leverage', 1)) / 2,
            'position_size': (parent1.get('position_size', 0.1) + parent2.get('position_size', 0.1)) / 2,
            'stop_loss': (parent1.get('stop_loss', 0.1) + parent2.get('stop_loss', 0.1)) / 2,
            'indicators': list(set(
                parent1.get('indicators', []) + parent2.get('indicators', [])
            ))[:5]
        }
        
        return hybrid
    
    def _extract_trade_data(self, backtest_results: Dict) -> List[Dict]:
        """Extract trade data for attribution analysis"""
        # Simplified - in production would extract real trade data
        num_trades = backtest_results.get('total_trades', 100)
        win_rate = backtest_results.get('win_rate', 0.5)
        
        trades = []
        for i in range(min(num_trades, 50)):
            is_win = np.random.random() < win_rate
            trades.append({
                'symbol': f'STOCK_{i}',
                'profit': np.random.normal(0.01 if is_win else -0.008, 0.005),
                'duration': np.random.randint(1, 20),
                'alpha': np.random.normal(0.002 if is_win else -0.001, 0.001)
            })
        
        return trades
    
    def _calculate_strategy_diversity(self, strategies: List[Dict]) -> float:
        """Calculate diversity of strategy population"""
        if not strategies:
            return 0.0
        
        types = [s.get('type', 'unknown') for s in strategies]
        type_diversity = len(set(types)) / len(types)
        
        leverages = [s.get('leverage', 1) for s in strategies]
        param_diversity = np.std(leverages) / (np.mean(leverages) + 1e-6)
        
        return 0.5 * type_diversity + 0.5 * min(param_diversity, 1.0)
    
    def _calculate_efficiency_gain(self) -> float:
        """Calculate computational efficiency gain over time"""
        if len(self.computational_metrics) < 2:
            return 0.0
        
        early_efficiency = np.mean(self.computational_metrics[:len(self.computational_metrics)//2])
        late_efficiency = np.mean(self.computational_metrics[len(self.computational_metrics)//2:])
        
        return (late_efficiency - early_efficiency) / max(early_efficiency, 0.01)
    
    def _generate_performance_analytics(self) -> Dict:
        """Generate detailed performance analytics"""
        if not self.performance_history:
            return {}
        
        return {
            'success_rate_trend': {
                'initial': self.performance_history[0]['success_rate'],
                'final': self.performance_history[-1]['success_rate'],
                'improvement': self.performance_history[-1]['success_rate'] - self.performance_history[0]['success_rate']
            },
            'fitness_progression': {
                'max_fitness': max(p['avg_fitness'] for p in self.performance_history),
                'final_fitness': self.performance_history[-1]['avg_fitness'],
                'consistency': np.std([p['avg_fitness'] for p in self.performance_history])
            },
            'regime_adaptations': len(self.adaptation_history)
        }
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate recommendations for system improvement"""
        recommendations = []
        
        if self.current_state.success_rate < 0.1:
            recommendations.append("Consider relaxing success criteria or adjusting target metrics")
        
        if self.current_state.learning_progress < 0.5:
            recommendations.append("System needs more training data - continue running")
        
        if len(self.adaptation_history) > 5:
            recommendations.append("High regime volatility detected - consider longer evaluation periods")
        
        if self.current_state.computational_efficiency < 0.5:
            recommendations.append("Implement parallel processing or early stopping for better efficiency")
        
        return recommendations


def main():
    """Main execution function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🧠 Initializing Adaptive Trading System")
    
    # Create adaptive system
    system = AdaptiveTradingSystem(initial_population_size=50)
    
    # Run adaptive evolution
    results = system.run_adaptive_evolution(
        max_generations=30,
        target_strategies=5
    )
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("🏁 ADAPTIVE EVOLUTION COMPLETE")
    logger.info("="*80)
    
    summary = results['execution_summary']
    logger.info(f"\nExecution Summary:")
    logger.info(f"  Time: {summary['total_time_minutes']:.1f} minutes")
    logger.info(f"  Generations: {summary['generations_completed']}")
    logger.info(f"  Success rate: {summary['final_success_rate']:.1%}")
    logger.info(f"  Strategies found: {summary['strategies_found']}")
    logger.info(f"  Final regime: {summary['final_regime']}")
    
    if results['successful_strategies']:
        logger.info(f"\nSuccessful Strategies:")
        for strategy in results['successful_strategies']:
            logger.info(f"  • {strategy['name']} ({strategy['type']})")
            logger.info(f"    CAGR: {strategy['metrics']['cagr']:.1%}, "
                       f"Sharpe: {strategy['metrics']['sharpe_ratio']:.2f}, "
                       f"Method: {strategy['generation_method']}")
    
    evolution = results['system_evolution']
    logger.info(f"\nSystem Evolution:")
    logger.info(f"  Learning progress: {evolution['learning_progress']:.1%}")
    logger.info(f"  Regime transitions: {evolution['regime_transitions']}")
    logger.info(f"  Efficiency gain: {evolution['computational_efficiency_gain']:.1%}")
    
    if results['system_recommendations']:
        logger.info(f"\nRecommendations:")
        for rec in results['system_recommendations']:
            logger.info(f"  • {rec}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"adaptive_evolution_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    main()