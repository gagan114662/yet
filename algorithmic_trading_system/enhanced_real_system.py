#!/usr/bin/env python3
"""
Enhanced Real Algorithmic Trading System
Replaces mock backtesting with real market data while debugging QuantConnect API
"""

import asyncio
import time
import logging
from typing import Dict, List
from datetime import datetime

# Import our systems
from integrated_dgm_claude_system import IntegratedDGMSystem
from hybrid_real_backtesting import hybrid_real_backtest
from staged_targets_system import TargetStage

logger = logging.getLogger(__name__)

class EnhancedRealTradingSystem:
    """
    Production-ready system using real market data for backtesting
    """
    
    def __init__(self):
        self.config = {
            'max_parallel_backtests': 8,  # Increase parallel processing
            'enable_real_data_backtesting': True,
            'enable_champion_breeding': True,
            'enable_micro_mutations': True,
            'target_cagr': 0.25,  # 25% CAGR target
            'enable_stage_progression': True
        }
        
        # Initialize integrated system
        self.system = IntegratedDGMSystem(self.config)
        
        # Override mock backtesting with real data
        self.system.streaming_dgm.simulate_detailed_backtest = self._real_backtest_wrapper
        
        # Set to Stage 2 for 20% targets initially
        self.system.staged_targets.current_stage = TargetStage.STAGE_2
        
        self.real_backtest_results = []
        self.evolution_history = []
        
        logger.info("🚀 Enhanced Real Trading System initialized with real market data")
    
    async def _real_backtest_wrapper(self, strategy: Dict, target_performance: float = None) -> Dict:
        """Wrapper to use hybrid real backtesting instead of mock"""
        try:
            # Add target performance hints to strategy
            if target_performance:
                strategy['target_performance_hint'] = target_performance
            
            # Use real market data backtesting
            result = await hybrid_real_backtest(strategy)
            
            # Store results for analysis
            if 'error' not in result:
                self.real_backtest_results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Real backtest wrapper failed: {e}")
            return {
                'error': f'Real backtest failed: {e}',
                'strategy_id': strategy.get('id', 'unknown')
            }
    
    def create_high_potential_strategies(self) -> List[Dict]:
        """Create strategies with high potential for 25% CAGR"""
        strategies = []
        
        # High-performance momentum strategies
        momentum_configs = [
            {'leverage': 2.5, 'position_size': 0.25, 'stop_loss': 0.08, 'rsi_period': 10},
            {'leverage': 2.8, 'position_size': 0.22, 'stop_loss': 0.07, 'rsi_period': 12},
            {'leverage': 3.0, 'position_size': 0.20, 'stop_loss': 0.06, 'rsi_period': 8}
        ]
        
        for i, config in enumerate(momentum_configs):
            strategies.append({
                'id': f'momentum_aggressive_{i+1}',
                'name': f'Aggressive Momentum Strategy {i+1}',
                'type': 'momentum',
                'symbol': 'SPY',
                'generation': 0,
                'creation_method': 'high_potential_seed',
                **config
            })
        
        # Mean reversion with higher frequency
        mean_reversion_configs = [
            {'leverage': 2.2, 'position_size': 0.18, 'stop_loss': 0.09, 'bb_period': 15},
            {'leverage': 2.0, 'position_size': 0.20, 'stop_loss': 0.08, 'bb_period': 18},
        ]
        
        for i, config in enumerate(mean_reversion_configs):
            strategies.append({
                'id': f'mean_reversion_tuned_{i+1}',
                'name': f'Tuned Mean Reversion {i+1}',
                'type': 'mean_reversion',
                'symbol': 'SPY',
                'generation': 0,
                'creation_method': 'high_potential_seed',
                **config
            })
        
        # Breakout strategies
        breakout_configs = [
            {'leverage': 2.6, 'position_size': 0.24, 'stop_loss': 0.07, 'atr_period': 12},
            {'leverage': 2.4, 'position_size': 0.26, 'stop_loss': 0.08, 'atr_period': 10}
        ]
        
        for i, config in enumerate(breakout_configs):
            strategies.append({
                'id': f'breakout_aggressive_{i+1}',
                'name': f'Aggressive Breakout {i+1}',
                'type': 'breakout',
                'symbol': 'SPY',
                'generation': 0,
                'creation_method': 'high_potential_seed',
                **config
            })
        
        logger.info(f"🌱 Created {len(strategies)} high-potential seed strategies")
        return strategies
    
    async def run_enhanced_evolution_cycle(self, generations: int = 5):
        """Run enhanced evolution with real market data"""
        print("🚀 ENHANCED REAL EVOLUTION SYSTEM")
        print("=" * 80)
        print(f"🎯 Target: {self.config['target_cagr']:.1%} CAGR")
        print(f"📊 Using: Real market data backtesting")
        print(f"🧬 Generations: {generations}")
        print(f"⚡ Parallel workers: {self.config['max_parallel_backtests']}")
        print()
        
        # Create initial high-potential strategies
        strategies = self.create_high_potential_strategies()
        
        best_ever_cagr = 0
        best_ever_strategy = None
        champions_found = []
        
        for generation in range(generations):
            gen_start = time.time()
            print(f"\n🧬 GENERATION {generation + 1}")
            print("=" * 60)
            
            # Current stage info
            current_targets = self.system.staged_targets.get_current_targets()
            print(f"🎯 Current Stage: {self.system.staged_targets.current_stage.value}")
            print(f"   Target CAGR: {current_targets.cagr:.1%}")
            print(f"   Target Sharpe: {current_targets.sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {current_targets.max_drawdown:.1%}")
            print()
            
            # Parallel backtesting with real data
            print(f"⚡ Running {len(strategies)} parallel real-data backtests...")
            backtest_tasks = []
            
            for strategy in strategies:
                strategy['generation'] = generation
                task = asyncio.create_task(self._real_backtest_wrapper(strategy))
                backtest_tasks.append(task)
            
            # Execute all backtests in parallel
            results = await asyncio.gather(*backtest_tasks)
            
            # Analyze results
            successful_results = [r for r in results if r and 'error' not in r]
            
            if successful_results:
                # Sort by CAGR
                successful_results.sort(key=lambda x: x['cagr'], reverse=True)
                
                best_gen_cagr = successful_results[0]['cagr']
                best_gen_strategy = successful_results[0]
                
                print(f"📈 Generation {generation + 1} Results:")
                print(f"   🔢 Strategies tested: {len(strategies)}")
                print(f"   ✅ Successful backtests: {len(successful_results)}")
                print(f"   🏆 Best CAGR: {best_gen_cagr:.1%}")
                print(f"   📊 Best Sharpe: {best_gen_strategy['sharpe_ratio']:.2f}")
                print(f"   📉 Best Max DD: {best_gen_strategy['max_drawdown']:.1%}")
                
                # Check for new overall champion
                if best_gen_cagr > best_ever_cagr:
                    best_ever_cagr = best_gen_cagr
                    best_ever_strategy = best_gen_strategy
                    print(f"   🎉 NEW CHAMPION! {best_gen_cagr:.1%} CAGR")
                
                # Identify stage successes and champions
                stage_successes = []
                generation_champions = []
                
                for result in successful_results[:10]:  # Top 10
                    success, reason = self.system.staged_targets.check_strategy_success(result)
                    if success:
                        stage_successes.append(result)
                        self.system.staged_targets.record_successful_strategy(
                            {'id': result['strategy_id'], 'name': result['strategy_name']}, 
                            result
                        )
                    
                    # Champion criteria: 22%+ CAGR, 0.8+ Sharpe, <20% DD
                    if (result['cagr'] >= 0.22 and 
                        result['sharpe_ratio'] >= 0.8 and 
                        result['max_drawdown'] <= 0.20):
                        generation_champions.append(result)
                        champions_found.append(result)
                        print(f"   🏆 CHAMPION: {result['strategy_name']} - "
                              f"{result['cagr']:.1%} CAGR, {result['sharpe_ratio']:.2f} Sharpe")
                
                print(f"   🎯 Stage successes: {len(stage_successes)}")
                print(f"   🏆 Champions found: {len(generation_champions)}")
                
                # Advanced breeding for next generation
                if generation < generations - 1:
                    print(f"\n🧬 Breeding next generation...")
                    
                    # Get top performers for breeding
                    top_performers = successful_results[:5]
                    bred_strategies = []
                    
                    # Champion breeding
                    if generation_champions:
                        for champion_result in generation_champions[:2]:  # Top 2 champions
                            champion_strategy = next(
                                (s for s in strategies if s['id'] == champion_result['strategy_id']), 
                                None
                            )
                            if champion_strategy:
                                offspring = self._breed_champion_focused(champion_strategy, champion_result, 4)
                                bred_strategies.extend(offspring)
                    
                    # Elite breeding from top performers
                    for i, result in enumerate(top_performers[:3]):
                        parent_strategy = next(
                            (s for s in strategies if s['id'] == result['strategy_id']), 
                            None
                        )
                        if parent_strategy:
                            mutations = self._create_targeted_mutations(parent_strategy, result, 3)
                            bred_strategies.extend(mutations)
                    
                    # Add some new exploration strategies
                    exploration_strategies = self._create_exploration_strategies(3)
                    bred_strategies.extend(exploration_strategies)
                    
                    strategies = bred_strategies
                    print(f"   🔬 Generated {len(strategies)} strategies for next generation")
                
                # Record generation history
                generation_summary = {
                    'generation': generation,
                    'best_cagr': best_gen_cagr,
                    'best_sharpe': best_gen_strategy['sharpe_ratio'],
                    'stage_successes': len(stage_successes),
                    'champions': len(generation_champions),
                    'total_tested': len(strategies),
                    'generation_time': time.time() - gen_start
                }
                self.evolution_history.append(generation_summary)
            
            else:
                print(f"   ❌ No successful backtests in generation {generation + 1}")
        
        # Final summary
        await self._display_final_results(best_ever_strategy, champions_found)
    
    def _breed_champion_focused(self, strategy: Dict, results: Dict, count: int) -> List[Dict]:
        """Create focused mutations of champion strategies"""
        offspring = []
        
        for i in range(count):
            child = strategy.copy()
            child['id'] = f"champion_focused_{strategy['id']}_{i}_{int(time.time())}"
            child['name'] = f"Champion Focused {i+1}"
            child['creation_method'] = 'champion_focused'
            child['parent'] = strategy['id']
            
            # Focused improvements targeting 25% CAGR
            cagr_gap = max(0, 0.25 - results['cagr'])
            
            if cagr_gap > 0:
                # Careful leverage increase
                leverage_mult = min(1.15, 1 + cagr_gap * 0.8)
                child['leverage'] = min(3.5, child['leverage'] * leverage_mult)
                
                # Position size optimization
                child['position_size'] = min(0.3, child['position_size'] * 1.1)
                
                # Tighter stops for better risk-adjusted returns
                child['stop_loss'] = max(0.05, child['stop_loss'] * 0.9)
            
            offspring.append(child)
        
        return offspring
    
    def _create_targeted_mutations(self, strategy: Dict, results: Dict, count: int) -> List[Dict]:
        """Create targeted mutations based on performance gaps"""
        mutations = []
        
        for i in range(count):
            mutant = strategy.copy()
            mutant['id'] = f"mutation_{strategy['id']}_{i}_{int(time.time())}"
            mutant['name'] = f"Targeted Mutation {i+1}"
            mutant['creation_method'] = 'targeted_mutation'
            mutant['parent'] = strategy['id']
            
            # Random targeted improvements
            import random
            mutation_type = random.choice(['leverage', 'position', 'timing', 'risk'])
            
            if mutation_type == 'leverage':
                mutant['leverage'] = min(3.0, mutant['leverage'] * random.uniform(1.05, 1.2))
            elif mutation_type == 'position':
                mutant['position_size'] = min(0.3, mutant['position_size'] * random.uniform(1.1, 1.3))
            elif mutation_type == 'timing':
                if 'rsi_period' in mutant:
                    mutant['rsi_period'] = max(5, mutant['rsi_period'] + random.randint(-3, 3))
            elif mutation_type == 'risk':
                mutant['stop_loss'] = max(0.04, mutant['stop_loss'] * random.uniform(0.8, 1.1))
            
            mutations.append(mutant)
        
        return mutations
    
    def _create_exploration_strategies(self, count: int) -> List[Dict]:
        """Create new exploration strategies"""
        import random
        strategies = []
        
        strategy_types = ['momentum', 'mean_reversion', 'breakout', 'trend_following']
        
        for i in range(count):
            strategy_type = random.choice(strategy_types)
            
            strategy = {
                'id': f'exploration_{strategy_type}_{i}_{int(time.time())}',
                'name': f'Exploration {strategy_type.title()} {i+1}',
                'type': strategy_type,
                'symbol': 'SPY',
                'leverage': random.uniform(1.5, 3.0),
                'position_size': random.uniform(0.15, 0.3),
                'stop_loss': random.uniform(0.05, 0.12),
                'creation_method': 'exploration',
                'generation': 999  # Mark as exploration
            }
            
            strategies.append(strategy)
        
        return strategies
    
    async def _display_final_results(self, best_strategy: Dict, champions: List[Dict]):
        """Display comprehensive final results"""
        print(f"\n🏆 ENHANCED EVOLUTION RESULTS")
        print("=" * 80)
        
        if best_strategy:
            print(f"🥇 BEST STRATEGY FOUND:")
            print(f"   📈 CAGR: {best_strategy['cagr']:.1%}")
            print(f"   📊 Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")
            print(f"   📉 Max Drawdown: {best_strategy['max_drawdown']:.1%}")
            print(f"   🎯 Win Rate: {best_strategy['win_rate']:.1%}")
            print(f"   🔢 Total Trades: {best_strategy['total_trades']}")
            print(f"   💰 Final Value: ${best_strategy['final_portfolio_value']:,.0f}")
            print(f"   📊 Data Source: {best_strategy['data_source']}")
            
            target_gap = 0.25 - best_strategy['cagr']
            gap_percentage = (target_gap / 0.25) * 100
            
            print(f"\n🎯 TARGET ANALYSIS:")
            print(f"   Gap to 25% target: {target_gap:.1%} ({gap_percentage:.1f}%)")
            
            if gap_percentage <= 0:
                print("   🎉 TARGET ACHIEVED! 25% CAGR REACHED!")
            elif gap_percentage <= 5:
                print("   🔥 EXTREMELY CLOSE! Within 5% of target!")
            elif gap_percentage <= 15:
                print("   ⚡ VERY CLOSE! Within 15% of target!")
            else:
                print("   🔧 MORE EVOLUTION NEEDED")
        
        if champions:
            print(f"\n🏆 CHAMPIONS DISCOVERED: {len(champions)}")
            for i, champion in enumerate(champions[:5]):  # Top 5 champions
                print(f"   {i+1}. {champion['strategy_name']}: "
                      f"{champion['cagr']:.1%} CAGR, {champion['sharpe_ratio']:.2f} Sharpe")
        
        print(f"\n📊 EVOLUTION METRICS:")
        print(f"   🧬 Total generations: {len(self.evolution_history)}")
        print(f"   📈 Real backtests completed: {len(self.real_backtest_results)}")
        print(f"   🏆 Champions found: {len(champions)}")
        
        if self.evolution_history:
            total_time = sum(g['generation_time'] for g in self.evolution_history)
            print(f"   ⏱️  Total evolution time: {total_time:.1f}s")
            print(f"   📈 Performance improvement: "
                  f"{self.evolution_history[-1]['best_cagr'] - self.evolution_history[0]['best_cagr']:+.1%}")
        
        print(f"\n✅ Enhanced real trading system evolution complete!")

async def main():
    """Main execution function"""
    system = EnhancedRealTradingSystem()
    
    try:
        await system.run_enhanced_evolution_cycle(generations=4)
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Evolution stopped by user")
    except Exception as e:
        print(f"\n❌ Evolution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("🚀 ENHANCED REAL ALGORITHMIC TRADING SYSTEM")
    print("=" * 80)
    print("Real market data backtesting with advanced evolution")
    print("Targeting 25% CAGR with real performance validation")
    print()
    
    asyncio.run(main())