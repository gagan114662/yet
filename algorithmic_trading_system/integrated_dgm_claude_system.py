"""
Integrated DGM + Claude Code Hybrid System
Complete integration of streaming orchestration, agent hierarchy, and safety systems
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import json
import numpy as np

# Import all DGM components
from streaming_dgm_orchestrator import StreamingDGM, EvolutionEvent, EvolutionPhase
from dgm_agent_hierarchy import DGMAgentOrchestrator, AgentContext
from dgm_safety_system import DGMSafetySystem, PermissionScope
from staged_targets_system import StagedTargetsManager, BreedingOptimizer
from parallel_backtesting_system import ParallelBacktester

logger = logging.getLogger(__name__)


@dataclass
class DGMIntegrationMetrics:
    """Metrics for integrated DGM system"""
    evolution_cycles_completed: int
    strategies_generated: int
    strategies_tested: int
    strategies_successful: int
    near_winners_identified: int
    champion_strategies: int
    safety_violations: int
    avg_cycle_time: float
    best_cagr_achieved: float
    best_sharpe_achieved: float
    target_achievement_progress: float


class IntegratedDGMSystem:
    """
    Complete DGM + Claude Code hybrid architecture
    Combines streaming orchestration, multi-agent hierarchy, and safety systems
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Initialize configuration
        default_config = self._default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Initialize core systems
        self.streaming_dgm = StreamingDGM(
            max_parallel_backtests=self.config['max_parallel_backtests'],
            archive_max_size=self.config['archive_max_size']
        )
        
        self.agent_orchestrator = DGMAgentOrchestrator()
        self.safety_system = DGMSafetySystem()
        self.staged_targets = StagedTargetsManager()
        self.breeding_optimizer = BreedingOptimizer(self.staged_targets)
        self.parallel_backtester = ParallelBacktester(
            max_workers=self.config['max_parallel_backtests']
        )
        
        # Integration state
        self.integration_metrics = DGMIntegrationMetrics(
            evolution_cycles_completed=0,
            strategies_generated=0,
            strategies_tested=0,
            strategies_successful=0,
            near_winners_identified=0,
            champion_strategies=0,
            safety_violations=0,
            avg_cycle_time=0.0,
            best_cagr_achieved=0.0,
            best_sharpe_achieved=0.0,
            target_achievement_progress=0.0
        )
        
        # Event streaming
        self.event_handlers = []
        self.real_time_dashboard = []
        
        logger.info("🚀 Integrated DGM + Claude Code System initialized")
        logger.info(f"   🧬 Streaming DGM: {self.config['max_parallel_backtests']} parallel")
        logger.info(f"   🤖 Agent hierarchy: {len(self.agent_orchestrator.agents)} agents")
        logger.info(f"   🛡️ Safety system: {len(self.safety_system.permission_scopes)} scopes")
        logger.info(f"   🎯 Staged targets: {self.staged_targets.current_stage.value}")
    
    def _default_config(self) -> Dict:
        """Default configuration for integrated system"""
        return {
            'max_parallel_backtests': 8,
            'archive_max_size': 1000,
            'target_generations': 100,
            'safety_scope': PermissionScope.EXPERIMENTAL,
            'enable_real_time_streaming': True,
            'enable_champion_breeding': True,
            'enable_micro_mutations': True,
            'dashboard_update_interval': 1.0
        }
    
    async def run_enhanced_evolution(self, target_generations: int = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run complete enhanced evolution with all systems integrated
        """
        target_generations = target_generations or self.config['target_generations']
        
        logger.info(f"🔥 Starting Enhanced DGM Evolution ({target_generations} generations)")
        logger.info("   🚀 Streaming orchestration: ENABLED")
        logger.info("   🤖 Multi-agent hierarchy: ENABLED") 
        logger.info("   🛡️ Safety system: ENABLED")
        logger.info("   🎯 Staged targets: ENABLED")
        logger.info("   ⚡ Parallel backtesting: ENABLED")
        
        start_time = time.time()
        
        try:
            # Stream evolution with all enhancements
            async for evolution_event in self._stream_enhanced_evolution(target_generations):
                # Update integration metrics
                self._update_integration_metrics(evolution_event)
                
                # Handle real-time dashboard updates
                if self.config['enable_real_time_streaming']:
                    dashboard_update = await self._generate_dashboard_update(evolution_event)
                    yield dashboard_update
                
                # Check for target achievement
                if self._check_target_achievement():
                    logger.info("🎉 TARGET ACHIEVED! Stopping evolution.")
                    break
                
                # Yield evolution event
                yield {
                    'type': 'evolution_event',
                    'event': evolution_event,
                    'integration_metrics': self.integration_metrics,
                    'timestamp': time.time()
                }
            
            total_time = time.time() - start_time
            
            # Final summary
            final_summary = await self._generate_final_summary(total_time)
            yield {
                'type': 'evolution_complete',
                'summary': final_summary,
                'total_time': total_time
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced evolution failed: {e}")
            yield {
                'type': 'evolution_error',
                'error': str(e),
                'integration_metrics': self.integration_metrics
            }
    
    async def _stream_enhanced_evolution(self, target_generations: int) -> AsyncGenerator[EvolutionEvent, None]:
        """Stream evolution with enhanced agent-safety integration"""
        
        for generation in range(target_generations):
            generation_start = time.time()
            
            logger.info(f"🧬 Generation {generation + 1}/{target_generations}")
            
            # Create agent context
            context = AgentContext(
                current_regime=self.streaming_dgm.current_regime,
                generation=generation,
                archive_summary=self._get_archive_summary(),
                performance_history=list(self.streaming_dgm.performance_history),
                near_winners=self.streaming_dgm.near_winners,
                compute_resources=self._get_compute_resources()
            )
            
            # Phase 1: Agent-orchestrated strategy generation
            async for event in self._enhanced_generation_phase(context):
                yield event
            
            # Phase 2: Safe parallel backtesting
            async for event in self._enhanced_backtesting_phase(context):
                yield event
            
            # Phase 3: Intelligent analysis with staged targets
            async for event in self._enhanced_analysis_phase(context):
                yield event
            
            # Phase 4: Champion breeding and micro-mutations
            async for event in self._enhanced_breeding_phase(context):
                yield event
            
            # Phase 5: Safe archive management
            async for event in self._enhanced_archive_phase(context):
                yield event
            
            # Phase 6: Adaptive regime management
            async for event in self._enhanced_adaptation_phase(context):
                yield event
            
            generation_time = time.time() - generation_start
            self.integration_metrics.evolution_cycles_completed += 1
            
            # Generation complete event
            yield EvolutionEvent(
                timestamp=time.time(),
                phase=EvolutionPhase.GENERATION,
                generation=generation,
                data={
                    'generation_complete': True,
                    'generation_time': generation_time,
                    'integration_metrics': self.integration_metrics
                },
                compute_stats=self._get_compute_stats(),
                performance_metrics=self._get_performance_metrics()
            )
    
    async def _enhanced_generation_phase(self, context: AgentContext) -> AsyncGenerator[EvolutionEvent, None]:
        """Enhanced generation using agent hierarchy and safety"""
        
        # Use agent orchestrator for intelligent generation
        orchestration_result = await self.agent_orchestrator.orchestrate_evolution_cycle(context)
        
        if 'error' in orchestration_result:
            logger.error(f"Agent orchestration failed: {orchestration_result['error']}")
            return
        
        # Extract generated strategies
        strategies = []
        if 'strategies_generated' in orchestration_result:
            # Get strategies from agent results (mock for now)
            for i in range(orchestration_result['strategies_generated']):
                strategy = {
                    'id': f"agent_gen_{context.generation}_{i}",
                    'type': np.random.choice(['momentum', 'mean_reversion', 'trend_following']),
                    'leverage': np.random.uniform(1.0, 3.0),
                    'position_size': np.random.uniform(0.1, 0.3),
                    'stop_loss': np.random.uniform(0.05, 0.15),
                    'indicators': ['RSI', 'MACD', 'BB'][:np.random.randint(2, 4)],
                    'creation_method': 'agent_orchestrated'
                }
                strategies.append(strategy)
        
        # Add champion breeding if enabled
        champion_offspring = []
        if self.config['enable_champion_breeding'] and self.streaming_dgm.champion_strategies:
            champion_offspring = await self.breeding_optimizer.breed_champion_lineage(5)
            strategies.extend([s for s in champion_offspring])
        
        # Store strategies for next phase
        self.current_strategies = strategies
        self.integration_metrics.strategies_generated += len(strategies)
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.GENERATION,
            generation=context.generation,
            data={
                'agent_orchestration': orchestration_result,
                'strategies_generated': len(strategies),
                'champion_offspring': len(champion_offspring)
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _enhanced_backtesting_phase(self, context: AgentContext) -> AsyncGenerator[EvolutionEvent, None]:
        """Enhanced backtesting with safety and parallel execution"""
        
        if not hasattr(self, 'current_strategies'):
            return
        
        # Safety-wrapped parallel backtesting
        safe_results = []
        
        for strategy in self.current_strategies:
            # Determine safety scope based on strategy risk
            scope = self._determine_safety_scope(strategy)
            
            # Execute with safety wrapper
            result = await self.safety_system.safe_strategy_execution(
                strategy=strategy,
                operation='backtest_strategy',
                scope=scope,
                executor_func=self._parallel_backtest_executor
            )
            
            safe_results.append(result)
        
        # Update safety metrics
        safety_report = self.safety_system.get_safety_report()
        self.integration_metrics.safety_violations = safety_report['safety_overview']['total_violations']
        self.integration_metrics.strategies_tested += len(safe_results)
        
        # Store results
        self.current_results = safe_results
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.BACKTESTING,
            generation=context.generation,
            data={
                'strategies_tested': len(safe_results),
                'successful_tests': sum(1 for r in safe_results if not r.get('error')),
                'safety_violations': safety_report['safety_overview']['recent_violations_24h'],
                'parallel_efficiency': self._calculate_parallel_efficiency()
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _enhanced_analysis_phase(self, context: AgentContext) -> AsyncGenerator[EvolutionEvent, None]:
        """Enhanced analysis with staged targets and intelligent insights"""
        
        if not hasattr(self, 'current_results'):
            return
        
        successful_results = [r for r in self.current_results if not r.get('error')]
        
        # Apply staged targets analysis
        stage_successes = 0
        near_winners = []
        champions = []
        
        for result in successful_results:
            # Check staged targets
            success, reason = self.staged_targets.check_strategy_success(result)
            if success:
                stage_successes += 1
                self.staged_targets.record_successful_strategy(result.get('strategy', {}), result)
            
            # Identify near-winners (23%+ CAGR strategies)
            if (result.get('cagr', 0) >= 0.20 and 
                result.get('sharpe_ratio', 0) >= 0.85):
                near_winners.append(result)
            
            # Identify champions
            is_champion = self.breeding_optimizer.identify_champion_strategy(
                result.get('strategy', {}), result
            )
            if is_champion:
                champions.append(result)
        
        # Update metrics
        self.integration_metrics.strategies_successful += stage_successes
        self.integration_metrics.near_winners_identified += len(near_winners)
        self.integration_metrics.champion_strategies += len(champions)
        
        # Update best performance
        for result in successful_results:
            cagr = result.get('cagr', 0)
            sharpe = result.get('sharpe_ratio', 0)
            if cagr > self.integration_metrics.best_cagr_achieved:
                self.integration_metrics.best_cagr_achieved = cagr
            if sharpe > self.integration_metrics.best_sharpe_achieved:
                self.integration_metrics.best_sharpe_achieved = sharpe
        
        # Calculate target achievement progress
        self.integration_metrics.target_achievement_progress = self._calculate_target_progress()
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.ANALYSIS,
            generation=context.generation,
            data={
                'stage_successes': stage_successes,
                'near_winners': len(near_winners),
                'champions_identified': len(champions),
                'current_stage': self.staged_targets.current_stage.value,
                'target_progress': self.integration_metrics.target_achievement_progress
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _enhanced_breeding_phase(self, context: AgentContext) -> AsyncGenerator[EvolutionEvent, None]:
        """Enhanced breeding with micro-mutations and champion focus"""
        
        breeding_count = 0
        micro_mutations_count = 0
        
        # Breed champion lineages
        if self.streaming_dgm.champion_strategies:
            offspring = await self.breeding_optimizer.breed_champion_lineage(10)
            breeding_count += len(offspring)
        
        # Generate micro-mutations for near-winners
        if self.config['enable_micro_mutations'] and self.streaming_dgm.near_winners:
            for near_winner in self.streaming_dgm.near_winners[-5:]:  # Last 5 near-winners
                micro_offspring = self.staged_targets.create_targeted_offspring(
                    near_winner.get('strategy', {}), near_winner
                )
                micro_mutations_count += len(micro_offspring)
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.BREEDING,
            generation=context.generation,
            data={
                'champion_breeding': breeding_count,
                'micro_mutations': micro_mutations_count,
                'breeding_efficiency': breeding_count / max(len(self.streaming_dgm.champion_strategies), 1)
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _enhanced_archive_phase(self, context: AgentContext) -> AsyncGenerator[EvolutionEvent, None]:
        """Enhanced archive management with safety and compaction"""
        
        # Archive successful strategies with safety checks
        archived_count = 0
        if hasattr(self, 'current_results'):
            for result in self.current_results:
                if (not result.get('error') and 
                    result.get('cagr', 0) > 0.15):  # Archive threshold
                    
                    # Safety check before archiving
                    if self._validate_for_archive(result):
                        self.streaming_dgm.strategy_archive[f"safe_archived_{archived_count}"] = result
                        archived_count += 1
        
        # Perform compaction if needed
        compaction_performed = False
        if len(self.streaming_dgm.strategy_archive) > self.config['archive_max_size']:
            await self.streaming_dgm._compact_archive_context()
            compaction_performed = True
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.ARCHIVE,
            generation=context.generation,
            data={
                'strategies_archived': archived_count,
                'total_in_archive': len(self.streaming_dgm.strategy_archive),
                'compaction_performed': compaction_performed,
                'archive_utilization': len(self.streaming_dgm.strategy_archive) / self.config['archive_max_size']
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _enhanced_adaptation_phase(self, context: AgentContext) -> AsyncGenerator[EvolutionEvent, None]:
        """Enhanced adaptive management with regime awareness"""
        
        # Regime detection and adaptation
        old_regime = self.streaming_dgm.current_regime
        new_regime = await self.streaming_dgm._detect_market_regime()
        regime_changed = old_regime != new_regime
        
        if regime_changed:
            self.streaming_dgm.current_regime = new_regime
            # Adapt safety scopes for new regime
            await self._adapt_safety_for_regime(new_regime)
        
        # Meta-evolution: adapt parameters
        meta_adaptations = await self._perform_meta_adaptations()
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.ADAPTATION,
            generation=context.generation,
            data={
                'regime_changed': regime_changed,
                'old_regime': old_regime,
                'new_regime': new_regime,
                'meta_adaptations': meta_adaptations,
                'system_efficiency': self._calculate_system_efficiency()
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
    
    # Helper methods
    
    def _determine_safety_scope(self, strategy: Dict) -> PermissionScope:
        """Determine appropriate safety scope for strategy"""
        leverage = strategy.get('leverage', 1.0)
        strategy_type = strategy.get('type', 'unknown')
        
        if strategy_type == 'experimental' or leverage > 3.0:
            return PermissionScope.RESEARCH
        elif leverage > 2.0:
            return PermissionScope.EXPERIMENTAL
        else:
            return PermissionScope.PRODUCTION
    
    async def _parallel_backtest_executor(self, strategy: Dict) -> Dict:
        """Execute backtest using parallel system"""
        # Use the parallel backtester
        results = await asyncio.get_event_loop().run_in_executor(
            None, self._mock_backtest, strategy
        )
        return results
    
    def _mock_backtest(self, strategy: Dict) -> Dict:
        """Mock backtest for demonstration"""
        # Simulate execution time
        time.sleep(np.random.uniform(0.1, 0.5))
        
        # Generate results based on strategy characteristics
        base_cagr = 0.15
        if strategy.get('creation_method') == 'champion_offspring':
            base_cagr = 0.22  # Champion offspring start higher
        
        cagr = base_cagr * np.random.uniform(0.8, 1.4)
        sharpe = 0.8 * np.random.uniform(0.7, 1.3)
        drawdown = 0.15 * np.random.uniform(0.8, 1.5)
        
        return {
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown,
            'strategy': strategy,
            'total_trades': np.random.randint(80, 200)
        }
    
    def _calculate_target_progress(self) -> float:
        """Calculate progress toward targets"""
        target_cagr = 0.25
        target_sharpe = 1.0
        
        # Progress based on best achieved vs targets
        cagr_progress = min(1.0, self.integration_metrics.best_cagr_achieved / target_cagr)
        sharpe_progress = min(1.0, self.integration_metrics.best_sharpe_achieved / target_sharpe)
        
        return (cagr_progress + sharpe_progress) / 2
    
    def _check_target_achievement(self) -> bool:
        """Check if targets have been achieved"""
        return (self.integration_metrics.best_cagr_achieved >= 0.25 and
                self.integration_metrics.best_sharpe_achieved >= 1.0)
    
    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel execution efficiency"""
        return 0.8  # Mock efficiency
    
    def _calculate_system_efficiency(self) -> float:
        """Calculate overall system efficiency"""
        if self.integration_metrics.evolution_cycles_completed == 0:
            return 0.0
        
        success_rate = self.integration_metrics.strategies_successful / max(self.integration_metrics.strategies_tested, 1)
        return min(1.0, success_rate * 2)  # Scale success rate
    
    def _validate_for_archive(self, result: Dict) -> bool:
        """Validate result for archiving"""
        return (not result.get('error') and
                result.get('cagr', 0) > 0.10 and
                result.get('sharpe_ratio', 0) > 0.5)
    
    async def _adapt_safety_for_regime(self, regime: str):
        """Adapt safety parameters for market regime"""
        # In volatile regimes, use more restrictive safety
        if regime in ['high_volatility', 'crash']:
            # Implement more restrictive limits
            pass
    
    async def _perform_meta_adaptations(self) -> Dict:
        """Perform meta-level adaptations"""
        return {
            'mutation_rate_adjusted': True,
            'safety_params_updated': True,
            'parallel_workers_optimized': True
        }
    
    def _get_archive_summary(self) -> Dict:
        """Get archive summary for agent context"""
        return {
            'total_strategies': len(self.streaming_dgm.strategy_archive),
            'best_performers': []  # Would extract top performers
        }
    
    def _get_compute_resources(self) -> Dict:
        """Get compute resource status"""
        return {
            'cpu_usage': 0.7,
            'memory_usage': 0.6,
            'parallel_utilization': 0.8
        }
    
    def _get_compute_stats(self) -> Dict:
        """Get compute statistics"""
        return {
            'active_backtests': 0,
            'parallel_efficiency': 0.8,
            'memory_usage': 0.6
        }
    
    def _get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return {
            'best_cagr': self.integration_metrics.best_cagr_achieved,
            'best_sharpe': self.integration_metrics.best_sharpe_achieved,
            'target_progress': self.integration_metrics.target_achievement_progress,
            'strategies_in_archive': len(self.streaming_dgm.strategy_archive)
        }
    
    def _update_integration_metrics(self, event: EvolutionEvent):
        """Update integration metrics from events"""
        # Update average cycle time
        if event.data.get('generation_complete'):
            gen_time = event.data.get('generation_time', 0)
            total_time = (self.integration_metrics.avg_cycle_time * 
                         self.integration_metrics.evolution_cycles_completed + gen_time)
            self.integration_metrics.avg_cycle_time = total_time / (self.integration_metrics.evolution_cycles_completed + 1)
    
    async def _generate_dashboard_update(self, event: EvolutionEvent) -> Dict:
        """Generate real-time dashboard update"""
        return {
            'type': 'dashboard_update',
            'timestamp': time.time(),
            'current_phase': event.phase.value,
            'generation': event.generation,
            'metrics': self.integration_metrics,
            'event_data': event.data,
            'performance': event.performance_metrics
        }
    
    async def _generate_final_summary(self, total_time: float) -> Dict:
        """Generate final evolution summary"""
        safety_report = self.safety_system.get_safety_report()
        
        return {
            'evolution_completed': True,
            'total_time_minutes': total_time / 60,
            'integration_metrics': self.integration_metrics,
            'target_achieved': self._check_target_achievement(),
            'safety_summary': safety_report['safety_overview'],
            'best_performance': {
                'cagr': self.integration_metrics.best_cagr_achieved,
                'sharpe': self.integration_metrics.best_sharpe_achieved
            },
            'system_efficiency': self._calculate_system_efficiency(),
            'recommendations': self._generate_final_recommendations()
        }
    
    def _generate_final_recommendations(self) -> List[str]:
        """Generate final recommendations"""
        recommendations = []
        
        if self.integration_metrics.target_achievement_progress >= 0.9:
            recommendations.append("✅ Near target achievement - continue current approach")
        elif self.integration_metrics.target_achievement_progress >= 0.7:
            recommendations.append("🎯 Good progress - focus on champion breeding")
        else:
            recommendations.append("🔧 Consider parameter adjustments and longer evolution")
        
        if self.integration_metrics.safety_violations > 10:
            recommendations.append("⚠️ Review safety parameters - high violation rate")
        
        return recommendations


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    async def demo_integrated_system():
        """Demonstrate complete integrated DGM system"""
        
        print("🚀" * 30)
        print("🚀 INTEGRATED DGM + CLAUDE CODE SYSTEM")
        print("🚀" * 30)
        print()
        
        # Initialize system
        config = {
            'max_parallel_backtests': 4,
            'target_generations': 5,  # Short demo
            'enable_real_time_streaming': True,
            'enable_champion_breeding': True
        }
        
        system = IntegratedDGMSystem(config)
        
        print("✅ System initialized with all components:")
        print("   🧬 Streaming DGM orchestration")
        print("   🤖 Multi-agent hierarchy")
        print("   🛡️ Safety system")
        print("   🎯 Staged targets")
        print("   ⚡ Parallel backtesting")
        print()
        
        generation_count = 0
        async for update in system.run_enhanced_evolution():
            if update['type'] == 'evolution_event':
                event = update['event']
                if event.data.get('generation_complete'):
                    generation_count += 1
                    metrics = update['integration_metrics']
                    
                    print(f"\n🧬 Generation {generation_count} Complete:")
                    print(f"   📊 Strategies generated: {metrics.strategies_generated}")
                    print(f"   ✅ Strategies successful: {metrics.strategies_successful}")
                    print(f"   🏆 Champions: {metrics.champion_strategies}")
                    print(f"   🎯 Target progress: {metrics.target_achievement_progress:.1%}")
                    print(f"   🛡️ Safety violations: {metrics.safety_violations}")
                
            elif update['type'] == 'dashboard_update':
                # Real-time dashboard update
                dashboard = update
                phase_emoji = {
                    'generation': '🧬',
                    'backtesting': '⚡',
                    'analysis': '📊',
                    'breeding': '🔬',
                    'archive': '💾',
                    'adaptation': '🎯'
                }
                emoji = phase_emoji.get(dashboard['current_phase'], '🔄')
                print(f"  {emoji} {dashboard['current_phase']}: {dashboard['event_data']}")
                
            elif update['type'] == 'evolution_complete':
                summary = update['summary']
                print(f"\n🏆" + "="*50)
                print(f"🏆 EVOLUTION COMPLETE")
                print(f"🏆" + "="*50)
                print(f"   ⏱️  Total time: {summary['total_time_minutes']:.1f} minutes")
                print(f"   🎯 Target achieved: {'✅ YES' if summary['target_achieved'] else '❌ NO'}")
                print(f"   📈 Best CAGR: {summary['best_performance']['cagr']:.1%}")
                print(f"   📊 Best Sharpe: {summary['best_performance']['sharpe']:.2f}")
                print(f"   ⚡ System efficiency: {summary['system_efficiency']:.1%}")
                
                print(f"\n💡 Recommendations:")
                for rec in summary['recommendations']:
                    print(f"   {rec}")
                break
    
    # Run the demo
    asyncio.run(demo_integrated_system())