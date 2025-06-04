"""
Streaming DGM Orchestration Engine
Enhanced Darwin Evolution with Claude Code's Streaming Patterns
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, AsyncGenerator, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import concurrent.futures
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class EvolutionPhase(Enum):
    """6-phase streaming evolution pipeline"""
    GENERATION = "generation"
    BACKTESTING = "backtesting" 
    ANALYSIS = "analysis"
    BREEDING = "breeding"
    ARCHIVE = "archive"
    ADAPTATION = "adaptation"


@dataclass
class EvolutionEvent:
    """Real-time evolution event for streaming"""
    timestamp: float
    phase: EvolutionPhase
    generation: int
    data: Dict[str, Any]
    compute_stats: Dict[str, float]
    performance_metrics: Dict[str, float]


@dataclass
class StrategyCandidate:
    """Enhanced strategy candidate with lineage tracking"""
    id: str
    config: Dict
    parent_ids: List[str]
    generation: int
    creation_method: str
    estimated_performance: Optional[Dict] = None
    actual_performance: Optional[Dict] = None
    lineage_depth: int = 0


class DGMOperationCategory:
    """Claude Code style operation categorization"""
    
    # Read-only operations (parallel safe)
    PARALLEL_SAFE = {
        'market_data_fetch',
        'indicator_calculation',
        'regime_detection', 
        'correlation_analysis',
        'performance_metrics',
        'risk_assessment',
        'strategy_validation'
    }
    
    # Write operations (must be serialized)
    SERIALIZE_REQUIRED = {
        'strategy_archive_update',
        'breeding_operations',
        'backtest_result_storage',
        'lineage_updates',
        'best_strategy_updates'
    }


class StreamingDGM:
    """
    Enhanced DGM with Claude Code's async generator streaming pattern
    6-phase streaming evolution with real-time progress updates
    """
    
    def __init__(self, max_parallel_backtests: int = 8, archive_max_size: int = 1000):
        self.max_parallel_backtests = max_parallel_backtests
        self.archive_max_size = archive_max_size
        
        # Core state
        self.generation = 0
        self.strategy_archive = {}
        self.performance_history = deque(maxlen=1000)
        self.current_regime = "neutral"
        
        # Streaming state
        self.active_backtests = {}
        self.streaming_events = []
        self.compute_utilization = {}
        
        # Best performers tracking
        self.best_cagr = 0.0
        self.best_sharpe = 0.0
        self.champion_strategies = []
        
        # Near-winner tracking (23% CAGR strategies)
        self.near_winners = []
        self.micro_mutations_queue = []
        
        logger.info("🚀 Streaming DGM initialized with Claude Code orchestration")
    
    async def evolve_strategies(self, target_generations: int = 100) -> AsyncGenerator[EvolutionEvent, None]:
        """
        6-phase streaming evolution based on Claude Code's async generator pattern
        """
        logger.info(f"🧬 Starting streaming evolution for {target_generations} generations")
        
        for gen in range(target_generations):
            self.generation = gen
            start_time = time.time()
            
            # Phase 1: Strategy Generation (Parallel)
            async for event in self._phase_generation():
                yield event
            
            # Phase 2: Backtesting (Parallel execution like Claude Code tools)
            async for event in self._phase_backtesting():
                yield event
            
            # Phase 3: Performance Analysis (Read-only parallel)
            async for event in self._phase_analysis():
                yield event
            
            # Phase 4: Selection & Breeding (Write operations - serialized)
            async for event in self._phase_breeding():
                yield event
            
            # Phase 5: Archive Management (Context compaction)
            async for event in self._phase_archive():
                yield event
            
            # Phase 6: Regime Adaptation (Meta-evolution)
            async for event in self._phase_adaptation():
                yield event
            
            # Generation summary
            gen_time = time.time() - start_time
            yield EvolutionEvent(
                timestamp=time.time(),
                phase=EvolutionPhase.GENERATION,
                generation=self.generation,
                data={
                    'generation_complete': True,
                    'generation_time': gen_time,
                    'strategies_in_archive': len(self.strategy_archive),
                    'near_winners_count': len(self.near_winners)
                },
                compute_stats=self._get_compute_stats(),
                performance_metrics=self._get_performance_metrics()
            )
    
    async def _phase_generation(self) -> AsyncGenerator[EvolutionEvent, None]:
        """Phase 1: Parallel strategy generation"""
        candidates = []
        
        # Generate base candidates
        base_candidates = await self._generate_base_candidates(20)
        candidates.extend(base_candidates)
        
        # Generate micro-mutations for near-winners (23% CAGR strategies)
        if self.near_winners:
            micro_candidates = await self._generate_micro_mutations(10)
            candidates.extend(micro_candidates)
        
        # Generate ensemble breeding candidates
        if len(self.champion_strategies) >= 2:
            ensemble_candidates = await self._generate_ensemble_candidates(5)
            candidates.extend(ensemble_candidates)
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.GENERATION,
            generation=self.generation,
            data={
                'candidates_generated': len(candidates),
                'base_candidates': len(base_candidates),
                'micro_mutations': len(micro_candidates) if self.near_winners else 0,
                'ensemble_candidates': len(ensemble_candidates) if len(self.champion_strategies) >= 2 else 0
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
        
        self.current_candidates = candidates
    
    async def _phase_backtesting(self) -> AsyncGenerator[EvolutionEvent, None]:
        """Phase 2: Parallel backtesting execution"""
        
        # Parallel backtesting with Claude Code's batch pattern
        batch_size = self.max_parallel_backtests
        results = []
        
        for i in range(0, len(self.current_candidates), batch_size):
            batch = self.current_candidates[i:i + batch_size]
            
            # Execute batch in parallel
            batch_results = await self._execute_parallel_backtests(batch)
            results.extend(batch_results)
            
            yield EvolutionEvent(
                timestamp=time.time(),
                phase=EvolutionPhase.BACKTESTING,
                generation=self.generation,
                data={
                    'batch_completed': i // batch_size + 1,
                    'total_batches': (len(self.current_candidates) + batch_size - 1) // batch_size,
                    'strategies_tested': len(results),
                    'successful_tests': sum(1 for r in results if r.get('success', False))
                },
                compute_stats=self._get_compute_stats(),
                performance_metrics=self._get_performance_metrics()
            )
        
        self.current_results = results
    
    async def _phase_analysis(self) -> AsyncGenerator[EvolutionEvent, None]:
        """Phase 3: Performance analysis (read-only parallel)"""
        
        # Parallel analysis operations
        analysis_tasks = [
            self._analyze_performance_metrics(self.current_results),
            self._analyze_strategy_lineages(self.current_results),
            self._identify_near_winners(self.current_results),
            self._detect_performance_patterns(self.current_results)
        ]
        
        analysis_results = await asyncio.gather(*analysis_tasks)
        
        performance_analysis = analysis_results[0]
        lineage_analysis = analysis_results[1] 
        new_near_winners = analysis_results[2]
        pattern_analysis = analysis_results[3]
        
        # Update near-winners list
        self.near_winners.extend(new_near_winners)
        self.near_winners = self.near_winners[-50:]  # Keep recent 50
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.ANALYSIS,
            generation=self.generation,
            data={
                'performance_analysis': performance_analysis,
                'lineage_insights': lineage_analysis,
                'new_near_winners': len(new_near_winners),
                'pattern_insights': pattern_analysis,
                'total_near_winners': len(self.near_winners)
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
        
        self.current_analysis = {
            'performance': performance_analysis,
            'lineages': lineage_analysis,
            'patterns': pattern_analysis
        }
    
    async def _phase_breeding(self) -> AsyncGenerator[EvolutionEvent, None]:
        """Phase 4: Selection & breeding (serialized writes)"""
        
        # Select best performers
        selected_strategies = self._select_best_performers(self.current_results)
        
        # Breed new lineages
        new_lineages = await self._breed_new_lineages(selected_strategies)
        
        # Update champion strategies
        champions_updated = self._update_champion_strategies(selected_strategies)
        
        # Generate micro-mutations for near-winners
        micro_mutations_generated = 0
        if self.near_winners:
            micro_mutations = await self._generate_micro_mutations_for_near_winners()
            micro_mutations_generated = len(micro_mutations)
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.BREEDING,
            generation=self.generation,
            data={
                'selected_strategies': len(selected_strategies),
                'new_lineages': len(new_lineages),
                'champions_updated': champions_updated,
                'micro_mutations_generated': micro_mutations_generated
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _phase_archive(self) -> AsyncGenerator[EvolutionEvent, None]:
        """Phase 5: Archive management with context compaction"""
        
        # Add successful strategies to archive
        archived_count = 0
        for result in self.current_results:
            if self._meets_archive_criteria(result):
                strategy_id = f"gen_{self.generation}_{archived_count}"
                self.strategy_archive[strategy_id] = result
                archived_count += 1
        
        # Context compaction if archive too large
        compacted = False
        if len(self.strategy_archive) > self.archive_max_size:
            await self._compact_archive_context()
            compacted = True
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.ARCHIVE,
            generation=self.generation,
            data={
                'strategies_archived': archived_count,
                'total_archived': len(self.strategy_archive),
                'archive_compacted': compacted,
                'archive_utilization': len(self.strategy_archive) / self.archive_max_size
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _phase_adaptation(self) -> AsyncGenerator[EvolutionEvent, None]:
        """Phase 6: Regime adaptation and meta-evolution"""
        
        # Detect regime changes
        old_regime = self.current_regime
        new_regime = await self._detect_market_regime()
        regime_changed = old_regime != new_regime
        
        if regime_changed:
            self.current_regime = new_regime
            # Adapt mutation parameters for new regime
            await self._adapt_to_regime(new_regime)
        
        # Meta-evolution: adapt algorithm parameters
        meta_adaptations = await self._perform_meta_evolution()
        
        yield EvolutionEvent(
            timestamp=time.time(),
            phase=EvolutionPhase.ADAPTATION,
            generation=self.generation,
            data={
                'regime_changed': regime_changed,
                'old_regime': old_regime,
                'new_regime': new_regime,
                'meta_adaptations': meta_adaptations
            },
            compute_stats=self._get_compute_stats(),
            performance_metrics=self._get_performance_metrics()
        )
    
    # Implementation methods
    
    async def _generate_base_candidates(self, count: int) -> List[StrategyCandidate]:
        """Generate base strategy candidates"""
        candidates = []
        
        for i in range(count):
            config = {
                'type': np.random.choice(['momentum', 'mean_reversion', 'trend_following']),
                'leverage': np.random.uniform(1.0, 3.0),
                'position_size': np.random.uniform(0.1, 0.3),
                'stop_loss': np.random.uniform(0.05, 0.15),
                'indicators': ['RSI', 'MACD', 'BB'][:np.random.randint(2, 4)]
            }
            
            candidate = StrategyCandidate(
                id=f"base_gen{self.generation}_{i}",
                config=config,
                parent_ids=[],
                generation=self.generation,
                creation_method='base_generation'
            )
            candidates.append(candidate)
        
        return candidates
    
    async def _generate_micro_mutations(self, count: int) -> List[StrategyCandidate]:
        """Generate micro-mutations for near-winner strategies (23% CAGR → 25%)"""
        candidates = []
        
        if not self.near_winners:
            return candidates
        
        for i in range(count):
            # Select a near-winner strategy
            base_strategy = np.random.choice(self.near_winners)
            
            # Apply micro-mutations: tiny adjustments to push over finish line
            config = base_strategy['config'].copy()
            
            # Micro-adjustments for 23% → 25% CAGR push
            if np.random.random() < 0.4:  # 40% chance: tiny leverage boost
                config['leverage'] *= np.random.uniform(1.02, 1.08)
            
            if np.random.random() < 0.3:  # 30% chance: position size optimization
                config['position_size'] *= np.random.uniform(1.01, 1.05)
            
            if np.random.random() < 0.5:  # 50% chance: stop loss tightening for Sharpe
                config['stop_loss'] *= np.random.uniform(0.92, 0.98)
            
            candidate = StrategyCandidate(
                id=f"micro_gen{self.generation}_{i}",
                config=config,
                parent_ids=[base_strategy['id']],
                generation=self.generation,
                creation_method='micro_mutation',
                lineage_depth=base_strategy.get('lineage_depth', 0) + 1
            )
            candidates.append(candidate)
        
        return candidates
    
    async def _generate_ensemble_candidates(self, count: int) -> List[StrategyCandidate]:
        """Generate ensemble breeding candidates"""
        candidates = []
        
        for i in range(count):
            # Select two champion strategies
            parents = np.random.choice(self.champion_strategies, 2, replace=False)
            
            # Blend configurations
            config = {}
            for key in ['leverage', 'position_size', 'stop_loss']:
                val1 = parents[0]['config'].get(key, 1.0)
                val2 = parents[1]['config'].get(key, 1.0)
                config[key] = (val1 + val2) / 2
            
            # Combine indicators
            indicators1 = set(parents[0]['config'].get('indicators', []))
            indicators2 = set(parents[1]['config'].get('indicators', []))
            config['indicators'] = list(indicators1.union(indicators2))[:4]
            
            config['type'] = 'ensemble'
            
            candidate = StrategyCandidate(
                id=f"ensemble_gen{self.generation}_{i}",
                config=config,
                parent_ids=[parents[0]['id'], parents[1]['id']],
                generation=self.generation,
                creation_method='ensemble_breeding'
            )
            candidates.append(candidate)
        
        return candidates
    
    async def _execute_parallel_backtests(self, candidates: List[StrategyCandidate]) -> List[Dict]:
        """Execute backtests in parallel using Claude Code's pattern"""
        
        # Use ThreadPoolExecutor for I/O bound backtesting
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_backtests) as executor:
            
            # Submit all backtest tasks
            futures = {
                executor.submit(self._backtest_strategy, candidate): candidate
                for candidate in candidates
            }
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                candidate = futures[future]
                try:
                    result = future.result(timeout=30)  # 30s timeout
                    result['candidate_id'] = candidate.id
                    result['success'] = True
                    results.append(result)
                except Exception as e:
                    logger.error(f"Backtest failed for {candidate.id}: {e}")
                    results.append({
                        'candidate_id': candidate.id,
                        'success': False,
                        'error': str(e)
                    })
            
            return results
    
    def _backtest_strategy(self, candidate: StrategyCandidate) -> Dict:
        """Mock backtest for demonstration - replace with real backtesting"""
        # Simulate backtest time
        time.sleep(np.random.uniform(0.1, 0.5))
        
        # Generate mock results
        base_cagr = 0.15
        base_sharpe = 0.8
        
        # Micro-mutations get boost toward targets
        if candidate.creation_method == 'micro_mutation':
            base_cagr = 0.22  # Start higher for near-winners
            base_sharpe = 0.92
        
        # Add some randomness
        cagr = base_cagr * np.random.uniform(0.8, 1.4)
        sharpe = base_sharpe * np.random.uniform(0.7, 1.3)
        drawdown = 0.15 * np.random.uniform(0.8, 1.5)
        
        return {
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown,
            'config': candidate.config,
            'id': candidate.id,
            'creation_method': candidate.creation_method
        }
    
    async def _analyze_performance_metrics(self, results: List[Dict]) -> Dict:
        """Analyze performance metrics"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        cagrs = [r['cagr'] for r in successful_results]
        sharpes = [r['sharpe_ratio'] for r in successful_results]
        
        return {
            'avg_cagr': np.mean(cagrs),
            'max_cagr': np.max(cagrs),
            'avg_sharpe': np.mean(sharpes),
            'max_sharpe': np.max(sharpes),
            'success_rate': len(successful_results) / len(results),
            'strategies_above_20_cagr': sum(1 for c in cagrs if c >= 0.20),
            'strategies_above_23_cagr': sum(1 for c in cagrs if c >= 0.23)
        }
    
    async def _identify_near_winners(self, results: List[Dict]) -> List[Dict]:
        """Identify near-winner strategies (23%+ CAGR, close to targets)"""
        near_winners = []
        
        for result in results:
            if not result.get('success', False):
                continue
            
            cagr = result.get('cagr', 0)
            sharpe = result.get('sharpe_ratio', 0)
            drawdown = result.get('max_drawdown', 1)
            
            # Near-winner criteria: close to 25% CAGR target
            if (cagr >= 0.20 and  # At least 20% CAGR
                sharpe >= 0.85 and  # Decent Sharpe
                drawdown <= 0.20):  # Reasonable drawdown
                
                near_winners.append(result)
        
        return near_winners
    
    def _get_compute_stats(self) -> Dict[str, float]:
        """Get current compute utilization stats"""
        return {
            'active_backtests': len(self.active_backtests),
            'max_parallel': self.max_parallel_backtests,
            'utilization': len(self.active_backtests) / self.max_parallel_backtests,
            'memory_usage': 0.7  # Mock
        }
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current best performance metrics"""
        return {
            'best_cagr': self.best_cagr,
            'best_sharpe': self.best_sharpe,
            'strategies_in_archive': len(self.strategy_archive),
            'near_winners_count': len(self.near_winners),
            'generation': self.generation
        }
    
    # Placeholder implementations for other methods
    async def _analyze_strategy_lineages(self, results): return {}
    async def _detect_performance_patterns(self, results): return {}
    def _select_best_performers(self, results): return results[:5]
    async def _breed_new_lineages(self, strategies): return []
    def _update_champion_strategies(self, strategies): return True
    async def _generate_micro_mutations_for_near_winners(self): return []
    def _meets_archive_criteria(self, result): return result.get('cagr', 0) > 0.15
    async def _compact_archive_context(self): pass
    async def _detect_market_regime(self): return self.current_regime
    async def _adapt_to_regime(self, regime): pass
    async def _perform_meta_evolution(self): return {}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    async def demo_streaming_dgm():
        """Demonstrate streaming DGM"""
        dgm = StreamingDGM(max_parallel_backtests=4)
        
        print("🚀 Starting Streaming DGM Demonstration")
        print("=" * 60)
        
        generation_count = 0
        async for event in dgm.evolve_strategies(target_generations=3):
            if event.phase == EvolutionPhase.GENERATION and event.data.get('generation_complete'):
                generation_count += 1
                print(f"\n🧬 Generation {event.generation} Complete")
                print(f"   Best CAGR: {event.performance_metrics['best_cagr']:.1%}")
                print(f"   Best Sharpe: {event.performance_metrics['best_sharpe']:.2f}")
                print(f"   Near Winners: {event.performance_metrics['near_winners_count']}")
            else:
                phase_emoji = {
                    EvolutionPhase.GENERATION: "🧬",
                    EvolutionPhase.BACKTESTING: "⚡",
                    EvolutionPhase.ANALYSIS: "📊", 
                    EvolutionPhase.BREEDING: "🔬",
                    EvolutionPhase.ARCHIVE: "💾",
                    EvolutionPhase.ADAPTATION: "🎯"
                }
                emoji = phase_emoji.get(event.phase, "🔄")
                print(f"{emoji} {event.phase.value}: {event.data}")
        
        print(f"\n🏆 Streaming DGM completed {generation_count} generations")
    
    # Run the demo
    asyncio.run(demo_streaming_dgm())