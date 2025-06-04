"""
DGM Agent Hierarchy - Multi-Agent DGM Enhancement
Hierarchical agents specialized for different aspects of evolution
"""

import asyncio
import time
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass 
class AgentContext:
    """Shared context between agents"""
    current_regime: str
    generation: int
    archive_summary: Dict
    performance_history: List[Dict]
    near_winners: List[Dict]
    compute_resources: Dict


class DGMAgent(ABC):
    """Base class for DGM agents following Claude Code's agent pattern"""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.execution_history = deque(maxlen=100)
        self.performance_metrics = {}
        
    @abstractmethod
    async def execute(self, context: AgentContext, **kwargs) -> Dict[str, Any]:
        """Execute agent's primary function"""
        pass
    
    def log_execution(self, start_time: float, result: Dict, success: bool):
        """Log execution metrics"""
        execution_time = time.time() - start_time
        self.execution_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'success': success,
            'result_size': len(str(result))
        })
        
        # Update performance metrics
        recent_executions = list(self.execution_history)[-10:]
        self.performance_metrics = {
            'avg_execution_time': np.mean([e['execution_time'] for e in recent_executions]),
            'success_rate': np.mean([e['success'] for e in recent_executions]),
            'total_executions': len(self.execution_history)
        }


class MarketRegimeAgent(DGMAgent):
    """Analyzes market conditions and regime for context-aware strategy generation"""
    
    def __init__(self):
        super().__init__("MarketRegimeAgent", ["regime_detection", "market_analysis", "volatility_assessment"])
        self.regime_history = deque(maxlen=50)
        self.regime_confidence_threshold = 0.7
        
    async def execute(self, context: AgentContext, market_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze current market regime and provide context"""
        start_time = time.time()
        
        try:
            # Mock market data if not provided
            if market_data is None:
                market_data = self._generate_mock_market_data()
            
            # Detect regime
            regime_analysis = await self._detect_regime(market_data)
            
            # Assess regime stability
            stability_analysis = self._assess_regime_stability()
            
            # Generate strategy recommendations
            strategy_recommendations = self._generate_strategy_recommendations(regime_analysis)
            
            result = {
                'current_regime': regime_analysis['regime'],
                'regime_confidence': regime_analysis['confidence'],
                'regime_change_probability': stability_analysis['change_probability'],
                'volatility_level': regime_analysis['volatility_level'],
                'strategy_recommendations': strategy_recommendations,
                'market_indicators': regime_analysis['indicators']
            }
            
            # Update regime history
            self.regime_history.append({
                'timestamp': time.time(),
                'regime': regime_analysis['regime'],
                'confidence': regime_analysis['confidence']
            })
            
            self.log_execution(start_time, result, True)
            return result
            
        except Exception as e:
            logger.error(f"MarketRegimeAgent execution failed: {e}")
            self.log_execution(start_time, {}, False)
            return {'error': str(e)}
    
    async def _detect_regime(self, market_data: Dict) -> Dict:
        """Detect current market regime"""
        # Mock regime detection logic
        spy_price = market_data.get('spy_price', 450)
        vix = market_data.get('vix', 20)
        momentum = market_data.get('momentum_20d', 0.02)
        
        # Simple regime classification
        if vix > 30:
            regime = "high_volatility"
            confidence = 0.8
        elif momentum > 0.05:
            regime = "bull_market"
            confidence = 0.9
        elif momentum < -0.05:
            regime = "bear_market"
            confidence = 0.85
        else:
            regime = "sideways"
            confidence = 0.7
        
        return {
            'regime': regime,
            'confidence': confidence,
            'volatility_level': 'high' if vix > 25 else 'medium' if vix > 15 else 'low',
            'indicators': {
                'vix': vix,
                'momentum': momentum,
                'price_level': spy_price
            }
        }
    
    def _assess_regime_stability(self) -> Dict:
        """Assess stability of current regime"""
        if len(self.regime_history) < 5:
            return {'change_probability': 0.3}
        
        recent_regimes = [r['regime'] for r in list(self.regime_history)[-5:]]
        regime_consistency = len(set(recent_regimes)) / len(recent_regimes)
        
        # High consistency = low change probability
        change_probability = 1.0 - regime_consistency
        
        return {
            'change_probability': change_probability,
            'consistency_score': regime_consistency
        }
    
    def _generate_strategy_recommendations(self, regime_analysis: Dict) -> Dict:
        """Generate strategy recommendations based on regime"""
        regime = regime_analysis['regime']
        
        recommendations = {
            'bull_market': {
                'preferred_types': ['momentum', 'trend_following'],
                'leverage_range': (1.5, 3.0),
                'position_size_range': (0.15, 0.3),
                'indicators': ['RSI', 'MACD', 'ADX']
            },
            'bear_market': {
                'preferred_types': ['defensive', 'short_momentum'],
                'leverage_range': (0.5, 1.5),
                'position_size_range': (0.05, 0.15),
                'indicators': ['VIX', 'PUT_CALL', 'BB']
            },
            'sideways': {
                'preferred_types': ['mean_reversion', 'range_trading'],
                'leverage_range': (1.0, 2.0),
                'position_size_range': (0.1, 0.2),
                'indicators': ['RSI', 'BB', 'STOCH']
            },
            'high_volatility': {
                'preferred_types': ['volatility', 'defensive'],
                'leverage_range': (0.5, 1.5),
                'position_size_range': (0.05, 0.15),
                'indicators': ['ATR', 'VIX', 'BB']
            }
        }
        
        return recommendations.get(regime, recommendations['sideways'])
    
    def _generate_mock_market_data(self) -> Dict:
        """Generate mock market data for testing"""
        return {
            'spy_price': 450 + np.random.normal(0, 10),
            'vix': 20 + np.random.normal(0, 5),
            'momentum_20d': np.random.normal(0.02, 0.05),
            'volume': 100000000 + np.random.normal(0, 20000000)
        }


class StrategyGeneratorAgent(DGMAgent):
    """Creates new strategy variants based on regime context and archive insights"""
    
    def __init__(self):
        super().__init__("StrategyGeneratorAgent", ["strategy_creation", "mutation", "crossover"])
        self.generation_templates = {}
        self.successful_patterns = defaultdict(list)
        
    async def execute(self, context: AgentContext, regime_context: Dict, 
                     generation_count: int = 20) -> Dict[str, Any]:
        """Generate new strategy candidates"""
        start_time = time.time()
        
        try:
            strategies = []
            
            # Generate regime-specific strategies (60%)
            regime_strategies = await self._generate_regime_specific_strategies(
                regime_context, int(generation_count * 0.6)
            )
            strategies.extend(regime_strategies)
            
            # Generate archive-based mutations (25%)
            if context.archive_summary:
                archive_strategies = await self._generate_archive_based_strategies(
                    context.archive_summary, int(generation_count * 0.25)
                )
                strategies.extend(archive_strategies)
            
            # Generate novel explorations (15%)
            novel_strategies = await self._generate_novel_strategies(
                int(generation_count * 0.15)
            )
            strategies.extend(novel_strategies)
            
            result = {
                'strategies_generated': len(strategies),
                'regime_specific': len(regime_strategies),
                'archive_based': len(archive_strategies) if context.archive_summary else 0,
                'novel_explorations': len(novel_strategies),
                'strategies': strategies
            }
            
            self.log_execution(start_time, result, True)
            return result
            
        except Exception as e:
            logger.error(f"StrategyGeneratorAgent execution failed: {e}")
            self.log_execution(start_time, {}, False)
            return {'error': str(e)}
    
    async def _generate_regime_specific_strategies(self, regime_context: Dict, count: int) -> List[Dict]:
        """Generate strategies optimized for current regime"""
        strategies = []
        recommendations = regime_context.get('strategy_recommendations', {})
        
        preferred_types = recommendations.get('preferred_types', ['momentum'])
        leverage_range = recommendations.get('leverage_range', (1.0, 2.0))
        position_range = recommendations.get('position_size_range', (0.1, 0.2))
        indicators = recommendations.get('indicators', ['RSI', 'MACD'])
        
        for i in range(count):
            strategy = {
                'id': f"regime_{regime_context.get('current_regime', 'unknown')}_{i}",
                'type': np.random.choice(preferred_types),
                'leverage': np.random.uniform(*leverage_range),
                'position_size': np.random.uniform(*position_range),
                'stop_loss': np.random.uniform(0.05, 0.15),
                'indicators': np.random.choice(indicators, size=min(3, len(indicators)), replace=False).tolist(),
                'regime_optimized': regime_context.get('current_regime'),
                'creation_method': 'regime_specific'
            }
            strategies.append(strategy)
        
        return strategies
    
    async def _generate_archive_based_strategies(self, archive_summary: Dict, count: int) -> List[Dict]:
        """Generate strategies based on successful patterns from archive"""
        strategies = []
        
        # Extract successful patterns
        best_performers = archive_summary.get('best_performers', [])
        if not best_performers:
            return strategies
        
        for i in range(count):
            # Select a successful strategy as base
            base_strategy = np.random.choice(best_performers)
            
            # Apply mutations
            mutated = base_strategy.copy()
            
            # Parameter mutations (small adjustments)
            if 'leverage' in mutated:
                mutated['leverage'] *= np.random.uniform(0.9, 1.1)
            if 'position_size' in mutated:
                mutated['position_size'] *= np.random.uniform(0.95, 1.05)
            if 'stop_loss' in mutated:
                mutated['stop_loss'] *= np.random.uniform(0.9, 1.1)
            
            mutated['id'] = f"archive_mutation_{i}"
            mutated['creation_method'] = 'archive_based'
            mutated['parent_id'] = base_strategy.get('id', 'unknown')
            
            strategies.append(mutated)
        
        return strategies
    
    async def _generate_novel_strategies(self, count: int) -> List[Dict]:
        """Generate novel strategy explorations"""
        strategies = []
        
        novel_types = ['experimental', 'hybrid', 'ensemble']
        
        for i in range(count):
            strategy = {
                'id': f"novel_{i}",
                'type': np.random.choice(novel_types),
                'leverage': np.random.uniform(0.5, 4.0),  # Wider range for exploration
                'position_size': np.random.uniform(0.05, 0.4),
                'stop_loss': np.random.uniform(0.02, 0.2),
                'indicators': ['RSI', 'MACD', 'BB', 'ADX', 'STOCH'][:np.random.randint(2, 5)],
                'experimental_features': {
                    'dynamic_sizing': np.random.choice([True, False]),
                    'regime_switching': np.random.choice([True, False])
                },
                'creation_method': 'novel_exploration'
            }
            strategies.append(strategy)
        
        return strategies


class RiskAnalyzerAgent(DGMAgent):
    """Specialized risk assessment focused on drawdown optimization"""
    
    def __init__(self):
        super().__init__("RiskAnalyzerAgent", ["risk_assessment", "drawdown_analysis", "correlation_analysis"])
        self.risk_thresholds = {
            'max_drawdown': 0.15,
            'max_leverage': 3.0,
            'min_sharpe': 0.8,
            'max_correlation': 0.7
        }
    
    async def execute(self, context: AgentContext, strategies: List[Dict]) -> Dict[str, Any]:
        """Assess risk for strategy candidates"""
        start_time = time.time()
        
        try:
            risk_assessments = []
            
            for strategy in strategies:
                assessment = await self._assess_strategy_risk(strategy, context)
                risk_assessments.append(assessment)
            
            # Aggregate risk insights
            risk_insights = self._generate_risk_insights(risk_assessments)
            
            result = {
                'strategies_assessed': len(strategies),
                'risk_assessments': risk_assessments,
                'risk_insights': risk_insights,
                'high_risk_count': sum(1 for a in risk_assessments if a['risk_level'] == 'high'),
                'approved_strategies': [a for a in risk_assessments if a['approved']]
            }
            
            self.log_execution(start_time, result, True)
            return result
            
        except Exception as e:
            logger.error(f"RiskAnalyzerAgent execution failed: {e}")
            self.log_execution(start_time, {}, False)
            return {'error': str(e)}
    
    async def _assess_strategy_risk(self, strategy: Dict, context: AgentContext) -> Dict:
        """Assess individual strategy risk"""
        risk_factors = []
        risk_score = 0.0
        
        # Leverage risk
        leverage = strategy.get('leverage', 1.0)
        if leverage > self.risk_thresholds['max_leverage']:
            risk_factors.append(f"High leverage: {leverage:.2f}")
            risk_score += 0.3
        
        # Position sizing risk
        position_size = strategy.get('position_size', 0.1)
        if position_size > 0.3:
            risk_factors.append(f"Large position size: {position_size:.2f}")
            risk_score += 0.2
        
        # Stop loss risk
        stop_loss = strategy.get('stop_loss', 0.1)
        if stop_loss > 0.2:
            risk_factors.append(f"Wide stop loss: {stop_loss:.2f}")
            risk_score += 0.1
        elif stop_loss < 0.03:
            risk_factors.append(f"Very tight stop loss: {stop_loss:.2f}")
            risk_score += 0.15
        
        # Strategy type risk
        strategy_type = strategy.get('type', 'unknown')
        if strategy_type in ['experimental', 'novel']:
            risk_factors.append("Experimental strategy type")
            risk_score += 0.2
        
        # Determine risk level
        if risk_score >= 0.6:
            risk_level = 'high'
        elif risk_score >= 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Approval decision
        approved = risk_level != 'high' and len(risk_factors) < 3
        
        return {
            'strategy_id': strategy.get('id', 'unknown'),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'approved': approved,
            'recommendations': self._generate_risk_recommendations(strategy, risk_factors)
        }
    
    def _generate_risk_recommendations(self, strategy: Dict, risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        if any('leverage' in factor.lower() for factor in risk_factors):
            recommendations.append("Consider reducing leverage to below 3.0")
        
        if any('position' in factor.lower() for factor in risk_factors):
            recommendations.append("Reduce position size for better risk management")
        
        if any('stop loss' in factor.lower() for factor in risk_factors):
            recommendations.append("Optimize stop loss between 5% and 15%")
        
        return recommendations
    
    def _generate_risk_insights(self, assessments: List[Dict]) -> Dict:
        """Generate aggregate risk insights"""
        total_strategies = len(assessments)
        if total_strategies == 0:
            return {}
        
        approved_count = sum(1 for a in assessments if a['approved'])
        avg_risk_score = np.mean([a['risk_score'] for a in assessments])
        
        # Most common risk factors
        all_factors = []
        for assessment in assessments:
            all_factors.extend(assessment['risk_factors'])
        
        factor_counts = defaultdict(int)
        for factor in all_factors:
            factor_counts[factor] += 1
        
        common_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'approval_rate': approved_count / total_strategies,
            'avg_risk_score': avg_risk_score,
            'common_risk_factors': [factor for factor, count in common_factors],
            'risk_distribution': {
                'high': sum(1 for a in assessments if a['risk_level'] == 'high'),
                'medium': sum(1 for a in assessments if a['risk_level'] == 'medium'),
                'low': sum(1 for a in assessments if a['risk_level'] == 'low')
            }
        }


class PerformanceSynthesizerAgent(DGMAgent):
    """Combines insights from all agents to make evolution decisions"""
    
    def __init__(self):
        super().__init__("PerformanceSynthesizerAgent", ["insight_synthesis", "decision_making", "optimization"])
        self.synthesis_history = deque(maxlen=100)
    
    async def execute(self, context: AgentContext, regime_analysis: Dict, 
                     generated_strategies: Dict, risk_assessments: Dict) -> Dict[str, Any]:
        """Synthesize all agent insights"""
        start_time = time.time()
        
        try:
            # Combine all insights
            synthesis = {
                'regime_context': regime_analysis,
                'generation_insights': generated_strategies,
                'risk_insights': risk_assessments,
                'synthesis_timestamp': time.time()
            }
            
            # Make evolution decisions
            evolution_decisions = self._make_evolution_decisions(synthesis)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(synthesis)
            
            # Update synthesis history
            self.synthesis_history.append(synthesis)
            
            result = {
                'synthesis_complete': True,
                'evolution_decisions': evolution_decisions,
                'recommendations': recommendations,
                'strategies_approved': len(risk_assessments.get('approved_strategies', [])),
                'regime_alignment': self._assess_regime_alignment(regime_analysis, generated_strategies)
            }
            
            self.log_execution(start_time, result, True)
            return result
            
        except Exception as e:
            logger.error(f"PerformanceSynthesizerAgent execution failed: {e}")
            self.log_execution(start_time, {}, False)
            return {'error': str(e)}
    
    def _make_evolution_decisions(self, synthesis: Dict) -> Dict:
        """Make decisions about evolution direction"""
        decisions = {}
        
        # Decide on mutation rates based on recent performance
        risk_insights = synthesis['risk_insights']
        approval_rate = risk_insights.get('risk_insights', {}).get('approval_rate', 0.5)
        
        if approval_rate < 0.3:
            decisions['mutation_rate'] = 'decrease'  # Too many risky strategies
        elif approval_rate > 0.8:
            decisions['mutation_rate'] = 'increase'  # Can be more aggressive
        else:
            decisions['mutation_rate'] = 'maintain'
        
        # Decide on exploration vs exploitation
        regime_confidence = synthesis['regime_context'].get('regime_confidence', 0.5)
        if regime_confidence > 0.8:
            decisions['exploration_mode'] = 'exploit'  # High confidence, exploit current regime
        else:
            decisions['exploration_mode'] = 'explore'  # Low confidence, explore more
        
        return decisions
    
    def _generate_recommendations(self, synthesis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        risk_insights = synthesis['risk_insights'].get('risk_insights', {})
        if risk_insights.get('avg_risk_score', 0) > 0.5:
            recommendations.append("Consider implementing more conservative parameter bounds")
        
        # Regime-based recommendations
        regime = synthesis['regime_context'].get('current_regime', 'unknown')
        if regime == 'high_volatility':
            recommendations.append("Focus on defensive strategies and reduced position sizes")
        elif regime == 'bull_market':
            recommendations.append("Increase allocation to momentum and trend-following strategies")
        
        # Generation-based recommendations
        generation_stats = synthesis['generation_insights']
        if generation_stats.get('novel_explorations', 0) < 2:
            recommendations.append("Increase novel strategy exploration for diversity")
        
        return recommendations
    
    def _assess_regime_alignment(self, regime_analysis: Dict, generated_strategies: Dict) -> float:
        """Assess how well generated strategies align with current regime"""
        current_regime = regime_analysis.get('current_regime', 'unknown')
        strategies = generated_strategies.get('strategies', [])
        
        if not strategies:
            return 0.0
        
        # Count regime-optimized strategies
        regime_aligned = sum(1 for s in strategies 
                           if s.get('regime_optimized') == current_regime or 
                              s.get('creation_method') == 'regime_specific')
        
        return regime_aligned / len(strategies)


class ArchiveManagerAgent(DGMAgent):
    """Manages strategy archive with context compaction"""
    
    def __init__(self, max_archive_size: int = 1000):
        super().__init__("ArchiveManagerAgent", ["archive_management", "context_compaction", "lineage_tracking"])
        self.max_archive_size = max_archive_size
        self.archive = {}
        self.lineage_tree = defaultdict(list)
    
    async def execute(self, context: AgentContext, synthesis_results: Dict) -> Dict[str, Any]:
        """Update archive with new strategies"""
        start_time = time.time()
        
        try:
            # Extract strategies to archive
            approved_strategies = synthesis_results.get('approved_strategies', [])
            
            # Archive successful strategies
            archived_count = 0
            for strategy in approved_strategies:
                if self._meets_archive_criteria(strategy):
                    await self._archive_strategy(strategy)
                    archived_count += 1
            
            # Perform context compaction if needed
            compaction_performed = False
            if len(self.archive) > self.max_archive_size:
                await self._compact_archive()
                compaction_performed = True
            
            # Update lineage tracking
            lineage_updates = self._update_lineage_tree(approved_strategies)
            
            result = {
                'strategies_archived': archived_count,
                'total_in_archive': len(self.archive),
                'compaction_performed': compaction_performed,
                'lineage_updates': lineage_updates,
                'archive_summary': self._generate_archive_summary()
            }
            
            self.log_execution(start_time, result, True)
            return result
            
        except Exception as e:
            logger.error(f"ArchiveManagerAgent execution failed: {e}")
            self.log_execution(start_time, {}, False)
            return {'error': str(e)}
    
    def _meets_archive_criteria(self, strategy: Dict) -> bool:
        """Check if strategy meets archival criteria"""
        # Mock criteria - in real implementation, check actual performance
        return (strategy.get('risk_level', 'high') != 'high' and
                strategy.get('approved', False))
    
    async def _archive_strategy(self, strategy: Dict):
        """Archive a strategy"""
        strategy_id = strategy.get('strategy_id', f"archived_{len(self.archive)}")
        self.archive[strategy_id] = {
            'strategy': strategy,
            'archived_at': time.time(),
            'performance_estimate': np.random.uniform(0.1, 0.3)  # Mock performance
        }
    
    async def _compact_archive(self):
        """Perform context compaction on archive"""
        # Keep top 70% by performance, remove bottom 30%
        archive_items = list(self.archive.items())
        archive_items.sort(key=lambda x: x[1]['performance_estimate'], reverse=True)
        
        keep_count = int(len(archive_items) * 0.7)
        kept_items = archive_items[:keep_count]
        
        self.archive = dict(kept_items)
        logger.info(f"Archive compacted: {len(archive_items)} → {len(self.archive)} strategies")
    
    def _update_lineage_tree(self, strategies: List[Dict]) -> int:
        """Update strategy lineage tree"""
        updates = 0
        for strategy in strategies:
            strategy_id = strategy.get('strategy_id', 'unknown')
            parent_id = strategy.get('parent_id')
            
            if parent_id:
                self.lineage_tree[parent_id].append(strategy_id)
                updates += 1
        
        return updates
    
    def _generate_archive_summary(self) -> Dict:
        """Generate summary of archive contents"""
        if not self.archive:
            return {}
        
        performances = [item['performance_estimate'] for item in self.archive.values()]
        
        return {
            'total_strategies': len(self.archive),
            'avg_performance': np.mean(performances),
            'best_performance': np.max(performances),
            'performance_distribution': {
                'top_10_percent': np.percentile(performances, 90),
                'median': np.median(performances),
                'bottom_10_percent': np.percentile(performances, 10)
            },
            'best_performers': [
                item['strategy'] for item in 
                sorted(self.archive.values(), key=lambda x: x['performance_estimate'], reverse=True)[:5]
            ]
        }


class DGMAgentOrchestrator:
    """Orchestrates all DGM agents following Claude Code's hierarchical pattern"""
    
    def __init__(self):
        self.agents = {
            'market_regime': MarketRegimeAgent(),
            'strategy_generator': StrategyGeneratorAgent(),
            'risk_analyzer': RiskAnalyzerAgent(),
            'performance_synthesizer': PerformanceSynthesizerAgent(),
            'archive_manager': ArchiveManagerAgent()
        }
        
        self.orchestration_history = deque(maxlen=100)
        logger.info("🧠 DGM Agent Hierarchy initialized")
    
    async def orchestrate_evolution_cycle(self, context: AgentContext) -> Dict[str, Any]:
        """Orchestrate complete evolution cycle through all agents"""
        cycle_start = time.time()
        
        logger.info(f"🔄 Starting evolution cycle {context.generation}")
        
        try:
            # 1. Market Regime Agent analyzes current conditions
            logger.info("📊 Market regime analysis...")
            regime_analysis = await self.agents['market_regime'].execute(context)
            
            # 2. Strategy Generator creates regime-specific variants
            logger.info("🧬 Strategy generation...")
            generated_strategies = await self.agents['strategy_generator'].execute(
                context, regime_analysis, generation_count=20
            )
            
            # 3. Risk Analyzer evaluates each candidate
            logger.info("🛡️ Risk assessment...")
            risk_assessments = await self.agents['risk_analyzer'].execute(
                context, generated_strategies.get('strategies', [])
            )
            
            # 4. Performance Synthesizer combines all insights
            logger.info("⚡ Performance synthesis...")
            synthesis = await self.agents['performance_synthesizer'].execute(
                context, regime_analysis, generated_strategies, risk_assessments
            )
            
            # 5. Archive Manager updates with best performers
            logger.info("💾 Archive management...")
            archive_updates = await self.agents['archive_manager'].execute(context, synthesis)
            
            cycle_time = time.time() - cycle_start
            
            # Record orchestration results
            orchestration_result = {
                'cycle_complete': True,
                'cycle_time': cycle_time,
                'generation': context.generation,
                'regime_analysis': regime_analysis,
                'strategies_generated': generated_strategies.get('strategies_generated', 0),
                'strategies_approved': synthesis.get('strategies_approved', 0),
                'archive_size': archive_updates.get('total_in_archive', 0),
                'agent_performance': {
                    agent_name: agent.performance_metrics 
                    for agent_name, agent in self.agents.items()
                }
            }
            
            self.orchestration_history.append(orchestration_result)
            
            logger.info(f"✅ Evolution cycle {context.generation} complete in {cycle_time:.2f}s")
            return orchestration_result
            
        except Exception as e:
            logger.error(f"❌ Evolution cycle {context.generation} failed: {e}")
            return {'error': str(e), 'generation': context.generation}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            agent_name: {
                'capabilities': agent.capabilities,
                'performance_metrics': agent.performance_metrics,
                'recent_executions': len(agent.execution_history)
            }
            for agent_name, agent in self.agents.items()
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    async def demo_agent_hierarchy():
        """Demonstrate DGM agent hierarchy"""
        orchestrator = DGMAgentOrchestrator()
        
        print("🧠 DGM Agent Hierarchy Demonstration")
        print("=" * 50)
        
        # Create mock context
        context = AgentContext(
            current_regime="bull_market",
            generation=1,
            archive_summary={},
            performance_history=[],
            near_winners=[],
            compute_resources={'cpu_usage': 0.7}
        )
        
        # Run evolution cycle
        result = await orchestrator.orchestrate_evolution_cycle(context)
        
        print(f"\n📊 Evolution Cycle Results:")
        print(f"   Strategies generated: {result.get('strategies_generated', 0)}")
        print(f"   Strategies approved: {result.get('strategies_approved', 0)}")
        print(f"   Archive size: {result.get('archive_size', 0)}")
        print(f"   Cycle time: {result.get('cycle_time', 0):.2f}s")
        
        # Show agent status
        agent_status = orchestrator.get_agent_status()
        print(f"\n🤖 Agent Status:")
        for agent_name, status in agent_status.items():
            print(f"   {agent_name}: {status['recent_executions']} executions")
    
    # Run demo
    asyncio.run(demo_agent_hierarchy())