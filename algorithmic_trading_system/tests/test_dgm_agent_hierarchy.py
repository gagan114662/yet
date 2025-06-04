import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from pathlib import Path
import sys
import asyncio
import time
import numpy as np # numpy is used by DGM code

# Adjust import paths
try:
    from algorithmic_trading_system.dgm_agent_hierarchy import (
        AgentContext,
        MarketRegimeAgent,
        StrategyGeneratorAgent,
        RiskAnalyzerAgent,
        PerformanceSynthesizerAgent,
        ArchiveManagerAgent,
        DGMAgentOrchestrator
    )
except ImportError:
    current_dir = Path(__file__).resolve().parent
    module_dir = current_dir.parent
    project_root = module_dir.parent
    sys.path.insert(0, str(project_root))
    from algorithmic_trading_system.dgm_agent_hierarchy import (
        AgentContext,
        MarketRegimeAgent,
        StrategyGeneratorAgent,
        RiskAnalyzerAgent,
        PerformanceSynthesizerAgent,
        ArchiveManagerAgent,
        DGMAgentOrchestrator
    )

class TestMarketRegimeAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = MarketRegimeAgent()
        self.mock_context = AgentContext(
            current_regime="initial", generation=1, archive_summary={},
            performance_history=[], near_winners=[], compute_resources={}
        )

    async def test_execute_market_regime_agent(self):
        """Test MarketRegimeAgent.execute() for plausible outputs."""
        print("\nTestMarketRegimeAgent: test_execute_market_regime_agent")
        mock_market_data = {'spy_price': 120, 'vix': 15, 'momentum_20d': 0.06} # Bullish

        result = await self.agent.execute(self.mock_context, market_data=mock_market_data)

        self.assertNotIn('error', result)
        self.assertIn('current_regime', result)
        self.assertIn('regime_confidence', result)
        self.assertIn('strategy_recommendations', result)
        self.assertEqual(result['current_regime'], "bull_market")
        self.assertGreater(result['regime_confidence'], 0)
        self.assertTrue(len(self.agent.regime_history) > 0)
        print("TestMarketRegimeAgent: test_execute_market_regime_agent PASSED")

class TestStrategyGeneratorAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = StrategyGeneratorAgent()
        self.mock_context_no_archive = AgentContext(
            current_regime="bull_market", generation=1, archive_summary={}, # Empty archive_summary
            performance_history=[], near_winners=[], compute_resources={}
        )
        self.mock_context_with_archive = AgentContext(
            current_regime="bear_market", generation=2,
            archive_summary={"best_performers": [{'id': 'arch1', 'type': 'momentum', 'leverage': 1.0}]},
            performance_history=[], near_winners=[], compute_resources={}
        )
        self.mock_regime_context = {
            'current_regime': 'bull_market',
            'strategy_recommendations': {
                'preferred_types': ['momentum', 'trend'], 'leverage_range': (1.0, 2.0),
                'position_size_range': (0.1, 0.2), 'indicators': ['RSI', 'MACD']
            }
        }

    async def test_execute_no_archive(self):
        """Test strategy generation without archive input."""
        print("\nTestStrategyGeneratorAgent: test_execute_no_archive")
        generation_count = 10
        # Expected counts:
        # regime_specific: int(10 * 0.6) = 6
        # archive_based: int(10 * 0.25) = 2, but 0 because no archive_summary.best_performers
        # novel: int(10 * 0.15) = 1
        # total = 6 + 0 + 1 = 7

        # In _generate_archive_based_strategies, if not context.archive_summary or not context.archive_summary.get('best_performers'), it returns []
        # So archive_strategies will be empty.
        # The novel strategies will be generation_count - len(regime_strategies) - len(archive_strategies)
        # novel_count = int(generation_count * 0.15) which is 1
        # The current code structure:
        # regime_strategies = ... (60% = 6)
        # archive_strategies = ... (0 if no archive)
        # novel_strategies = ... (15% = 1)
        # total = sum of these.

        result = await self.agent.execute(self.mock_context_no_archive, self.mock_regime_context, generation_count=generation_count)

        expected_regime_count = int(generation_count * 0.6)
        expected_archive_count = 0 # No archive data
        expected_novel_count = int(generation_count * 0.15)
        # The way it was structured, the remainder logic for novel was not there. It was fixed percentages.
        # So total is sum of these.
        expected_total_generated = expected_regime_count + expected_archive_count + expected_novel_count


        self.assertNotIn('error', result)
        self.assertEqual(result['strategies_generated'], expected_total_generated)
        self.assertEqual(result['regime_specific'], expected_regime_count)
        self.assertEqual(result['archive_based'], expected_archive_count)
        self.assertEqual(result['novel_explorations'], expected_novel_count)
        self.assertEqual(len(result['strategies']), expected_total_generated)

        if result['strategies']:
            self.assertIn('id', result['strategies'][0])
            self.assertIn('type', result['strategies'][0])
        print("TestStrategyGeneratorAgent: test_execute_no_archive PASSED")

    async def test_execute_with_archive(self):
        """Test strategy generation with archive input."""
        print("\nTestStrategyGeneratorAgent: test_execute_with_archive")
        generation_count = 20
        expected_regime_count = int(generation_count * 0.6) # 12
        expected_archive_count = int(generation_count * 0.25) # 5
        expected_novel_count = int(generation_count * 0.15) # 3
        expected_total_generated = expected_regime_count + expected_archive_count + expected_novel_count # 20

        result = await self.agent.execute(self.mock_context_with_archive, self.mock_regime_context, generation_count=generation_count)

        self.assertNotIn('error', result)
        self.assertEqual(result['strategies_generated'], expected_total_generated)
        self.assertEqual(result['regime_specific'], expected_regime_count)
        self.assertEqual(result['archive_based'], expected_archive_count)
        self.assertEqual(result['novel_explorations'], expected_novel_count)
        self.assertEqual(len(result['strategies']), expected_total_generated)

        archive_based_count = sum(1 for s in result['strategies'] if s['creation_method'] == 'archive_based')
        self.assertEqual(archive_based_count, expected_archive_count)
        print("TestStrategyGeneratorAgent: test_execute_with_archive PASSED")


class TestRiskAnalyzerAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = RiskAnalyzerAgent()
        self.mock_context = AgentContext("bull", 1, {}, [], [], {})
        self.mock_strategies = [
            {'id': 's1', 'type': 'momentum', 'leverage': 2.0, 'position_size': 0.1, 'stop_loss': 0.1}, # Low risk
            {'id': 's2', 'type': 'experimental', 'leverage': 4.0, 'position_size': 0.4, 'stop_loss': 0.25}, # High risk
            {'id': 's3', 'type': 'trend', 'leverage': 2.5, 'position_size': 0.3, 'stop_loss': 0.02}, # Medium/High risk (tight stop loss adds risk)
        ]

    async def test_execute_risk_analysis(self):
        """Test risk analysis and approval logic."""
        print("\nTestRiskAnalyzerAgent: test_execute_risk_analysis")
        result = await self.agent.execute(self.mock_context, self.mock_strategies)

        self.assertNotIn('error', result)
        self.assertEqual(result['strategies_assessed'], 3)

        assessment_s1 = next(r for r in result['risk_assessments'] if r['strategy_id'] == 's1')
        assessment_s2 = next(r for r in result['risk_assessments'] if r['strategy_id'] == 's2')
        assessment_s3 = next(r for r in result['risk_assessments'] if r['strategy_id'] == 's3')

        self.assertEqual(assessment_s1['risk_level'], 'low')
        self.assertTrue(assessment_s1['approved'])

        self.assertEqual(assessment_s2['risk_level'], 'high') # leverage > 3 (0.3) + position_size > 0.3 (0.2) + experimental (0.2) = 0.7
        self.assertFalse(assessment_s2['approved'])

        self.assertEqual(assessment_s3['risk_level'], 'low') # stop_loss < 0.03 (0.15 risk score), overall low
        self.assertTrue(assessment_s3['approved'])

        # s1 and s3 are approved by the risk analyzer's logic
        approved_by_risk_analyzer_count = sum(1 for r in result['risk_assessments'] if r['approved'])
        self.assertEqual(approved_by_risk_analyzer_count, 2)
        print("TestRiskAnalyzerAgent: test_execute_risk_analysis PASSED")

class TestPerformanceSynthesizerAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = PerformanceSynthesizerAgent()
        self.mock_context = AgentContext("bull", 1, {}, [], [], {})
        self.mock_regime_analysis = {'current_regime': 'bull', 'regime_confidence': 0.8} # regime_confidence >= 0.8 for exploit
        self.mock_generated_strategies = {'strategies': [{'id':'s1', 'type':'momentum', 'creation_method':'regime_specific', 'regime_optimized':'bull'}], 'strategies_generated':1, 'novel_explorations':0}
        self.mock_risk_assessments = {'approved_strategies': [{'id':'s1', 'risk_level':'low'}], 'risk_insights': {'approval_rate': 1.0, 'avg_risk_score':0.2}}

    async def test_execute_synthesis(self):
        """Test performance synthesis and decision making."""
        print("\nTestPerformanceSynthesizerAgent: test_execute_synthesis")
        result = await self.agent.execute(
            self.mock_context, self.mock_regime_analysis,
            self.mock_generated_strategies, self.mock_risk_assessments
        )
        self.assertNotIn('error', result)
        self.assertTrue(result['synthesis_complete'])
        self.assertIn('evolution_decisions', result)
        self.assertIn('recommendations', result)
        self.assertEqual(result['evolution_decisions']['exploration_mode'], 'exploit')
        self.assertEqual(result['evolution_decisions']['mutation_rate'], 'increase') # approval_rate 1.0 > 0.8
        print("TestPerformanceSynthesizerAgent: test_execute_synthesis PASSED")

class TestArchiveManagerAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_context = AgentContext("bull", 1, {}, [], [], {})
        self.mock_synthesis_results_approve_some = {
            'approved_strategies': [
                {'strategy_id': 's1_approved', 'risk_level': 'low', 'approved': True, 'parent_id': None},
                {'strategy_id': 's2_not_approved', 'risk_level': 'high', 'approved': False}, # This one won't be passed to _meets_archive_criteria by loop
                {'strategy_id': 's3_approved', 'risk_level': 'medium', 'approved': True, 'parent_id': 's1_approved'},
            ]
        }
        self.mock_synthesis_results_for_compaction = {
            'approved_strategies': [ # These are already "approved"
                {'strategy_id': 's4_approved', 'risk_level': 'low', 'approved': True},
                {'strategy_id': 's5_approved', 'risk_level': 'low', 'approved': True},
                {'strategy_id': 's6_approved', 'risk_level': 'low', 'approved': True},
            ]
        }

    async def test_execute_archival_and_summary(self):
        print("\nTestArchiveManagerAgent: test_execute_archival_and_summary")
        agent = ArchiveManagerAgent(max_archive_size=5) # Ensure no compaction

        # Mock _meets_archive_criteria to control which of the "approved" strategies are archived
        def mock_meets_criteria_func(strategy_dict):
            return strategy_dict['strategy_id'] != 's2_not_approved' # s2 is already not approved

        with patch.object(agent, '_meets_archive_criteria', side_effect=mock_meets_criteria_func) as mocked_criteria_check:
            # Patch _archive_strategy to assign a fixed mock performance for predictability
            async def mock_archive_strategy(strategy_dict):
                strategy_id = strategy_dict.get('strategy_id')
                performance_map = {'s1_approved': 0.25, 's3_approved': 0.35}
                agent.archive[strategy_id] = {
                    'strategy': strategy_dict, 'archived_at': time.time(),
                    'performance_estimate': performance_map.get(strategy_id, 0.1) # Default for any others
                }

            with patch.object(agent, '_archive_strategy', new=AsyncMock(wraps=mock_archive_strategy)) as mocked_archive_strat:
                result = await agent.execute(self.mock_context, self.mock_synthesis_results_approve_some)

        self.assertNotIn('error', result)
        # The loop in agent.execute iterates over synthesis_results['approved_strategies']
        # which has 3 items. _meets_archive_criteria is called for each.
        self.assertEqual(mocked_criteria_check.call_count, 3)
        # Strategies s1_approved and s3_approved will pass mock_meets_criteria_func
        self.assertEqual(result['strategies_archived'], 2)
        self.assertEqual(result['total_in_archive'], 2)
        self.assertIn('s1_approved', agent.archive)
        self.assertIn('s3_approved', agent.archive)
        self.assertIn('archive_summary', result)
        self.assertEqual(result['archive_summary']['total_strategies'], 2)
        self.assertEqual(result['lineage_updates'], 1)
        print("TestArchiveManagerAgent: test_execute_archival_and_summary PASSED")

    async def test_execute_compaction(self):
        print("\nTestArchiveManagerAgent: test_execute_compaction")
        agent = ArchiveManagerAgent(max_archive_size=2)

        async def mock_archive_strategy_for_compaction(strategy_dict):
            strategy_id = strategy_dict.get('strategy_id')
            performance_map = {'s4_approved': 0.2, 's5_approved': 0.3, 's6_approved': 0.1}
            agent.archive[strategy_id] = {
                'strategy': strategy_dict, 'archived_at': time.time(),
                'performance_estimate': performance_map.get(strategy_id)
            }

        # Assume all approved strategies meet criteria for simplicity here
        with patch.object(agent, '_meets_archive_criteria', return_value=True):
            with patch.object(agent, '_archive_strategy', new=AsyncMock(wraps=mock_archive_strategy_for_compaction)):
                result = await agent.execute(self.mock_context, self.mock_synthesis_results_for_compaction)

        self.assertTrue(result['compaction_performed'])
        self.assertEqual(result['total_in_archive'], 2)
        self.assertIn('s5_approved', agent.archive) # Kept due to higher perf_estimate
        self.assertIn('s4_approved', agent.archive) # Kept
        self.assertNotIn('s6_approved', agent.archive) # Removed due to lower perf_estimate
        print("TestArchiveManagerAgent: test_execute_compaction PASSED")


class TestDGMAgentOrchestrator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orchestrator = DGMAgentOrchestrator()
        self.mock_context = AgentContext("bull", 1, {}, [], [], {})

    async def test_orchestrate_evolution_cycle_call_order_and_flow(self):
        print("\nTestDGMAgentOrchestrator: test_orchestrate_evolution_cycle_call_order_and_flow")

        mock_regime_output = {"current_regime": "bull", "regime_confidence": 0.9, "strategy_recommendations": {}}
        mock_gen_output = {"strategies_generated": 5, "strategies": [{'id':'s1'}]}
        mock_risk_output = {"strategies_assessed": 5, "approved_strategies": [{'id':'s1'}]}
        mock_synth_output = {"synthesis_complete": True, "evolution_decisions": {}, "recommendations": [], "strategies_approved":1}
        mock_archive_output = {"strategies_archived": 1, "total_in_archive": 1}

        # Create MagicMock instances that are also AsyncMock compatible for their execute methods
        self.orchestrator.agents['market_regime'] = MagicMock(spec=MarketRegimeAgent)
        self.orchestrator.agents['market_regime'].execute = AsyncMock(return_value=mock_regime_output)
        self.orchestrator.agents['market_regime'].performance_metrics = {} # Add missing attribute

        self.orchestrator.agents['strategy_generator'] = MagicMock(spec=StrategyGeneratorAgent)
        self.orchestrator.agents['strategy_generator'].execute = AsyncMock(return_value=mock_gen_output)
        self.orchestrator.agents['strategy_generator'].performance_metrics = {}

        self.orchestrator.agents['risk_analyzer'] = MagicMock(spec=RiskAnalyzerAgent)
        self.orchestrator.agents['risk_analyzer'].execute = AsyncMock(return_value=mock_risk_output)
        self.orchestrator.agents['risk_analyzer'].performance_metrics = {}

        self.orchestrator.agents['performance_synthesizer'] = MagicMock(spec=PerformanceSynthesizerAgent)
        self.orchestrator.agents['performance_synthesizer'].execute = AsyncMock(return_value=mock_synth_output)
        self.orchestrator.agents['performance_synthesizer'].performance_metrics = {}

        self.orchestrator.agents['archive_manager'] = MagicMock(spec=ArchiveManagerAgent)
        self.orchestrator.agents['archive_manager'].execute = AsyncMock(return_value=mock_archive_output)
        self.orchestrator.agents['archive_manager'].performance_metrics = {}

        result = await self.orchestrator.orchestrate_evolution_cycle(self.mock_context)

        self.assertTrue(result['cycle_complete'])
        self.orchestrator.agents['market_regime'].execute.assert_called_once_with(self.mock_context)
        self.orchestrator.agents['strategy_generator'].execute.assert_called_once_with(self.mock_context, mock_regime_output, generation_count=20)
        self.orchestrator.agents['risk_analyzer'].execute.assert_called_once_with(self.mock_context, mock_gen_output.get('strategies', []))
        self.orchestrator.agents['performance_synthesizer'].execute.assert_called_once_with(self.mock_context, mock_regime_output, mock_gen_output, mock_risk_output)
        self.orchestrator.agents['archive_manager'].execute.assert_called_once_with(self.mock_context, mock_synth_output)

        self.assertEqual(result['strategies_generated'], 5)
        self.assertEqual(result['strategies_approved'], 1)
        self.assertEqual(result['archive_size'], 1)
        print("TestDGMAgentOrchestrator: test_orchestrate_evolution_cycle_call_order_and_flow PASSED")

if __name__ == '__main__':
    if "PYTHONPATH" not in os.environ or "/app" not in os.environ["PYTHONPATH"]:
        print("Adjusting PYTHONPATH for test execution...")
        os.environ["PYTHONPATH"] = f"/app{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    try:
        import numpy
    except ImportError:
        print("Missing numpy. Please install for these tests.")
        sys.exit(1)

    unittest.main(verbosity=2)
