import unittest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

# Adjust import paths
try:
    from algorithmic_trading_system.ensemble_strategy_generator import EnsembleStrategyGenerator
    from algorithmic_trading_system.strategy_importer import StrategyImporter
except ImportError:
    current_dir = Path(__file__).resolve().parent
    module_dir = current_dir.parent
    project_root = module_dir.parent
    sys.path.insert(0, str(project_root))
    from algorithmic_trading_system.ensemble_strategy_generator import EnsembleStrategyGenerator
    from algorithmic_trading_system.strategy_importer import StrategyImporter


class TestEnsembleStrategyGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = EnsembleStrategyGenerator()

        self.mock_strategies_data = [
            {
                "strategy": {"name": "StratA", "type": "momentum", "indicators": ["RSI", "MACD"], "asset_classes": {"equities": 1.0}},
                "metrics": {"sharpe_ratio": 1.5, "cagr": 0.25, "max_drawdown": 0.10, "total_trades": 100}
            },
            {
                "strategy": {"name": "StratB", "type": "mean_reversion", "indicators": ["BB", "ADX"], "asset_classes": {"forex": 1.0}},
                "metrics": {"sharpe_ratio": 1.2, "cagr": 0.15, "max_drawdown": 0.08, "total_trades": 150}
            },
            {
                "strategy": {"name": "StratC", "type": "trend_following", "indicators": ["SMA", "EMA"], "asset_classes": {"commodities": 1.0}},
                "metrics": {"sharpe_ratio": 1.8, "cagr": 0.30, "max_drawdown": 0.12, "total_trades": 120}
            },
            {
                "strategy": {"name": "StratD", "type": "momentum", "indicators": ["RSI", "STOCH"], "asset_classes": {"crypto": 1.0}},
                "metrics": {"sharpe_ratio": 1.0, "cagr": 0.20, "max_drawdown": 0.15, "total_trades": 200}
            },
            {
                "strategy": {"name": "StratE", "type": "multi_factor", "indicators": ["PCA", "LDA"], "asset_classes": {"equities": 0.5, "bonds": 0.5}},
                "metrics": {"sharpe_ratio": 1.6, "cagr": 0.28, "max_drawdown": 0.09, "total_trades": 180}
            }
        ]

        # Clear the pool for each test, tests will add as needed
        self.generator.strategy_pool = []


    def test_add_strategy_to_pool(self):
        """Test if strategies are added to the pool correctly."""
        self.assertEqual(len(self.generator.strategy_pool), 0)
        strategy_data = self.mock_strategies_data[0]
        self.generator.add_strategy_to_pool(strategy_data["strategy"], strategy_data["metrics"])
        self.assertEqual(len(self.generator.strategy_pool), 1)
        self.assertEqual(self.generator.strategy_pool[0]['strategy']['name'], strategy_data['strategy']['name'])
        self.assertIn('id', self.generator.strategy_pool[0])
        print("\nTestEnsembleStrategyGenerator: test_add_strategy_to_pool PASSED")

    @patch.object(EnsembleStrategyGenerator, '_estimate_correlation')
    def test_select_uncorrelated_strategies(self, mock_estimate_correlation):
        """Test the selection of uncorrelated strategies."""
        for s_data in self.mock_strategies_data:
            self.generator.add_strategy_to_pool(s_data["strategy"], s_data["metrics"])

        def mock_corr_func(s1_data, s2_data):
            s1_name = s1_data['strategy']['name']
            s2_name = s2_data['strategy']['name']
            pairs = {frozenset({"StratC", "StratA"}): 0.7,
                     frozenset({"StratC", "StratB"}): 0.1,
                     frozenset({"StratC", "StratE"}): 0.2,
                     frozenset({"StratC", "StratD"}): 0.8,
                     frozenset({"StratE", "StratB"}): 0.3,
                    }
            return pairs.get(frozenset({s1_name, s2_name}), 0.0)

        mock_estimate_correlation.side_effect = mock_corr_func

        selected = self.generator._select_uncorrelated_strategies(target_size=3)
        selected_names = [s['strategy']['name'] for s in selected]

        self.assertIn("StratC", selected_names)
        self.assertIn("StratE", selected_names)
        self.assertIn("StratB", selected_names)
        self.assertNotIn("StratA", selected_names)
        self.assertNotIn("StratD", selected_names)
        self.assertEqual(len(selected), 3)
        print("\nTestEnsembleStrategyGenerator: test_select_uncorrelated_strategies PASSED")

    def test_calculate_equal_weights(self):
        """Test equal weight calculation."""
        selected = [self.mock_strategies_data[i] for i in range(3)]
        for s_data in selected: # Manually add to this test's generator instance
            self.generator.add_strategy_to_pool(s_data["strategy"], s_data["metrics"])

        weights = self.generator._calculate_equal_weights(self.generator.strategy_pool)
        self.assertEqual(len(weights), 3)
        for weight in weights:
            self.assertAlmostEqual(weight, 1/3)
        print("\nTestEnsembleStrategyGenerator: test_calculate_equal_weights PASSED")

    def test_calculate_risk_parity_weights(self):
        """Test risk parity weight calculation (inverse volatility)."""
        selected_data = [
            self.mock_strategies_data[0], # DD: 0.10
            self.mock_strategies_data[1], # DD: 0.08
            self.mock_strategies_data[2]  # DD: 0.12
        ]
        for s_data in selected_data: # Manually add to this test's generator instance
            self.generator.add_strategy_to_pool(s_data["strategy"], s_data["metrics"])

        weights = self.generator._calculate_risk_parity_weights(self.generator.strategy_pool)
        self.assertEqual(len(weights), 3)
        # Expect StratB (DD 0.08, pool index 1) to have highest weight
        # Expect StratA (DD 0.10, pool index 0)
        # Expect StratC (DD 0.12, pool index 2) to have lowest weight
        # Order in pool is A, B, C. So weights[1] > weights[0] > weights[2]
        self.assertTrue(weights[1] > weights[0] > weights[2])
        self.assertAlmostEqual(sum(weights), 1.0)
        print("\nTestEnsembleStrategyGenerator: test_calculate_risk_parity_weights PASSED")

    @patch('algorithmic_trading_system.ensemble_strategy_generator.minimize')
    def test_calculate_max_sharpe_weights_success(self, mock_minimize):
        """Test max Sharpe weight calculation on successful optimization."""
        for s_data in self.mock_strategies_data[:3]:
             self.generator.add_strategy_to_pool(s_data["strategy"], s_data["metrics"])

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.x = np.array([0.5, 0.3, 0.2])
        mock_minimize.return_value = mock_result

        weights = self.generator._calculate_max_sharpe_weights(self.generator.strategy_pool)

        mock_minimize.assert_called_once()
        self.assertTrue(np.array_equal(weights, mock_result.x))
        print("\nTestEnsembleStrategyGenerator: test_calculate_max_sharpe_weights_success PASSED")

    @patch('algorithmic_trading_system.ensemble_strategy_generator.minimize')
    def test_calculate_max_sharpe_weights_failure_fallback_to_equal(self, mock_minimize):
        """Test max Sharpe fallback to equal weights on optimization failure."""
        for s_data in self.mock_strategies_data[:3]:
             self.generator.add_strategy_to_pool(s_data["strategy"], s_data["metrics"])

        mock_result = MagicMock()
        mock_result.success = False
        mock_minimize.return_value = mock_result

        weights = self.generator._calculate_max_sharpe_weights(self.generator.strategy_pool)

        mock_minimize.assert_called_once()
        self.assertEqual(len(weights), 3)
        for weight in weights:
            self.assertAlmostEqual(weight, 1/3)
        print("\nTestEnsembleStrategyGenerator: test_calculate_max_sharpe_weights_failure_fallback_to_equal PASSED")

    def test_create_ensemble_definition(self):
        """Test the structure of the created ensemble definition."""
        for s_data in self.mock_strategies_data[:2]:
             self.generator.add_strategy_to_pool(s_data["strategy"], s_data["metrics"])

        selected_in_pool = self.generator.strategy_pool
        weights = np.array([0.6, 0.4])
        ensemble_def = self.generator._create_ensemble_definition(selected_in_pool, weights, "test_method")

        self.assertEqual(ensemble_def['method'], "test_method")
        self.assertEqual(len(ensemble_def['components']), 2)
        self.assertEqual(ensemble_def['components'][0]['name'], selected_in_pool[0]['strategy']['name'])
        self.assertEqual(ensemble_def['weights'][0], 0.6)
        self.assertIn('expected_metrics', ensemble_def)
        self.assertIn('risk_controls', ensemble_def)
        print("\nTestEnsembleStrategyGenerator: test_create_ensemble_definition PASSED")

    def test_generate_ensemble_code(self):
        """Test the generation of QuantConnect code for an ensemble."""
        ensemble_def = {
            'name': 'TestEnsembleCodeStrategy',
            'method': 'equal',
            'components': [
                {'name': 'StratA_ID', 'weight': 0.5, 'metrics': {}}, # Name here is component name, not strategy name
                {'name': 'StratB_ID', 'weight': 0.5, 'metrics': {}}
            ],
            'weights': [0.5, 0.5],
            'expected_metrics': {'expected_cagr': 0.20, 'expected_sharpe_ratio': 1.0},
            'risk_controls': {'rebalance_frequency': 'daily', 'drawdown_limit': 0.20}
        }

        generated_code = self.generator._generate_ensemble_code(ensemble_def)

        self.assertIn(f"class {ensemble_def['name']}(QCAlgorithm):", generated_code)
        self.assertIn(f"self.ensemble_method = \"{ensemble_def['method']}\"", generated_code)
        self.assertIn(f"self.component_weights = {ensemble_def['weights']}", generated_code)
        self.assertIn(f"# Component 0: {ensemble_def['components'][0]['name']}", generated_code)
        self.assertIn(f"'weight': {ensemble_def['components'][0]['weight']}", generated_code)
        self.assertIn("def InitializeComponents(self):", generated_code)
        self.assertIn("def RebalanceEnsemble(self):", generated_code)
        self.assertIn("def GenerateComponentSignals(self, component_index):", generated_code)
        self.assertIn("def ExecuteEnsembleSignals(self, ensemble_signals):", generated_code)
        print("\nTestEnsembleStrategyGenerator: test_generate_ensemble_code PASSED")

    @patch.object(EnsembleStrategyGenerator, '_estimate_correlation', return_value=0.1)
    @patch('algorithmic_trading_system.ensemble_strategy_generator.minimize')
    def test_generate_ensemble_strategy_full_flow(self, mock_minimize, mock_estimate_corr):
        """Test the full flow of generate_ensemble_strategy."""
        for s_data in self.mock_strategies_data[:3]: # Ensure pool has enough for min_strategies
             self.generator.add_strategy_to_pool(s_data["strategy"], s_data["metrics"])

        mock_result = MagicMock()
        mock_result.success = True
        # Ensure weights sum to 1 and match length of selected strategies (target_size=3)
        mock_result.x = np.array([0.4, 0.3, 0.3])
        mock_minimize.return_value = mock_result

        ensemble_output = self.generator.generate_ensemble(method='max_sharpe', target_size=3)

        print(f"Ensemble output in full flow test: {ensemble_output}") # DEBUG PRINT

        self.assertIsNotNone(ensemble_output, "generate_ensemble returned None unexpectedly.")
        # If ensemble_output is None, the following lines will fail.
        # This assertion helps confirm if minimize was called if output is None.
        if ensemble_output is not None: # Only assert these if output is not None
            mock_minimize.assert_called_once()
            self.assertIn('name', ensemble_output)
        self.assertTrue(ensemble_output['name'].startswith("Ensemble_max_sharpe"))
        self.assertIn('components', ensemble_output)
        self.assertEqual(len(ensemble_output['components']), 3)
        self.assertIn('weights', ensemble_output)
        self.assertEqual(len(ensemble_output['weights']), 3)
        self.assertAlmostEqual(sum(ensemble_output['weights']), 1.0)
        self.assertIn('expected_metrics', ensemble_output)
        self.assertIn('code', ensemble_output)
        self.assertTrue(ensemble_output['code'].strip().startswith("from AlgorithmImports import *"))

        mock_estimate_corr.assert_called() # This will be called regardless of minimize if selection happens
        # mock_minimize.assert_called_once() # Moved this up, conditional on ensemble_output
        print("\nTestEnsembleStrategyGenerator: test_generate_ensemble_strategy_full_flow PASSED")

if __name__ == '__main__':
    if "PYTHONPATH" not in os.environ or "/app" not in os.environ["PYTHONPATH"]:
        print("Adjusting PYTHONPATH for test execution...")
        os.environ["PYTHONPATH"] = f"/app{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    unittest.main(verbosity=2)
