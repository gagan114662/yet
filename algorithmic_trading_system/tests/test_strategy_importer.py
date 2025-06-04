import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import random
from pathlib import Path

# Adjust import path for StrategyImporter
try:
    from algorithmic_trading_system.strategy_importer import StrategyImporter
except ImportError:
    # Fallback for local execution if paths are not set up in the environment
    import sys
    current_dir = Path(__file__).resolve().parent
    # Assuming 'tests' is a subdirectory of 'algorithmic_trading_system'
    # and 'algorithmic_trading_system' is a sibling of 'quantconnect_integration'
    # and the root of the project is one level above 'algorithmic_trading_system'
    # Project Root -> algorithmic_trading_system -> tests
    # Project Root -> algorithmic_trading_system -> strategy_importer.py
    # Project Root -> lean_workspace

    # Path to 'algorithmic_trading_system' directory
    module_dir = current_dir.parent
    # Path to project root (one level above 'algorithmic_trading_system')
    project_root = module_dir.parent

    sys.path.insert(0, str(project_root)) # Add project root to sys.path
    from algorithmic_trading_system.strategy_importer import StrategyImporter

class TestStrategyImporter(unittest.TestCase):

    def setUp(self):
        # The StrategyImporter uses a hardcoded relative path "../lean_workspace"
        # In the context of where StrategyImporter itself is (algorithmic_trading_system/),
        # this resolves to /app/lean_workspace/
        # For testing, we might need to mock os.path.join or Path resolution if
        # the actual lean_workspace is not available or if we want to isolate tests.
        # For now, we assume the default path is used by the class.
        self.importer = StrategyImporter()
        # This is the expected absolute path based on where StrategyImporter is.
        self.expected_lean_workspace_abs_path = \
            (Path(__file__).parent.parent / "../lean_workspace").resolve()


    def test_list_available_strategies(self):
        """
        Verify that list_available_strategies returns the expected list of top strategies
        and that their paths are constructed as expected.
        """
        strategies = self.importer.list_available_strategies()
        self.assertIsInstance(strategies, list)
        self.assertTrue(len(strategies) > 0)

        expected_keys = ["name", "path", "type", "complexity", "target_cagr", "target_sharpe", "description"]
        for strategy_info in strategies:
            for key in expected_keys:
                self.assertIn(key, strategy_info)

            # Verify path construction logic implicitly
            # StrategyImporter's lean_workspace_path is relative to its own location.
            # For a test running from /app/algorithmic_trading_system/tests/test_strategy_importer.py,
            # Path(self.importer.lean_workspace_path).resolve() would point to /app/lean_workspace

            # The path stored in strategy_info["path"] is just the subdirectory name.
            # The full path is constructed inside get_strategy_code.
            self.assertIsInstance(strategy_info["path"], str)
        print("\nTestStrategyImporter: test_list_available_strategies PASSED")

    def test_get_strategy_code_path_construction(self):
        """
        Test the internal logic for constructing the file path to a strategy's main.py.
        This primarily tests how os.path.join is used within get_strategy_code.
        """
        # Pick the first strategy from the hardcoded list for testing path construction
        first_strategy_info = self.importer.top_strategies[0]
        strategy_name = first_strategy_info["name"]
        strategy_sub_path = first_strategy_info["path"]

        # Expected path: <resolved_lean_workspace_path>/<strategy_sub_path>/main.py
        # self.importer.lean_workspace_path is "../lean_workspace"
        # When StrategyImporter is at /app/algorithmic_trading_system/strategy_importer.py
        # its self.lean_workspace_path points to /app/lean_workspace

        # We use a mock to intercept the open call and verify the path passed to it.
        with patch('builtins.open', mock_open(read_data="mock code")) as mocked_file:
            self.importer.get_strategy_code(strategy_name)

            # Construct the path as get_strategy_code would
            # Path to strategy_importer.py: /app/algorithmic_trading_system/strategy_importer.py
            # self.importer.lean_workspace_path = "../lean_workspace"
            # So, Path(self.importer.lean_workspace_path) relative to strategy_importer.py's dir is /app/lean_workspace

            # We need to determine the path from where StrategyImporter is located
            # Assuming StrategyImporter is in algorithmic_trading_system/
            # The StrategyImporter class uses os.path.join with its self.lean_workspace_path
            expected_path = os.path.join(self.importer.lean_workspace_path, strategy_sub_path, "main.py")

            mocked_file.assert_called_once_with(expected_path, 'r')
        print("\nTestStrategyImporter: test_get_strategy_code_path_construction PASSED")


    @patch("builtins.open", new_callable=mock_open, read_data="mock strategy code")
    def test_get_strategy_code_success(self, mock_file_open):
        """Test successfully fetching strategy code."""
        strategy_name = self.importer.top_strategies[0]["name"]
        code = self.importer.get_strategy_code(strategy_name)
        self.assertEqual(code, "mock strategy code")

        # Verify the path used by open
        strategy_info = self.importer.top_strategies[0]
        # The StrategyImporter class uses os.path.join with its self.lean_workspace_path
        expected_path = os.path.join(self.importer.lean_workspace_path, strategy_info["path"], "main.py")
        mock_file_open.assert_called_with(expected_path, 'r')
        print("\nTestStrategyImporter: test_get_strategy_code_success PASSED")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_get_strategy_code_file_not_found(self, mock_file_open):
        """Test handling when strategy file is not found; should return fallback."""
        strategy_name = self.importer.top_strategies[0]["name"]
        fallback_code = self.importer._generate_fallback_strategy()
        code = self.importer.get_strategy_code(strategy_name)
        self.assertEqual(code, fallback_code)
        print("\nTestStrategyImporter: test_get_strategy_code_file_not_found PASSED")

    def test_get_strategy_code_name_not_found(self):
        """Test handling when strategy name is not in top_strategies; should return fallback."""
        fallback_code = self.importer._generate_fallback_strategy()
        code = self.importer.get_strategy_code("NonExistentStrategyName")
        self.assertEqual(code, fallback_code)
        print("\nTestStrategyImporter: test_get_strategy_code_name_not_found PASSED")

    def test_generate_strategy_from_template(self):
        """
        Test the generation of a strategy idea from a template.
        Focuses on the dictionary manipulation and parameter randomization.
        """
        # Mock random.choice to always pick the first top strategy for predictability
        with patch.object(random, 'choice', return_value=self.importer.top_strategies[0]):
            strategy_idea = self.importer.generate_strategy_from_template()

        self.assertIsInstance(strategy_idea, dict)
        self.assertIn("name", strategy_idea)
        self.assertTrue(strategy_idea["name"].startswith(self.importer.top_strategies[0]["name"]))
        self.assertIn("_Variant_", strategy_idea["name"])

        self.assertEqual(strategy_idea["base_template"], self.importer.top_strategies[0]["name"])
        self.assertEqual(strategy_idea["type"], self.importer.top_strategies[0]["type"])

        # Check for presence of randomized parameters
        self.assertIn("lookback_period", strategy_idea)
        self.assertTrue(10 <= strategy_idea["lookback_period"] <= 50)
        self.assertIn("leverage", strategy_idea)
        self.assertTrue(1.5 <= strategy_idea["leverage"] <= 4.0)

        # Check for type-specific parameters (if the chosen template was 'leveraged_etf')
        if self.importer.top_strategies[0]["type"] == "leveraged_etf":
            self.assertIn("etf_universe", strategy_idea)

        print("\nTestStrategyImporter: test_generate_strategy_from_template PASSED")

    def test_adapt_strategy_for_targets(self):
        """Test the adaptation of strategy parameters to meet targets."""
        strategy_idea = {
            "name": "TestStrategy",
            "type": "momentum",
            "target_cagr": 0.10, # Below target
            "target_sharpe": 0.5, # Below target
            "leverage": 1.5,
            "position_size": 0.15,
            "stop_loss": 0.20,
            "rebalance_frequency": 7
        }
        adapted_idea = self.importer.adapt_strategy_for_targets(strategy_idea.copy()) # Pass a copy

        # Check CAGR related adaptations
        self.assertAlmostEqual(adapted_idea["leverage"], 1.5 * 1.2)
        self.assertAlmostEqual(adapted_idea["position_size"], min(0.15 * 1.3, 0.5))

        # Check Sharpe related adaptations
        self.assertEqual(adapted_idea["stop_loss"], 0.12) # min(0.20, 0.12)
        self.assertEqual(adapted_idea["rebalance_frequency"], 3) # min(7, 3)

        # Check new parameters
        self.assertEqual(adapted_idea["volatility_target"], 0.15)
        self.assertEqual(adapted_idea["max_portfolio_drawdown"], 0.15)
        print("\nTestStrategyImporter: test_adapt_strategy_for_targets PASSED")

    def test_get_random_high_performance_strategy(self):
        """Test the combined generation and adaptation."""
        with patch.object(random, 'choice', return_value=self.importer.top_strategies[0]):
            strategy_idea = self.importer.get_random_high_performance_strategy()

        self.assertIsInstance(strategy_idea, dict)
        self.assertTrue(strategy_idea["name"].startswith(self.importer.top_strategies[0]["name"]))
        # Check if some adaptation has occurred (e.g., volatility_target should be present)
        self.assertIn("volatility_target", strategy_idea)
        self.assertEqual(strategy_idea["volatility_target"], 0.15)
        print("\nTestStrategyImporter: test_get_random_high_performance_strategy PASSED")

if __name__ == '__main__':
    # Ensure PYTHONPATH is set correctly if running this file directly for testing
    # This is a simplified way; typically, you'd run tests with `python -m unittest discover`
    # or a test runner like pytest from the project root.

    # Assuming the test file is in /app/algorithmic_trading_system/tests/
    # We need /app/ to be in PYTHONPATH for `from algorithmic_trading_system...` to work
    # if the fallback sys.path modification in the test file itself isn't sufficient
    # or if other modules are structured expecting /app as a root.

    # This setup is primarily for the `run_in_bash_session` tool.
    # If running locally, ensure your environment's PYTHONPATH is set up or run from project root.

    if "PYTHONPATH" not in os.environ or "/app" not in os.environ["PYTHONPATH"]:
        print("Adjusting PYTHONPATH for test execution...")
        os.environ["PYTHONPATH"] = f"/app{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        # Re-import to try and catch the module if the initial import failed
        # This is a bit of a hack for direct execution; normally, test runners handle this.
        try:
            from algorithmic_trading_system.strategy_importer import StrategyImporter
        except ImportError as e:
            print(f"Re-import failed even after PYTHONPATH adjustment: {e}")
            print(f"Current sys.path: {sys.path}")
            # For the tool's execution, the earlier sys.path modification in the file should work.
            # This os.environ adjustment is more for local `python test_file.py` scenarios.

    unittest.main(verbosity=2)
