import unittest
from unittest.mock import patch, MagicMock, call
import os
from pathlib import Path
import sys

# Adjust import paths
try:
    from algorithmic_trading_system.strategy_utils import generate_next_strategy
    from algorithmic_trading_system.strategy_importer import StrategyImporter
    from algorithmic_trading_system.backtester import Backtester
    from algorithmic_trading_system.config import INITIAL_CAPITAL, BACKTEST_START_DATE, BACKTEST_END_DATE
    from quantconnect_integration.rd_agent_qc_bridge import QuantConnectIntegration
    from algorithmic_trading_system.enhanced_backtester import EnhancedBacktester, MarketRegime
except ImportError:
    current_dir = Path(__file__).resolve().parent
    module_dir = current_dir.parent
    project_root = module_dir.parent
    sys.path.insert(0, str(project_root))
    from algorithmic_trading_system.strategy_utils import generate_next_strategy
    from algorithmic_trading_system.strategy_importer import StrategyImporter
    from algorithmic_trading_system.backtester import Backtester
    from algorithmic_trading_system.config import INITIAL_CAPITAL, BACKTEST_START_DATE, BACKTEST_END_DATE
    from quantconnect_integration.rd_agent_qc_bridge import QuantConnectIntegration
    from algorithmic_trading_system.enhanced_backtester import EnhancedBacktester, MarketRegime

class TestDataFlow(unittest.TestCase):

    def setUp(self):
        self.sample_strategy_idea_simple = {
            'name': 'TestMomentum',
            'type': 'momentum', # This will trigger _generate_default_strategy in Backtester
            'start_date': '2022,1,1',
            'end_date': '2023,1,1',
            'lookback_period': 20,
            'position_size': 0.1,
            'leverage': 1.5,
            'universe_size': 50,
            'min_price': 5.0,
            'min_volume': 100000,
            'rebalance_frequency': 5, # days
            'stop_loss': 0.10,
            'profit_target': 0.20,
            'indicator_setup': '"rsi": self.RSI(symbol, 14), "momentum": self.MOMP(symbol, self.lookback_period)',
            'signal_generation_logic': '''
indicators = self.indicators[symbol]
momentum_val = indicators["momentum"].Current.Value
if momentum_val > 0.01: trade_signal = 1
elif momentum_val < -0.01: trade_signal = -1
else: trade_signal = 0
            '''
        }

        self.sample_strategy_idea_with_template = {
            'name': 'TestTemplateBased',
            'base_template': 'QuantumEdgeDominator', # A name from StrategyImporter's list
            'type': 'multi_factor', # Should match the type of the base_template
            'start_date': '2021,1,1',
            'end_date': '2022,1,1',
            'lookback_period': 30,
            'position_size': 0.25,
            'leverage': 2.0,
            'stop_loss': 0.05,
            # Other parameters that _adapt_template_code might use
            'initial_capital': 200000
        }
        self.mock_qc_template_code = """
class MyTemplateStrategy(QCAlgorithm):
    def Initialize(self):
        # These default/hardcoded values are expected by _adapt_template_code's .replace() calls
        self.SetStartDate(2020,1,1)
        self.SetEndDate(2023,12,31)
        self.SetCash(100000) # Default initial capital in template
        self.leverage = 1.0 # Default leverage for re.sub to find
        self.position_size = 0.1 # Default position_size for re.sub
        self.stop_loss = 0.15 # Default stop_loss for re.sub
        # More template code...
    def OnData(self, data):
        pass
"""
    @patch.object(StrategyImporter, 'get_strategy_code')
    def test_strategy_idea_to_qc_code_conversion_with_template(self, mock_get_strategy_code):
        """
        Tests Backtester._generate_strategy_code_from_idea when a 'base_template' is provided.
        Mocks StrategyImporter.get_strategy_code.
        Verifies that the parameters from the strategy idea are incorporated into the template.
        """
        mock_get_strategy_code.return_value = self.mock_qc_template_code

        backtester = Backtester()
        # Temporarily disable qc_integration to force usage of _generate_strategy_code_from_idea directly for testing this part
        backtester.use_qc_integration = False

        generated_code = backtester._generate_strategy_code_from_idea(self.sample_strategy_idea_with_template)

        mock_get_strategy_code.assert_called_once_with(self.sample_strategy_idea_with_template['base_template'])

        self.assertIn("class MyTemplateStrategy(QCAlgorithm):", generated_code)
        # Dates from strategy_idea are directly used by the .replace() call that matches the template's default dates
        self.assertIn(f"self.SetStartDate({self.sample_strategy_idea_with_template['start_date']})", generated_code)
        self.assertIn(f"self.SetEndDate({self.sample_strategy_idea_with_template['end_date']})", generated_code)
        self.assertIn(f"self.SetCash({self.sample_strategy_idea_with_template['initial_capital']})", generated_code)
        self.assertIn(f"self.leverage = {self.sample_strategy_idea_with_template['leverage']}", generated_code)
        print("\nTestDataFlow: test_strategy_idea_to_qc_code_conversion_with_template PASSED")

    def test_strategy_idea_to_qc_code_conversion_no_template(self):
        """
        Tests Backtester._generate_strategy_code_from_idea for a strategy idea without a 'base_template'.
        Verifies that the generated code incorporates parameters into the default template.
        """
        backtester = Backtester()
        backtester.use_qc_integration = False # Ensure local generation path

        generated_code = backtester._generate_strategy_code_from_idea(self.sample_strategy_idea_simple)

        self.assertIn("class EnhancedStrategy(QCAlgorithm):", generated_code) # Default strategy class name
        self.assertIn(f"self.SetStartDate(2022,1,1)", generated_code)
        self.assertIn(f"self.SetEndDate(2023,1,1)", generated_code)
        self.assertIn(f"self.leverage = {self.sample_strategy_idea_simple['leverage']}", generated_code)
        self.assertIn(f"self.lookback_period = {self.sample_strategy_idea_simple['lookback_period']}", generated_code)
        self.assertIn(self.sample_strategy_idea_simple['indicator_setup'], generated_code)
        self.assertIn(self.sample_strategy_idea_simple['signal_generation_logic'], generated_code)
        print("\nTestDataFlow: test_strategy_idea_to_qc_code_conversion_no_template PASSED")

    @patch.object(QuantConnectIntegration, 'run_backtest')
    @patch.object(QuantConnectIntegration, 'generate_strategy_code')
    @patch.object(QuantConnectIntegration, 'create_lean_project')
    def test_backtester_to_qc_integration_handoff(self, mock_create_project, mock_generate_code, mock_run_backtest):
        """
        Tests the handoff from Backtester to QuantConnectIntegration when use_qc_integration is True.
        """
        mock_create_project.return_value = "/fake/project/path/TestMomentum_XYZ"
        mock_generate_code.return_value = "# Fake QC Code from QuantConnectIntegration"
        mock_run_backtest.return_value = {"cagr": 0.1, "sharpe_ratio": 1.0} # Mocked backtest results

        backtester = Backtester()
        self.assertTrue(backtester.use_qc_integration) # Assuming QuantConnectIntegration can be imported

        backtester.backtest_strategy(self.sample_strategy_idea_simple)

        mock_create_project.assert_called_once()
        # Name will have a timestamp, so check that it starts with the strategy name
        self.assertTrue(mock_create_project.call_args[1]['project_name'].startswith(self.sample_strategy_idea_simple['name']))

        mock_generate_code.assert_called_once_with(self.sample_strategy_idea_simple)
        mock_run_backtest.assert_called_once_with("# Fake QC Code from QuantConnectIntegration", "/fake/project/path/TestMomentum_XYZ")
        print("\nTestDataFlow: test_backtester_to_qc_integration_handoff PASSED")

    @patch.object(QuantConnectIntegration, 'run_backtest')
    @patch.object(QuantConnectIntegration, 'create_lean_project')
    @patch.object(EnhancedBacktester, '_generate_enhanced_strategy_code') # Mocking this internal method
    def test_enhanced_backtester_to_qc_integration_handoff(self, mock_generate_enhanced_code, mock_create_project, mock_run_backtest):
        """
        Tests the handoff from EnhancedBacktester to QuantConnectIntegration.
        Focuses on the _single_backtest method.
        """
        mock_create_project.return_value = "/fake/enhanced_project/path/TestEnhanced_XYZ"
        mock_generate_enhanced_code.return_value = "# Fake Enhanced QC Code"
        mock_run_backtest.return_value = {"cagr": 0.15, "sharpe_ratio": 1.2} # Mocked backtest results

        # Need to ensure qc_integration is an instance of QuantConnectIntegration for EnhancedBacktester
        # This should be handled by EnhancedBacktester's __init__ if QuantConnectIntegration can be imported
        try:
            enhanced_backtester = EnhancedBacktester(force_cloud=True)
        except RuntimeError as e:
            # This can happen if the real QuantConnectIntegration init fails (e.g. network or other setup)
            # For this test, we can mock the qc_integration attribute directly if needed.
            if "QuantConnect integration failed" in str(e):
                enhanced_backtester = EnhancedBacktester(force_cloud=False) # Create instance without forcing cloud
                enhanced_backtester.qc_integration = MagicMock(spec=QuantConnectIntegration)
                # Re-assign mocked methods to this specific instance's mock
                enhanced_backtester.qc_integration.create_lean_project = mock_create_project
                enhanced_backtester.qc_integration.run_backtest = mock_run_backtest
            else:
                raise e

        # If not forcing cloud, qc_integration might be None.
        # For this test, we explicitly set a mock if it's None (or if init failed)
        if enhanced_backtester.qc_integration is None:
             enhanced_backtester.qc_integration = MagicMock(spec=QuantConnectIntegration)
             enhanced_backtester.qc_integration.create_lean_project = mock_create_project
             enhanced_backtester.qc_integration.run_backtest = mock_run_backtest


        sample_idea = self.sample_strategy_idea_simple.copy()
        sample_idea['name'] = "TestEnhanced" # Give it a different name for clarity

        enhanced_backtester._single_backtest(sample_idea)

        mock_create_project.assert_called_once()
        self.assertTrue(mock_create_project.call_args[1]['project_name'].startswith(sample_idea['name']))

        mock_generate_enhanced_code.assert_called_once_with(sample_idea)
        mock_run_backtest.assert_called_once_with("# Fake Enhanced QC Code", "/fake/enhanced_project/path/TestEnhanced_XYZ")
        print("\nTestDataFlow: test_enhanced_backtester_to_qc_integration_handoff PASSED")

if __name__ == '__main__':
    if "PYTHONPATH" not in os.environ or "/app" not in os.environ["PYTHONPATH"]:
        print("Adjusting PYTHONPATH for test execution...")
        os.environ["PYTHONPATH"] = f"/app{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    unittest.main(verbosity=2)
