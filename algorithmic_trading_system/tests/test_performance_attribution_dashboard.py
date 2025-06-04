import unittest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Adjust import paths
try:
    from algorithmic_trading_system.performance_attribution_dashboard import PerformanceAttributionDashboard, PerformanceMetrics, AttributionFactors
except ImportError:
    current_dir = Path(__file__).resolve().parent
    module_dir = current_dir.parent
    project_root = module_dir.parent
    sys.path.insert(0, str(project_root))
    from algorithmic_trading_system.performance_attribution_dashboard import PerformanceAttributionDashboard, PerformanceMetrics, AttributionFactors

class TestPerformanceAttributionDashboard(unittest.TestCase):

    def setUp(self):
        self.dashboard = PerformanceAttributionDashboard()

        self.mock_strategy_results = {
            'strategy_name': 'TestStrategy1',
            'strategy_type': 'momentum', # Used in _analyze_regime_performance
            'cagr': 0.22,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 1.8, # Explicitly provide to test pass-through
            'max_drawdown': 0.10,
            'total_return': 0.50, # Used by _generate_summary
            'win_rate': 0.60,
            'profit_factor': 2.1,
            'avg_win': 0.02,
            'avg_loss': -0.01,
            'total_trades': 150,
            'avg_trade_duration': 5,
            'total_fees': 300, # For cost analysis
            'total_slippage': 150, # For cost analysis
            'total_value': 100000, # For cost analysis (denominator)
            'volatility': 0.18, # For risk decomposition -> VaR
            'regime_aligned': True # For attribution
        }

        self.mock_trade_data_list = [
            {'symbol': 'AAPL', 'profit': 0.03, 'duration': 5, 'alpha': 0.005},
            {'symbol': 'MSFT', 'profit': -0.01, 'duration': 3, 'alpha': -0.002},
            {'symbol': 'GOOG', 'profit': 0.05, 'duration': 10, 'alpha': 0.008},
        ]
        # trade_data will be converted to DataFrame inside _analyze_trades if it's called with list

        self.mock_market_data_dict = {
            'spy_return': 0.12, # For market timing attribution
            'market_volatility': 0.15, # Not directly used by current dashboard logic but good to have
            'regime': 'bull' # For regime analysis context
        }

        # Expected structure from analyze_strategy_performance
        self.expected_top_level_keys = [
            'summary', 'attribution', 'regime_analysis', 'trade_analysis',
            'risk_decomposition', 'cost_analysis', 'failure_analysis', 'recommendations'
        ]

    def test_analyze_strategy_performance_structure(self):
        """
        Test the overall structure of the dictionary returned by analyze_strategy_performance.
        """
        print("\nTestPerformanceAttributionDashboard: test_analyze_strategy_performance_structure")
        analysis_report = self.dashboard.analyze_strategy_performance(
            self.mock_strategy_results,
            self.mock_trade_data_list,
            self.mock_market_data_dict
        )
        self.assertIsInstance(analysis_report, dict)
        for key in self.expected_top_level_keys:
            self.assertIn(key, analysis_report, f"Expected key '{key}' not found in analysis report.")
        print("TestPerformanceAttributionDashboard: test_analyze_strategy_performance_structure PASSED")

    def test_summary_stats_calculation_and_presence(self):
        """
        Test the 'summary' section, ensuring metrics are passed through or defaulted,
        and that a rating is generated.
        """
        print("\nTestPerformanceAttributionDashboard: test_summary_stats_calculation_and_presence")
        analysis_report = self.dashboard.analyze_strategy_performance(
            self.mock_strategy_results,
            self.mock_trade_data_list,
            self.mock_market_data_dict
        )
        summary = analysis_report['summary']
        self.assertIn('metrics', summary)
        self.assertIn('rating', summary)

        metrics = summary['metrics']
        self.assertEqual(metrics['cagr'], self.mock_strategy_results['cagr'])
        self.assertEqual(metrics['sharpe_ratio'], self.mock_strategy_results['sharpe_ratio'])
        self.assertEqual(metrics['sortino_ratio'], self.mock_strategy_results['sortino_ratio']) # Test pass-through
        self.assertEqual(metrics['max_drawdown'], self.mock_strategy_results['max_drawdown'])
        self.assertEqual(metrics['win_rate'], self.mock_strategy_results['win_rate'])

        # Test default sortino if not provided in input
        results_no_sortino = self.mock_strategy_results.copy()
        del results_no_sortino['sortino_ratio']
        analysis_no_sortino = self.dashboard.analyze_strategy_performance(results_no_sortino)
        self.assertAlmostEqual(
            analysis_no_sortino['summary']['metrics']['sortino_ratio'],
            results_no_sortino['sharpe_ratio'] * 1.2
        )

        self.assertIn('grade', summary['rating'])
        self.assertIn('overall', summary['rating']) # Corrected key name
        print("TestPerformanceAttributionDashboard: test_summary_stats_calculation_and_presence PASSED")

    def test_benchmark_comparison_absence(self):
        """
        Verify that the dashboard does not compute alpha/beta itself,
        as this is handled by EnhancedBacktester.
        """
        print("\nTestPerformanceAttributionDashboard: test_benchmark_comparison_absence")
        analysis_report = self.dashboard.analyze_strategy_performance(
            self.mock_strategy_results,
            self.mock_trade_data_list,
            self.mock_market_data_dict
        )
        # This dashboard focuses on attribution from given results, not recalculating benchmark stats like alpha/beta
        # It has 'market_timing' in its 'attribution' section.
        self.assertIn('attribution', analysis_report)
        self.assertIn('market_timing', analysis_report['attribution']['factors'])
        # No top-level 'performance_vs_benchmark' with alpha/beta is expected from this specific class.
        self.assertNotIn('performance_vs_benchmark', analysis_report)
        print("TestPerformanceAttributionDashboard: test_benchmark_comparison_absence PASSED")


    @patch('builtins.open', new_callable=MagicMock) # For generate_html_report if it writes to file
    @patch('algorithmic_trading_system.performance_attribution_dashboard.plt') # Mock matplotlib
    @patch('algorithmic_trading_system.performance_attribution_dashboard.sns') # Mock seaborn
    def test_html_report_generation_mocked(self, mock_seaborn, mock_matplotlib_pyplot, mock_builtin_open):
        """
        Test the generate_html_report method, mocking plotting and file I/O.
        Verifies that an HTML string is produced.
        """
        print("\nTestPerformanceAttributionDashboard: test_html_report_generation_mocked")
        # First, get an analysis report structure
        analysis_report_data = self.dashboard.analyze_strategy_performance(
            self.mock_strategy_results,
            self.mock_trade_data_list,
            self.mock_market_data_dict
        )

        html_output = self.dashboard.generate_html_report(analysis_report_data)

        self.assertIsInstance(html_output, str)
        self.assertTrue(html_output.strip().startswith("<!DOCTYPE html>"))
        self.assertIn("Strategy Performance Attribution Report", html_output)
        self.assertIn(self.mock_strategy_results['strategy_name'], html_output) # Check if some data is in the report
        self.assertIn(f"{self.mock_strategy_results['cagr']:.3f}", html_output)

        # If generate_html_report were to save the file:
        # mock_builtin_open.assert_called_once() # Or assert specific path
        # For now, it only returns a string, so no file op to check.
        print("TestPerformanceAttributionDashboard: test_html_report_generation_mocked PASSED")

if __name__ == '__main__':
    if "PYTHONPATH" not in os.environ or "/app" not in os.environ["PYTHONPATH"]:
        print("Adjusting PYTHONPATH for test execution...")
        os.environ["PYTHONPATH"] = f"/app{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    # Need to ensure pandas is available for pd.DataFrame call in _analyze_trades
    try:
        import pandas
        import numpy
    except ImportError:
        print("Missing pandas or numpy. Please install for these tests.")
        sys.exit(1)

    unittest.main(verbosity=2)
