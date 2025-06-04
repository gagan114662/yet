import unittest
from unittest.mock import patch
import os
from pathlib import Path
import sys
import numpy as np # For np.mean if used by any tested method indirectly, or for test data generation

# Adjust import paths
try:
    from algorithmic_trading_system.market_regime_detector import MarketRegimeDetector, MarketRegime
except ImportError:
    current_dir = Path(__file__).resolve().parent
    module_dir = current_dir.parent
    project_root = module_dir.parent
    sys.path.insert(0, str(project_root))
    from algorithmic_trading_system.market_regime_detector import MarketRegimeDetector, MarketRegime

class TestMarketRegimeDetector(unittest.TestCase):

    def setUp(self):
        self.detector = MarketRegimeDetector()

    def test_get_regime_strategy_params_logic(self):
        """
        Test that get_regime_strategy_params returns the correct, predefined parameters
        for each regime in the MarketRegime enum.
        """
        print("\nTestMarketRegimeDetector: test_get_regime_strategy_params_logic")
        for regime in MarketRegime:
            with self.subTest(regime=regime):
                params = self.detector.get_regime_strategy_params(regime)
                self.assertIsNotNone(params, f"Parameters for {regime.value} should not be None")
                self.assertIsInstance(params, dict, f"Parameters for {regime.value} should be a dict")
                # Check against the internally stored strategies
                expected_params = self.detector.regime_strategies.get(regime)
                self.assertEqual(params, expected_params, f"Parameters for {regime.value} do not match expected")
        print("TestMarketRegimeDetector: test_get_regime_strategy_params_logic PASSED")


    def test_detect_regime_logic_scenarios(self):
        """
        Test detect_regime with various market data scenarios to ensure it classifies
        them into plausible regimes. Focuses on clear-cut scenarios.
        """
        print("\nTestMarketRegimeDetector: test_detect_regime_logic_scenarios")
        scenarios = {
            "strong_bull": {
                "market_data": {
                    'spy_price': 110, 'spy_sma_50': 105, 'spy_sma_200': 100, # Strong uptrend
                    'vix': 10, 'vix_sma_20': 12, # Low volatility
                    'momentum_20d': 0.06, # Strong positive momentum
                    'advance_decline_ratio': 2.5, 'new_highs_lows_ratio': 3.5, 'percent_above_200ma': 80, # Strong breadth
                    'put_call_ratio': 0.7, 'fear_greed_index': 85, # Greed
                    'yield_curve_10y2y': 2.0, 'gdp_growth': 3.5, 'unemployment_trend': -0.1, 'inflation_rate': 2.0 # Strong economy
                },
                "expected_dominant_regimes": [MarketRegime.STRONG_BULL, MarketRegime.BULL] # Expect one of these to be dominant
            },
            "strong_bear_crash": {
                "market_data": {
                    'spy_price': 90, 'spy_sma_50': 95, 'spy_sma_200': 100, # Strong downtrend
                    'vix': 45, 'vix_sma_20': 35, # Extreme volatility
                    'momentum_20d': -0.08, # Strong negative momentum
                    'advance_decline_ratio': 0.2, 'new_highs_lows_ratio': 0.1, 'percent_above_200ma': 10, # Weak breadth
                    'put_call_ratio': 1.8, 'fear_greed_index': 10, # Extreme fear
                    'yield_curve_10y2y': -0.5, 'gdp_growth': -1.0, 'unemployment_trend': 0.2, 'inflation_rate': 1.0 # Weak economy
                },
                "expected_dominant_regimes": [MarketRegime.STRONG_BEAR, MarketRegime.CRASH, MarketRegime.HIGH_VOLATILITY, MarketRegime.BEAR]
            },
            "sideways_low_vol": {
                "market_data": {
                    'spy_price': 100, 'spy_sma_50': 99.8, 'spy_sma_200': 100.2, # Sideways trend
                    'vix': 10, 'vix_sma_20': 11, # Low volatility
                    'momentum_20d': 0.001, # No momentum
                    'advance_decline_ratio': 1.0, 'new_highs_lows_ratio': 1.0, 'percent_above_200ma': 50, # Neutral breadth
                    'put_call_ratio': 1.0, 'fear_greed_index': 50, # Neutral sentiment
                    'yield_curve_10y2y': 0.5, 'gdp_growth': 1.5, 'unemployment_trend': 0.0, 'inflation_rate': 2.0 # Stable economy
                },
                "expected_dominant_regimes": [MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY]
            },
             "high_vol_bearish_bias": {
                "market_data": {
                    'spy_price': 98, 'spy_sma_50': 100, 'spy_sma_200': 102, # Slight downtrend
                    'vix': 35, 'vix_sma_20': 28, # High volatility
                    'momentum_20d': -0.01, # Slight negative momentum
                    'advance_decline_ratio': 0.8, 'new_highs_lows_ratio': 0.7, 'percent_above_200ma': 40, # Weakening breadth
                    'put_call_ratio': 1.3, 'fear_greed_index': 30, # Fear
                    'yield_curve_10y2y': 0.2, 'gdp_growth': 0.5, 'unemployment_trend': 0.1, 'inflation_rate': 3.0 # Weakening economy
                },
                "expected_dominant_regimes": [MarketRegime.HIGH_VOLATILITY, MarketRegime.BEAR, MarketRegime.WEAK_BEAR]
            }
        }

        for scenario_name, data in scenarios.items():
            with self.subTest(scenario=scenario_name):
                regime, confidence = self.detector.detect_regime(data["market_data"])
                self.assertIn(regime, data["expected_dominant_regimes"],
                              f"For {scenario_name}, detected {regime.value} not in expected {data['expected_dominant_regimes']}")
                self.assertTrue(0 <= confidence <= 1.0, f"Confidence {confidence} out of bounds for {scenario_name}")
        print("TestMarketRegimeDetector: test_detect_regime_logic_scenarios PASSED")

    def test_detect_and_get_params_integration(self):
        """
        Test the integration of detect_regime and get_regime_strategy_params.
        """
        print("\nTestMarketRegimeDetector: test_detect_and_get_params_integration")
        # Using the "strong_bull" scenario data
        market_data_bull = {
            'spy_price': 110, 'spy_sma_50': 105, 'spy_sma_200': 100,
            'vix': 10, 'vix_sma_20': 12, 'momentum_20d': 0.06,
            'advance_decline_ratio': 2.5, 'new_highs_lows_ratio': 3.5, 'percent_above_200ma': 80,
            'put_call_ratio': 0.7, 'fear_greed_index': 85,
            'yield_curve_10y2y': 2.0, 'gdp_growth': 3.5, 'unemployment_trend': -0.1, 'inflation_rate': 2.0
        }

        detected_regime, _ = self.detector.detect_regime(market_data_bull)

        # Now get params for this detected regime
        params_for_detected_regime = self.detector.get_regime_strategy_params(detected_regime)

        # Get expected params directly from the detector's storage for the detected regime
        expected_params = self.detector.regime_strategies.get(detected_regime)

        self.assertIsNotNone(params_for_detected_regime, f"Params for detected regime {detected_regime.value} should not be None")
        self.assertEqual(params_for_detected_regime, expected_params,
                         f"Parameters for detected regime {detected_regime.value} do not match expected")

        # Verify that if no regime is passed to get_regime_strategy_params, it uses the last detected one
        params_for_current_regime = self.detector.get_regime_strategy_params() # No argument
        self.assertEqual(params_for_current_regime, expected_params,
                         "Parameters for current (last detected) regime do not match expected")
        print("TestMarketRegimeDetector: test_detect_and_get_params_integration PASSED")


if __name__ == '__main__':
    if "PYTHONPATH" not in os.environ or "/app" not in os.environ["PYTHONPATH"]:
        print("Adjusting PYTHONPATH for test execution...")
        os.environ["PYTHONPATH"] = f"/app{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    unittest.main(verbosity=2)
