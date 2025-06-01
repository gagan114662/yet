#!/usr/bin/env python3
"""
Main Pipeline: RD-Agent + QuantConnect Integration

This script orchestrates the automated strategy development and backtesting pipeline using:
1. RD-Agent for AI-powered strategy generation
2. QuantConnect Lean for backtesting
3. OpenRouter API with DeepSeek R1 model
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add RD-Agent to path
sys.path.append(str(Path(__file__).parent.parent / "RD-Agent"))

from rd_agent_qc_bridge import QuantConnectIntegration
from rdagent.app.data_mining.conf import MED_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.scenario import Scenario
from rdagent.core.proposal import Hypothesis
from rdagent.log import rdagent_logger as logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class QuantConnectScenario(Scenario):
    """Custom scenario for QuantConnect strategy development"""
    
    def __init__(self):
        super().__init__()
        self.target_metrics = {
            "cagr": 0.25,  # 25% CAGR
            "sharpe_ratio": 1.0,  # Sharpe > 1
            "max_drawdown": 0.20,  # Max DD < 20%
            "avg_profit": 0.0075  # Average profit > 0.75%
        }
        self._rich_style_description = "Trading strategy development using QuantConnect"
        
    def background(self, tag=None) -> str:
        return """
You are developing automated trading strategies for the US stock market using QuantConnect.
Your goal is to create strategies that meet the following performance criteria:
- CAGR (Compound Annual Growth Rate) > 25%
- Sharpe Ratio > 1.0 (with 5% risk-free rate)
- Maximum Drawdown < 20%
- Average Profit per Trade > 0.75%

Focus on:
1. Momentum and trend-following strategies
2. Mean reversion strategies
3. Factor-based strategies (value, quality, low volatility)
4. Machine learning-based signals
5. Risk management and position sizing

Use real market data and ensure strategies are robust across different market conditions.
"""
    
    def get_source_data_desc(self) -> str:
        return """
Available data sources:
- US Equity price and volume data (daily resolution)
- Fundamental data (P/E, P/B, market cap, etc.)
- Technical indicators (SMA, EMA, RSI, MACD, etc.)
- Market microstructure data
- Economic indicators

Universe: US stocks listed on major exchanges (NYSE, NASDAQ)
Time period: 2010-2023 for backtesting
"""
    
    def output_format(self, tag=None) -> str:
        return """
Output a trading strategy with:
1. Clear entry and exit signals
2. Position sizing rules
3. Risk management parameters
4. Universe selection criteria
5. Rebalancing frequency
"""
    
    def interface(self, tag=None) -> str:
        return """
Strategy should be implemented as a QuantConnect algorithm with:
- Initialize() method for setup
- CoarseSelectionFunction() for universe selection
- OnData() or scheduled methods for trading logic
- Proper risk management
"""
    
    def simulator(self, tag=None) -> str:
        return "QuantConnect Lean backtesting engine"
    
    def get_scenario_all_desc(self, tag=None) -> str:
        """Get all scenario descriptions"""
        return f"""
Background:
{self.background(tag)}

Data Sources:
{self.get_source_data_desc()}

Output Format:
{self.output_format(tag)}

Interface:
{self.interface(tag)}

Simulator:
{self.simulator(tag)}
"""
    
    @property
    def rich_style_description(self) -> str:
        """Get rich style description"""
        return self._rich_style_description


class StrategyDevelopmentLoop:
    """Main loop for automated strategy development"""
    
    def __init__(self):
        self.qc_integration = QuantConnectIntegration()
        self.scenario = QuantConnectScenario()
        self.strategies_tested = []
        self.successful_strategies = []
        
    def generate_strategy_hypothesis(self, iteration: int) -> Dict[str, Any]:
        """Generate a new strategy hypothesis using RD-Agent"""
        
        # Define strategy templates
        strategy_templates = [
            {
                "type": "momentum",
                "lookback_periods": [10, 20, 30, 60],
                "indicators": ["MOMP", "ROC", "RSI"],
                "rebalance_days": [1, 5, 10, 20]
            },
            {
                "type": "mean_reversion",
                "lookback_periods": [5, 10, 20],
                "indicators": ["BB", "RSI", "ZSCORE"],
                "rebalance_days": [1, 3, 5]
            },
            {
                "type": "factor",
                "factors": ["value", "quality", "low_vol", "size"],
                "rebalance_days": [20, 60, 90]
            },
            {
                "type": "ml_based",
                "features": ["price_momentum", "volume_profile", "fundamental_ratios"],
                "models": ["linear", "tree", "ensemble"],
                "rebalance_days": [5, 10, 20]
            }
        ]
        
        # Select strategy type based on iteration
        template = strategy_templates[iteration % len(strategy_templates)]
        
        # Generate specific parameters
        if template["type"] == "momentum":
            lookback = template["lookback_periods"][iteration % len(template["lookback_periods"])]
            indicator = template["indicators"][iteration % len(template["indicators"])]
            rebalance = template["rebalance_days"][iteration % len(template["rebalance_days"])]
            
            strategy_idea = {
                "name": f"Momentum_{indicator}_{lookback}D_v{iteration}",
                "description": f"Momentum strategy using {indicator} with {lookback}-day lookback",
                "type": "momentum",
                "lookback_period": lookback,
                "rebalance_frequency": rebalance,
                "position_size": 0.1,
                "universe_size": 50,
                "min_price": 10,
                "min_volume": 5000000,
                "indicator_setup": f'"{indicator.lower()}": self.{indicator}(symbol, {lookback})',
                "signal_generation_logic": f'''
        indicators = self.indicators[symbol]
        {indicator.lower()}_value = indicators["{indicator.lower()}"].Current.Value
        
        # Dynamic thresholds based on market conditions
        threshold = 0.02 if self.Portfolio.TotalPortfolioValue > self.StartingCapital * 1.1 else 0.015
        
        if {indicator.lower()}_value > threshold:
            signal = 1
        elif {indicator.lower()}_value < -threshold:
            signal = -1
        else:
            signal = 0
        '''
            }
            
        elif template["type"] == "mean_reversion":
            lookback = template["lookback_periods"][iteration % len(template["lookback_periods"])]
            indicator = template["indicators"][iteration % len(template["indicators"])]
            rebalance = template["rebalance_days"][iteration % len(template["rebalance_days"])]
            
            strategy_idea = {
                "name": f"MeanReversion_{indicator}_{lookback}D_v{iteration}",
                "description": f"Mean reversion strategy using {indicator}",
                "type": "mean_reversion",
                "lookback_period": lookback,
                "rebalance_frequency": rebalance,
                "position_size": 0.05,
                "universe_size": 100,
                "min_price": 5,
                "min_volume": 2000000,
                "indicator_setup": self._get_mean_reversion_indicators(indicator, lookback),
                "signal_generation_logic": self._get_mean_reversion_logic(indicator)
            }
            
        elif template["type"] == "factor":
            factor = template["factors"][iteration % len(template["factors"])]
            rebalance = template["rebalance_days"][iteration % len(template["rebalance_days"])]
            
            strategy_idea = {
                "name": f"Factor_{factor}_v{iteration}",
                "description": f"Factor-based strategy focusing on {factor}",
                "type": "factor",
                "lookback_period": 60,
                "rebalance_frequency": rebalance,
                "position_size": 0.02,
                "universe_size": 200,
                "min_price": 10,
                "min_volume": 10000000,
                "indicator_setup": self._get_factor_indicators(factor),
                "signal_generation_logic": self._get_factor_logic(factor)
            }
            
        else:  # ml_based
            features = template["features"]
            model = template["models"][iteration % len(template["models"])]
            rebalance = template["rebalance_days"][iteration % len(template["rebalance_days"])]
            
            strategy_idea = {
                "name": f"ML_{model}_v{iteration}",
                "description": f"ML-based strategy using {model} model",
                "type": "ml_based",
                "lookback_period": 30,
                "rebalance_frequency": rebalance,
                "position_size": 0.05,
                "universe_size": 75,
                "min_price": 15,
                "min_volume": 5000000,
                "indicator_setup": self._get_ml_indicators(features),
                "signal_generation_logic": self._get_ml_logic(model)
            }
            
        # Add common parameters
        strategy_idea.update({
            "start_date": "2018,1,1",
            "end_date": "2023,12,31",
            "schedule_rule": "EveryDay" if rebalance == 1 else f"Every({rebalance} * Days)",
            "max_dd_per_security": 0.10,
            "holding_period": rebalance * 2
        })
        
        return strategy_idea
        
    def _get_mean_reversion_indicators(self, indicator: str, lookback: int) -> str:
        if indicator == "BB":
            return f'"bb": self.BB(symbol, {lookback}, 2), "sma": self.SMA(symbol, {lookback})'
        elif indicator == "RSI":
            return f'"rsi": self.RSI(symbol, {lookback}), "sma": self.SMA(symbol, {lookback})'
        else:  # ZSCORE
            return f'"sma": self.SMA(symbol, {lookback}), "std": self.STD(symbol, {lookback})'
            
    def _get_mean_reversion_logic(self, indicator: str) -> str:
        if indicator == "BB":
            return '''
        indicators = self.indicators[symbol]
        price = self.Securities[symbol].Price
        upper_band = indicators["bb"].UpperBand.Current.Value
        lower_band = indicators["bb"].LowerBand.Current.Value
        
        if price < lower_band:
            signal = 1  # Oversold - Buy
        elif price > upper_band:
            signal = -1  # Overbought - Sell
        else:
            signal = 0
        '''
        elif indicator == "RSI":
            return '''
        indicators = self.indicators[symbol]
        rsi = indicators["rsi"].Current.Value
        
        if rsi < 30:
            signal = 1  # Oversold
        elif rsi > 70:
            signal = -1  # Overbought
        else:
            signal = 0
        '''
        else:  # ZSCORE
            return '''
        indicators = self.indicators[symbol]
        price = self.Securities[symbol].Price
        sma = indicators["sma"].Current.Value
        std = indicators["std"].Current.Value
        
        if std > 0:
            zscore = (price - sma) / std
            if zscore < -2:
                signal = 1
            elif zscore > 2:
                signal = -1
            else:
                signal = 0
        else:
            signal = 0
        '''
            
    def _get_factor_indicators(self, factor: str) -> str:
        if factor == "value":
            return '"pe_ratio": self.PE(symbol), "pb_ratio": self.PB(symbol)'
        elif factor == "quality":
            return '"roe": self.ROE(symbol), "roa": self.ROA(symbol)'
        elif factor == "low_vol":
            return '"volatility": self.STD(symbol, 60), "beta": self.BETA(symbol, "SPY", 60)'
        else:  # size
            return '"market_cap": self.MarketCap(symbol), "volume": self.V(symbol, 20)'
            
    def _get_factor_logic(self, factor: str) -> str:
        if factor == "value":
            return '''
        # Value factor - look for low P/E and P/B ratios
        fundamentals = self.Securities[symbol].Fundamentals
        if fundamentals:
            pe_ratio = fundamentals.ValuationRatios.PERatio
            pb_ratio = fundamentals.ValuationRatios.PBRatio
            
            if pe_ratio > 0 and pe_ratio < 15 and pb_ratio > 0 and pb_ratio < 2:
                signal = 1
            else:
                signal = 0
        else:
            signal = 0
        '''
        elif factor == "quality":
            return '''
        # Quality factor - high ROE/ROA
        fundamentals = self.Securities[symbol].Fundamentals
        if fundamentals:
            roe = fundamentals.OperationRatios.ROE
            roa = fundamentals.OperationRatios.ROA
            
            if roe > 0.15 and roa > 0.10:
                signal = 1
            else:
                signal = 0
        else:
            signal = 0
        '''
        elif factor == "low_vol":
            return '''
        # Low volatility factor
        indicators = self.indicators[symbol]
        volatility = indicators["volatility"].Current.Value
        
        # Select stocks with volatility in bottom quartile
        if volatility > 0 and volatility < 0.20:  # Annual vol < 20%
            signal = 1
        else:
            signal = 0
        '''
        else:  # size
            return '''
        # Size factor - focus on mid-cap stocks
        fundamentals = self.Securities[symbol].Fundamentals
        if fundamentals:
            market_cap = fundamentals.MarketCap
            
            if market_cap > 2e9 and market_cap < 10e9:  # $2B - $10B
                signal = 1
            else:
                signal = 0
        else:
            signal = 0
        '''
            
    def _get_ml_indicators(self, features: List[str]) -> str:
        indicators = []
        if "price_momentum" in features:
            indicators.append('"momentum_10": self.MOMP(symbol, 10)')
            indicators.append('"momentum_30": self.MOMP(symbol, 30)')
        if "volume_profile" in features:
            indicators.append('"volume_ratio": self.V(symbol, 5) / self.V(symbol, 20)')
        if "fundamental_ratios" in features:
            indicators.append('"pe_ratio": self.PE(symbol)')
            
        return ", ".join(indicators)
        
    def _get_ml_logic(self, model: str) -> str:
        if model == "linear":
            return '''
        # Simple linear combination of features
        indicators = self.indicators[symbol]
        
        score = 0
        if "momentum_10" in indicators:
            score += indicators["momentum_10"].Current.Value * 0.3
        if "momentum_30" in indicators:
            score += indicators["momentum_30"].Current.Value * 0.2
            
        if score > 0.01:
            signal = 1
        elif score < -0.01:
            signal = -1
        else:
            signal = 0
        '''
        else:
            return '''
        # Placeholder for more complex ML models
        # In practice, this would use pre-trained models
        indicators = self.indicators[symbol]
        
        # Simple rule-based approximation
        momentum = indicators.get("momentum_10", {"Current": {"Value": 0}})["Current"]["Value"]
        
        if momentum > 0.02:
            signal = 1
        elif momentum < -0.02:
            signal = -1
        else:
            signal = 0
        '''
            
    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run a single iteration of strategy development and testing"""
        
        logger.info(f"Starting iteration {iteration}")
        
        # Generate strategy hypothesis
        strategy_idea = self.generate_strategy_hypothesis(iteration)
        logger.info(f"Generated strategy: {strategy_idea['name']}")
        
        # Create project
        project_path = self.qc_integration.create_lean_project()
        
        # Generate strategy code
        strategy_code = self.qc_integration.generate_strategy_code(strategy_idea)
        
        # Save strategy code for reference
        with open(f"{project_path}/strategy_code.txt", 'w') as f:
            f.write(strategy_code)
            
        # Run backtest
        logger.info("Running backtest...")
        metrics = self.qc_integration.run_backtest(strategy_code, project_path)
        
        if "error" in metrics:
            logger.error(f"Backtest failed: {metrics['error']}")
            return {
                "iteration": iteration,
                "strategy": strategy_idea,
                "status": "failed",
                "error": metrics["error"]
            }
            
        # Evaluate performance
        evaluation = self.qc_integration.evaluate_performance(metrics)
        
        # Generate report
        report = self.qc_integration.generate_report(strategy_idea, metrics, evaluation)
        
        # Save report
        with open(f"{project_path}/report.md", 'w') as f:
            f.write(report)
            
        # Store results
        result = {
            "iteration": iteration,
            "strategy": strategy_idea,
            "metrics": metrics,
            "evaluation": evaluation,
            "report": report,
            "project_path": project_path,
            "status": "success" if evaluation["meets_criteria"] else "below_target"
        }
        
        self.strategies_tested.append(result)
        
        if evaluation["meets_criteria"]:
            self.successful_strategies.append(result)
            logger.info(f"✅ Strategy {strategy_idea['name']} meets all criteria!")
        else:
            logger.info(f"❌ Strategy {strategy_idea['name']} does not meet criteria")
            
        return result
        
    def run_pipeline(self, max_iterations: int = 50):
        """Run the full pipeline for multiple iterations"""
        
        logger.info(f"Starting automated strategy development pipeline")
        logger.info(f"Target metrics: {self.scenario.target_metrics}")
        logger.info(f"Maximum iterations: {max_iterations}")
        
        start_time = time.time()
        
        for i in range(max_iterations):
            try:
                result = self.run_iteration(i)
                
                # Log progress
                logger.info(f"Completed iteration {i + 1}/{max_iterations}")
                logger.info(f"Successful strategies found: {len(self.successful_strategies)}")
                
                # Save intermediate results
                self.save_results()
                
                # Early stopping if we have enough successful strategies
                if len(self.successful_strategies) >= 5:
                    logger.info("Found 5 successful strategies. Stopping early.")
                    break
                    
                # Add delay to avoid overwhelming the system
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in iteration {i}: {str(e)}")
                continue
                
        elapsed_time = time.time() - start_time
        
        # Generate final report
        self.generate_final_report(elapsed_time)
        
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total strategies tested: {len(self.strategies_tested)}")
        logger.info(f"Successful strategies: {len(self.successful_strategies)}")
        
    def save_results(self):
        """Save intermediate results to file"""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "strategies_tested": len(self.strategies_tested),
            "successful_strategies": len(self.successful_strategies),
            "all_results": self.strategies_tested
        }
        
        with open("strategy_development_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
    def generate_final_report(self, elapsed_time: float):
        """Generate comprehensive final report"""
        
        report = f"""
# Automated Strategy Development Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Runtime: {elapsed_time:.2f} seconds

## Summary
- **Total Strategies Tested**: {len(self.strategies_tested)}
- **Successful Strategies**: {len(self.successful_strategies)}
- **Success Rate**: {len(self.successful_strategies) / len(self.strategies_tested) * 100:.1f}%

## Target Metrics
- CAGR > 25%
- Sharpe Ratio > 1.0
- Max Drawdown < 20%
- Average Profit > 0.75%

"""
        
        if self.successful_strategies:
            report += "## Successful Strategies\n\n"
            for i, strategy in enumerate(self.successful_strategies, 1):
                metrics = strategy["metrics"]
                report += f"""
### {i}. {strategy['strategy']['name']}
- **Type**: {strategy['strategy']['type']}
- **CAGR**: {metrics['cagr']:.2%}
- **Sharpe Ratio**: {metrics['sharpe_ratio']:.2f}
- **Max Drawdown**: {metrics['max_drawdown']:.2%}
- **Win Rate**: {metrics['win_rate']:.2%}
- **Total Trades**: {metrics['total_trades']}

"""
        
        # Add performance distribution
        if self.strategies_tested:
            report += "## Performance Distribution\n\n"
            
            cagrs = [s["metrics"]["cagr"] for s in self.strategies_tested if "metrics" in s and "error" not in s["metrics"]]
            sharpes = [s["metrics"]["sharpe_ratio"] for s in self.strategies_tested if "metrics" in s and "error" not in s["metrics"]]
            
            if cagrs:
                report += f"- **Average CAGR**: {sum(cagrs) / len(cagrs):.2%}\n"
                report += f"- **Best CAGR**: {max(cagrs):.2%}\n"
                report += f"- **Worst CAGR**: {min(cagrs):.2%}\n\n"
                
            if sharpes:
                report += f"- **Average Sharpe**: {sum(sharpes) / len(sharpes):.2f}\n"
                report += f"- **Best Sharpe**: {max(sharpes):.2f}\n"
                report += f"- **Worst Sharpe**: {min(sharpes):.2f}\n"
                
        with open("final_strategy_report.md", "w") as f:
            f.write(report)
            
        logger.info("Final report saved to final_strategy_report.md")


def main():
    """Main entry point"""
    
    # Initialize pipeline
    pipeline = StrategyDevelopmentLoop()
    
    # Run pipeline
    pipeline.run_pipeline(max_iterations=20)
    
    # Return results
    return {
        "strategies_tested": len(pipeline.strategies_tested),
        "successful_strategies": len(pipeline.successful_strategies),
        "results": pipeline.successful_strategies
    }


if __name__ == "__main__":
    results = main()
    print(f"\nPipeline completed!")
    print(f"Strategies tested: {results['strategies_tested']}")
    print(f"Successful strategies: {results['successful_strategies']}")