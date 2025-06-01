"""
QuantConnect - RD-Agent Integration Bridge

This module connects RD-Agent's AI-powered strategy generation with QuantConnect's backtesting engine.
It automates the development and testing of trading strategies based on target performance metrics.
"""

import os
import json
import subprocess
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import pandas as pd

class QuantConnectIntegration:
    def __init__(self, 
                 user_id: str = "357130",
                 api_token: str = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912",
                 target_metrics: Dict[str, float] = None):
        self.user_id = user_id
        self.api_token = api_token
        self.target_metrics = target_metrics or {
            "cagr": 0.25,  # 25% CAGR
            "sharpe_ratio": 1.0,  # Sharpe > 1
            "max_drawdown": 0.20,  # Max DD < 20%
            "avg_profit": 0.0075  # Average profit > 0.75%
        }
        self.project_name = None
        
    def create_lean_project(self, project_path: str = None) -> str:
        """Create a new Lean project for backtesting"""
        if not self.project_name:
            self.project_name = f"rd_agent_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
        if not project_path:
            project_path = f"./{self.project_name}"
            
        # Create project in current directory
        cmd = f"lean create-project '{self.project_name}' --language python"
        subprocess.run(cmd, shell=True, check=True, cwd=".")
        
        return project_path
        
    def generate_strategy_code(self, strategy_idea: Dict[str, Any]) -> str:
        """Convert strategy idea from RD-Agent into QuantConnect algorithm code"""
        
        strategy_template = '''
from AlgorithmImports import *
import numpy as np
import pandas as pd

class RDAgentStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({start_year}, {start_month}, {start_day})
        self.SetEndDate({end_year}, {end_month}, {end_day})
        self.SetCash(100000)
        
        # Strategy parameters from RD-Agent
        self.lookback_period = {lookback_period}
        self.rebalance_frequency = {rebalance_frequency}
        self.position_size = {position_size}
        
        # Add universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.{schedule_rule}(), 
                        self.TimeRules.AfterMarketOpen("SPY", 10),
                        self.Rebalance)
        
        # Risk management
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity({max_dd_per_security}))
        
        # Initialize indicators and variables
        self.symbols = []
        self.indicators = {{}}
        
    def CoarseSelectionFunction(self, coarse):
        """Select universe based on RD-Agent criteria"""
        # Filter by price and volume
        filtered = [x for x in coarse if x.HasFundamentalData 
                    and x.Price > {min_price} 
                    and x.DollarVolume > {min_volume}]
        
        # Sort by selection criteria
        sorted_stocks = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        
        # Return top N stocks
        return [x.Symbol for x in sorted_stocks[:{universe_size}]]
        
    def OnSecuritiesChanged(self, changes):
        """Handle universe changes"""
        # Remove indicators for removed securities
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.indicators:
                del self.indicators[symbol]
                
        # Add indicators for new securities
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            self.indicators[symbol] = {{
                {indicator_setup}
            }}
            
    def Rebalance(self):
        """Execute trading logic based on RD-Agent strategy"""
        insights = []
        
        for symbol in self.indicators:
            if not self.IsWarmingUp:
                # Generate trading signal
                signal = self.GenerateSignal(symbol)
                
                if signal > 0:
                    insights.append(Insight.Price(symbol, timedelta(days={holding_period}), InsightDirection.Up))
                elif signal < 0:
                    insights.append(Insight.Price(symbol, timedelta(days={holding_period}), InsightDirection.Down))
                    
        # Execute trades based on insights
        self.SetHoldings(insights)
        
    def GenerateSignal(self, symbol):
        """Generate trading signal based on RD-Agent logic"""
        {signal_generation_logic}
        
        return signal
        
    def OnData(self, data):
        """Process incoming data"""
        # Update custom calculations if needed
        pass
'''
        
        # Fill in the template with strategy parameters
        params = {
            'start_year': strategy_idea.get('start_date', '2020,1,1').split(',')[0],
            'start_month': strategy_idea.get('start_date', '2020,1,1').split(',')[1],
            'start_day': strategy_idea.get('start_date', '2020,1,1').split(',')[2],
            'end_year': strategy_idea.get('end_date', '2023,12,31').split(',')[0],
            'end_month': strategy_idea.get('end_date', '2023,12,31').split(',')[1],
            'end_day': strategy_idea.get('end_date', '2023,12,31').split(',')[2],
            'lookback_period': strategy_idea.get('lookback_period', 20),
            'rebalance_frequency': strategy_idea.get('rebalance_frequency', 5),
            'position_size': strategy_idea.get('position_size', 0.1),
            'schedule_rule': strategy_idea.get('schedule_rule', 'EveryDay'),
            'max_dd_per_security': strategy_idea.get('max_dd_per_security', 0.05),
            'min_price': strategy_idea.get('min_price', 5),
            'min_volume': strategy_idea.get('min_volume', 1000000),
            'universe_size': strategy_idea.get('universe_size', 50),
            'holding_period': strategy_idea.get('holding_period', 5),
            'indicator_setup': strategy_idea.get('indicator_setup', '"momentum": self.MOMP(symbol, self.lookback_period)'),
            'signal_generation_logic': strategy_idea.get('signal_generation_logic', '''
        # Example momentum-based signal
        indicators = self.indicators[symbol]
        momentum = indicators["momentum"].Current.Value
        
        if momentum > 0.02:  # 2% momentum threshold
            signal = 1
        elif momentum < -0.02:
            signal = -1
        else:
            signal = 0
            ''')
        }
        
        return strategy_template.format(**params)
        
    def run_backtest(self, strategy_code: str, project_path: str) -> Dict[str, Any]:
        """Run backtest using Lean CLI and return results"""
        
        # Write strategy code to main.py
        main_file = Path(project_path) / "main.py"
        with open(main_file, 'w') as f:
            f.write(strategy_code)
            
        # Run backtest
        cmd = f"cd '{project_path}' && lean backtest '{Path(project_path).name}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"error": result.stderr}
            
        # Parse results
        results_file = Path(project_path) / "backtests" / "1" / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            # Extract key metrics
            metrics = self.parse_backtest_results(results)
            return metrics
        else:
            return {"error": "Results file not found"}
            
    def parse_backtest_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Parse QuantConnect backtest results and extract key metrics"""
        
        statistics = results.get("Statistics", {})
        
        # Calculate CAGR
        total_return = float(statistics.get("Total Return", "0").rstrip('%')) / 100
        years = float(statistics.get("Total Trading Days", "0")) / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Extract other metrics
        metrics = {
            "cagr": cagr,
            "sharpe_ratio": float(statistics.get("Sharpe Ratio", "0")),
            "max_drawdown": abs(float(statistics.get("Drawdown", "0").rstrip('%')) / 100),
            "total_return": total_return,
            "win_rate": float(statistics.get("Win Rate", "0").rstrip('%')) / 100,
            "profit_loss_ratio": float(statistics.get("Profit-Loss Ratio", "0")),
            "avg_profit": float(statistics.get("Average Win", "0").rstrip('%')) / 100,
            "total_trades": int(statistics.get("Total Trades", "0")),
            "annual_volatility": float(statistics.get("Annual Standard Deviation", "0"))
        }
        
        return metrics
        
    def evaluate_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate if strategy meets target criteria"""
        
        evaluation = {
            "meets_criteria": True,
            "details": {}
        }
        
        # Check each target metric
        if metrics["cagr"] < self.target_metrics["cagr"]:
            evaluation["meets_criteria"] = False
            evaluation["details"]["cagr"] = f"Target: {self.target_metrics['cagr']:.2%}, Actual: {metrics['cagr']:.2%}"
            
        if metrics["sharpe_ratio"] < self.target_metrics["sharpe_ratio"]:
            evaluation["meets_criteria"] = False
            evaluation["details"]["sharpe_ratio"] = f"Target: {self.target_metrics['sharpe_ratio']:.2f}, Actual: {metrics['sharpe_ratio']:.2f}"
            
        if metrics["max_drawdown"] > self.target_metrics["max_drawdown"]:
            evaluation["meets_criteria"] = False
            evaluation["details"]["max_drawdown"] = f"Target: <{self.target_metrics['max_drawdown']:.2%}, Actual: {metrics['max_drawdown']:.2%}"
            
        if metrics["avg_profit"] < self.target_metrics["avg_profit"]:
            evaluation["meets_criteria"] = False
            evaluation["details"]["avg_profit"] = f"Target: >{self.target_metrics['avg_profit']:.2%}, Actual: {metrics['avg_profit']:.2%}"
            
        return evaluation
        
    def generate_report(self, strategy_idea: Dict[str, Any], metrics: Dict[str, float], 
                       evaluation: Dict[str, Any]) -> str:
        """Generate a detailed report of the backtest results"""
        
        report = f"""
# RD-Agent QuantConnect Strategy Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Strategy Overview
- **Name**: {strategy_idea.get('name', 'RD-Agent Generated Strategy')}
- **Description**: {strategy_idea.get('description', 'AI-generated trading strategy')}
- **Type**: {strategy_idea.get('type', 'Momentum-based')}

## Backtest Results

### Performance Metrics
- **CAGR**: {metrics['cagr']:.2%}
- **Sharpe Ratio**: {metrics['sharpe_ratio']:.2f}
- **Max Drawdown**: {metrics['max_drawdown']:.2%}
- **Total Return**: {metrics['total_return']:.2%}
- **Win Rate**: {metrics['win_rate']:.2%}
- **Average Profit**: {metrics['avg_profit']:.2%}
- **Total Trades**: {metrics['total_trades']}
- **Annual Volatility**: {metrics['annual_volatility']:.2%}

### Target Evaluation
**Meets All Criteria**: {'✅ Yes' if evaluation['meets_criteria'] else '❌ No'}

"""
        if not evaluation['meets_criteria']:
            report += "**Failed Criteria:**\n"
            for criterion, detail in evaluation['details'].items():
                report += f"- {criterion}: {detail}\n"
                
        report += f"""
## Strategy Parameters
- **Lookback Period**: {strategy_idea.get('lookback_period', 20)} days
- **Rebalance Frequency**: Every {strategy_idea.get('rebalance_frequency', 5)} days
- **Universe Size**: {strategy_idea.get('universe_size', 50)} stocks
- **Position Size**: {strategy_idea.get('position_size', 10)}%

## Next Steps
"""
        if evaluation['meets_criteria']:
            report += "✅ Strategy meets all target criteria and is ready for further validation or live deployment.\n"
        else:
            report += "⚠️ Strategy does not meet all criteria. Consider:\n"
            report += "- Adjusting strategy parameters\n"
            report += "- Modifying signal generation logic\n"
            report += "- Changing universe selection criteria\n"
            report += "- Implementing additional risk management rules\n"
            
        return report


# Example usage function
def run_rd_agent_quantconnect_pipeline():
    """Main pipeline to run RD-Agent with QuantConnect"""
    
    # Initialize integration
    qc_integration = QuantConnectIntegration()
    
    # Example strategy idea (this would come from RD-Agent)
    strategy_idea = {
        "name": "AI Momentum Strategy",
        "description": "Momentum-based strategy with risk management",
        "type": "Momentum",
        "start_date": "2020,1,1",
        "end_date": "2023,12,31",
        "lookback_period": 20,
        "rebalance_frequency": 5,
        "position_size": 0.1,
        "universe_size": 50,
        "min_price": 5,
        "min_volume": 1000000,
        "indicator_setup": '"momentum": self.MOMP(symbol, self.lookback_period), "rsi": self.RSI(symbol, 14)',
        "signal_generation_logic": '''
        indicators = self.indicators[symbol]
        momentum = indicators["momentum"].Current.Value
        rsi = indicators["rsi"].Current.Value
        
        # Buy signal: positive momentum and RSI not overbought
        if momentum > 0.02 and rsi < 70:
            signal = 1
        # Sell signal: negative momentum or RSI overbought
        elif momentum < -0.02 or rsi > 80:
            signal = -1
        else:
            signal = 0
        '''
    }
    
    # Create project
    project_path = qc_integration.create_lean_project()
    
    # Generate strategy code
    strategy_code = qc_integration.generate_strategy_code(strategy_idea)
    
    # Run backtest
    metrics = qc_integration.run_backtest(strategy_code, project_path)
    
    # Evaluate performance
    evaluation = qc_integration.evaluate_performance(metrics)
    
    # Generate report
    report = qc_integration.generate_report(strategy_idea, metrics, evaluation)
    
    print(report)
    
    return {
        "strategy_idea": strategy_idea,
        "metrics": metrics,
        "evaluation": evaluation,
        "report": report,
        "project_path": project_path
    }


if __name__ == "__main__":
    run_rd_agent_quantconnect_pipeline()