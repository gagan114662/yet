import subprocess
import json
import os
from typing import Dict
from datetime import datetime
try:
    from quantconnect_integration.rd_agent_qc_bridge import QuantConnectIntegration
except ImportError:
    print("Warning: QuantConnect integration not available. Falling back to basic Lean CLI.")
    QuantConnectIntegration = None
import config # To access Lean CLI configurations

class Backtester:
    def __init__(self):
        """
        Initializes the Backtester with QuantConnect integration if available,
        otherwise falls back to basic Lean CLI setup.
        """
        # Try to use QuantConnect integration first
        if QuantConnectIntegration:
            try:
                self.qc_integration = QuantConnectIntegration()
                self.use_qc_integration = True
                print("Using QuantConnect integration for backtesting.")
            except Exception as e:
                print(f"Failed to initialize QuantConnect integration: {e}")
                self.use_qc_integration = False
        else:
            self.use_qc_integration = False
            
        # Fallback to basic Lean CLI setup
        if not self.use_qc_integration:
            self.lean_cli_user_id = config.LEAN_CLI_USER_ID
            self.lean_cli_api_token = config.LEAN_CLI_API_TOKEN
            self.lean_cli_path = config.LEAN_CLI_PATH
            self.temp_lean_project_path = "lean_workspace/TempBacktestStrategy"
            # Ensure the base workspace directory exists
            os.makedirs("lean_workspace", exist_ok=True)
            print("Using basic Lean CLI for backtesting.")

    def backtest_strategy(self, strategy_idea: Dict) -> Dict:
        """
        Backtests a strategy using either QuantConnect integration or basic Lean CLI.

        Args:
            strategy_idea: A dictionary containing the strategy definition.

        Returns:
            A dictionary containing performance metrics or error information.
        """
        strategy_name = strategy_idea.get('name', 'UnnamedStrategy')
        print(f"Backtesting strategy: {strategy_name}")
        
        # Use QuantConnect integration if available
        if self.use_qc_integration:
            return self._backtest_with_qc_integration(strategy_idea)
        else:
            return self._backtest_with_lean_cli(strategy_idea)
    
    def _backtest_with_qc_integration(self, strategy_idea: Dict) -> Dict:
        """Backtest using QuantConnect integration."""
        strategy_name = strategy_idea.get('name', 'UnnamedStrategy').replace(' ', '_')
        unique_project_name = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        print(f"Preparing to backtest strategy: {unique_project_name}")
        print(f"Strategy details: {strategy_idea}")

        try:
            # 1. Create a LEAN project directory
            print(f"Attempting to create LEAN project: {unique_project_name}")
            project_path = self.qc_integration.create_lean_project(project_name=unique_project_name)
            print(f"LEAN project created at: {project_path}")

            # 2. Generate strategy code from the idea
            print("Generating strategy code...")
            strategy_code = self.qc_integration.generate_strategy_code(strategy_idea)

            # 3. Run the backtest
            print(f"Running backtest for project: {unique_project_name} at path: {project_path}...")
            results = self.qc_integration.run_backtest(strategy_code, project_path)

            if "error" in results:
                print(f"Warning: Backtest for {unique_project_name} encountered an error: {results['error']}")
            else:
                print(f"Backtest completed for {unique_project_name}. Results: {results}")

            return results

        except Exception as e:
            print(f"An unexpected error occurred during backtesting strategy {unique_project_name}: {e}")
            return {
                "error": str(e),
                "details": "Exception in Backtester._backtest_with_qc_integration",
                "project_name": unique_project_name,
                "strategy_idea": strategy_idea
            }
    
    def _backtest_with_lean_cli(self, strategy_idea: Dict) -> Dict:
        """Backtest using basic Lean CLI."""
        strategy_name = strategy_idea.get('name', 'UnnamedStrategy')
        print(f"Backtesting strategy: {strategy_name} using basic Lean CLI.")

        # Generate strategy code from strategy_idea
        strategy_code = self._generate_strategy_code_from_idea(strategy_idea)

        try:
            os.makedirs(self.temp_lean_project_path, exist_ok=True)

            # Write strategy code to main.py
            with open(os.path.join(self.temp_lean_project_path, "main.py"), "w") as f:
                f.write(strategy_code)

            # Write minimal config.json
            lean_config = {
                "algorithm-language": "Python",
                "parameters": {}
            }
            with open(os.path.join(self.temp_lean_project_path, "config.json"), "w") as f:
                json.dump(lean_config, f)

        except IOError as e:
            print(f"Error writing strategy or config files: {e}")
            return {
                'error': 'File system error preparing Lean project', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }

        # --- Construct Lean CLI Command ---
        output_dir = os.path.join(self.temp_lean_project_path, "lean_output")
        os.makedirs(output_dir, exist_ok=True)

        command = [
            self.lean_cli_path,
            "backtest",
            self.temp_lean_project_path,
            "--output", output_dir,
            "--json"
        ]

        print(f"Executing Lean CLI command: {' '.join(command)}")

        # --- Execute Command ---
        try:
            env = os.environ.copy()
            process = subprocess.run(command, capture_output=True, text=True, env=env, check=False)

            if process.returncode != 0:
                print(f"Error executing Lean CLI: {process.stderr}")
                results_json_path = self.find_results_json(output_dir)
                if results_json_path:
                    print(f"Attempting to parse results from {results_json_path} despite CLI error.")
                    return self.parse_lean_results_from_file(results_json_path, process.stderr)
                return {
                    'error': 'Lean CLI execution failed', 'details': process.stderr,
                    'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
                }

            # --- Parse Results ---
            if process.stdout:
                try:
                    lean_results_data = json.loads(process.stdout)
                    print("Successfully parsed Lean CLI JSON output from stdout.")
                    return self.parse_metrics_from_lean_json(lean_results_data, process.stdout)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON from Lean CLI stdout: {e}. Stdout was: {process.stdout[:500]}...")

            # Fallback: look for results.json in the output directory
            results_json_path = self.find_results_json(output_dir)
            if not results_json_path:
                project_backtests_dir = os.path.join(self.temp_lean_project_path, "backtests")
                results_json_path = self.find_results_json(project_backtests_dir)

            if results_json_path:
                print(f"Parsing Lean CLI output from file: {results_json_path}")
                return self.parse_lean_results_from_file(results_json_path)
            else:
                print(f"Lean CLI stdout was not JSON, and no results.json found.")
                return {
                    'error': 'Lean CLI output not JSON and results.json not found', 'details': process.stdout[:1000],
                    'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
                }

        except FileNotFoundError:
            print(f"Error: Lean CLI executable not found at '{self.lean_cli_path}'. Please check config.py.")
            return {
                'error': 'Lean CLI executable not found', 'details': f"Path '{self.lean_cli_path}' is invalid.",
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }
        except Exception as e:
            print(f"An unexpected error occurred during Lean CLI execution: {e}")
            return {
                'error': 'Unexpected error during backtest', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }

    def _generate_strategy_code_from_idea(self, strategy_idea: Dict) -> str:
        """Generate Lean algorithm code from strategy idea."""
        strategy_type = strategy_idea.get('type', 'momentum')
        start_date = strategy_idea.get('start_date', '2020,1,1')
        end_date = strategy_idea.get('end_date', '2023,12,31')
        lookback_period = strategy_idea.get('lookback_period', 20)
        position_size = strategy_idea.get('position_size', 0.1)
        universe_size = strategy_idea.get('universe_size', 50)
        min_price = strategy_idea.get('min_price', 10.0)
        min_volume = strategy_idea.get('min_volume', 1000000)
        rebalance_frequency = strategy_idea.get('rebalance_frequency', 5)
        
        # Get strategy-specific logic
        indicator_setup = strategy_idea.get('indicator_setup', '"rsi": self.RSI("SPY", 14)')
        signal_generation_logic = strategy_idea.get('signal_generation_logic', '''
indicators = self.indicators["SPY"]
rsi = indicators["rsi"].Current.Value
signal = 0
if self.Securities["SPY"].Price > 0 and rsi < 30:
    signal = 1
elif rsi > 70:
    signal = -1
''')
        
        code = f'''
from AlgorithmImports import *
import random

class GeneratedStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({start_date})
        self.SetEndDate({end_date})
        self.SetCash(100000)
        
        # Strategy parameters
        self.lookback_period = {lookback_period}
        self.position_size = {position_size}
        self.universe_size = {universe_size}
        self.min_price = {min_price}
        self.min_volume = {min_volume}
        self.rebalance_frequency = {rebalance_frequency}
        
        # Universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Storage for indicators and data
        self.indicators = {{}}
        self.last_rebalance = datetime.min
        self.rebalance_count = 0
        
    def CoarseSelectionFunction(self, coarse):
        # Filter by price and volume
        filtered = [x for x in coarse if x.Price > self.min_price and x.DollarVolume > self.min_volume]
        
        # Sort by dollar volume and take top stocks
        sorted_by_dollar_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_dollar_volume[:self.universe_size]]
    
    def OnSecuritiesChanged(self, changes):
        # Add indicators for new securities
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.indicators:
                self.indicators[symbol] = {{{indicator_setup}}}
        
        # Clean up removed securities
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.indicators:
                del self.indicators[symbol]
    
    def OnData(self, data):
        # Check if it's time to rebalance
        if (self.Time - self.last_rebalance).days < self.rebalance_frequency:
            return
            
        # Generate signals for each security
        signals = {{}}
        for symbol in self.indicators.keys():
            if symbol in data and data[symbol] is not None:
                try:
{signal_generation_logic}
                    signals[symbol] = signal
                except:
                    signals[symbol] = 0
        
        # Execute trades based on signals
        if signals:
            self.Rebalance(signals)
            self.last_rebalance = self.Time
            self.rebalance_count += 1
    
    def Rebalance(self, signals):
        # Count positive signals for position sizing
        positive_signals = [s for s in signals.values() if s > 0]
        if not positive_signals:
            self.Liquidate()
            return
            
        # Calculate position size per stock
        position_size_per_stock = self.position_size / len(positive_signals)
        
        # Liquidate positions without signals
        for symbol in self.Portfolio.Keys:
            if symbol not in signals or signals[symbol] <= 0:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
        
        # Enter new positions
        for symbol, signal in signals.items():
            if signal > 0:
                self.SetHoldings(symbol, position_size_per_stock)
'''
        
        return code

    def find_results_json(self, search_dir: str) -> str | None:
        """Find the latest results.json file in the search directory."""
        if not os.path.isdir(search_dir):
            return None

        latest_time = 0
        results_file = None

        # Option 1: results.json directly in search_dir
        direct_results_json = os.path.join(search_dir, "results.json")
        if os.path.isfile(direct_results_json):
            return direct_results_json

        # Option 2: Search in subdirectories (timestamped folders)
        for item in os.listdir(search_dir):
            item_path = os.path.join(search_dir, item)
            if os.path.isdir(item_path):
                current_results_json = os.path.join(item_path, "results.json")
                if os.path.isfile(current_results_json):
                    try:
                        folder_time = int(item)
                    except ValueError:
                        folder_time = os.path.getmtime(current_results_json)

                    if folder_time > latest_time:
                        latest_time = folder_time
                        results_file = current_results_json

        if results_file:
            print(f"Found results.json at: {results_file}")
        return results_file

    def parse_lean_results_from_file(self, file_path: str, cli_stderr_if_any: str = None) -> Dict:
        try:
            with open(file_path, 'r') as f:
                lean_results_data = json.load(f)
            print(f"Successfully parsed Lean CLI JSON output from file: {file_path}")
            full_output_for_debugging = f"File: {file_path}"
            if cli_stderr_if_any:
                full_output_for_debugging += f"\\nCLI Stderr:\\n{cli_stderr_if_any}"

            return self.parse_metrics_from_lean_json(lean_results_data, full_output_for_debugging)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from Lean results file {file_path}: {e}")
            return {
                'error': 'Failed to parse results.json', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }
        except IOError as e:
            print(f"Failed to read results.json file {file_path}: {e}")
            return {
                'error': 'Failed to read results.json', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }

    def parse_metrics_from_lean_json(self, lean_data: Dict, raw_output_for_debugging: str) -> Dict:
        """Parse metrics from Lean's JSON output structure."""
        try:
            # Default values for metrics (matching config.py format)
            metrics = {
                'cagr': 0.0,
                'max_drawdown': 1.0, # Positive value where smaller is better
                'sharpe_ratio': 0.0,
                'avg_profit': 0.0,
                'total_trades': 0,
                'lean_cli_output': raw_output_for_debugging[:2000]
            }

            # Parse Lean output structure
            stats = lean_data.get('Statistics', lean_data)

            # Map Lean metrics to our format
            metrics['sharpe_ratio'] = float(stats.get('Sharpe Ratio', stats.get('SharpeRatio', 0.0)))
            metrics['cagr'] = float(stats.get('Compounding Annual Return', stats.get('CompoundingAnnualReturn', 0.0)))
            
            # Max Drawdown - ensure it's positive (smaller is better)
            max_drawdown_lean = float(stats.get('Drawdown', stats.get('Max Drawdown', stats.get('Maximum Drawdown', 1.0))))
            metrics['max_drawdown'] = abs(max_drawdown_lean) if max_drawdown_lean != 1.0 else 1.0
            
            # Calculate average profit per trade
            total_trades = int(stats.get('Total Trades', stats.get('TotalTrades', 0)))
            metrics['total_trades'] = total_trades
            if total_trades > 0:
                total_return = metrics['cagr']
                metrics['avg_profit'] = total_return / total_trades if total_return != 0 else 0.0

            print(f"Parsed metrics: {metrics}")
            return metrics

        except (KeyError, ValueError, TypeError) as e:
            print(f"Error parsing Lean metrics: {e}. Data was: {str(lean_data)[:500]}")
            return {
                'error': 'Error parsing Lean metrics', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0,
                'lean_cli_output': raw_output_for_debugging[:2000]
            }

# Example usage
if __name__ == '__main__':
    from strategy_utils import generate_next_strategy

    print("Generating a sample strategy idea for backtesting...")
    strategy_idea_to_test = generate_next_strategy()
    print(f"Strategy Idea: {strategy_idea_to_test}")

    backtester = Backtester()
    print("\\nInitializing backtester and starting backtest process...")
    backtest_results = backtester.backtest_strategy(strategy_idea_to_test)

    print(f"\\nBacktester execution finished.")
    print(f"Strategy Idea Tested: {strategy_idea_to_test.get('name')}")
    print(f"Full Results: {backtest_results}")

    if "error" in backtest_results:
        print(f"Error during backtest: {backtest_results['error']}")
    elif backtest_results:
        print(f"CAGR: {backtest_results.get('cagr')}")
        print(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio')}")
        print(f"Max Drawdown: {backtest_results.get('max_drawdown')}")
        print(f"Total Trades: {backtest_results.get('total_trades')}")