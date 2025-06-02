import subprocess
import json
import os
from typing import Dict
from .strategy_utils import Strategy # Assuming Strategy is in strategy_utils.py
from . import config # To access Lean CLI configurations

class Backtester:
    def __init__(self):
        """
        Initializes the Backtester.
        Loads Lean CLI configuration and sets up paths.
        """
        self.lean_cli_user_id = config.LEAN_CLI_USER_ID
        self.lean_cli_api_token = config.LEAN_CLI_API_TOKEN
        self.lean_cli_path = config.LEAN_CLI_PATH
        self.temp_lean_project_path = "lean_workspace/TempBacktestStrategy"
        # Ensure the base workspace directory exists
        os.makedirs("lean_workspace", exist_ok=True)


    def backtest_strategy(self, strategy: Strategy) -> Dict:
        """
        Backtests a strategy using Lean CLI.

        Args:
            strategy: The strategy object to backtest.

        Returns:
            A dictionary containing performance metrics or error information.
        """
        print(f"Backtesting strategy ID: {strategy.id}, Type: {strategy.type}, Params: {strategy.parameters} using Lean CLI.")

        # --- Prepare Strategy File ---
        # For now, using a default strategy code as strategy.get_code() is not available.
        default_strategy_code = """
from AlgorithmImports import *
class DefaultQCAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 10, 1)
        self.SetEndDate(2023, 10, 11) # Short period for faster testing
        self.SetCash(100000)
        self.AddEquity("SPY", Resolution.Daily)
    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1)
"""
        strategy_code = default_strategy_code # Replace with strategy.get_code() when available

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
                'annual_return': 0, 'max_drawdown': -1, 'sharpe_ratio': 0, 'win_rate': 0, 'trades_executed': 0
            }

        # --- Construct Lean CLI Command ---
        # Adjust output_dir if Lean CLI saves files there instead of stdout
        output_dir = os.path.join(self.temp_lean_project_path, "lean_output")
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

        # Command structure: lean backtest <project> --output <output_path_for_results_json> --json
        # The --json flag is key. Some versions of Lean CLI might output JSON to stdout,
        # others might save it in the --output path. We'll first try to parse stdout.
        # If that fails, we'll look for a results.json in the output_dir.
        # The project path should be the directory containing main.py and config.json
        command = [
            self.lean_cli_path,
            "backtest",
            self.temp_lean_project_path, # Path to the project directory
            "--output", output_dir,      # Directory where Lean stores detailed results
            "--json"                     # Request JSON output (hopefully to stdout or a known file)
        ]

        print(f"Executing Lean CLI command: {' '.join(command)}")

        # --- Execute Command ---
        try:
            # Environment variables for Lean (if needed, typically for cloud operations or specific auth)
            # For local backtesting with a pre-configured CLI, this might not be strictly necessary
            # but good to keep in mind. User ID and API Token are more for cloud.
            env = os.environ.copy()
            # env["QC_USER_ID"] = self.lean_cli_user_id # Example if needed
            # env["QC_API_TOKEN"] = self.lean_cli_api_token # Example if needed

            process = subprocess.run(command, capture_output=True, text=True, env=env, check=False) # check=False to handle errors manually

            if process.returncode != 0:
                print(f"Error executing Lean CLI: {process.stderr}")
                # Attempt to read results.json even if CLI reports error, as it might contain partial data or error details
                # This is a fallback, primary expectation is stdout or success.
                results_json_path = self.find_results_json(output_dir)
                if results_json_path:
                    print(f"Attempting to parse results from {results_json_path} despite CLI error.")
                    return self.parse_lean_results_from_file(results_json_path, process.stderr)
                return {
                    'error': 'Lean CLI execution failed', 'details': process.stderr,
                    'annual_return': 0, 'max_drawdown': -1, 'sharpe_ratio': 0, 'win_rate': 0, 'trades_executed': 0
                }

            # --- Parse Results ---
            # Try parsing stdout first
            if process.stdout:
                try:
                    lean_results_data = json.loads(process.stdout)
                    print("Successfully parsed Lean CLI JSON output from stdout.")
                    return self.parse_metrics_from_lean_json(lean_results_data, process.stdout)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON from Lean CLI stdout: {e}. Stdout was: {process.stdout[:500]}...")
                    # Fallback: Check if results.json was created in the output directory

            # Fallback: look for results.json in the output directory or project's backtests folder
            results_json_path = self.find_results_json(output_dir)
            if not results_json_path: # If not in output_dir, check standard backtest location
                project_backtests_dir = os.path.join(self.temp_lean_project_path, "backtests")
                results_json_path = self.find_results_json(project_backtests_dir)

            if results_json_path:
                print(f"Parsing Lean CLI output from file: {results_json_path}")
                return self.parse_lean_results_from_file(results_json_path)
            else:
                print(f"Lean CLI stdout was not JSON, and no results.json found in {output_dir} or project backtests.")
                return {
                    'error': 'Lean CLI output not JSON and results.json not found', 'details': process.stdout[:1000],
                    'annual_return': 0, 'max_drawdown': -1, 'sharpe_ratio': 0, 'win_rate': 0, 'trades_executed': 0
                }

        except FileNotFoundError:
            print(f"Error: Lean CLI executable not found at '{self.lean_cli_path}'. Please check config.py.")
            return {
                'error': 'Lean CLI executable not found', 'details': f"Path '{self.lean_cli_path}' is invalid.",
                'annual_return': 0, 'max_drawdown': -1, 'sharpe_ratio': 0, 'win_rate': 0, 'trades_executed': 0
            }
        except Exception as e:
            print(f"An unexpected error occurred during Lean CLI execution: {e}")
            return {
                'error': 'Unexpected error during backtest', 'details': str(e),
                'annual_return': 0, 'max_drawdown': -1, 'sharpe_ratio': 0, 'win_rate': 0, 'trades_executed': 0
            }

    def find_results_json(self, search_dir: str) -> str | None:
        """
        Finds the 'results.json' file, typically the latest one if multiple backtests exist.
        Lean CLI usually stores results in <output_dir>/<timestamp>/results.json or <project_dir>/backtests/<timestamp>/results.json
        """
        if not os.path.isdir(search_dir):
            return None

        latest_time = 0
        results_file = None

        # Option 1: results.json directly in search_dir (e.g. if --output points to a file, or a specific folder)
        direct_results_json = os.path.join(search_dir, "results.json")
        if os.path.isfile(direct_results_json):
            return direct_results_json # Found it directly

        # Option 2: Search in subdirectories (timestamped folders)
        for item in os.listdir(search_dir):
            item_path = os.path.join(search_dir, item)
            if os.path.isdir(item_path):
                # Heuristic: timestamped folders are often numeric or date-like
                # More robustly, find the most recently modified folder if names aren't predictable
                current_results_json = os.path.join(item_path, "results.json")
                if os.path.isfile(current_results_json):
                    try:
                        # If directory name is a timestamp (e.g. "1678886400")
                        folder_time = int(item)
                    except ValueError:
                        # Or use modification time of the results.json file itself
                        folder_time = os.path.getmtime(current_results_json)

                    if folder_time > latest_time:
                        latest_time = folder_time
                        results_file = current_results_json

        if results_file:
            print(f"Found results.json at: {results_file}")
        else:
            print(f"No results.json found in {search_dir} or its subdirectories.")
        return results_file

    def parse_lean_results_from_file(self, file_path: str, cli_stderr_if_any: str = None) -> Dict:
        try:
            with open(file_path, 'r') as f:
                lean_results_data = json.load(f)
            print(f"Successfully parsed Lean CLI JSON output from file: {file_path}")
            # Include stderr in the output if the CLI command itself had an error,
            # but we still managed to parse a JSON file.
            full_output_for_debugging = f"File: {file_path}"
            if cli_stderr_if_any:
                full_output_for_debugging += f"\nCLI Stderr:\n{cli_stderr_if_any}"

            return self.parse_metrics_from_lean_json(lean_results_data, full_output_for_debugging)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from Lean results file {file_path}: {e}")
            return {
                'error': 'Failed to parse results.json', 'details': str(e),
                'annual_return': 0, 'max_drawdown': -1, 'sharpe_ratio': 0, 'win_rate': 0, 'trades_executed': 0
            }
        except IOError as e:
            print(f"Failed to read results.json file {file_path}: {e}")
            return {
                'error': 'Failed to read results.json', 'details': str(e),
                'annual_return': 0, 'max_drawdown': -1, 'sharpe_ratio': 0, 'win_rate': 0, 'trades_executed': 0
            }

    def parse_metrics_from_lean_json(self, lean_data: Dict, raw_output_for_debugging: str) -> Dict:
        """
        Parses metrics from Lean's JSON output structure.
        The exact keys depend on Lean's output format.
        Common keys: 'SharpeRatio', 'CompoundingAnnualReturn', 'TotalTrades', 'WinRate', 'Drawdown'.
        """
        try:
            # Example: Extracting common metrics. Adjust keys based on actual Lean output.
            # These are common names but might vary (e.g. "Sharpe Ratio" vs "SharpeRatio")
            # The 'statistics' dictionary within 'results' is a common place for these.
            # Or they might be in a 'Strategy Equity' chart's summary.
            # For overall backtest statistics, they are usually in a top-level dictionary like 'Statistics' or 'summary'.
            # Let's assume the JSON output itself is the final backtest result dictionary.

            # Default values for metrics
            metrics = {
                'annual_return': 0.0,
                'max_drawdown': -1.0, # Indicate failure with -1
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'trades_executed': 0,
                'lean_cli_output': raw_output_for_debugging[:2000] # Store sample of output
            }

            # Actual Lean output parsing can be complex. The final summary is often in `oResults.Statistics`
            # or similar. The exact structure can vary.
            # Example: if results are flat in the JSON:
            # metrics['sharpe_ratio'] = float(lean_data.get('Sharpe Ratio', lean_data.get('SharpeRatio', 0)))
            # metrics['annual_return'] = float(lean_data.get('Compounding Annual Return', lean_data.get('CompoundingAnnualReturn', 0)))
            # metrics['max_drawdown'] = float(lean_data.get('Drawdown', lean_data.get('Max Drawdown', -1))) # Ensure it's negative
            # metrics['win_rate'] = float(lean_data.get('Win Rate', lean_data.get('WinRate', 0)))
            # metrics['trades_executed'] = int(lean_data.get('Total Trades', lean_data.get('TotalTrades', 0)))

            # A common structure for Lean CLI JSON output (especially with --json flag)
            # is that `lean_data` directly contains the statistics dictionary.
            # Or it might be nested, e.g., lean_data['statistics'] or lean_data['results']['Statistics']
            # For this subtask, I'll assume the keys are directly available or in a "Statistics" dictionary.

            stats = lean_data.get('Statistics', lean_data) # Try top-level, then 'Statistics' key

            metrics['sharpe_ratio'] = float(stats.get('Sharpe Ratio', stats.get('SharpeRatio', 0.0)))
            metrics['annual_return'] = float(stats.get('Compounding Annual Return', stats.get('CompoundingAnnualReturn', 0.0)))
            # Max Drawdown in Lean is typically positive, so we make it negative.
            max_drawdown_lean = float(stats.get('Drawdown', stats.get('Max Drawdown', stats.get('Maximum Drawdown', 1.0))))
            metrics['max_drawdown'] = -abs(max_drawdown_lean) if max_drawdown_lean != 1.0 else -1.0 # Ensure negative, handle default

            metrics['win_rate'] = float(stats.get('Win Rate', stats.get('WinRate', 0.0)))
            metrics['trades_executed'] = int(stats.get('Total Trades', stats.get('TotalTrades', 0)))

            # Additional check for required metrics, if some are absolutely critical
            if metrics['sharpe_ratio'] == 0.0 and metrics['annual_return'] == 0.0 and metrics['trades_executed'] == 0:
                 # This might indicate an empty or problematic backtest (e.g. no trades)
                print("Warning: Parsed metrics seem to indicate no trading activity or problematic backtest.")


            print(f"Parsed metrics: {metrics}")
            return metrics

        except KeyError as e:
            print(f"KeyError parsing Lean metrics: {e}. Data was: {str(lean_data)[:500]}")
            return {
                'error': 'KeyError parsing Lean metrics', 'details': str(e),
                'annual_return': 0, 'max_drawdown': -1, 'sharpe_ratio': 0, 'win_rate': 0, 'trades_executed': 0,
                'lean_cli_output': raw_output_for_debugging[:2000]
            }
        except (ValueError, TypeError) as e:
            print(f"ValueError/TypeError parsing Lean metrics: {e}. Data was: {str(lean_data)[:500]}")
            return {
                'error': 'Data type error parsing Lean metrics', 'details': str(e),
                'annual_return': 0, 'max_drawdown': -1, 'sharpe_ratio': 0, 'win_rate': 0, 'trades_executed': 0,
                'lean_cli_output': raw_output_for_debugging[:2000]
            }


# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    from .strategy_utils import generate_next_strategy # For standalone testing

    # Create a dummy strategy for testing
    # Note: For this to run, config.py needs to be correctly populated with placeholder paths at least.
    # And Lean CLI needs to be installed and executable via the path in config.py.
    # This example will likely fail if Lean CLI is not configured and executable.

    print("Attempting to initialize Backtester for standalone test...")
    try:
        backtester = Backtester()
        print("Backtester initialized.")

        test_strategy = generate_next_strategy()
        print(f"Generated strategy for test: {test_strategy}")

        print("Running backtest_strategy...")
        # This will try to run Lean CLI. Ensure config.py has a valid LEAN_CLI_PATH (even if it's just "lean")
        # and that "lean" is in your system PATH or the full path is provided.
        # The placeholder credentials in config.py are fine for local backtesting.
        results = backtester.backtest_strategy(test_strategy)

        print(f"\nBacktester executed for strategy {test_strategy.id}.")
        print(f"Results: {results}")

    except ImportError as e:
        print(f"ImportError during standalone test: {e}. Make sure config.py exists and is accessible.")
        print("You might need to run this as a module if direct execution fails due to relative imports: python -m algorithmic_trading_system.backtester")
    except Exception as e:
        print(f"An error occurred during standalone test: {e}")

