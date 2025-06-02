from typing import Dict
from datetime import datetime
from quantconnect_integration.rd_agent_qc_bridge import QuantConnectIntegration
# from strategy_utils import Strategy # No longer needed, using dict for strategy_idea

# Assuming strategy_utils is in the same directory or accessible in PYTHONPATH
# For the main application, ensure algorithmic_trading_system is in PYTHONPATH
# or use relative imports if this becomes part of a larger package.


class Backtester:
    def __init__(self):
        """
        Initializes the Backtester with QuantConnectIntegration.
        """
        self.qc_integration = QuantConnectIntegration()

    def backtest_strategy(self, strategy_idea: Dict) -> Dict:
        """
        Backtests a strategy idea using QuantConnect LEAN engine.

        Args:
            strategy_idea: A dictionary containing the strategy definition.

        Returns:
            A dictionary containing performance metrics from QuantConnect,
            or an error dictionary if the backtest failed.
        """
        strategy_name = strategy_idea.get('name', 'UnnamedStrategy').replace(' ', '_')
        unique_project_name = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        print(f"Preparing to backtest strategy: {unique_project_name}")
        print(f"Strategy details: {strategy_idea}")

        try:
            # 1. Create a LEAN project directory
            # The bridge script creates projects relative to its own CWD or a specified base path.
            # For now, let's assume it creates it in a location accessible for LEAN execution.
            print(f"Attempting to create LEAN project: {unique_project_name}")
            project_path = self.qc_integration.create_lean_project(project_name=unique_project_name)
            print(f"LEAN project created at: {project_path}")

            # 2. Generate strategy code from the idea
            print("Generating strategy code...")
            strategy_code = self.qc_integration.generate_strategy_code(strategy_idea)
            # (generate_strategy_code in the bridge writes the main.py to project_path)

            # 3. Run the backtest
            print(f"Running backtest for project: {unique_project_name} at path: {project_path}...")
            # The run_backtest method in rd_agent_qc_bridge.py handles writing the code to main.py
            # and then executing the backtest.
            results = self.qc_integration.run_backtest(strategy_code, project_path) # project_path is used by bridge to know where main.py is and where to run 'lean backtest'

            if "error" in results:
                print(f"Warning: Backtest for {unique_project_name} encountered an error: {results['error']}")
            else:
                print(f"Backtest completed for {unique_project_name}. Results: {results}")

            return results

        except Exception as e:
            print(f"An unexpected error occurred during backtesting strategy {unique_project_name}: {e}")
            return {
                "error": str(e),
                "details": "Exception in Backtester.backtest_strategy",
                "project_name": unique_project_name,
                "strategy_idea": strategy_idea
            }

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # This import path might need adjustment based on how the project is structured
    # and how PYTHONPATH is configured. Assuming 'strategy_utils.py' is in the same
    # directory 'algorithmic_trading_system' which is added to PYTHONPATH.
    from strategy_utils import generate_next_strategy

    print("Generating a sample strategy idea for backtesting...")
    # generate_next_strategy now returns a dict
    strategy_idea_to_test = generate_next_strategy()
    print(f"Strategy Idea: {strategy_idea_to_test}")

    backtester = Backtester()
    print("\nInitializing backtester and starting backtest process...")
    backtest_results = backtester.backtest_strategy(strategy_idea_to_test)

    print(f"\nBacktester execution finished.")
    print(f"Strategy Idea Tested: {strategy_idea_to_test.get('name')}")
    print(f"Full Results: {backtest_results}")

    if "error" in backtest_results:
        print(f"Error during backtest: {backtest_results['error']}")
    elif backtest_results:
        print(f"CAGR: {backtest_results.get('cagr')}")
        print(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio')}")
        print(f"Max Drawdown: {backtest_results.get('max_drawdown')}")
        print(f"Total Trades: {backtest_results.get('total_trades')}")
    else:
        print("No results returned from backtest.")
