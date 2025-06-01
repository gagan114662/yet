import os
import json
import itertools
import subprocess
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

class StrategyOptimizer:
    """
    Utility to optimize the multi-timeframe strategy by running backtests
    with different parameter combinations and analyzing the results.
    """
    
    def __init__(self, strategy_dir, lean_cli_path="lean"):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        strategy_dir : str
            Path to the strategy directory
        lean_cli_path : str
            Path to the Lean CLI executable
        """
        self.strategy_dir = strategy_dir
        self.lean_cli_path = lean_cli_path
        self.results = []
        
    def generate_parameter_combinations(self, param_ranges):
        """
        Generate all combinations of parameters to test
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary of parameter names and their possible values
            
        Returns:
        --------
        list
            List of parameter dictionaries for each combination
        """
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
        print(f"Generated {len(combinations)} parameter combinations to test")
        return combinations
        
    def update_config(self, params):
        """
        Update the config.json file with new parameters
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to update
        """
        config_path = os.path.join(self.strategy_dir, "config.json")
        
        # Read existing config
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Update parameters
        config["parameters"] = params
        
        # Write updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    def run_backtest(self, params):
        """
        Run a backtest with the given parameters
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to use for the backtest
            
        Returns:
        --------
        dict
            Dictionary with backtest results
        """
        # Update config with parameters
        self.update_config(params)
        
        # Run backtest using Lean CLI
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.strategy_dir, "optimization", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save parameters to output directory
        with open(os.path.join(output_dir, "parameters.json"), 'w') as f:
            json.dump(params, f, indent=4)
            
        # Command to run backtest
        cmd = [
            self.lean_cli_path,
            "backtest",
            "--output", output_dir,
            self.strategy_dir
        ]
        
        try:
            # Run backtest
            print(f"Running backtest with parameters: {params}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Save output
            with open(os.path.join(output_dir, "backtest_output.txt"), 'w') as f:
                f.write(result.stdout)
                f.write("\n\n")
                f.write(result.stderr)
                
            # Parse results
            results = self.parse_backtest_results(output_dir)
            results.update(params)
            
            return results
        except Exception as e:
            print(f"Error running backtest: {str(e)}")
            return {"error": str(e), **params}
            
    def parse_backtest_results(self, output_dir):
        """
        Parse backtest results from the output directory
        
        Parameters:
        -----------
        output_dir : str
            Path to the backtest output directory
            
        Returns:
        --------
        dict
            Dictionary with parsed results
        """
        # Find statistics file
        stats_file = None
        for file in os.listdir(output_dir):
            if file.endswith("statistics.json"):
                stats_file = os.path.join(output_dir, file)
                break
                
        if not stats_file:
            return {"error": "No statistics file found"}
            
        # Parse statistics
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            
        # Extract key metrics
        results = {
            "total_trades": stats.get("TotalOrders", 0),
            "sharpe_ratio": stats.get("SharpeRatio", 0),
            "annual_return": stats.get("CompoundingAnnualReturn", 0),
            "drawdown": stats.get("Drawdown", 0),
            "profit": stats.get("TotalNetProfit", 0),
            "win_rate": stats.get("WinRate", 0),
            "output_dir": output_dir
        }
        
        return results
        
    def run_optimization(self, param_ranges, max_combinations=None):
        """
        Run optimization with all parameter combinations
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary of parameter names and their possible values
        max_combinations : int, optional
            Maximum number of combinations to test
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all results
        """
        combinations = self.generate_parameter_combinations(param_ranges)
        
        # Limit combinations if specified
        if max_combinations and len(combinations) > max_combinations:
            print(f"Limiting to {max_combinations} combinations")
            combinations = combinations[:max_combinations]
            
        # Run backtests for each combination
        for i, params in enumerate(combinations):
            print(f"Running combination {i+1}/{len(combinations)}")
            result = self.run_backtest(params)
            self.results.append(result)
            
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.strategy_dir, f"optimization_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        
        print(f"Optimization complete. Results saved to {results_path}")
        return results_df
        
    def analyze_results(self, results_df=None, metric="sharpe_ratio"):
        """
        Analyze optimization results
        
        Parameters:
        -----------
        results_df : pd.DataFrame, optional
            DataFrame with results, if None uses self.results
        metric : str, optional
            Metric to optimize for
            
        Returns:
        --------
        dict
            Best parameters
        """
        if results_df is None:
            results_df = pd.DataFrame(self.results)
            
        if results_df.empty:
            print("No results to analyze")
            return None
            
        # Sort by metric
        results_df = results_df.sort_values(by=metric, ascending=False)
        
        # Get best parameters
        best_params = results_df.iloc[0].to_dict()
        
        print(f"Best parameters based on {metric}:")
        for param, value in best_params.items():
            if param not in ["error", "output_dir", "total_trades", "sharpe_ratio", 
                           "annual_return", "drawdown", "profit", "win_rate"]:
                print(f"  {param}: {value}")
                
        print(f"Performance metrics:")
        print(f"  Sharpe Ratio: {best_params.get('sharpe_ratio', 'N/A')}")
        print(f"  Annual Return: {best_params.get('annual_return', 'N/A')}")
        print(f"  Drawdown: {best_params.get('drawdown', 'N/A')}")
        print(f"  Win Rate: {best_params.get('win_rate', 'N/A')}")
        print(f"  Total Trades: {best_params.get('total_trades', 'N/A')}")
        
        # Plot parameter impact
        self.plot_parameter_impact(results_df, metric)
        
        return best_params
        
    def plot_parameter_impact(self, results_df, metric):
        """
        Plot the impact of each parameter on the target metric
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with results
        metric : str
            Metric to analyze
        """
        # Identify parameter columns
        param_cols = [col for col in results_df.columns if col not in 
                     ["error", "output_dir", "total_trades", "sharpe_ratio", 
                      "annual_return", "drawdown", "profit", "win_rate"]]
                      
        if not param_cols:
            print("No parameter columns found for analysis")
            return
            
        # Create figure
        n_params = len(param_cols)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 4 * n_params))
        
        if n_params == 1:
            axes = [axes]
            
        # Plot each parameter's impact
        for i, param in enumerate(param_cols):
            # Group by parameter and calculate mean metric
            param_impact = results_df.groupby(param)[metric].mean().reset_index()
            
            # Plot
            axes[i].bar(param_impact[param].astype(str), param_impact[metric])
            axes[i].set_title(f"Impact of {param} on {metric}")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(metric)
            
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.strategy_dir, f"parameter_impact_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Parameter impact plot saved to {plot_path}")
        
if __name__ == "__main__":
    # Example usage
    optimizer = StrategyOptimizer(".")
    
    # Define parameter ranges to test
    param_ranges = {
        "daily_allocation": [0.1, 0.2, 0.3],
        "weekly_allocation": [0.2, 0.3, 0.4],
        "monthly_allocation": [0.3, 0.4, 0.5],
        "rsi_period": [7, 14, 21],
        "rsi_oversold": [20, 30],
        "rsi_overbought": [70, 80],
        "stop_loss_pct": [3, 5, 7],
        "take_profit_pct": [15, 20, 25]
    }
    
    # Run optimization (limited to 5 combinations for example)
    results = optimizer.run_optimization(param_ranges, max_combinations=5)
    
    # Analyze results
    best_params = optimizer.analyze_results(results, metric="sharpe_ratio")
