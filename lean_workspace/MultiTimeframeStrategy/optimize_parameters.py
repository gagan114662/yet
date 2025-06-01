import os
import json
import subprocess
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import time

class StrategyOptimizer:
    """
    Optimizes strategy parameters using QuantConnect's Lean CLI
    """
    
    def __init__(self, project_dir, config_path):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        project_dir : str
            Path to the QuantConnect project directory
        config_path : str
            Path to the strategy configuration file
        """
        self.project_dir = project_dir
        self.config_path = config_path
        self.results = []
        
        # Load current configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Create results directory if it doesn't exist
        self.results_dir = os.path.join(project_dir, 'optimization_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def generate_parameter_grid(self, param_ranges):
        """
        Generate a grid of parameter combinations to test
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary of parameter names and their ranges to test
            
        Returns:
        --------
        list
            List of parameter combinations to test
        """
        # Extract parameter names and values
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        # Generate all combinations
        combinations = list(product(*param_values))
        
        # Convert to list of dictionaries
        param_grid = []
        for combo in combinations:
            param_dict = {}
            for i, name in enumerate(param_names):
                param_dict[name] = combo[i]
            param_grid.append(param_dict)
            
        return param_grid
        
    def update_config(self, params):
        """
        Update the configuration file with new parameters
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to update
        """
        # Make a copy of the current config
        updated_config = self.config.copy()
        
        # Update parameters
        for key, value in params.items():
            # Handle nested parameters with dot notation
            if '.' in key:
                parts = key.split('.')
                current = updated_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                updated_config[key] = value
                
        # Save updated config
        with open(self.config_path, 'w') as f:
            json.dump(updated_config, f, indent=4)
            
        return updated_config
        
    def run_backtest(self, params, project_name="QuantitativeTradingStrategy"):
        """
        Run a backtest with the given parameters
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters to test
        project_name : str
            Name of the QuantConnect project
            
        Returns:
        --------
        dict
            Dictionary of backtest results
        """
        # Update config with new parameters
        self.update_config(params)
        
        # Run backtest using Lean CLI
        cmd = f"cd {self.project_dir} && lean backtest {project_name} --output {self.results_dir}"
        
        try:
            print(f"Running backtest with parameters: {params}")
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"Error running backtest: {stderr.decode()}")
                return None
                
            # Parse results
            results_file = os.path.join(self.results_dir, "backtests", "latest", "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                # Extract key metrics
                metrics = {
                    'total_trades': results.get('TotalNumberOfTrades', 0),
                    'win_rate': results.get('WinRate', 0),
                    'annual_return': results.get('CompoundingAnnualReturn', 0),
                    'sharpe_ratio': results.get('SharpeRatio', 0),
                    'max_drawdown': results.get('MaximumDrawdown', 0),
                    'params': params
                }
                
                return metrics
            else:
                print(f"Results file not found: {results_file}")
                return None
                
        except Exception as e:
            print(f"Error running backtest: {e}")
            return None
            
    def optimize(self, param_ranges, metric='sharpe_ratio', maximize=True):
        """
        Optimize parameters by running backtests with different parameter combinations
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary of parameter names and their ranges to test
        metric : str
            Metric to optimize (default: 'sharpe_ratio')
        maximize : bool
            Whether to maximize or minimize the metric (default: True)
            
        Returns:
        --------
        dict
            Dictionary of optimal parameters and results
        """
        # Generate parameter grid
        param_grid = self.generate_parameter_grid(param_ranges)
        print(f"Generated {len(param_grid)} parameter combinations to test")
        
        # Run backtests for each parameter combination
        results = []
        for i, params in enumerate(param_grid):
            print(f"Testing combination {i+1}/{len(param_grid)}")
            result = self.run_backtest(params)
            if result:
                results.append(result)
                
            # Save intermediate results
            self.results = results
            self.save_results()
            
        # Find optimal parameters
        if maximize:
            optimal_result = max(results, key=lambda x: x[metric])
        else:
            optimal_result = min(results, key=lambda x: x[metric])
            
        # Update config with optimal parameters
        self.update_config(optimal_result['params'])
        
        # Save final results
        self.results = results
        self.save_results()
        
        return optimal_result
        
    def save_results(self):
        """Save optimization results to file"""
        results_file = os.path.join(self.results_dir, "optimization_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
            
    def plot_results(self, x_param, y_param, metric='sharpe_ratio'):
        """
        Plot optimization results for two parameters
        
        Parameters:
        -----------
        x_param : str
            Parameter name for x-axis
        y_param : str
            Parameter name for y-axis
        metric : str
            Metric to plot (default: 'sharpe_ratio')
        """
        if not self.results:
            print("No results to plot")
            return
            
        # Extract unique parameter values
        x_values = sorted(list(set([r['params'][x_param] for r in self.results])))
        y_values = sorted(list(set([r['params'][y_param] for r in self.results])))
        
        # Create grid for heatmap
        grid = np.zeros((len(y_values), len(x_values)))
        
        # Fill grid with metric values
        for result in self.results:
            x_idx = x_values.index(result['params'][x_param])
            y_idx = y_values.index(result['params'][y_param])
            grid[y_idx, x_idx] = result[metric]
            
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(label=metric)
        plt.xticks(range(len(x_values)), x_values, rotation=45)
        plt.yticks(range(len(y_values)), y_values)
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.title(f"Optimization Results: {metric}")
        
        # Add values to cells
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                plt.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center', 
                         color='white' if grid[i, j] < np.max(grid)/2 else 'black')
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"optimization_{x_param}_{y_param}_{metric}.png"))
        plt.close()

def main():
    """Main function to run optimization"""
    # Define paths
    project_dir = "/workspace/finaly/OpenAlpha_Evolve/OpenAlpha_Evolve/quant_strategy/QuantitativeTradingStrategy"
    config_path = os.path.join(project_dir, "config.json")
    
    # Create optimizer
    optimizer = StrategyOptimizer(project_dir, config_path)
    
    # Define parameter ranges to test
    param_ranges = {
        # Daily strategy parameters
        "parameters.rsi_period": [7, 14, 21],
        "parameters.rsi_overbought": [70, 75, 80],
        "parameters.rsi_oversold": [20, 25, 30],
        "parameters.bb_period": [14, 20, 26],
        "parameters.bb_std_dev": [1.5, 2.0, 2.5],
        
        # Weekly strategy parameters
        "parameters.macd_fast": [8, 12, 16],
        "parameters.macd_slow": [21, 26, 30],
        "parameters.macd_signal": [7, 9, 11],
        
        # Monthly strategy parameters
        "parameters.ema_fast": [20, 50, 100],
        "parameters.ema_slow": [50, 100, 200],
        
        # Risk management parameters
        "parameters.stop_loss_pct": [3, 5, 7],
        "parameters.take_profit_pct": [10, 15, 20],
        "parameters.risk_per_trade_pct": [0.5, 1, 2],
        
        # Asset allocation
        "parameters.daily_allocation": [0.1, 0.2, 0.3],
        "parameters.weekly_allocation": [0.2, 0.3, 0.4],
        "parameters.monthly_allocation": [0.4, 0.5, 0.6]
    }
    
    # Run optimization
    optimal_result = optimizer.optimize(param_ranges, metric='sharpe_ratio', maximize=True)
    
    # Print optimal parameters
    print("\n=== Optimal Parameters ===")
    for param, value in optimal_result['params'].items():
        print(f"{param}: {value}")
        
    # Print optimal results
    print("\n=== Optimal Results ===")
    for metric, value in optimal_result.items():
        if metric != 'params':
            print(f"{metric}: {value}")
            
    # Plot results for selected parameter pairs
    optimizer.plot_results('daily.rsi_period', 'daily.rsi_oversold')
    optimizer.plot_results('weekly.macd_fast', 'weekly.macd_slow')
    optimizer.plot_results('monthly.ema_fast', 'monthly.ema_slow')
    optimizer.plot_results('risk.stop_loss_pct', 'risk.take_profit_pct')
    
if __name__ == "__main__":
    main()
