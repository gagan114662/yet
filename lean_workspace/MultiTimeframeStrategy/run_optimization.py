#!/usr/bin/env python3
"""
Run optimization for the Enhanced Multi-Timeframe Strategy
This script uses the QuantConnect Lean CLI to run parameter optimization
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def generate_parameter_grid(config):
    """Generate parameter grid from optimization configuration"""
    param_grid = []
    
    # Extract parameters to optimize
    params = config['optimization']['parameters']
    
    # Generate all combinations
    import itertools
    
    # Extract parameter values
    param_values = {}
    for param in params:
        name = param['name']
        values = list(range(param['min'], param['max'] + 1, param['step']))
        param_values[name] = values
    
    # Generate all combinations
    param_names = list(param_values.keys())
    combinations = list(itertools.product(*[param_values[name] for name in param_names]))
    
    # Convert to list of dictionaries
    for combo in combinations:
        param_dict = {}
        for i, name in enumerate(param_names):
            param_dict[name] = combo[i]
        param_grid.append(param_dict)
    
    return param_grid

def update_config(config, params, output_path):
    """Update configuration with new parameters"""
    # Make a copy of the config
    updated_config = config.copy()
    
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
    with open(output_path, 'w') as f:
        json.dump(updated_config, f, indent=4)
    
    return updated_config

def run_backtest(project_dir, project_name, params, config_path, results_dir):
    """Run a backtest with the given parameters"""
    # Create temporary config file
    temp_config_path = os.path.join(project_dir, 'temp_config.json')
    
    # Load original config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update config with new parameters
    update_config(config, params, temp_config_path)
    
    # Run backtest using Lean CLI
    cmd = f"cd {project_dir} && lean cloud backtest {project_name} --push"
    
    try:
        print(f"Running backtest with parameters: {params}")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running backtest: {stderr.decode()}")
            return None
        
        # Extract backtest ID from output
        output = stdout.decode()
        import re
        match = re.search(r'Backtest id: ([a-f0-9]+)', output)
        if match:
            backtest_id = match.group(1)
            print(f"Backtest ID: {backtest_id}")
            
            # Get backtest results
            cmd = f"cd {project_dir} && lean cloud backtest-report {backtest_id} --output {results_dir}"
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"Error getting backtest report: {stderr.decode()}")
                return None
            
            # Parse results
            results_file = os.path.join(results_dir, f"{backtest_id}.json")
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
                    'params': params,
                    'backtest_id': backtest_id
                }
                
                return metrics
            else:
                print(f"Results file not found: {results_file}")
                return None
        else:
            print("Could not extract backtest ID from output")
            return None
    
    except Exception as e:
        print(f"Error running backtest: {e}")
        return None

def calculate_score(result, config):
    """Calculate optimization score based on weighted metrics"""
    if not result:
        return -float('inf')
    
    metrics = config['optimization']['metrics']
    score = 0
    
    for metric in metrics:
        name = metric['name']
        weight = metric['weight']
        minimize = metric.get('minimize', False)
        
        if name in result:
            value = result[name]
            if minimize:
                value = -value  # Negate value for metrics to minimize
            score += value * weight
    
    return score

def optimize(config_path, project_dir, project_name, max_iterations=10):
    """Run optimization process"""
    # Load configuration
    config = load_config(config_path)
    
    # Create results directory
    results_dir = os.path.join(project_dir, 'optimization_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate parameter grid
    param_grid = generate_parameter_grid(config)
    print(f"Generated {len(param_grid)} parameter combinations")
    
    # Limit to max_iterations if specified
    if max_iterations > 0 and max_iterations < len(param_grid):
        import random
        param_grid = random.sample(param_grid, max_iterations)
        print(f"Limited to {max_iterations} random combinations")
    
    # Run backtests
    results = []
    for i, params in enumerate(param_grid):
        print(f"Testing combination {i+1}/{len(param_grid)}")
        result = run_backtest(project_dir, project_name, params, config_path, results_dir)
        if result:
            # Calculate score
            score = calculate_score(result, config)
            result['score'] = score
            results.append(result)
            print(f"Score: {score:.4f}")
        
        # Save intermediate results
        save_results(results, results_dir)
    
    # Find optimal parameters
    if results:
        optimal_result = max(results, key=lambda x: x['score'])
        
        # Update config with optimal parameters
        optimal_config_path = os.path.join(results_dir, 'optimal_config.json')
        update_config(config, optimal_result['params'], optimal_config_path)
        
        # Print optimal parameters
        print("\n=== Optimal Parameters ===")
        for param, value in optimal_result['params'].items():
            print(f"{param}: {value}")
        
        # Print optimal results
        print("\n=== Optimal Results ===")
        for metric, value in optimal_result.items():
            if metric not in ['params', 'backtest_id']:
                print(f"{metric}: {value}")
        
        # Generate optimization report
        generate_report(results, results_dir, config)
        
        return optimal_result
    else:
        print("No valid results found")
        return None

def save_results(results, results_dir):
    """Save optimization results to file"""
    results_file = os.path.join(results_dir, "optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

def generate_report(results, results_dir, config):
    """Generate optimization report with visualizations"""
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(results_dir, "optimization_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Create report directory
    report_dir = os.path.join(results_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate parameter distribution plots
    for param in config['optimization']['parameters']:
        param_name = param['name']
        if param_name in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[param_name], df['score'])
            plt.xlabel(param_name)
            plt.ylabel('Score')
            plt.title(f'Parameter Optimization: {param_name}')
            plt.grid(True)
            plt.savefig(os.path.join(report_dir, f"{param_name}_optimization.png"))
            plt.close()
    
    # Generate metric correlation matrix
    metrics = [metric['name'] for metric in config['optimization']['metrics']]
    metrics = [m for m in metrics if m in df.columns]
    
    if metrics:
        plt.figure(figsize=(10, 8))
        correlation = df[metrics].corr()
        plt.imshow(correlation, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(metrics)), metrics, rotation=45)
        plt.yticks(range(len(metrics)), metrics)
        plt.title('Metric Correlation Matrix')
        
        # Add correlation values
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                plt.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha='center', va='center',
                        color='white' if correlation.iloc[i, j] < 0.5 else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "metric_correlation.png"))
        plt.close()
    
    # Generate HTML report
    html_report = os.path.join(report_dir, "optimization_report.html")
    with open(html_report, 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>Strategy Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ font-weight: bold; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Strategy Optimization Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Optimal Parameters</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
        """)
        
        # Add optimal parameters
        optimal_result = max(results, key=lambda x: x['score'])
        for param, value in optimal_result['params'].items():
            f.write(f"<tr><td>{param}</td><td>{value}</td></tr>\n")
        
        f.write("""
            </table>
            
            <h2>Optimal Results</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """)
        
        # Add optimal results
        for metric, value in optimal_result.items():
            if metric not in ['params', 'backtest_id']:
                f.write(f"<tr><td>{metric}</td><td>{value}</td></tr>\n")
        
        f.write("""
            </table>
            
            <h2>Parameter Optimization Plots</h2>
        """)
        
        # Add parameter plots
        for param in config['optimization']['parameters']:
            param_name = param['name']
            plot_path = f"{param_name}_optimization.png"
            if os.path.exists(os.path.join(report_dir, plot_path)):
                f.write(f"<h3>{param_name}</h3>\n")
                f.write(f"<img src='{plot_path}' alt='{param_name} optimization'>\n")
        
        f.write("""
            <h2>Metric Correlation</h2>
            <img src='metric_correlation.png' alt='Metric correlation matrix'>
            
            <h2>All Results</h2>
            <table>
                <tr>
                    <th>Score</th>
        """)
        
        # Add column headers
        for col in df.columns:
            if col not in ['params', 'backtest_id']:
                f.write(f"<th>{col}</th>\n")
        
        f.write("</tr>\n")
        
        # Add all results
        for _, row in df.sort_values('score', ascending=False).iterrows():
            f.write("<tr>\n")
            f.write(f"<td>{row['score']:.4f}</td>\n")
            for col in df.columns:
                if col not in ['params', 'score', 'backtest_id']:
                    f.write(f"<td>{row[col]}</td>\n")
            f.write("</tr>\n")
        
        f.write("""
            </table>
        </body>
        </html>
        """)
    
    print(f"Report generated at {html_report}")

def main():
    """Main function"""
    # Define paths
    project_dir = "/workspace/finaly/MultiTimeframeStrategy"
    config_path = os.path.join(project_dir, "enhanced_config.json")
    project_name = "MultiTimeframeStrategy"
    
    # Run optimization
    optimize(config_path, project_dir, project_name, max_iterations=5)

if __name__ == "__main__":
    main()
