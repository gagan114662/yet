#!/usr/bin/env python3
"""
Deploy and backtest high-performance trading strategies
Targets: 25%+ CAGR, 1.0+ Sharpe Ratio, <20% Max Drawdown
"""

import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

class StrategyDeployer:
    def __init__(self):
        self.workspace_dir = Path("/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace")
        self.strategies = [
            "quantum_edge_dominator",
            "apex_performance_engine", 
            "ultimate_alpha_generator",
            "gamma_scalper_pro",
            "microstructure_hunter",
            "crisis_alpha_harvester",
            "master_strategy_rotator",
            "volatility_harvester",
            "statistical_arbitrage",
            "multi_factor_alpha"
        ]
        self.results = {}
        
    def run_backtest(self, strategy_name):
        """Run backtest for a single strategy"""
        print(f"\n{'='*60}")
        print(f"Backtesting: {strategy_name}")
        print(f"{'='*60}")
        
        strategy_path = self.workspace_dir / strategy_name
        
        # Check if strategy exists
        if not strategy_path.exists():
            print(f"Strategy {strategy_name} not found, skipping...")
            return None
            
        # Create backtest directory
        backtest_dir = strategy_path / "backtests"
        backtest_dir.mkdir(exist_ok=True)
        
        # Run backtest
        try:
            cmd = ["lean", "backtest", str(strategy_name)]
            
            # Change to workspace directory
            os.chdir(self.workspace_dir)
            
            # Execute backtest
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Backtest completed successfully for {strategy_name}")
                
                # Parse results from output
                return self.parse_results(result.stdout, strategy_name)
            else:
                print(f"✗ Backtest failed for {strategy_name}")
                print(f"Error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"✗ Error running backtest for {strategy_name}: {str(e)}")
            return None
    
    def parse_results(self, output, strategy_name):
        """Parse backtest results from output"""
        results = {
            "strategy": strategy_name,
            "cagr": 0,
            "sharpe": 0,
            "max_drawdown": 0,
            "total_trades": 0,
            "win_rate": 0,
            "profit_loss_ratio": 0,
            "avg_profit": 0
        }
        
        # Parse statistics from output
        lines = output.split('\n')
        for line in lines:
            if "Compounding Annual Return" in line:
                try:
                    results["cagr"] = float(line.split()[-1].strip('%'))
                except:
                    pass
            elif "Sharpe Ratio" in line and "Probabilistic" not in line:
                try:
                    results["sharpe"] = float(line.split()[-1])
                except:
                    pass
            elif "Drawdown" in line and "%" in line:
                try:
                    results["max_drawdown"] = abs(float(line.split()[-1].strip('%')))
                except:
                    pass
            elif "Total Orders" in line:
                try:
                    results["total_trades"] = int(line.split()[-1])
                except:
                    pass
            elif "Win Rate" in line:
                try:
                    results["win_rate"] = float(line.split()[-1].strip('%'))
                except:
                    pass
            elif "Profit-Loss Ratio" in line:
                try:
                    results["profit_loss_ratio"] = float(line.split()[-1])
                except:
                    pass
            elif "Average Win" in line:
                try:
                    results["avg_profit"] = float(line.split()[-1].strip('%'))
                except:
                    pass
        
        return results
    
    def deploy_all_strategies(self):
        """Deploy and backtest all strategies"""
        print(f"\nStarting deployment of {len(self.strategies)} high-performance strategies")
        print(f"Target metrics: CAGR > 25%, Sharpe > 1.0, Max DD < 20%")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        successful = 0
        failed = 0
        
        for strategy in self.strategies:
            result = self.run_backtest(strategy)
            
            if result:
                self.results[strategy] = result
                successful += 1
                
                # Check if meets targets
                meets_targets = (
                    result["cagr"] > 25 and
                    result["sharpe"] > 1.0 and
                    result["max_drawdown"] < 20 and
                    result["avg_profit"] > 0.75
                )
                
                if meets_targets:
                    print(f"🎯 {strategy} MEETS ALL TARGETS!")
                else:
                    print(f"📊 {strategy} results: CAGR={result['cagr']:.1f}%, Sharpe={result['sharpe']:.2f}, DD={result['max_drawdown']:.1f}%")
            else:
                failed += 1
            
            # Small delay between backtests
            time.sleep(2)
        
        print(f"\n{'='*60}")
        print(f"Deployment Summary")
        print(f"{'='*60}")
        print(f"Total strategies: {len(self.strategies)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        # Save results
        self.save_results()
        
        # Generate report
        self.generate_report()
    
    def save_results(self):
        """Save results to JSON file"""
        results_file = self.workspace_dir / "backtest_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def generate_report(self):
        """Generate performance report"""
        if not self.results:
            print("No results to report")
            return
        
        print(f"\n{'='*60}")
        print(f"Performance Report - Target Achievement")
        print(f"{'='*60}")
        print(f"{'Strategy':<30} {'CAGR':<10} {'Sharpe':<10} {'Max DD':<10} {'Status':<10}")
        print(f"{'-'*60}")
        
        strategies_meeting_targets = []
        
        for strategy, result in self.results.items():
            cagr = result['cagr']
            sharpe = result['sharpe']
            max_dd = result['max_drawdown']
            
            # Check targets
            meets_all = (
                cagr > 25 and
                sharpe > 1.0 and
                max_dd < 20 and
                result.get('avg_profit', 0) > 0.75
            )
            
            status = "✓ PASS" if meets_all else "✗ FAIL"
            
            if meets_all:
                strategies_meeting_targets.append(strategy)
            
            print(f"{strategy:<30} {cagr:>8.1f}% {sharpe:>9.2f} {max_dd:>8.1f}% {status:<10}")
        
        print(f"\n{'='*60}")
        print(f"Strategies Meeting ALL Targets: {len(strategies_meeting_targets)}")
        
        if strategies_meeting_targets:
            print("\nTop Performers:")
            for strategy in strategies_meeting_targets:
                print(f"  ✓ {strategy}")
                print(f"    - CAGR: {self.results[strategy]['cagr']:.1f}%")
                print(f"    - Sharpe: {self.results[strategy]['sharpe']:.2f}")
                print(f"    - Max DD: {self.results[strategy]['max_drawdown']:.1f}%")
                print(f"    - Avg Profit: {self.results[strategy].get('avg_profit', 0):.2f}%")
        
        # Create deployment script for winning strategies
        if strategies_meeting_targets:
            self.create_deployment_script(strategies_meeting_targets)
    
    def create_deployment_script(self, winning_strategies):
        """Create script to deploy winning strategies"""
        script_content = f"""#!/bin/bash
# Deploy winning strategies to QuantConnect Cloud
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

echo "Deploying {len(winning_strategies)} high-performance strategies to QuantConnect Cloud"

STRATEGIES=(
"""
        
        for strategy in winning_strategies:
            script_content += f'    "{strategy}"\n'
        
        script_content += """)

for strategy in "${STRATEGIES[@]}"; do
    echo "Deploying $strategy..."
    lean cloud push --project "$strategy"
    
    # Optional: Start live trading
    # lean cloud live --project "$strategy" --brokerage "Interactive Brokers" --data-provider "QuantConnect"
    
    sleep 5
done

echo "Deployment complete!"
"""
        
        script_path = self.workspace_dir / "deploy_winners.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"\nDeployment script created: {script_path}")
        print(f"Run './deploy_winners.sh' to deploy winning strategies to cloud")

def main():
    """Main execution"""
    deployer = StrategyDeployer()
    
    # First, let's just test with the strategies we created
    deployer.strategies = [
        "quantum_edge_dominator",
        "apex_performance_engine",
        "ultimate_alpha_generator"
    ]
    
    deployer.deploy_all_strategies()

if __name__ == "__main__":
    main()