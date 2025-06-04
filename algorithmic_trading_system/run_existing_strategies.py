#!/usr/bin/env python3
"""
Run Existing High-Performance Strategies
========================================

This script runs backtests on your existing lean_workspace strategies 
to identify which ones meet your aggressive targets without generating new strategies.
This is more efficient and uses proven algorithms.
"""

import sys
import os
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

def analyze_strategy_results(results_file: str) -> dict:
    """Analyze Lean backtest results to extract key metrics"""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract statistics from Lean results
        stats = data.get('Statistics', {})
        
        # Parse key metrics
        metrics = {
            'cagr': float(stats.get('Compounding Annual Return', '0').strip('%')) / 100,
            'sharpe_ratio': float(stats.get('Sharpe Ratio', '0')),
            'max_drawdown': abs(float(stats.get('Drawdown', '0').strip('%'))) / 100,
            'total_trades': int(stats.get('Total Trades', '0')),
            'win_rate': float(stats.get('Win Rate', '0').strip('%')) / 100,
            'total_fees': float(stats.get('Total Fees', '$0').replace('$', '').replace(',', '')),
        }
        
        # Calculate average profit per trade
        if metrics['total_trades'] > 0:
            metrics['avg_profit'] = metrics['cagr'] / metrics['total_trades']
        else:
            metrics['avg_profit'] = 0
            
        return metrics
        
    except Exception as e:
        print(f"Error parsing results: {e}")
        return None

def meets_targets(metrics: dict, targets: dict) -> bool:
    """Check if strategy metrics meet targets"""
    if not metrics:
        return False
        
    try:
        # Check each target
        if metrics['cagr'] < targets['cagr']:
            return False
        if metrics['max_drawdown'] > targets['max_drawdown']:  # Lower is better
            return False
        if metrics['sharpe_ratio'] < targets['sharpe_ratio']:
            return False
        if metrics['avg_profit'] < targets['avg_profit']:
            return False
            
        return True
    except:
        return False

def run_backtest_on_strategy(strategy_path: str, lean_cli_path: str) -> dict:
    """Run Lean CLI backtest on a strategy directory"""
    try:
        print(f"   🚀 Running backtest on {os.path.basename(strategy_path)}...")
        
        # Run lean backtest
        cmd = [lean_cli_path, "backtest", strategy_path, "--quiet"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(strategy_path))
        
        if result.returncode != 0:
            print(f"   ❌ Backtest failed: {result.stderr}")
            return None
            
        # Find results file
        backtests_dir = os.path.join(strategy_path, "backtests")
        if not os.path.exists(backtests_dir):
            print(f"   ❌ No backtests directory found")
            return None
            
        # Find latest results.json
        latest_results = None
        latest_time = 0
        
        for item in os.listdir(backtests_dir):
            item_path = os.path.join(backtests_dir, item)
            if os.path.isdir(item_path):
                results_file = os.path.join(item_path, "results.json")
                if os.path.exists(results_file):
                    mtime = os.path.getmtime(results_file)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_results = results_file
        
        if latest_results:
            metrics = analyze_strategy_results(latest_results)
            print(f"   ✅ Backtest completed, metrics extracted")
            return metrics
        else:
            print(f"   ❌ No results.json found")
            return None
            
    except Exception as e:
        print(f"   ❌ Error running backtest: {e}")
        return None

def main():
    print("=" * 70)
    print("🎯 TESTING EXISTING HIGH-PERFORMANCE STRATEGIES")
    print("=" * 70)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Import config
    try:
        import config
        targets = config.TARGET_METRICS
        lean_cli_path = config.LEAN_CLI_PATH
        lean_workspace = "../lean_workspace"
        
        print(f"🎯 Targets: {targets}")
        print(f"🔧 Lean CLI: {lean_cli_path}")
        print(f"📁 Workspace: {lean_workspace}")
        print()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Priority strategies to test (known high performers)
    priority_strategies = [
        "target_crusher_ultimate",
        "quantum_edge_dominator", 
        "ultimate_alpha_generator",
        "extreme_performance_2025",
        "microstructure_strategy",
        "VandanStrategyBurke",
        "MultiTimeframeStrategy"
    ]
    
    print("🔍 Testing Priority Strategies:")
    successful_strategies = []
    
    for strategy_name in priority_strategies:
        strategy_path = os.path.join(lean_workspace, strategy_name)
        
        if os.path.exists(strategy_path) and os.path.exists(os.path.join(strategy_path, "main.py")):
            print(f"📊 Testing {strategy_name}...")
            
            # Check if recent backtest exists
            backtests_dir = os.path.join(strategy_path, "backtests")
            recent_results = None
            
            if os.path.exists(backtests_dir):
                # Look for recent results (within last day)
                for item in os.listdir(backtests_dir):
                    item_path = os.path.join(backtests_dir, item)
                    if os.path.isdir(item_path):
                        results_file = os.path.join(item_path, "results.json")
                        if os.path.exists(results_file):
                            # Check if recent (within 24 hours)
                            if (time.time() - os.path.getmtime(results_file)) < 86400:
                                recent_results = results_file
                                break
            
            if recent_results:
                print(f"   📋 Using recent backtest results...")
                metrics = analyze_strategy_results(recent_results)
            else:
                # Run new backtest
                metrics = run_backtest_on_strategy(strategy_path, lean_cli_path)
            
            if metrics:
                print(f"   📈 CAGR: {metrics['cagr']*100:.1f}%")
                print(f"   📊 Sharpe: {metrics['sharpe_ratio']:.2f}")
                print(f"   📉 Max DD: {metrics['max_drawdown']*100:.1f}%") 
                print(f"   💰 Avg Profit: {metrics['avg_profit']*100:.3f}%")
                print(f"   📋 Trades: {metrics['total_trades']}")
                
                if meets_targets(metrics, targets):
                    print(f"   🏆 SUCCESS! {strategy_name} meets ALL targets!")
                    successful_strategies.append((strategy_name, metrics))
                else:
                    print(f"   📈 Doesn't meet all targets yet")
            else:
                print(f"   ❌ Failed to get metrics")
        else:
            print(f"❌ Strategy {strategy_name} not found or missing main.py")
        
        print()
    
    # Results summary
    print("=" * 70)
    print("🏆 RESULTS SUMMARY")
    print("=" * 70)
    
    if successful_strategies:
        print(f"✅ Found {len(successful_strategies)} strategies meeting ALL targets:")
        print()
        
        for i, (strategy_name, metrics) in enumerate(successful_strategies, 1):
            print(f"{i}. 🎯 {strategy_name.upper()}")
            print(f"   📈 CAGR: {metrics['cagr']*100:.1f}% (target: {targets['cagr']*100:.0f}%+)")
            print(f"   📊 Sharpe: {metrics['sharpe_ratio']:.2f} (target: {targets['sharpe_ratio']:.1f}+)")
            print(f"   📉 Max DD: {metrics['max_drawdown']*100:.1f}% (target: <{targets['max_drawdown']*100:.0f}%)")
            print(f"   💰 Avg Profit: {metrics['avg_profit']*100:.3f}% (target: {targets['avg_profit']*100:.3f}%+)")
            print(f"   📋 Total Trades: {metrics['total_trades']}")
            print()
        
        print("🚀 CONGRATULATIONS!")
        print("Your existing strategies already meet your aggressive targets!")
        print("These strategies are ready for live trading or further optimization.")
        
    else:
        print("📈 No existing strategies currently meet ALL targets.")
        print("Recommendations:")
        print("  1. Run the target-seeking system to generate optimized strategies")
        print("  2. Modify existing strategies with higher leverage or position sizing")
        print("  3. Combine multiple strategies for portfolio approach")
        
    print()
    print("🔄 Next steps:")
    if successful_strategies:
        print("  ✅ Your existing strategies are performing excellently!")
        print("  🚀 Consider deploying these for live trading")
        print("  📊 Run: python3 run_target_seeking_system.py for additional strategies")
    else:
        print("  🔄 Run: python3 run_target_seeking_system.py")
        print("  🎯 The system will generate strategies to meet your targets")
    
    print("=" * 70)

if __name__ == '__main__':
    main()