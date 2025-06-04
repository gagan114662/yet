#!/usr/bin/env python3
"""
Test Real Backtesting with Lean CLI
===================================

This script tests a single strategy backtest using real Lean CLI to ensure
everything is working correctly before running the full target-seeking system.
"""

import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

def test_real_backtest():
    """Test a single real backtest to verify the setup"""
    
    print("=" * 60)
    print("🧪 TESTING REAL BACKTEST WITH LEAN CLI")
    print("=" * 60)
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import modules
        from strategy_utils import generate_next_strategy
        from backtester import Backtester
        import config
        
        print("✅ All modules imported successfully")
        print(f"🎯 Targets: {config.TARGET_METRICS}")
        print(f"🔧 Lean CLI Path: {config.LEAN_CLI_PATH}")
        print(f"👤 User ID: {config.LEAN_CLI_USER_ID}")
        print(f"📅 Date Range: {config.BACKTEST_START_DATE} to {config.BACKTEST_END_DATE}")
        print()
        
        # Test strategy generation
        print("🧬 Generating test strategy...")
        strategy = generate_next_strategy()
        print(f"   Generated: {strategy['name']}")
        print(f"   Type: {strategy['type']}")
        print(f"   Target CAGR: {strategy.get('target_cagr', 'N/A')}")
        print(f"   Leverage: {strategy.get('leverage', 'N/A')}")
        print()
        
        # Initialize backtester
        print("🏗️  Initializing backtester...")
        backtester = Backtester()
        print(f"   Using QC Integration: {backtester.use_qc_integration}")
        print(f"   Lean CLI Path: {backtester.lean_cli_path}")
        print()
        
        # Test strategy code generation
        print("📝 Generating strategy code...")
        strategy_code = backtester._generate_strategy_code_from_idea(strategy)
        print(f"   ✅ Code generated ({len(strategy_code)} characters)")
        print(f"   ✅ Contains class: {'class' in strategy_code}")
        print(f"   ✅ Contains indicators: {'RSI' in strategy_code or 'MACD' in strategy_code}")
        print()
        
        # Show sample of generated code
        print("📄 Sample of generated strategy code:")
        lines = strategy_code.split('\\n')[:15]
        for i, line in enumerate(lines, 1):
            print(f"   {i:2d}: {line}")
        print("   ... (truncated)")
        print()
        
        # Run actual backtest
        print("🚀 RUNNING REAL BACKTEST...")
        print("   This may take 30-60 seconds depending on strategy complexity...")
        print()
        
        start_time = time.time()
        results = backtester.backtest_strategy(strategy)
        end_time = time.time()
        
        print(f"⏱️  Backtest completed in {end_time - start_time:.1f} seconds")
        print()
        
        # Analyze results
        print("📊 BACKTEST RESULTS:")
        print("=" * 40)
        
        if 'error' in results:
            print("❌ BACKTEST FAILED:")
            print(f"   Error: {results['error']}")
            print(f"   Details: {results.get('details', 'No details')}")
            
            # Provide troubleshooting
            print()
            print("🔧 TROUBLESHOOTING:")
            print("   1. Check Lean CLI installation: lean --version")
            print("   2. Verify credentials in config.py")
            print("   3. Check internet connection")
            print("   4. Ensure lean_workspace directory has proper permissions")
            
        else:
            print("✅ BACKTEST SUCCESSFUL!")
            print()
            for metric, value in results.items():
                if metric == 'lean_cli_output':
                    continue  # Skip raw output
                if isinstance(value, float):
                    if metric == 'cagr':
                        print(f"   📈 {metric.upper()}: {value*100:.2f}%")
                    elif metric == 'max_drawdown':
                        print(f"   📉 {metric.upper()}: {value*100:.2f}%")
                    elif metric == 'sharpe_ratio':
                        print(f"   📊 {metric.upper()}: {value:.2f}")
                    elif metric == 'avg_profit':
                        print(f"   💰 {metric.upper()}: {value*100:.3f}%")
                    else:
                        print(f"   📋 {metric.upper()}: {value:.4f}")
                else:
                    print(f"   📋 {metric.upper()}: {value}")
            
            print()
            print("🎯 TARGET ANALYSIS:")
            target_met = True
            for metric, target in config.TARGET_METRICS.items():
                if metric in results:
                    value = results[metric]
                    if metric == 'max_drawdown':
                        # Lower is better for drawdown
                        meets_target = value <= target
                        status = "✅" if meets_target else "❌"
                        print(f"   {status} {metric}: {value*100:.2f}% (target: <{target*100:.1f}%)")
                    else:
                        # Higher is better for other metrics
                        meets_target = value >= target
                        status = "✅" if meets_target else "❌"
                        if metric == 'cagr':
                            print(f"   {status} {metric}: {value*100:.2f}% (target: >{target*100:.1f}%)")
                        elif metric == 'avg_profit':
                            print(f"   {status} {metric}: {value*100:.3f}% (target: >{target*100:.3f}%)")
                        else:
                            print(f"   {status} {metric}: {value:.2f} (target: >{target:.1f})")
                    
                    if not meets_target:
                        target_met = False
            
            print()
            if target_met:
                print("🏆 AMAZING! This strategy meets ALL targets!")
                print("🚀 Ready to run full target-seeking system!")
            else:
                print("📈 Strategy doesn't meet all targets yet (expected)")
                print("🔄 The full system will iterate to find better strategies!")
    
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you're in the algorithmic_trading_system directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
        return False
    
    print()
    print("=" * 60)
    print("✅ REAL BACKTEST TEST COMPLETED")
    print("=" * 60)
    print()
    print("🚀 Next steps:")
    print("   1. If test passed: python3 run_target_seeking_system.py")
    print("   2. If test failed: Check troubleshooting steps above")
    print()
    
    return True

if __name__ == '__main__':
    test_real_backtest()