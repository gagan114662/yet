#!/usr/bin/env python3
"""
Enhanced Target-Seeking Algorithmic Trading System
==================================================

This system generates and tests sophisticated trading strategies to meet aggressive targets:
- 25% annual return (CAGR)
- 1.0+ Sharpe ratio
- Maximum 15% drawdown
- 0.2% average profit per trade

The system uses:
1. Sophisticated strategy templates from lean_workspace
2. Real Lean CLI backtesting with QuantConnect integration
3. Multi-factor strategies with leverage and risk management
4. Adaptive strategy generation and optimization

Usage:
    python3 run_target_seeking_system.py [--iterations=2000] [--mock] [--verbose]
"""

import sys
import argparse
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, '.')

def main():
    parser = argparse.ArgumentParser(description='Run the Target-Seeking Trading System')
    parser.add_argument('--iterations', type=int, default=2000, 
                       help='Maximum iterations to run (default: 2000)')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock backtesting for testing (faster)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--targets-only', action='store_true',
                       help='Only show target configuration and exit')
    
    args = parser.parse_args()
    
    # Import after argument parsing to handle potential import errors
    try:
        from controller import TargetSeekingController
        import config
        from strategy_importer import StrategyImporter
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the algorithmic_trading_system directory.")
        sys.exit(1)
    
    # Display system information
    print("=" * 70)
    print("🚀 ENHANCED TARGET-SEEKING ALGORITHMIC TRADING SYSTEM")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Targets: {config.TARGET_METRICS}")
    print(f"📊 Required Successful Strategies: {config.REQUIRED_SUCCESSFUL_STRATEGIES}")
    print(f"⚙️  Max Iterations: {args.iterations}")
    print(f"🔧 Mock Mode: {args.mock}")
    print()
    
    if args.targets_only:
        print("Target configuration displayed. Exiting.")
        return
    
    # Initialize strategy importer and show available templates
    try:
        importer = StrategyImporter()
        available_strategies = importer.list_available_strategies()
        print("📚 Available Strategy Templates:")
        for i, strategy in enumerate(available_strategies[:5], 1):
            print(f"  {i}. {strategy['name']} ({strategy['type']}) - Target: {strategy['target_cagr']*100:.0f}% CAGR")
        print(f"  ... and {len(available_strategies)-5} more sophisticated templates")
        print()
    except Exception as e:
        print(f"⚠️  Warning: Strategy importer error: {e}")
        print("Continuing with basic strategy generation...")
        print()
    
    # Initialize the controller
    print("🏗️  Initializing Target-Seeking Controller...")
    try:
        controller = TargetSeekingController()
        print(f"✅ Controller initialized successfully")
        print(f"🔄 Backtester type: {'QuantConnect' if controller.backtester.use_qc_integration else 'Lean CLI'}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize controller: {e}")
        sys.exit(1)
    
    # Set up mock mode if requested
    if args.mock:
        print("🧪 Setting up mock backtesting for rapid testing...")
        setup_mock_backtesting(controller)
        print("✅ Mock backtesting enabled")
        print()
    
    # Display strategy generation test
    print("🧬 Testing Strategy Generation (3 samples):")
    try:
        from strategy_utils import generate_next_strategy
        for i in range(3):
            strategy = generate_next_strategy()
            print(f"  Sample {i+1}: {strategy['name']}")
            print(f"    Type: {strategy['type']}, Target CAGR: {strategy.get('target_cagr', 'N/A')}")
            print(f"    Leverage: {strategy.get('leverage', 'N/A')}, Position Size: {strategy.get('position_size', 'N/A')}")
        print("✅ Strategy generation working correctly")
        print()
    except Exception as e:
        print(f"⚠️  Strategy generation test failed: {e}")
        print()
    
    # Run the main loop
    print("🎯 STARTING TARGET-SEEKING LOOP")
    print("=" * 50)
    print("Seeking strategies that meet ALL targets:")
    for metric, target in config.TARGET_METRICS.items():
        if metric == 'max_drawdown':
            print(f"  📉 {metric}: < {target*100:.1f}% (lower is better)")
        elif metric in ['cagr', 'sharpe_ratio']:
            print(f"  📈 {metric}: > {target*100 if metric=='cagr' else target:.1f}{'%' if metric=='cagr' else ''}")
        else:
            print(f"  📊 {metric}: > {target*100:.2f}%")
    print()
    print("🔥 Starting optimization loop...")
    print("   Use Ctrl+C to stop gracefully")
    print()
    
    # Modify iteration limit for this run
    original_limit = getattr(controller, 'iteration_limit', 2000)
    controller.iteration_limit = args.iterations
    
    try:
        start_time = time.time()
        controller.run_until_success()
        end_time = time.time()
        
        print()
        print("=" * 70)
        print("🏆 TARGET-SEEKING SESSION COMPLETED")
        print("=" * 70)
        print(f"⏱️  Total Runtime: {end_time - start_time:.1f} seconds")
        print(f"🔄 Total Iterations: {controller.iteration_count}")
        print(f"✅ Successful Strategies Found: {len(controller.successful_strategies)}")
        print(f"🎯 Target Achievement: {len(controller.successful_strategies)}/{controller.required_successful_strategies}")
        
        if len(controller.successful_strategies) >= controller.required_successful_strategies:
            print()
            print("🎉 SUCCESS! All targets achieved!")
            print("🚀 The system has found strategies that meet your aggressive targets.")
        else:
            print()
            print("⏳ Session ended before all targets were met.")
            print(f"💡 Consider running longer or adjusting target parameters.")
            
            # Suggest optimizations
            print()
            print("🔧 Optimization Suggestions:")
            print("  - Increase iterations: --iterations=5000")
            print("  - Use mock mode for testing: --mock")
            print("  - Check Lean CLI configuration in config.py")
            
    except KeyboardInterrupt:
        print()
        print("🛑 Stopped by user (Ctrl+C)")
        print(f"🔄 Completed {controller.iteration_count} iterations")
        print(f"✅ Found {len(controller.successful_strategies)} successful strategies")
    except Exception as e:
        print()
        print(f"❌ Error during execution: {e}")
        print("Check logs above for details.")
        sys.exit(1)
    finally:
        # Restore original limit
        controller.iteration_limit = original_limit
    
    print()
    print("💾 Session data and successful strategies saved in controller object.")
    print("🔄 Run again with different parameters to continue optimization.")
    print("=" * 70)

def setup_mock_backtesting(controller):
    """Set up mock backtesting for rapid testing"""
    import config
    
    original_backtest_method = controller.backtester.backtest_strategy
    generated_successful_count = [0]
    
    def mock_backtest_strategy(strategy_idea_dict):
        strategy_name = strategy_idea_dict.get('name', 'UnnamedMockStrategy')
        
        # Print abbreviated info for mock mode
        if controller.iteration_count % 50 == 0:  # Print every 50th iteration
            print(f"   Mock testing: {strategy_name[:30]}... (iteration {controller.iteration_count})")
        
        # Generate successful results every 15 iterations to test success path
        if (generated_successful_count[0] < controller.required_successful_strategies and 
            controller.iteration_count % 15 == 0):
            generated_successful_count[0] += 1
            
            # Generate results that exceed targets
            results = {
                'cagr': config.TARGET_METRICS['cagr'] * (1.1 + generated_successful_count[0] * 0.1),
                'max_drawdown': config.TARGET_METRICS['max_drawdown'] * 0.8,  # Better drawdown
                'sharpe_ratio': config.TARGET_METRICS['sharpe_ratio'] * (1.2 + generated_successful_count[0] * 0.2),
                'avg_profit': config.TARGET_METRICS['avg_profit'] * (1.5 + generated_successful_count[0] * 0.3),
                'total_trades': 150 + generated_successful_count[0] * 20,
                'mock_mode': True
            }
            
            if controller.iteration_count % 50 == 0:
                print(f"   🎯 MOCK SUCCESS: {strategy_name[:30]} meets targets!")
        else:
            # Generate failing results
            results = {
                'cagr': config.TARGET_METRICS['cagr'] * 0.6,  # Below target
                'max_drawdown': config.TARGET_METRICS['max_drawdown'] * 1.4,  # Worse drawdown
                'sharpe_ratio': config.TARGET_METRICS['sharpe_ratio'] * 0.7,  # Below target
                'avg_profit': config.TARGET_METRICS['avg_profit'] * 0.5,  # Below target
                'total_trades': 80,
                'mock_mode': True
            }
        
        return results
    
    # Override the backtest method
    controller.backtester.backtest_strategy = mock_backtest_strategy
    
    # Also reduce sleep time for faster testing
    controller.original_sleep_time = getattr(controller, 'sleep_time', 0.001)
    controller.sleep_time = 0.001

if __name__ == '__main__':
    main()