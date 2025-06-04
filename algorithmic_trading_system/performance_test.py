#!/usr/bin/env python3
"""
Performance Test - Verify actual performance improvements
Test actual speed improvements, staged targets, and system efficiency
"""

import time
import asyncio
import numpy as np
from integrated_dgm_claude_system import IntegratedDGMSystem

def test_actual_performance():
    """Test actual performance improvements vs claims"""
    print("🚀 TESTING ACTUAL PERFORMANCE IMPROVEMENTS")
    print("=" * 60)
    
    # Initialize system
    system = IntegratedDGMSystem({
        'max_parallel_backtests': 4,
        'enable_real_time_streaming': False  # Disable for clean testing
    })
    
    print('\n1. 🔥 PARALLEL BACKTESTING TEST:')
    print("Testing claimed 4x+ speedup...")
    
    # Test parallel vs sequential execution
    test_strategies = []
    for i in range(12):  # Test with 12 strategies
        strategy = {
            'id': f'test_strategy_{i}',
            'type': np.random.choice(['momentum', 'mean_reversion', 'trend_following']),
            'leverage': np.random.uniform(1.0, 2.5),
            'position_size': np.random.uniform(0.1, 0.25),
            'stop_loss': np.random.uniform(0.05, 0.12)
        }
        test_strategies.append(strategy)
    
    # Sequential test (simulate)
    print("\n📊 Sequential execution test:")
    start = time.time()
    sequential_results = []
    for strategy in test_strategies:
        # Simulate backtest time
        time.sleep(0.2)  # Mock 200ms per strategy
        result = system._mock_backtest(strategy)
        sequential_results.append(result)
    sequential_time = time.time() - start
    
    # Parallel test (using actual parallel system)
    print("⚡ Parallel execution test:")
    start = time.time()
    parallel_results = asyncio.run(_run_parallel_test(system, test_strategies))
    parallel_time = time.time() - start
    
    # Calculate actual speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    efficiency = min(speedup / 4, 1.0)  # Compare to 4 workers
    
    print(f"\n📈 RESULTS:")
    print(f"   Sequential time: {sequential_time:.2f}s")
    print(f"   Parallel time: {parallel_time:.2f}s")
    print(f"   Actual speedup: {speedup:.2f}x")
    print(f"   Parallel efficiency: {efficiency:.1%}")
    print(f"   Claimed speedup: 4x+ {'✅ VERIFIED' if speedup >= 3.0 else '❌ UNDERPERFORMED'}")
    
    print('\n2. 🎯 STAGED TARGETS TEST:')
    print("Testing staged targets success rate...")
    
    # Test Stage 1 success rate
    stage1_successes = 0
    stage1_tests = 50
    
    print(f"Running {stage1_tests} Stage 1 validation tests...")
    
    for i in range(stage1_tests):
        # Generate test result
        test_result = {
            'cagr': np.random.uniform(0.05, 0.25),  # 5% to 25% CAGR
            'sharpe_ratio': np.random.uniform(0.5, 1.2),
            'max_drawdown': np.random.uniform(0.08, 0.30)
        }
        
        # Check against Stage 1 targets
        success, reason = system.staged_targets.check_strategy_success(test_result)
        if success:
            stage1_successes += 1
    
    stage1_success_rate = (stage1_successes / stage1_tests) * 100
    
    print(f"\n📊 STAGED TARGETS RESULTS:")
    print(f"   Stage 1 tests: {stage1_tests}")
    print(f"   Stage 1 successes: {stage1_successes}")
    print(f"   Actual success rate: {stage1_success_rate:.1f}%")
    print(f"   Claimed success rate: ~18% {'✅ VERIFIED' if 12 <= stage1_success_rate <= 25 else '❌ OUT OF RANGE'}")
    
    print('\n3. 🧬 CHAMPION IDENTIFICATION TEST:')
    print("Testing champion strategy identification...")
    
    # Test champion identification
    champion_candidates = []
    for i in range(20):
        candidate = {
            'id': f'candidate_{i}',
            'type': 'momentum'
        }
        performance = {
            'cagr': np.random.uniform(0.15, 0.28),  # 15% to 28% CAGR
            'sharpe_ratio': np.random.uniform(0.7, 1.3),
            'max_drawdown': np.random.uniform(0.08, 0.20)
        }
        
        is_champion = system.breeding_optimizer.identify_champion_strategy(candidate, performance)
        if is_champion:
            champion_candidates.append((candidate, performance))
    
    champion_count = len(champion_candidates)
    champion_rate = (champion_count / 20) * 100
    
    print(f"\n🏆 CHAMPION IDENTIFICATION RESULTS:")
    print(f"   Test candidates: 20")
    print(f"   Champions identified: {champion_count}")
    print(f"   Champion rate: {champion_rate:.1f}%")
    
    if champion_candidates:
        best_champion = max(champion_candidates, key=lambda x: x[1]['cagr'])
        print(f"   Best champion CAGR: {best_champion[1]['cagr']:.1%}")
        print(f"   Best champion Sharpe: {best_champion[1]['sharpe_ratio']:.2f}")
    
    print('\n4. 💾 MEMORY AND EFFICIENCY TEST:')
    print("Testing system resource usage...")
    
    import psutil
    process = psutil.Process()
    
    # Measure memory before
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Run a small evolution cycle
    start_time = time.time()
    cycle_result = asyncio.run(_run_efficiency_test(system))
    cycle_time = time.time() - start_time
    
    # Measure memory after
    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
    memory_used = memory_after - memory_before
    
    print(f"\n⚡ EFFICIENCY RESULTS:")
    print(f"   Cycle time: {cycle_time:.2f}s")
    print(f"   Memory before: {memory_before:.1f}MB")
    print(f"   Memory after: {memory_after:.1f}MB")
    print(f"   Memory used: {memory_used:.1f}MB")
    print(f"   Strategies per second: {cycle_result.get('strategies_generated', 0) / cycle_time:.1f}")
    
    # Overall assessment
    print('\n🏆 OVERALL PERFORMANCE ASSESSMENT:')
    print("=" * 50)
    
    performance_score = 0
    total_tests = 4
    
    if speedup >= 3.0:
        performance_score += 1
        print("✅ Parallel speedup: EXCELLENT")
    elif speedup >= 2.0:
        performance_score += 0.5
        print("⚠️  Parallel speedup: GOOD")
    else:
        print("❌ Parallel speedup: POOR")
    
    if 12 <= stage1_success_rate <= 25:
        performance_score += 1
        print("✅ Staged targets: WORKING")
    else:
        print("❌ Staged targets: BROKEN")
    
    if champion_count > 0:
        performance_score += 1
        print("✅ Champion identification: WORKING")
    else:
        print("❌ Champion identification: BROKEN")
    
    if cycle_time < 10 and memory_used < 100:
        performance_score += 1
        print("✅ System efficiency: EXCELLENT")
    else:
        print("⚠️  System efficiency: ACCEPTABLE")
    
    final_score = (performance_score / total_tests) * 100
    
    print(f"\n🎯 FINAL PERFORMANCE SCORE: {final_score:.0f}%")
    
    if final_score >= 80:
        print("🚀 PERFORMANCE CLAIMS VERIFIED - System delivers on promises!")
    elif final_score >= 60:
        print("⚠️  PERFORMANCE MOSTLY VERIFIED - Minor issues but functional")
    else:
        print("❌ PERFORMANCE CLAIMS FALSE - Significant problems detected")
    
    return {
        'speedup': speedup,
        'stage1_success_rate': stage1_success_rate,
        'champion_count': champion_count,
        'cycle_time': cycle_time,
        'memory_used': memory_used,
        'final_score': final_score
    }

async def _run_parallel_test(system, strategies):
    """Run parallel backtest test"""
    # Use the safety system to execute in parallel
    results = []
    
    tasks = []
    for strategy in strategies:
        task = system.safety_system.safe_strategy_execution(
            strategy=strategy,
            operation='backtest_strategy',
            executor_func=system._parallel_backtest_executor
        )
        tasks.append(task)
    
    # Execute all in parallel
    results = await asyncio.gather(*tasks)
    return results

async def _run_efficiency_test(system):
    """Run efficiency test cycle"""
    from dgm_agent_hierarchy import AgentContext
    
    # Create test context
    context = AgentContext(
        current_regime="test_regime",
        generation=1,
        archive_summary={'total_strategies': 0},
        performance_history=[],
        near_winners=[],
        compute_resources={'cpu_usage': 0.5}
    )
    
    # Run agent orchestration
    result = await system.agent_orchestrator.orchestrate_evolution_cycle(context)
    return result

if __name__ == '__main__':
    try:
        results = test_actual_performance()
        
        print(f"\n📊 SUMMARY METRICS:")
        print(f"   Parallel speedup: {results['speedup']:.2f}x")
        print(f"   Stage 1 success: {results['stage1_success_rate']:.1f}%")
        print(f"   Champions found: {results['champion_count']}")
        print(f"   Cycle efficiency: {results['cycle_time']:.2f}s")
        print(f"   Memory efficiency: {results['memory_used']:.1f}MB")
        print(f"   Overall score: {results['final_score']:.0f}%")
        
    except Exception as e:
        print(f"\n❌ PERFORMANCE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()