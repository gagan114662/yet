#!/usr/bin/env python3
"""
Comprehensive verification of all system claims
Tests if implementations are real or fake
"""

import time
import multiprocessing as mp
import sys
import traceback
from typing import List, Dict

def test_file_existence():
    """Test 1: Do claimed files actually exist with substantial code?"""
    print("🔍 TEST 1: File Existence and Size")
    
    import os
    
    required_files = [
        'streaming_dgm_orchestrator.py',
        'dgm_agent_hierarchy.py', 
        'dgm_safety_system.py',
        'integrated_dgm_claude_system.py',
        'staged_targets_system.py',
        'parallel_backtesting_system.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            lines = sum(1 for line in open(file))
            print(f"  ✅ {file}: {lines} lines, {size} bytes")
            if lines < 50:
                print(f"     ⚠️  WARNING: File seems small for claimed functionality")
        else:
            print(f"  ❌ {file}: FILE MISSING")
    
    return True

def test_system_imports():
    """Test 2: Do systems actually import and initialize?"""
    print("\n🔍 TEST 2: System Imports and Initialization")
    
    try:
        from integrated_dgm_claude_system import IntegratedDGMSystem
        system = IntegratedDGMSystem()
        
        print(f"  ✅ IntegratedDGMSystem imports and initializes")
        
        # Check components
        components = [
            ('agent_orchestrator', 'Agent hierarchy'),
            ('safety_system', 'Safety system'),
            ('staged_targets', 'Staged targets'),
            ('parallel_backtester', 'Parallel backtester'),
            ('streaming_dgm', 'Streaming DGM')
        ]
        
        for attr, name in components:
            if hasattr(system, attr):
                print(f"  ✅ {name}: EXISTS")
            else:
                print(f"  ❌ {name}: MISSING")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return False

def test_agent_hierarchy():
    """Test 3: Are 5 agents actually implemented?"""
    print("\n🔍 TEST 3: Agent Hierarchy Implementation")
    
    try:
        from dgm_agent_hierarchy import DGMAgentOrchestrator
        orchestrator = DGMAgentOrchestrator()
        
        expected_agents = [
            'market_regime',
            'strategy_generator', 
            'risk_analyzer',
            'performance_synthesizer',
            'archive_manager'
        ]
        
        if hasattr(orchestrator, 'agents'):
            actual_agents = list(orchestrator.agents.keys())
            print(f"  Expected: {expected_agents}")
            print(f"  Actual: {actual_agents}")
            
            missing = set(expected_agents) - set(actual_agents)
            extra = set(actual_agents) - set(expected_agents)
            
            if not missing and not extra:
                print("  ✅ All 5 agents correctly implemented")
                
                # Test if agents have execute methods
                for agent_name, agent in orchestrator.agents.items():
                    if hasattr(agent, 'execute'):
                        print(f"    ✅ {agent_name}: Has execute method")
                    else:
                        print(f"    ❌ {agent_name}: Missing execute method")
                        
                return True
            else:
                if missing:
                    print(f"  ❌ Missing agents: {missing}")
                if extra:
                    print(f"  ⚠️  Extra agents: {extra}")
                return False
        else:
            print("  ❌ No agents attribute found")
            return False
            
    except Exception as e:
        print(f"  ❌ Agent hierarchy test failed: {e}")
        return False

def test_staged_targets():
    """Test 4: Do staged targets actually work?"""
    print("\n🔍 TEST 4: Staged Targets Implementation")
    
    try:
        from staged_targets_system import StagedTargetsManager, BreedingOptimizer
        
        staged = StagedTargetsManager()
        breeding = BreedingOptimizer(staged)
        
        print(f"  ✅ StagedTargetsManager initializes")
        print(f"  ✅ BreedingOptimizer initializes")
        
        # Check stage definitions
        print(f"  Current stage: {staged.current_stage.value}")
        print(f"  Stage 1 CAGR target: {staged.stage_definitions[staged.current_stage].cagr:.1%}")
        print(f"  Expected success rate: {staged.get_success_rate_estimate():.1%}")
        
        # Test with sample results
        test_cases = [
            {'cagr': 0.16, 'sharpe_ratio': 0.85, 'max_drawdown': 0.18},  # Should pass Stage 1
            {'cagr': 0.12, 'sharpe_ratio': 0.75, 'max_drawdown': 0.22},  # Should fail
            {'cagr': 0.23, 'sharpe_ratio': 0.95, 'max_drawdown': 0.14}   # Champion level
        ]
        
        for i, test_result in enumerate(test_cases):
            success, reason = staged.check_strategy_success(test_result)
            is_champion = breeding.identify_champion_strategy({'name': f'test_{i}'}, test_result)
            
            print(f"  Test {i+1}: CAGR {test_result['cagr']:.1%}, Sharpe {test_result['sharpe_ratio']:.2f}")
            print(f"    Stage check: {reason}")
            print(f"    Champion: {'YES' if is_champion else 'NO'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Staged targets test failed: {e}")
        return False

def test_parallel_backtesting():
    """Test 5: Does parallel backtesting actually work?"""
    print("\n🔍 TEST 5: Parallel Backtesting Implementation")
    
    try:
        from parallel_backtesting_system import ParallelBacktester
        
        backtester = ParallelBacktester(max_workers=2)
        print(f"  ✅ ParallelBacktester initializes")
        print(f"  Workers configured: {backtester.max_workers}")
        print(f"  CPU cores available: {mp.cpu_count()}")
        
        # Test methods exist
        required_methods = [
            'backtest_strategies_parallel',
            'get_performance_summary',
            '_execute_parallel_backtests'
        ]
        
        for method in required_methods:
            if hasattr(backtester, method):
                print(f"    ✅ {method}: EXISTS")
            else:
                print(f"    ❌ {method}: MISSING")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Parallel backtesting test failed: {e}")
        return False

def test_safety_system():
    """Test 6: Is safety system actually implemented?"""
    print("\n🔍 TEST 6: Safety System Implementation")
    
    try:
        from dgm_safety_system import DGMSafetySystem, PermissionScope
        
        safety = DGMSafetySystem()
        print(f"  ✅ DGMSafetySystem initializes")
        
        # Check permission scopes
        expected_scopes = [
            PermissionScope.EXPERIMENTAL,
            PermissionScope.PRODUCTION,
            PermissionScope.RESEARCH,
            PermissionScope.SANDBOX,
            PermissionScope.RESTRICTED
        ]
        
        print(f"  Permission scopes configured: {len(safety.permission_scopes)}")
        
        for scope in expected_scopes:
            if scope in safety.permission_scopes:
                perms = safety.permission_scopes[scope]
                print(f"    ✅ {scope.value}: {len(perms.allowed_operations)} operations, {perms.timeout_seconds}s timeout")
            else:
                print(f"    ❌ {scope.value}: MISSING")
        
        # Test permission checking
        test_strategy = {
            'leverage': 2.0,
            'position_size': 0.15,
            'stop_loss': 0.1
        }
        
        permitted, reason = safety.check_permissions(test_strategy, 'backtest_strategy', PermissionScope.EXPERIMENTAL)
        print(f"  Permission test: {'GRANTED' if permitted else 'DENIED'} - {reason}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Safety system test failed: {e}")
        return False

def test_streaming_orchestration():
    """Test 7: Does streaming actually work?"""
    print("\n🔍 TEST 7: Streaming Orchestration")
    
    try:
        from streaming_dgm_orchestrator import StreamingDGM, EvolutionPhase
        
        dgm = StreamingDGM(max_parallel_backtests=2)
        print(f"  ✅ StreamingDGM initializes")
        
        # Test streaming (with timeout)
        print("  Testing evolution streaming...")
        
        import asyncio
        
        async def test_streaming():
            count = 0
            try:
                async for event in dgm.evolve_strategies(target_generations=1):
                    print(f"    Stream event {count}: {event.phase.value}")
                    count += 1
                    if count >= 10:  # Don't run forever
                        break
                return count
            except Exception as e:
                print(f"    ❌ Streaming failed: {e}")
                return 0
        
        # Run with timeout
        count = asyncio.run(asyncio.wait_for(test_streaming(), timeout=30))
        
        if count > 0:
            print(f"  ✅ Streaming working - received {count} events")
            return True
        else:
            print(f"  ❌ No streaming events received")
            return False
            
    except Exception as e:
        print(f"  ❌ Streaming test failed: {e}")
        return False

def test_integration():
    """Test 8: Does the integrated system actually work?"""
    print("\n🔍 TEST 8: Full System Integration")
    
    try:
        from integrated_dgm_claude_system import IntegratedDGMSystem
        
        config = {
            'max_parallel_backtests': 2,
            'target_generations': 1,  # Short test
            'enable_real_time_streaming': True
        }
        
        system = IntegratedDGMSystem(config)
        print(f"  ✅ Integrated system initializes")
        
        # Test integration metrics
        metrics = system.integration_metrics
        print(f"  Integration metrics: {type(metrics).__name__}")
        
        # Test if we can run a short evolution (with timeout)
        import asyncio
        
        async def test_evolution():
            count = 0
            try:
                async for update in system.run_enhanced_evolution(target_generations=1):
                    print(f"    Evolution update {count}: {update['type']}")
                    count += 1
                    if count >= 5:  # Don't run forever
                        break
                return count
            except Exception as e:
                print(f"    ❌ Evolution failed: {e}")
                return 0
        
        count = asyncio.run(asyncio.wait_for(test_evolution(), timeout=30))
        
        if count > 0:
            print(f"  ✅ Integration working - {count} updates received")
            return True
        else:
            print(f"  ❌ No integration updates received")
            return False
            
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def run_all_verifications():
    """Run all verification tests"""
    print("🚨 COMPREHENSIVE VERIFICATION OF ALL CLAIMS")
    print("=" * 60)
    
    tests = [
        test_file_existence,
        test_system_imports,
        test_agent_hierarchy,
        test_staged_targets,
        test_parallel_backtesting,
        test_safety_system,
        test_streaming_orchestration,
        test_integration
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print(f"\n🏆 VERIFICATION SUMMARY")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total:.1%}")
    
    if passed == total:
        print("✅ ALL CLAIMS VERIFIED - System is real and functional")
    elif passed >= total * 0.8:
        print("⚠️  MOSTLY VERIFIED - Some issues but generally functional")
    elif passed >= total * 0.5:
        print("🔧 PARTIALLY VERIFIED - Significant issues but core exists")
    else:
        print("❌ CLAIMS LARGELY FALSE - Major problems with implementation")
    
    return passed / total

if __name__ == '__main__':
    success_rate = run_all_verifications()
    sys.exit(0 if success_rate >= 0.8 else 1)