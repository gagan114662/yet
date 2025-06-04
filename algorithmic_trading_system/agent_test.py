#!/usr/bin/env python3
"""
Agent Coordination Test - Verify multi-agent functionality
Test if agents actually work together and coordinate properly
"""

import time
import asyncio
import numpy as np
from integrated_dgm_claude_system import IntegratedDGMSystem
from dgm_agent_hierarchy import AgentContext

def test_agent_coordination():
    """Test if agents actually work together"""
    print("🧠 TESTING MULTI-AGENT COORDINATION")
    print("=" * 60)
    
    # Initialize system
    system = IntegratedDGMSystem({
        'max_parallel_backtests': 2,
        'enable_real_time_streaming': False
    })
    
    print('\n1. 🌍 MARKET REGIME AGENT TEST:')
    print("Testing market regime analysis...")
    
    # Test Market Regime Agent
    regime_agent = system.agent_orchestrator.agents['market_regime']
    
    # Create test context
    context = AgentContext(
        current_regime="unknown",
        generation=1,
        archive_summary={},
        performance_history=[],
        near_winners=[],
        compute_resources={'cpu_usage': 0.5}
    )
    
    # Test regime analysis
    regime_result = asyncio.run(regime_agent.execute(context))
    
    if 'error' not in regime_result:
        print(f"✅ Market regime detected: {regime_result.get('current_regime', 'unknown')}")
        print(f"   Confidence: {regime_result.get('regime_confidence', 0):.2f}")
        print(f"   Volatility: {regime_result.get('volatility_level', 'unknown')}")
        regime_working = True
    else:
        print(f"❌ Market regime agent failed: {regime_result['error']}")
        regime_working = False
    
    print('\n2. 🧬 STRATEGY GENERATOR AGENT TEST:')
    print("Testing strategy generation...")
    
    # Test Strategy Generator Agent
    generator_agent = system.agent_orchestrator.agents['strategy_generator']
    
    if regime_working:
        # Use actual regime context
        generation_result = asyncio.run(generator_agent.execute(
            context, 
            regime_context=regime_result, 
            generation_count=8
        ))
    else:
        # Use mock regime context
        mock_regime = {
            'current_regime': 'test_regime',
            'strategy_recommendations': {
                'preferred_types': ['momentum'],
                'leverage_range': (1.0, 2.0),
                'position_size_range': (0.1, 0.2)
            }
        }
        generation_result = asyncio.run(generator_agent.execute(
            context, 
            regime_context=mock_regime, 
            generation_count=8
        ))
    
    if 'error' not in generation_result:
        strategies_generated = generation_result.get('strategies_generated', 0)
        print(f"✅ Strategies generated: {strategies_generated}")
        print(f"   Regime-specific: {generation_result.get('regime_specific', 0)}")
        print(f"   Novel explorations: {generation_result.get('novel_explorations', 0)}")
        generator_working = strategies_generated > 0
        generated_strategies = generation_result.get('strategies', [])
    else:
        print(f"❌ Strategy generator failed: {generation_result['error']}")
        generator_working = False
        generated_strategies = []
    
    print('\n3. 🛡️ RISK ANALYZER AGENT TEST:')
    print("Testing risk assessment...")
    
    # Test Risk Analyzer Agent
    risk_agent = system.agent_orchestrator.agents['risk_analyzer']
    
    if generator_working and generated_strategies:
        # Use actual generated strategies
        test_strategies = generated_strategies[:5]  # Test first 5
    else:
        # Create mock strategies for testing
        test_strategies = []
        for i in range(5):
            strategy = {
                'id': f'test_strategy_{i}',
                'type': 'momentum',
                'leverage': np.random.uniform(1.0, 3.0),
                'position_size': np.random.uniform(0.1, 0.3),
                'stop_loss': np.random.uniform(0.05, 0.15)
            }
            test_strategies.append(strategy)
    
    risk_result = asyncio.run(risk_agent.execute(context, test_strategies))
    
    if 'error' not in risk_result:
        assessed_count = risk_result.get('strategies_assessed', 0)
        approved_count = len(risk_result.get('approved_strategies', []))
        high_risk_count = risk_result.get('high_risk_count', 0)
        
        print(f"✅ Strategies assessed: {assessed_count}")
        print(f"   Approved strategies: {approved_count}")
        print(f"   High risk strategies: {high_risk_count}")
        print(f"   Approval rate: {(approved_count/assessed_count*100) if assessed_count > 0 else 0:.1f}%")
        risk_working = assessed_count > 0
        risk_assessments = risk_result
    else:
        print(f"❌ Risk analyzer failed: {risk_result['error']}")
        risk_working = False
        risk_assessments = {}
    
    print('\n4. ⚡ PERFORMANCE SYNTHESIZER TEST:')
    print("Testing insight synthesis...")
    
    # Test Performance Synthesizer Agent
    synthesizer_agent = system.agent_orchestrator.agents['performance_synthesizer']
    
    # Prepare inputs for synthesizer
    if regime_working:
        regime_input = regime_result
    else:
        regime_input = {'current_regime': 'test', 'regime_confidence': 0.5}
    
    if generator_working:
        generation_input = generation_result
    else:
        generation_input = {'strategies_generated': 0}
    
    if risk_working:
        risk_input = risk_assessments
    else:
        risk_input = {'strategies_assessed': 0}
    
    synthesis_result = asyncio.run(synthesizer_agent.execute(
        context, regime_input, generation_input, risk_input
    ))
    
    if 'error' not in synthesis_result:
        print(f"✅ Synthesis completed: {synthesis_result.get('synthesis_complete', False)}")
        print(f"   Evolution decisions: {len(synthesis_result.get('evolution_decisions', {}))}")
        print(f"   Recommendations: {len(synthesis_result.get('recommendations', []))}")
        synthesizer_working = synthesis_result.get('synthesis_complete', False)
    else:
        print(f"❌ Performance synthesizer failed: {synthesis_result['error']}")
        synthesizer_working = False
    
    print('\n5. 💾 ARCHIVE MANAGER TEST:')
    print("Testing archive management...")
    
    # Test Archive Manager Agent
    archive_agent = system.agent_orchestrator.agents['archive_manager']
    
    # Create mock synthesis for archive manager
    if synthesizer_working:
        archive_input = synthesis_result
    else:
        archive_input = {
            'approved_strategies': [
                {'strategy_id': 'test_1', 'risk_level': 'low', 'approved': True},
                {'strategy_id': 'test_2', 'risk_level': 'medium', 'approved': True}
            ]
        }
    
    archive_result = asyncio.run(archive_agent.execute(context, archive_input))
    
    if 'error' not in archive_result:
        archived_count = archive_result.get('strategies_archived', 0)
        total_in_archive = archive_result.get('total_in_archive', 0)
        
        print(f"✅ Strategies archived: {archived_count}")
        print(f"   Total in archive: {total_in_archive}")
        print(f"   Compaction performed: {archive_result.get('compaction_performed', False)}")
        archive_working = True
    else:
        print(f"❌ Archive manager failed: {archive_result['error']}")
        archive_working = False
    
    print('\n6. 🔄 FULL ORCHESTRATION TEST:')
    print("Testing complete agent orchestration...")
    
    # Test full orchestration
    orchestration_result = asyncio.run(
        system.agent_orchestrator.orchestrate_evolution_cycle(context)
    )
    
    if 'error' not in orchestration_result:
        print(f"✅ Full cycle completed: {orchestration_result.get('cycle_complete', False)}")
        print(f"   Cycle time: {orchestration_result.get('cycle_time', 0):.2f}s")
        print(f"   Strategies generated: {orchestration_result.get('strategies_generated', 0)}")
        print(f"   Strategies approved: {orchestration_result.get('strategies_approved', 0)}")
        orchestration_working = orchestration_result.get('cycle_complete', False)
    else:
        print(f"❌ Full orchestration failed: {orchestration_result['error']}")
        orchestration_working = False
    
    # Agent Performance Summary
    print('\n7. 📊 AGENT PERFORMANCE SUMMARY:')
    print("Checking individual agent performance metrics...")
    
    agent_status = system.agent_orchestrator.get_agent_status()
    working_agents = 0
    total_agents = len(agent_status)
    
    for agent_name, status in agent_status.items():
        executions = status.get('recent_executions', 0)
        metrics = status.get('performance_metrics', {})
        success_rate = metrics.get('success_rate', 0)
        
        print(f"   {agent_name}:")
        print(f"     Executions: {executions}")
        print(f"     Success rate: {success_rate:.1%}")
        
        if executions > 0 and success_rate > 0:
            working_agents += 1
    
    # Overall Assessment
    print('\n🏆 AGENT COORDINATION ASSESSMENT:')
    print("=" * 50)
    
    agent_scores = {
        'Market Regime': regime_working,
        'Strategy Generator': generator_working,
        'Risk Analyzer': risk_working,
        'Performance Synthesizer': synthesizer_working,
        'Archive Manager': archive_working,
        'Full Orchestration': orchestration_working
    }
    
    working_count = sum(agent_scores.values())
    total_count = len(agent_scores)
    
    for agent, working in agent_scores.items():
        status = "✅ WORKING" if working else "❌ BROKEN"
        print(f"   {agent}: {status}")
    
    coordination_score = (working_count / total_count) * 100
    
    print(f"\n🎯 COORDINATION SCORE: {coordination_score:.0f}%")
    print(f"Working agents: {working_agents}/{total_agents}")
    
    if coordination_score >= 80:
        print("🚀 AGENT COORDINATION EXCELLENT - Multi-agent system working!")
    elif coordination_score >= 60:
        print("⚠️  AGENT COORDINATION GOOD - Most agents functional")
    else:
        print("❌ AGENT COORDINATION POOR - Major coordination issues")
    
    return {
        'individual_agents': agent_scores,
        'coordination_score': coordination_score,
        'working_agents': working_agents,
        'total_agents': total_agents,
        'orchestration_working': orchestration_working
    }

def test_agent_communication():
    """Test inter-agent communication and data flow"""
    print('\n8. 📡 AGENT COMMUNICATION TEST:')
    print("Testing data flow between agents...")
    
    system = IntegratedDGMSystem()
    
    # Test data compatibility between agents
    context = AgentContext(
        current_regime="bull_market",
        generation=1,
        archive_summary={},
        performance_history=[],
        near_winners=[],
        compute_resources={'cpu_usage': 0.6}
    )
    
    try:
        # Chain agent outputs as inputs
        regime_agent = system.agent_orchestrator.agents['market_regime']
        generator_agent = system.agent_orchestrator.agents['strategy_generator']
        
        # Step 1: Get regime analysis
        regime_output = asyncio.run(regime_agent.execute(context))
        
        # Step 2: Use regime output as generator input
        if 'error' not in regime_output:
            generator_output = asyncio.run(generator_agent.execute(
                context, regime_output, generation_count=5
            ))
            
            if 'error' not in generator_output:
                print("✅ Agent communication working - data flows correctly")
                communication_working = True
            else:
                print("❌ Generator failed to process regime data")
                communication_working = False
        else:
            print("❌ Regime agent failed - cannot test communication")
            communication_working = False
            
    except Exception as e:
        print(f"❌ Agent communication failed: {e}")
        communication_working = False
    
    return communication_working

if __name__ == '__main__':
    try:
        print("Starting comprehensive agent coordination test...\n")
        
        # Run main coordination test
        coordination_results = test_agent_coordination()
        
        # Test communication
        communication_working = test_agent_communication()
        
        # Final summary
        print('\n📋 FINAL AGENT TEST SUMMARY:')
        print("=" * 50)
        print(f"Individual agent score: {coordination_results['coordination_score']:.0f}%")
        print(f"Inter-agent communication: {'✅ WORKING' if communication_working else '❌ BROKEN'}")
        print(f"Full orchestration: {'✅ WORKING' if coordination_results['orchestration_working'] else '❌ BROKEN'}")
        
        overall_agent_score = (
            coordination_results['coordination_score'] * 0.7 +
            (100 if communication_working else 0) * 0.2 +
            (100 if coordination_results['orchestration_working'] else 0) * 0.1
        )
        
        print(f"\n🎯 OVERALL AGENT SYSTEM SCORE: {overall_agent_score:.0f}%")
        
        if overall_agent_score >= 80:
            print("🚀 MULTI-AGENT SYSTEM VERIFIED - Agents coordinate excellently!")
        elif overall_agent_score >= 60:
            print("⚠️  MULTI-AGENT SYSTEM MOSTLY WORKING - Minor issues")
        else:
            print("❌ MULTI-AGENT SYSTEM BROKEN - Major coordination failures")
            
    except Exception as e:
        print(f"\n❌ AGENT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()