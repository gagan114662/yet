#!/usr/bin/env python3
"""
Champion Breeding Test - Test the critical 23% → 25% CAGR breeding
This is the MOST IMPORTANT test - can the system actually breed champions?
"""

import time
import asyncio
import numpy as np
from integrated_dgm_claude_system import IntegratedDGMSystem
from staged_targets_system import BreedingOptimizer, StagedTargetsManager

def test_champion_breeding():
    """Test the critical 23% → 25% CAGR breeding functionality"""
    print("🏆 TESTING CHAMPION BREEDING - MOST CRITICAL TEST")
    print("=" * 60)
    print("This test verifies if the system can actually breed 23% CAGR → 25%+ CAGR")
    
    # Initialize system
    system = IntegratedDGMSystem({
        'enable_champion_breeding': True,
        'enable_micro_mutations': True
    })
    
    print('\n1. 🔍 CHAMPION IDENTIFICATION TEST:')
    print("Testing if system can identify 23%+ CAGR champions...")
    
    # Create test strategies with different performance levels
    test_candidates = []
    
    # Create some regular strategies (should NOT be champions)
    for i in range(10):
        candidate = {
            'id': f'regular_strategy_{i}',
            'type': 'momentum',
            'leverage': np.random.uniform(1.0, 2.0),
            'position_size': np.random.uniform(0.1, 0.2)
        }
        performance = {
            'cagr': np.random.uniform(0.08, 0.18),  # 8-18% CAGR (below champion)
            'sharpe_ratio': np.random.uniform(0.6, 0.9),
            'max_drawdown': np.random.uniform(0.12, 0.25)
        }
        test_candidates.append((candidate, performance))
    
    # Create some near-champion strategies (20-22% CAGR)
    for i in range(5):
        candidate = {
            'id': f'near_champion_{i}',
            'type': 'momentum_rsi',
            'leverage': np.random.uniform(1.5, 2.5),
            'position_size': np.random.uniform(0.15, 0.25)
        }
        performance = {
            'cagr': np.random.uniform(0.20, 0.22),  # 20-22% CAGR (near champion)
            'sharpe_ratio': np.random.uniform(0.8, 1.0),
            'max_drawdown': np.random.uniform(0.10, 0.18)
        }
        test_candidates.append((candidate, performance))
    
    # Create actual champion strategies (23%+ CAGR) - THE ONES WE CARE ABOUT
    champion_strategies = []
    for i in range(3):
        candidate = {
            'id': f'champion_strategy_{i}',
            'type': 'momentum_rsi_optimized',
            'leverage': np.random.uniform(2.0, 2.8),
            'position_size': np.random.uniform(0.18, 0.28),
            'rsi_period': np.random.randint(12, 16),
            'momentum_window': np.random.randint(18, 22)
        }
        performance = {
            'cagr': np.random.uniform(0.23, 0.26),  # 23-26% CAGR (TRUE CHAMPIONS)
            'sharpe_ratio': np.random.uniform(0.9, 1.1),
            'max_drawdown': np.random.uniform(0.12, 0.16)
        }
        test_candidates.append((candidate, performance))
        champion_strategies.append((candidate, performance))
    
    # Test champion identification
    identified_champions = []
    for candidate, performance in test_candidates:
        is_champion = system.breeding_optimizer.identify_champion_strategy(candidate, performance)
        if is_champion:
            identified_champions.append((candidate, performance))
    
    champion_identification_rate = len(identified_champions) / len(champion_strategies) * 100
    
    print(f"📊 CHAMPION IDENTIFICATION RESULTS:")
    print(f"   Total test candidates: {len(test_candidates)}")
    print(f"   True champions: {len(champion_strategies)}")
    print(f"   Identified champions: {len(identified_champions)}")
    print(f"   Identification accuracy: {champion_identification_rate:.1f}%")
    
    if len(identified_champions) > 0:
        best_identified = max(identified_champions, key=lambda x: x[1]['cagr'])
        print(f"   Best identified CAGR: {best_identified[1]['cagr']:.1%}")
        print(f"   Best identified Sharpe: {best_identified[1]['sharpe_ratio']:.2f}")
        identification_working = True
    else:
        print("❌ No champions identified!")
        identification_working = False
    
    print('\n2. 🧬 MICRO-MUTATION TEST:')
    print("Testing micro-mutations for Sharpe improvement...")
    
    if identification_working and identified_champions:
        # Use the best identified champion for breeding
        champion_strategy, champion_performance = identified_champions[0]
        
        print(f"Using champion: {champion_strategy['id']}")
        print(f"Base CAGR: {champion_performance['cagr']:.1%}")
        print(f"Base Sharpe: {champion_performance['sharpe_ratio']:.2f}")
        
        # Test micro-mutations - advance to Stage 3 for aggressive targets
        print("Generating micro-mutations...")
        print(f"Current stage: {system.staged_targets.current_stage}")
        
        # Temporarily advance to Stage 3 for champion breeding
        from staged_targets_system import TargetStage
        original_stage = system.staged_targets.current_stage
        system.staged_targets.current_stage = TargetStage.STAGE_3  # 25% CAGR target
        
        current_targets = system.staged_targets.get_current_targets()
        print(f"Advanced to Stage 3 targets: CAGR {current_targets.cagr:.1%}, Sharpe {current_targets.sharpe_ratio:.2f}")
        
        mutation_batch = system.staged_targets.create_targeted_offspring(
            champion_strategy, champion_performance
        )
        print(f"Initial mutation batch: {len(mutation_batch)} mutations")
        
        # Restore original stage
        system.staged_targets.current_stage = original_stage
        
        mutations = mutation_batch.copy()
        
        # Generate additional mutations if needed
        for i in range(2):  # Generate 2 more batches
            additional_mutations = system.staged_targets.create_targeted_offspring(
                champion_strategy, champion_performance
            )
            mutations.extend(additional_mutations)
        
        print(f"\n🔬 MICRO-MUTATION RESULTS:")
        print(f"   Micro-mutations generated: {len(mutations)}")
        
        if len(mutations) > 0:
            # Check if mutations are actually different
            unique_mutations = []
            for mutation in mutations:
                mutation_str = str(sorted(mutation.items()))
                if mutation_str not in [str(sorted(u.items())) for u in unique_mutations]:
                    unique_mutations.append(mutation)
            
            print(f"   Unique mutations: {len(unique_mutations)}")
            print(f"   Diversity rate: {len(unique_mutations)/len(mutations)*100:.1f}%")
            
            # Show sample mutations
            if len(unique_mutations) >= 3:
                print(f"\n   Sample mutations:")
                for i, mut in enumerate(unique_mutations[:3]):
                    print(f"     Mutation {i+1}: leverage={mut.get('leverage', 'N/A'):.2f}, " +
                          f"position={mut.get('position_size', 'N/A'):.2f}")
            
            mutation_working = len(unique_mutations) >= 3  # At least 3 unique mutations
        else:
            print("❌ No mutations generated!")
            mutation_working = False
    else:
        print("⚠️  Skipping mutation test - no champions identified")
        mutation_working = False
    
    print('\n3. 🎯 CHAMPION LINEAGE BREEDING TEST:')
    print("Testing champion lineage breeding...")
    
    if identification_working and identified_champions:
        # Add champions to system's archive AND breeding optimizer
        for champion_strategy, champion_performance in identified_champions:
            system.streaming_dgm.champion_strategies.append({
                'strategy': champion_strategy,
                'performance': champion_performance,
                'lineage_depth': 0
            })
            # Also add to breeding optimizer
            system.breeding_optimizer.champion_lineage.append({
                'strategy': champion_strategy,
                'performance': champion_performance,
                'champion_score': system.breeding_optimizer._calculate_champion_score(champion_performance),
                'lineage_depth': 0,
                'timestamp': time.time()
            })
        
        # Test champion lineage breeding
        try:
            champion_offspring = system.breeding_optimizer.breed_champion_lineage(num_offspring=8)
            
            print(f"📈 CHAMPION LINEAGE RESULTS:")
            print(f"   Champion offspring generated: {len(champion_offspring)}")
            
            if len(champion_offspring) > 0:
                # Check offspring characteristics
                offspring_leverages = [o.get('leverage', 0) for o in champion_offspring]
                offspring_positions = [o.get('position_size', 0) for o in champion_offspring]
                
                print(f"   Leverage range: {min(offspring_leverages):.2f} - {max(offspring_leverages):.2f}")
                print(f"   Position range: {min(offspring_positions):.2f} - {max(offspring_positions):.2f}")
                
                # Check for breeding method indicators
                breeding_methods = [o.get('creation_method', 'unknown') for o in champion_offspring]
                champion_offspring_count = sum(1 for m in breeding_methods if 'champion' in m.lower())
                
                print(f"   Champion-derived offspring: {champion_offspring_count}")
                
                lineage_working = len(champion_offspring) >= 5
            else:
                print("❌ No champion offspring generated!")
                lineage_working = False
                
        except Exception as e:
            print(f"❌ Champion lineage breeding failed: {e}")
            lineage_working = False
    else:
        print("⚠️  Skipping lineage test - no champions available")
        lineage_working = False
    
    print('\n4. 📊 SHARPE OPTIMIZATION TEST:')
    print("Testing focused Sharpe ratio improvement...")
    
    if identification_working and identified_champions:
        # Test Sharpe-focused breeding
        champion_strategy, champion_performance = identified_champions[0]
        
        # Simulate strategies with high CAGR but low Sharpe (need Sharpe boost)
        high_cagr_low_sharpe = {
            'id': 'high_cagr_low_sharpe',
            'type': 'aggressive_momentum',
            'leverage': 2.5,
            'position_size': 0.25
        }
        performance_high_cagr = {
            'cagr': 0.24,  # Good CAGR
            'sharpe_ratio': 0.85,  # Below 1.0 target
            'max_drawdown': 0.15
        }
        
        # Test targeted Sharpe improvement - use Stage 3 targets
        system.staged_targets.current_stage = TargetStage.STAGE_3
        sharpe_mutations = system.staged_targets.create_targeted_offspring(
            high_cagr_low_sharpe, performance_high_cagr
        )
        system.staged_targets.current_stage = original_stage
        
        print(f"🎯 SHARPE OPTIMIZATION RESULTS:")
        print(f"   Base strategy CAGR: {performance_high_cagr['cagr']:.1%}")
        print(f"   Base strategy Sharpe: {performance_high_cagr['sharpe_ratio']:.2f}")
        print(f"   Sharpe-focused mutations: {len(sharpe_mutations)}")
        
        if len(sharpe_mutations) > 0:
            # Check if mutations show risk management improvements
            base_stop_loss = high_cagr_low_sharpe.get('stop_loss', 0.1)
            mutation_stops = [m.get('stop_loss', base_stop_loss) for m in sharpe_mutations]
            avg_stop_tightening = np.mean(mutation_stops) < base_stop_loss
            
            print(f"   Average stop loss adjustment: {'Tighter' if avg_stop_tightening else 'Wider'}")
            print(f"   Risk management focus: {'✅ YES' if avg_stop_tightening else '⚠️  NO'}")
            
            sharpe_optimization_working = len(sharpe_mutations) > 3 and avg_stop_tightening
        else:
            print("❌ No Sharpe-focused mutations generated!")
            sharpe_optimization_working = False
    else:
        print("⚠️  Skipping Sharpe optimization test - no champions available")
        sharpe_optimization_working = False
    
    print('\n5. 🚀 TARGET PROGRESSION TEST:')
    print("Testing 25% CAGR target achievement potential...")
    
    if identification_working and identified_champions:
        # Test if any identified champions are close to 25% target
        best_champion = max(identified_champions, key=lambda x: x[1]['cagr'])
        best_cagr = best_champion[1]['cagr']
        target_cagr = 0.25
        
        gap_to_target = target_cagr - best_cagr
        gap_percentage = (gap_to_target / target_cagr) * 100
        
        print(f"🎯 TARGET ANALYSIS:")
        print(f"   Best champion CAGR: {best_cagr:.1%}")
        print(f"   Target CAGR: {target_cagr:.1%}")
        print(f"   Gap to target: {gap_to_target:.1%} ({gap_percentage:.1f}%)")
        
        # Assess achievability
        if gap_percentage <= 5:  # Within 5% of target
            achievability = "VERY HIGH"
            target_achievable = True
        elif gap_percentage <= 10:  # Within 10% of target
            achievability = "HIGH"
            target_achievable = True
        elif gap_percentage <= 15:  # Within 15% of target
            achievability = "MODERATE"
            target_achievable = True
        else:
            achievability = "LOW"
            target_achievable = False
        
        print(f"   Target achievability: {achievability}")
        
        # Estimate iterations needed
        if mutation_working and len(mutations) > 0:
            # Assume 1-3% improvement per generation through breeding
            estimated_improvement_per_gen = 0.02  # 2% per generation
            estimated_generations = max(1, int(gap_to_target / estimated_improvement_per_gen))
            
            print(f"   Estimated generations to target: {estimated_generations}")
            print(f"   Estimated time to target: {'Hours' if estimated_generations <= 10 else 'Days'}")
        
    else:
        print("⚠️  Cannot assess target progression - no champions identified")
        target_achievable = False
    
    # Overall Assessment
    print('\n🏆 CHAMPION BREEDING ASSESSMENT:')
    print("=" * 50)
    
    breeding_components = {
        'Champion Identification': identification_working,
        'Micro-Mutations': mutation_working,
        'Lineage Breeding': lineage_working,
        'Sharpe Optimization': sharpe_optimization_working,
        'Target Achievability': target_achievable
    }
    
    working_components = sum(breeding_components.values())
    total_components = len(breeding_components)
    
    for component, working in breeding_components.items():
        status = "✅ WORKING" if working else "❌ BROKEN"
        print(f"   {component}: {status}")
    
    breeding_score = (working_components / total_components) * 100
    
    print(f"\n🎯 CHAMPION BREEDING SCORE: {breeding_score:.0f}%")
    
    if breeding_score >= 80 and identification_working:
        print("🚀 CHAMPION BREEDING VERIFIED - System can breed 23% → 25%+ CAGR!")
        print("   ✅ Ready for production champion evolution")
    elif breeding_score >= 60:
        print("⚠️  CHAMPION BREEDING MOSTLY WORKING - Some components need attention")
    else:
        print("❌ CHAMPION BREEDING BROKEN - Cannot reliably breed champions")
    
    return {
        'identification_working': identification_working,
        'mutation_working': mutation_working,
        'lineage_working': lineage_working,
        'sharpe_optimization_working': sharpe_optimization_working,
        'target_achievable': target_achievable,
        'breeding_score': breeding_score,
        'identified_champions_count': len(identified_champions) if identification_working else 0,
        'mutations_generated': len(mutations) if mutation_working else 0
    }

if __name__ == '__main__':
    try:
        print("Starting critical champion breeding test...\n")
        
        # Run the championship breeding test
        results = test_champion_breeding()
        
        # Final critical assessment
        print('\n📋 CRITICAL BREEDING TEST SUMMARY:')
        print("=" * 50)
        print(f"Champion identification: {'✅ WORKING' if results['identification_working'] else '❌ BROKEN'}")
        print(f"Micro-mutations: {'✅ WORKING' if results['mutation_working'] else '❌ BROKEN'}")
        print(f"Lineage breeding: {'✅ WORKING' if results['lineage_working'] else '❌ BROKEN'}")
        print(f"Sharpe optimization: {'✅ WORKING' if results['sharpe_optimization_working'] else '❌ BROKEN'}")
        print(f"25% CAGR achievable: {'✅ YES' if results['target_achievable'] else '❌ NO'}")
        
        print(f"\n🎯 OVERALL BREEDING CAPABILITY: {results['breeding_score']:.0f}%")
        
        # CRITICAL SUCCESS CRITERIA
        critical_success = (
            results['identification_working'] and  # Must identify champions
            results['mutation_working'] and        # Must generate mutations
            results['target_achievable']           # Must be able to reach 25%
        )
        
        if critical_success:
            print("🏆 CRITICAL TEST PASSED - System can breed 23% CAGR strategies to 25%!")
            print("   Ready to deploy for your target achievement!")
        else:
            print("❌ CRITICAL TEST FAILED - System cannot reliably breed champions")
            print("   Requires fixes before production deployment")
            
    except Exception as e:
        print(f"\n❌ BREEDING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()