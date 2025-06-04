#!/usr/bin/env python3
"""
Run Darwin Gödel Trading Machine evolution
"""

import os
import sys
from dgm_core import DarwinGodelTradingMachine

def main():
    # Configuration
    LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
    BASE_AGENT_PATH = os.path.join(os.path.dirname(__file__), "base_agent")
    TARGET_CAGR = 0.25  # 25% target
    MAX_GENERATIONS = 20
    
    print("🧬 Darwin Gödel Trading Machine")
    print("=" * 60)
    print(f"Target CAGR: {TARGET_CAGR*100:.0f}%")
    print(f"Max Generations: {MAX_GENERATIONS}")
    print(f"Lean Workspace: {LEAN_WORKSPACE}")
    print("=" * 60)
    
    # Initialize DGM
    dgm = DarwinGodelTradingMachine(
        initial_agent_path=BASE_AGENT_PATH,
        lean_workspace=LEAN_WORKSPACE,
        target_cagr=TARGET_CAGR
    )
    
    # Run evolution
    print("\nStarting evolution...")
    best_agent = dgm.run(max_generations=MAX_GENERATIONS)
    
    # Report results
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    
    if best_agent:
        print(f"Best Agent: {best_agent.agent_id}")
        print(f"Generation: {best_agent.generation}")
        print(f"CAGR: {best_agent.performance_metrics.get('cagr', 0)*100:.1f}%")
        print(f"Sharpe: {best_agent.performance_metrics.get('sharpe', 0):.2f}")
        print(f"Max Drawdown: {best_agent.performance_metrics.get('max_drawdown', 1)*100:.1f}%")
        print(f"\nMutations applied:")
        for i, mutation in enumerate(best_agent.mutations, 1):
            print(f"{i}. {mutation}")
            
        if best_agent.performance_metrics.get('cagr', 0) >= TARGET_CAGR:
            print(f"\n🎉 TARGET ACHIEVED! {TARGET_CAGR*100:.0f}%+ CAGR!")
        else:
            print(f"\n📈 Best result: {best_agent.performance_metrics.get('cagr', 0)*100:.1f}% CAGR")
            print(f"   Still need: {(TARGET_CAGR - best_agent.performance_metrics.get('cagr', 0))*100:.1f}% more")
    else:
        print("❌ No successful agents found")
        
    print(f"\nTotal agents in archive: {len(dgm.archive)}")
    print(f"Check dgm_evolution.log for detailed evolution history")

if __name__ == "__main__":
    main()