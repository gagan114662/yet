#!/usr/bin/env python3
"""
Real Darwin Gödel Trading Machine with actual Lean backtests
"""

import os
import json
import shutil
import subprocess
import time
from datetime import datetime

# Configuration
LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
LEAN_CLI = "/home/vandan/.local/bin/lean"
TARGET_CAGR = 0.25

class RealTradingAgent:
    """Trading agent that runs real backtests"""
    
    def __init__(self, agent_id, base_code):
        self.agent_id = agent_id
        self.base_code = base_code
        self.mutations = []
        self.performance = {"cagr": 0.0, "sharpe": 0.0, "drawdown": 1.0}
        self.generation = 0
        
    def apply_mutation(self, mutation_type):
        """Apply specific mutation to code"""
        code = self.base_code
        
        if mutation_type == "increase_leverage":
            # Find and increase leverage
            if "set_leverage(2.0)" in code:
                code = code.replace("set_leverage(2.0)", "set_leverage(4.0)")
            elif "set_leverage(3.0)" in code:
                code = code.replace("set_leverage(3.0)", "set_leverage(5.0)")
            elif "set_leverage(4.0)" in code:
                code = code.replace("set_leverage(4.0)", "set_leverage(5.0)")
            else:
                code = code.replace("set_leverage(", "set_leverage(3.0) # was set_leverage(")
                
        elif mutation_type == "add_rsi":
            if "self.rsi" not in code:
                # Add RSI indicator
                init_point = code.find("self.sma_slow")
                if init_point > 0:
                    insert = code.find("\n", init_point) + 1
                    code = code[:insert] + "        self.rsi = self.rsi('SPY', 14)\n" + code[insert:]
                    
                    # Add RSI logic
                    old_logic = "if self.sma_fast.current.value > self.sma_slow.current.value:"
                    new_logic = """if (self.sma_fast.current.value > self.sma_slow.current.value and 
            self.rsi.is_ready and self.rsi.current.value < 70):"""
                    code = code.replace(old_logic, new_logic)
                    
        elif mutation_type == "change_to_qqq":
            code = code.replace('"SPY"', '"QQQ"')
            code = code.replace("'SPY'", "'QQQ'")
            
        elif mutation_type == "faster_trading":
            # Look for rebalancing logic
            if "days < 7" in code:
                code = code.replace("days < 7", "days < 2")
            elif "days < 5" in code:
                code = code.replace("days < 5", "days < 1")
                
        elif mutation_type == "add_momentum":
            if "self.mom" not in code:
                init_point = code.find("self.sma_slow")
                if init_point > 0:
                    insert = code.find("\n", init_point) + 1
                    code = code[:insert] + "        self.mom = self.momp('SPY', 20)\n" + code[insert:]
                    
        elif mutation_type == "aggressive_position":
            # Increase position sizes
            code = code.replace("set_holdings('SPY', 1.0)", "set_holdings('SPY', 2.0)")
            code = code.replace("set_holdings('QQQ', 1.0)", "set_holdings('QQQ', 2.0)")
            
        self.base_code = code
        self.mutations.append(mutation_type)
        return code
    
    def run_backtest(self):
        """Run actual backtest using Lean CLI"""
        # Create unique project name
        project_name = f"dgm_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        # Create project directory
        os.makedirs(project_path, exist_ok=True)
        
        # Write strategy code
        with open(os.path.join(project_path, "main.py"), 'w') as f:
            f.write(self.base_code)
            
        # Write config
        config = {
            "algorithm-language": "Python",
            "parameters": {},
            "local-id": hash(self.agent_id) % 1000000
        }
        with open(os.path.join(project_path, "config.json"), 'w') as f:
            json.dump(config, f)
            
        print(f"  Running backtest for {self.agent_id}...")
        
        # Run backtest
        try:
            result = subprocess.run(
                [LEAN_CLI, "backtest", project_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Parse results
                output = result.stdout
                
                # Extract CAGR
                cagr = 0.0
                if "Compounding Annual Return" in output:
                    for line in output.split('\n'):
                        if "Compounding Annual Return" in line:
                            try:
                                cagr = float(line.split()[-1].replace('%', '')) / 100
                            except:
                                pass
                            break
                            
                # Extract Sharpe
                sharpe = 0.0
                if "Sharpe Ratio" in output:
                    for line in output.split('\n'):
                        if "Sharpe Ratio" in line and "Probabilistic" not in line:
                            try:
                                sharpe = float(line.split()[-1])
                            except:
                                pass
                            break
                            
                # Extract Drawdown
                drawdown = 1.0
                if "Drawdown" in output:
                    for line in output.split('\n'):
                        if "Drawdown" in line:
                            try:
                                drawdown = float(line.split()[-1].replace('%', '')) / 100
                            except:
                                pass
                            break
                
                self.performance = {
                    "cagr": cagr,
                    "sharpe": sharpe,
                    "drawdown": drawdown
                }
                
                print(f"    CAGR: {cagr*100:.1f}%, Sharpe: {sharpe:.2f}, Drawdown: {drawdown*100:.1f}%")
                
            else:
                print(f"    Backtest failed!")
                self.performance = {"cagr": -1.0, "sharpe": 0.0, "drawdown": 1.0}
                
        except Exception as e:
            print(f"    Error: {e}")
            self.performance = {"cagr": -1.0, "sharpe": 0.0, "drawdown": 1.0}
            
        # Cleanup
        try:
            shutil.rmtree(project_path)
        except:
            pass
            
        return self.performance


def run_real_evolution():
    """Run real DGM evolution with actual backtests"""
    
    print("🧬 REAL Darwin Gödel Trading Machine")
    print("=" * 60)
    print(f"Target: {TARGET_CAGR*100:.0f}% CAGR")
    print(f"Using Lean workspace: {LEAN_WORKSPACE}")
    print("=" * 60)
    
    # Base strategy code
    base_code = '''from AlgorithmImports import *

class DGMAgent(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("SPY", Resolution.DAILY)
        self.symbol.set_leverage(2.0)
        
        self.sma_fast = self.sma("SPY", 10)
        self.sma_slow = self.sma("SPY", 30)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready:
            return
            
        # Trade frequency control
        if (self.time - self.last_trade).days < 7:
            return
            
        self.last_trade = self.time
        
        if self.sma_fast.current.value > self.sma_slow.current.value:
            self.set_holdings("SPY", 1.0)
        else:
            self.liquidate()
'''
    
    # Initialize with base agent
    print("\nInitializing base agent...")
    base_agent = RealTradingAgent("base", base_code)
    base_agent.run_backtest()
    
    best_agent = base_agent
    best_cagr = base_agent.performance["cagr"]
    
    # Evolution mutations to try
    mutations = [
        "increase_leverage",
        "add_rsi",
        "change_to_qqq",
        "faster_trading",
        "add_momentum",
        "aggressive_position"
    ]
    
    archive = [base_agent]
    generation = 0
    
    # Run evolution
    while generation < 10 and best_cagr < TARGET_CAGR:
        generation += 1
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        
        # Select top agents as parents
        parents = sorted(archive, key=lambda a: a.performance["cagr"], reverse=True)[:3]
        
        new_agents = []
        
        for i, parent in enumerate(parents):
            # Try different mutations
            for mutation in mutations[:2]:  # Try 2 mutations per parent
                child_id = f"gen{generation}_p{i}_{mutation}"
                child = RealTradingAgent(child_id, parent.base_code)
                child.generation = generation
                child.mutations = parent.mutations.copy()
                
                print(f"\n  Creating child: {child_id}")
                print(f"  Applying mutation: {mutation}")
                
                # Apply mutation
                child.apply_mutation(mutation)
                
                # Run backtest
                child.run_backtest()
                
                # Check if better
                if child.performance["cagr"] > best_cagr:
                    best_cagr = child.performance["cagr"]
                    best_agent = child
                    print(f"  🎯 NEW BEST! CAGR: {best_cagr*100:.1f}%")
                    
                new_agents.append(child)
                
                # Stop if target reached
                if best_cagr >= TARGET_CAGR:
                    print(f"\n🏆 TARGET ACHIEVED! {best_cagr*100:.1f}% CAGR!")
                    break
                    
            if best_cagr >= TARGET_CAGR:
                break
                
        # Add successful agents to archive
        archive.extend([a for a in new_agents if a.performance["cagr"] > 0])
        
        print(f"\nGeneration {generation} complete")
        print(f"Archive size: {len(archive)}")
        print(f"Best CAGR: {best_cagr*100:.1f}%")
        
    # Final report
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    
    print(f"\nBest Agent: {best_agent.agent_id}")
    print(f"CAGR: {best_agent.performance['cagr']*100:.1f}%")
    print(f"Sharpe: {best_agent.performance['sharpe']:.2f}")
    print(f"Drawdown: {best_agent.performance['drawdown']*100:.1f}%")
    print(f"Mutations: {best_agent.mutations}")
    
    if best_cagr >= TARGET_CAGR:
        print(f"\n🎉 SUCCESS! Achieved {best_cagr*100:.1f}% CAGR (target: {TARGET_CAGR*100:.0f}%)")
        
        # Save winning strategy
        with open("winning_strategy.py", 'w') as f:
            f.write(best_agent.base_code)
        print("\nWinning strategy saved to winning_strategy.py")
    else:
        print(f"\n📈 Best result: {best_cagr*100:.1f}% (target: {TARGET_CAGR*100:.0f}%)")
        print(f"   Need {(TARGET_CAGR - best_cagr)*100:.1f}% more")


if __name__ == "__main__":
    run_real_evolution()