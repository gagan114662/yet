#!/usr/bin/env python3
"""
Optimized Darwin Gödel Trading Machine - Faster Evolution to 25% CAGR
Based on learnings from previous evolution run
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

class OptimizedTradingAgent:
    """Optimized trading agent with focused high-impact mutations"""
    
    def __init__(self, agent_id, base_code):
        self.agent_id = agent_id
        self.base_code = base_code
        self.mutations = []
        self.performance = {"cagr": 0.0, "sharpe": 0.0, "drawdown": 1.0}
        self.generation = 0
        
    def apply_mutation(self, mutation_type):
        """Apply high-impact mutations focused on 25% CAGR target"""
        code = self.base_code
        
        if mutation_type == "switch_to_qqq":
            # QQQ typically outperforms SPY
            code = code.replace('"SPY"', '"QQQ"')
            code = code.replace("'SPY'", "'QQQ'")
            
        elif mutation_type == "switch_to_tqqq":
            # TQQQ is 3x leveraged QQQ
            code = code.replace('"SPY"', '"TQQQ"')
            code = code.replace('"QQQ"', '"TQQQ"')
            code = code.replace("'SPY'", "'TQQQ'")
            code = code.replace("'QQQ'", "'TQQQ'")
            
        elif mutation_type == "max_leverage":
            # Increase to maximum leverage
            if "set_leverage(5.0)" in code:
                code = code.replace("set_leverage(5.0)", "set_leverage(10.0)")
            elif "set_leverage(4.0)" in code:
                code = code.replace("set_leverage(4.0)", "set_leverage(8.0)")
            elif "set_leverage(3.0)" in code:
                code = code.replace("set_leverage(3.0)", "set_leverage(6.0)")
            elif "set_leverage(2.0)" in code:
                code = code.replace("set_leverage(2.0)", "set_leverage(5.0)")
            else:
                code = code.replace("add_equity(", "add_equity(")
                insert_point = code.find("add_equity(")
                if insert_point > 0:
                    line_end = code.find("\\n", insert_point)
                    code = code[:line_end] + "\\n        self.symbol.set_leverage(5.0)" + code[line_end:]
                    
        elif mutation_type == "daily_trading":
            # Trade daily instead of weekly
            code = code.replace("days < 7", "days < 1")
            code = code.replace("days < 5", "days < 1")
            code = code.replace("days < 3", "days < 1")
            
        elif mutation_type == "aggressive_signals":
            # More aggressive entry/exit
            if "sma_fast" in code and "sma_slow" in code:
                # Reduce SMA periods for faster signals
                code = code.replace("sma(\\\"SPY\\\", 10)", "sma(\\\"SPY\\\", 5)")
                code = code.replace("sma(\\\"QQQ\\\", 10)", "sma(\\\"QQQ\\\", 5)")
                code = code.replace("sma(\\\"TQQQ\\\", 10)", "sma(\\\"TQQQ\\\", 5)")
                code = code.replace("sma(\\\"SPY\\\", 30)", "sma(\\\"SPY\\\", 15)")
                code = code.replace("sma(\\\"QQQ\\\", 30)", "sma(\\\"QQQ\\\", 15)")
                code = code.replace("sma(\\\"TQQQ\\\", 30)", "sma(\\\"TQQQ\\\", 15)")
                
        elif mutation_type == "add_momentum_filter":
            # Add MACD or momentum
            if "self.macd" not in code:
                init_point = code.find("self.sma_slow")
                if init_point > 0:
                    insert = code.find("\\n", init_point) + 1
                    code = code[:insert] + "        self.macd = self.macd('SPY', 12, 26, 9)\\n" + code[insert:]
                    
                    # Update logic
                    old_logic = "if (self.sma_fast.current.value > self.sma_slow.current.value"
                    new_logic = "if (self.sma_fast.current.value > self.sma_slow.current.value and \\n            self.macd.is_ready and self.macd.current.value > 0"
                    code = code.replace(old_logic, new_logic)
                    
        elif mutation_type == "remove_rsi_filter":
            # Remove RSI constraint to allow more trades
            if "rsi.current.value < 70" in code:
                code = code.replace(" and \\n            self.rsi.is_ready and self.rsi.current.value < 70", "")
                code = code.replace("and self.rsi.is_ready and self.rsi.current.value < 70", "")
                
        elif mutation_type == "portfolio_margin":
            # Use portfolio margin (2x effective leverage)
            if "set_leverage(10.0)" not in code:
                code = code.replace("set_cash(100000)", "set_cash(100000)\\n        self.settings.automatic_indicator_warm_up = True")
                
        self.base_code = code
        self.mutations.append(mutation_type)
        return code
    
    def run_backtest(self):
        """Run actual backtest using Lean CLI"""
        project_name = f"opt_dgm_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        # Create project directory
        os.makedirs(project_path, exist_ok=True)
        
        try:
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
                
            print(f"  Testing {self.agent_id} (mutations: {len(self.mutations)})...")
            
            # Run backtest
            result = subprocess.run(
                [LEAN_CLI, "backtest", project_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if result.returncode == 0:
                # Parse results
                output = result.stdout
                
                # Extract CAGR
                cagr = 0.0
                if "Compounding Annual Return" in output:
                    for line in output.split('\\n'):
                        if "Compounding Annual Return" in line:
                            try:
                                cagr = float(line.split()[-1].replace('%', '')) / 100
                            except:
                                pass
                            break
                            
                # Extract Sharpe
                sharpe = 0.0
                if "Sharpe Ratio" in output:
                    for line in output.split('\\n'):
                        if "Sharpe Ratio" in line and "Probabilistic" not in line:
                            try:
                                sharpe = float(line.split()[-1])
                            except:
                                pass
                            break
                            
                # Extract Drawdown
                drawdown = 1.0
                if "Drawdown" in output:
                    for line in output.split('\\n'):
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
                
                print(f"    → CAGR: {cagr*100:.1f}%, Sharpe: {sharpe:.2f}, DD: {drawdown*100:.1f}%")
                
            else:
                print(f"    → FAILED: {result.stderr[:100] if result.stderr else 'Unknown error'}")
                self.performance = {"cagr": -1.0, "sharpe": 0.0, "drawdown": 1.0}
                
        except Exception as e:
            print(f"    → ERROR: {str(e)[:100]}")
            self.performance = {"cagr": -1.0, "sharpe": 0.0, "drawdown": 1.0}
        finally:
            # Cleanup
            try:
                if os.path.exists(project_path):
                    shutil.rmtree(project_path)
            except:
                pass
                
        return self.performance


def run_optimized_evolution():
    """Run optimized DGM evolution focused on 25% CAGR target"""
    
    print("🚀 OPTIMIZED Darwin Gödel Trading Machine")
    print("=" * 60)
    print(f"Target: {TARGET_CAGR*100:.0f}% CAGR")
    print("Focus: High-impact mutations for maximum returns")
    print("=" * 60)
    
    # Start with best evolved strategy from previous run
    best_code = '''from AlgorithmImports import *

class DGMAgent(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("SPY", Resolution.DAILY)
        self.symbol.set_leverage(5.0)
        
        self.sma_fast = self.sma("SPY", 10)
        self.sma_slow = self.sma("SPY", 30)
        self.rsi = self.rsi('SPY', 14)
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready:
            return
            
        # Trade frequency control
        if (self.time - self.last_trade).days < 7:
            return
            
        self.last_trade = self.time
        
        if (self.sma_fast.current.value > self.sma_slow.current.value and 
            self.rsi.is_ready and self.rsi.current.value < 70):
            self.set_holdings("SPY", 1.0)
        else:
            self.liquidate()
'''
    
    # Initialize with current best
    print("\\nStarting with evolved strategy (5x SPY + RSI)...")
    current_best = OptimizedTradingAgent("best_evolved", best_code)
    current_best.run_backtest()
    
    best_agent = current_best
    best_cagr = current_best.performance["cagr"]
    
    print(f"Baseline: {best_cagr*100:.1f}% CAGR")
    
    # High-impact mutations to test
    mutations = [
        "switch_to_qqq",           # QQQ > SPY historically
        "switch_to_tqqq",          # 3x leveraged QQQ
        "max_leverage",            # Higher leverage
        "daily_trading",           # More frequent trades
        "aggressive_signals",      # Faster moving averages
        "add_momentum_filter",     # MACD filter
        "remove_rsi_filter",       # More trades
        "portfolio_margin"         # Margin optimization
    ]
    
    generation = 0
    archive = [current_best]
    
    # Run targeted evolution
    while generation < 8 and best_cagr < TARGET_CAGR:
        generation += 1
        print(f"\\n{'='*60}")
        print(f"OPTIMIZATION ROUND {generation}")
        print(f"Current Best: {best_cagr*100:.1f}% CAGR (need {(TARGET_CAGR-best_cagr)*100:.1f}% more)")
        
        # Test all mutations on current best
        new_agents = []
        
        for mutation in mutations:
            child_id = f"round{generation}_{mutation}"
            child = OptimizedTradingAgent(child_id, best_agent.base_code)
            child.generation = generation
            child.mutations = best_agent.mutations.copy()
            
            print(f"\\n  Testing mutation: {mutation}")
            
            # Apply mutation
            child.apply_mutation(mutation)
            
            # Run backtest
            child.run_backtest()
            
            # Check if better
            if child.performance["cagr"] > best_cagr:
                improvement = (child.performance["cagr"] - best_cagr) * 100
                best_cagr = child.performance["cagr"]
                best_agent = child
                print(f"  🎯 NEW BEST! +{improvement:.1f}% improvement → {best_cagr*100:.1f}% CAGR")
                
                # Stop if target reached
                if best_cagr >= TARGET_CAGR:
                    print(f"\\n🏆 TARGET ACHIEVED! {best_cagr*100:.1f}% CAGR!")
                    break
                    
            new_agents.append(child)
            
        # Update archive with successful agents
        archive.extend([a for a in new_agents if a.performance["cagr"] > 0])
        
        print(f"\\nRound {generation} Summary:")
        print(f"  Best CAGR: {best_cagr*100:.1f}%")
        print(f"  Progress: {(best_cagr/TARGET_CAGR)*100:.1f}% of target")
        print(f"  Archive size: {len(archive)}")
        
        if best_cagr >= TARGET_CAGR:
            break
    
    # Final report
    print("\\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    print(f"\\n🏆 BEST RESULT")
    print(f"Agent: {best_agent.agent_id}")
    print(f"CAGR: {best_agent.performance['cagr']*100:.1f}%")
    print(f"Sharpe: {best_agent.performance['sharpe']:.2f}")
    print(f"Max Drawdown: {best_agent.performance['drawdown']*100:.1f}%")
    print(f"Mutations Applied: {len(best_agent.mutations)}")
    print(f"Mutation History: {' → '.join(best_agent.mutations)}")
    
    if best_cagr >= TARGET_CAGR:
        print(f"\\n🎉 SUCCESS! Achieved {best_cagr*100:.1f}% CAGR (target: {TARGET_CAGR*100:.0f}%)")
        
        # Save winning strategy
        final_name = f"winning_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        final_path = os.path.join(LEAN_WORKSPACE, final_name)
        os.makedirs(final_path, exist_ok=True)
        
        with open(os.path.join(final_path, "main.py"), 'w') as f:
            f.write(best_agent.base_code)
        with open(os.path.join(final_path, "config.json"), 'w') as f:
            json.dump({"algorithm-language": "Python", "parameters": {}}, f)
            
        print(f"\\n💾 Winning strategy saved to: {final_path}")
        
    else:
        print(f"\\n📈 Best achieved: {best_cagr*100:.1f}% (target: {TARGET_CAGR*100:.0f}%)")
        print(f"   Gap remaining: {(TARGET_CAGR - best_cagr)*100:.1f}%")
        
        # Suggest next steps
        print("\\n🔧 NEXT OPTIMIZATION SUGGESTIONS:")
        if "switch_to_tqqq" not in best_agent.mutations:
            print("   • Try TQQQ (3x leveraged QQQ)")
        if "daily_trading" not in best_agent.mutations:
            print("   • Switch to daily trading frequency")
        if best_agent.performance['drawdown'] < 0.25:
            print("   • Increase leverage further")
        print("   • Add options strategies")
        print("   • Test crypto or futures")
        print("   • Implement machine learning signals")


if __name__ == "__main__":
    run_optimized_evolution()