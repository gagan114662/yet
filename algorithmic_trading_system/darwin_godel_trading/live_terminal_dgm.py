#!/usr/bin/env python3
"""
LIVE Terminal Darwin Gödel Machine
Real-time progress display in your terminal - no log truncation
Shows EVERYTHING happening under the hood as it happens
"""

import os
import json
import shutil
import subprocess
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Configuration
LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
LEAN_CLI = "/home/vandan/.local/bin/lean"

# ALL TARGET CRITERIA
TARGET_CRITERIA = {
    "cagr": 0.25,           # >25%
    "sharpe": 1.0,          # >1.0
    "max_drawdown": 0.20,   # <20%
    "avg_profit": 0.0075    # >0.75%
}

def live_print(message: str, level: str = "INFO"):
    """Print with immediate flush to terminal"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] [{level}] {message}"
    print(full_msg)
    sys.stdout.flush()

def status_update(message: str):
    """Show status updates prominently"""
    print("\n" + "="*60)
    live_print(message, "STATUS")
    print("="*60)

class LiveTerminalAgent:
    """Trading agent with live terminal output"""
    
    def __init__(self, agent_id: str, generation: int = 0):
        self.agent_id = agent_id
        self.generation = generation
        self.mutations = []
        self.performance = {
            "cagr": -999.0, 
            "sharpe": 0.0, 
            "drawdown": 1.0,
            "avg_profit": 0.0,
            "total_trades": 0,
            "win_rate": 0.0
        }
        # Start from WORKING base and incrementally improve to 25%+ CAGR
        self.code_components = {
            "asset": "QQQ",    # Known to work
            "leverage": 8.0,   # Higher than previous best
            "sma_fast": 10,    # Proven working signals
            "sma_slow": 30,    # Proven working signals
            "has_rsi": False,
            "has_macd": False,
            "trade_frequency": 1,  # Daily
            "position_size": 2.0   # Aggressive but working
        }
        self.is_valid = False
        self.criteria_score = 0.0
    
    def generate_code(self) -> str:
        """Generate code with live output"""
        live_print(f"🔧 Generating code for {self.agent_id}", "CODE")
        
        try:
            asset = self.code_components["asset"]
            leverage = self.code_components["leverage"]
            sma_fast = self.code_components["sma_fast"]
            sma_slow = self.code_components["sma_slow"]
            has_rsi = self.code_components["has_rsi"]
            has_macd = self.code_components["has_macd"]
            trade_freq = self.code_components["trade_frequency"]
            position_size = self.code_components["position_size"]
            
            live_print(f"   Asset: {asset}, Leverage: {leverage}x, Position: {position_size}x", "CODE")
            live_print(f"   SMA: ({sma_fast},{sma_slow}), RSI: {has_rsi}, MACD: {has_macd}", "CODE")
            live_print(f"   Trading frequency: Every {trade_freq} day(s)", "CODE")
            
            # Build indicators
            indicators_lines = [
                f'        self.sma_fast = self.sma("{asset}", {sma_fast})',
                f'        self.sma_slow = self.sma("{asset}", {sma_slow})'
            ]
            
            if has_rsi:
                indicators_lines.append(f'        self.rsi = self.rsi("{asset}", 14)')
                live_print(f"   Added RSI(14) filter", "CODE")
            
            if has_macd:
                indicators_lines.append(f'        self.macd = self.macd("{asset}", 12, 26, 9)')
                live_print(f"   Added MACD(12,26,9) filter", "CODE")
            
            indicators = '\n'.join(indicators_lines)
            
            # Build conditions
            conditions = ["self.sma_fast.current.value > self.sma_slow.current.value"]
            
            if has_rsi:
                conditions.append("self.rsi.is_ready and self.rsi.current.value < 70")
            
            if has_macd:
                conditions.append("self.macd.is_ready and self.macd.current.value > 0")
            
            if len(conditions) == 1:
                full_condition = conditions[0]
                live_print(f"   Logic: Simple SMA crossover", "CODE")
            else:
                condition_str = " and \\\n            ".join(conditions)
                full_condition = f"({condition_str})"
                live_print(f"   Logic: SMA + {len(conditions)-1} filters", "CODE")
            
            # Generate code
            code = f'''from AlgorithmImports import *

class LiveTerminalStrategy(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("{asset}", Resolution.DAILY)
        self.symbol.set_leverage({leverage})
        
{indicators}
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        if (self.time - self.last_trade).days < {trade_freq}:
            return
            
        self.last_trade = self.time
        
        if {full_condition}:
            self.set_holdings("{asset}", {position_size})
        else:
            self.liquidate()
'''
            
            # Test compilation
            compile(code, f"agent_{self.agent_id}", 'exec')
            self.is_valid = True
            live_print(f"   ✅ Code compiled successfully ({len(code)} chars)", "CODE")
            return code
            
        except Exception as e:
            live_print(f"   ❌ Code generation FAILED: {e}", "ERROR")
            self.is_valid = False
            return ""
    
    def apply_mutation(self, mutation_type: str) -> bool:
        """Apply mutation with live feedback"""
        live_print(f"🧬 Applying mutation: {mutation_type}", "MUTATION")
        
        try:
            original = self.code_components.copy()
            
            if mutation_type == "boost_leverage_15x":
                old_lev = self.code_components["leverage"]
                self.code_components["leverage"] = 15.0
                live_print(f"   Boost leverage: {old_lev}x → 15.0x", "MUTATION")
            elif mutation_type == "boost_leverage_20x":
                old_lev = self.code_components["leverage"]
                self.code_components["leverage"] = 20.0
                live_print(f"   High leverage: {old_lev}x → 20.0x", "MUTATION")
            elif mutation_type == "switch_to_tqqq":
                self.code_components["asset"] = "TQQQ"
                self.code_components["leverage"] = 3.0  # Lower leverage on leveraged ETF
                live_print(f"   Switch to TQQQ with 3x leverage", "MUTATION")
            elif mutation_type == "aggressive_position_3x":
                old_pos = self.code_components["position_size"]
                self.code_components["position_size"] = 3.0
                live_print(f"   Aggressive position: {old_pos}x → 3.0x", "MUTATION")
            elif mutation_type == "faster_signals_8_20":
                self.code_components["sma_fast"] = 8
                self.code_components["sma_slow"] = 20
                live_print(f"   Faster signals: SMA(8,20)", "MUTATION")
            elif mutation_type == "ultra_fast_5_15":
                self.code_components["sma_fast"] = 5
                self.code_components["sma_slow"] = 15
                live_print(f"   Ultra fast signals: SMA(5,15)", "MUTATION")
            elif mutation_type == "reduce_for_sharpe":
                self.code_components["leverage"] = 10.0
                self.code_components["has_rsi"] = True
                self.code_components["trade_frequency"] = 3
                live_print(f"   Optimize Sharpe: 10x leverage, RSI filter, 3-day frequency", "MUTATION")
            elif mutation_type == "add_rsi_management":
                self.code_components["has_rsi"] = True
                live_print(f"   Added RSI risk management", "MUTATION")
            else:
                live_print(f"   Unknown mutation: {mutation_type}", "ERROR")
                return False
            
            # Test code generation
            test_code = self.generate_code()
            if test_code and self.is_valid:
                self.mutations.append(mutation_type)
                live_print(f"   ✅ Mutation applied successfully", "MUTATION")
                return True
            else:
                self.code_components = original
                live_print(f"   ❌ Mutation failed - reverted", "ERROR")
                return False
                
        except Exception as e:
            self.code_components = original
            live_print(f"   ❌ Mutation error: {e}", "ERROR")
            return False
    
    def run_backtest(self) -> Dict:
        """Run backtest with live progress"""
        if not self.is_valid:
            live_print(f"❌ Cannot backtest - invalid strategy", "ERROR")
            return self.performance
        
        live_print(f"⚡ Starting backtest for {self.agent_id}", "BACKTEST")
        project_name = f"live_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        try:
            # Create project
            os.makedirs(project_path, exist_ok=True)
            live_print(f"   📁 Created project: {project_name}", "BACKTEST")
            
            # Write files
            code = self.generate_code()
            with open(os.path.join(project_path, "main.py"), 'w') as f:
                f.write(code)
            
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "local-id": abs(hash(self.agent_id)) % 1000000
            }
            with open(os.path.join(project_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            live_print(f"   📝 Files written, executing lean backtest...", "BACKTEST")
            
            start_time = time.time()
            
            # Run backtest
            result = subprocess.run(
                [LEAN_CLI, "backtest", project_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            backtest_time = time.time() - start_time
            live_print(f"   ⏱️ Backtest completed in {backtest_time:.1f}s", "BACKTEST")
            
            if result.returncode == 0:
                live_print(f"   ✅ Backtest successful - parsing results...", "BACKTEST")
                metrics = self._parse_metrics(result.stdout)
                self.performance = metrics
                self.criteria_score = self._calculate_score(metrics)
                
                # Show results immediately
                self._show_results()
                
                return metrics
            else:
                live_print(f"   ❌ Backtest failed: {result.stderr[:100] if result.stderr else 'Unknown error'}", "ERROR")
                return self.performance
                
        except Exception as e:
            live_print(f"   ❌ Backtest error: {str(e)[:100]}", "ERROR")
            return self.performance
        finally:
            # Cleanup
            try:
                if os.path.exists(project_path):
                    shutil.rmtree(project_path)
                    live_print(f"   🧹 Cleaned up project files", "BACKTEST")
            except:
                pass
    
    def _parse_metrics(self, output: str) -> Dict:
        """Parse metrics from backtest output"""
        live_print(f"📊 Parsing backtest results...", "PARSING")
        
        metrics = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "drawdown": 1.0,
            "avg_profit": 0.0,
            "total_trades": 0,
            "win_rate": 0.0
        }
        
        try:
            lines = output.split('\n')
            avg_win = 0.0
            
            for line in lines:
                if "Compounding Annual Return" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["cagr"] = float(value) / 100
                        live_print(f"   Found CAGR: {metrics['cagr']*100:.2f}%", "PARSING")
                    except:
                        pass
                elif "Sharpe Ratio" in line and "Probabilistic" not in line:
                    try:
                        metrics["sharpe"] = float(line.split()[-1])
                        live_print(f"   Found Sharpe: {metrics['sharpe']:.3f}", "PARSING")
                    except:
                        pass
                elif "Drawdown" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["drawdown"] = float(value) / 100
                        live_print(f"   Found Max DD: {metrics['drawdown']*100:.1f}%", "PARSING")
                    except:
                        pass
                elif "Total Orders" in line:
                    try:
                        metrics["total_trades"] = int(line.split()[-1])
                        live_print(f"   Found Total Trades: {metrics['total_trades']}", "PARSING")
                    except:
                        pass
                elif "Average Win" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        avg_win = float(value) / 100
                        live_print(f"   Found Avg Win: {avg_win*100:.2f}%", "PARSING")
                    except:
                        pass
                elif "Win Rate" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["win_rate"] = float(value) / 100
                        live_print(f"   Found Win Rate: {metrics['win_rate']*100:.1f}%", "PARSING")
                        
                        # Calculate avg profit
                        if metrics["total_trades"] > 0:
                            metrics["avg_profit"] = avg_win * metrics["win_rate"] * 0.5
                            live_print(f"   Calculated Avg Profit: {metrics['avg_profit']*100:.3f}%", "PARSING")
                    except:
                        pass
            
            return metrics
            
        except Exception as e:
            live_print(f"   ❌ Parsing error: {e}", "ERROR")
            return metrics
    
    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate criteria score with live feedback"""
        live_print(f"🎯 Checking criteria compliance...", "CRITERIA")
        
        score = 0.0
        
        # Check each criterion
        if metrics["cagr"] > TARGET_CRITERIA["cagr"]:
            score += 1
            live_print(f"   ✅ CAGR: {metrics['cagr']*100:.2f}% > 25% ✓", "CRITERIA")
        else:
            gap = (TARGET_CRITERIA["cagr"] - metrics["cagr"]) * 100
            live_print(f"   ❌ CAGR: {metrics['cagr']*100:.2f}% < 25% (need +{gap:.1f}%)", "CRITERIA")
        
        if metrics["sharpe"] > TARGET_CRITERIA["sharpe"]:
            score += 1
            live_print(f"   ✅ SHARPE: {metrics['sharpe']:.3f} > 1.0 ✓", "CRITERIA")
        else:
            gap = TARGET_CRITERIA["sharpe"] - metrics["sharpe"]
            live_print(f"   ❌ SHARPE: {metrics['sharpe']:.3f} < 1.0 (need +{gap:.3f})", "CRITERIA")
        
        if metrics["drawdown"] < TARGET_CRITERIA["max_drawdown"]:
            score += 1
            live_print(f"   ✅ DRAWDOWN: {metrics['drawdown']*100:.1f}% < 20% ✓", "CRITERIA")
        else:
            excess = (metrics["drawdown"] - TARGET_CRITERIA["max_drawdown"]) * 100
            live_print(f"   ❌ DRAWDOWN: {metrics['drawdown']*100:.1f}% > 20% (reduce by {excess:.1f}%)", "CRITERIA")
        
        if metrics["avg_profit"] > TARGET_CRITERIA["avg_profit"]:
            score += 1
            live_print(f"   ✅ AVG PROFIT: {metrics['avg_profit']*100:.3f}% > 0.75% ✓", "CRITERIA")
        else:
            gap = (TARGET_CRITERIA["avg_profit"] - metrics["avg_profit"]) * 100
            live_print(f"   ❌ AVG PROFIT: {metrics['avg_profit']*100:.3f}% < 0.75% (need +{gap:.3f}%)", "CRITERIA")
        
        live_print(f"   🎯 TOTAL SCORE: {score}/4 criteria met", "CRITERIA")
        return score
    
    def _show_results(self):
        """Show results prominently"""
        status_update(f"RESULTS FOR {self.agent_id.upper()}")
        live_print(f"📊 CAGR: {self.performance['cagr']*100:.2f}% {'✅' if self.performance['cagr'] > TARGET_CRITERIA['cagr'] else '❌'}", "RESULTS")
        live_print(f"📈 Sharpe: {self.performance['sharpe']:.3f} {'✅' if self.performance['sharpe'] > TARGET_CRITERIA['sharpe'] else '❌'}", "RESULTS")
        live_print(f"📉 Max DD: {self.performance['drawdown']*100:.1f}% {'✅' if self.performance['drawdown'] < TARGET_CRITERIA['max_drawdown'] else '❌'}", "RESULTS")
        live_print(f"💰 Avg Profit: {self.performance['avg_profit']*100:.3f}% {'✅' if self.performance['avg_profit'] > TARGET_CRITERIA['avg_profit'] else '❌'}", "RESULTS")
        live_print(f"🎯 CRITERIA SCORE: {self.criteria_score}/4", "RESULTS")


class LiveTerminalDGM:
    """DGM with live terminal output"""
    
    def __init__(self):
        self.target_criteria = TARGET_CRITERIA
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_score = 0.0
        self.start_time = time.time()
        
        # Incremental mutations from working base to 25%+ CAGR
        self.mutations = [
            "boost_leverage_15x",
            "boost_leverage_20x", 
            "switch_to_tqqq",
            "aggressive_position_3x",
            "faster_signals_8_20",
            "ultra_fast_5_15",
            "reduce_for_sharpe",
            "add_rsi_management"
        ]
        
        status_update("LIVE TERMINAL DARWIN GÖDEL MACHINE STARTED")
        live_print(f"🎯 TARGET CRITERIA (ALL MUST BE MET):", "SYSTEM")
        live_print(f"   📊 CAGR: >{self.target_criteria['cagr']*100:.0f}%", "SYSTEM")
        live_print(f"   📈 Sharpe: >{self.target_criteria['sharpe']:.1f}", "SYSTEM")
        live_print(f"   📉 Max DD: <{self.target_criteria['max_drawdown']*100:.0f}%", "SYSTEM")
        live_print(f"   💰 Avg Profit: >{self.target_criteria['avg_profit']*100:.2f}%", "SYSTEM")
    
    def evolve_live(self):
        """Evolution with live terminal updates"""
        
        # Start with base agent
        status_update("CREATING BASE AGENT")
        base_agent = LiveTerminalAgent("base", 0)
        base_agent.performance = base_agent.run_backtest()
        
        self.archive.append(base_agent)
        self.best_agent = base_agent
        self.best_score = base_agent.criteria_score
        
        status_update(f"BASE AGENT SCORE: {self.best_score}/4 CRITERIA MET")
        
        # Evolution loop
        while self.best_score < 4.0:
            self.generation += 1
            runtime = (time.time() - self.start_time) / 60
            
            status_update(f"GENERATION {self.generation} - RUNTIME: {runtime:.1f}min")
            live_print(f"🎯 Current best score: {self.best_score}/4", "EVOLUTION")
            live_print(f"📊 Current best CAGR: {self.best_agent.performance['cagr']*100:.1f}%", "EVOLUTION")
            
            # Test mutations on best agent
            parent = self.best_agent
            live_print(f"👨‍👩‍👧‍👦 Using parent: {parent.agent_id} (score: {parent.criteria_score}/4)", "EVOLUTION")
            
            new_agents = []
            
            for mutation in self.mutations:
                status_update(f"TESTING MUTATION: {mutation.upper()}")
                
                child_id = f"gen{self.generation}_{mutation}"
                child = LiveTerminalAgent(child_id, self.generation)
                child.code_components = parent.code_components.copy()
                child.mutations = parent.mutations.copy()
                
                if child.apply_mutation(mutation):
                    child.performance = child.run_backtest()
                    
                    # Check for improvement
                    if child.criteria_score > self.best_score:
                        old_score = self.best_score
                        old_cagr = self.best_agent.performance['cagr'] * 100
                        
                        self.best_score = child.criteria_score
                        self.best_agent = child
                        
                        status_update("🎉 NEW CHAMPION FOUND!")
                        live_print(f"📈 Score improved: {old_score} → {self.best_score}/4", "CHAMPION")
                        live_print(f"💰 CAGR improved: {old_cagr:.1f}% → {child.performance['cagr']*100:.1f}%", "CHAMPION")
                        live_print(f"🧬 Evolution path: {' → '.join(child.mutations)}", "CHAMPION")
                        
                        # Check if all criteria met
                        if self.best_score >= 4.0:
                            status_update("🏆🏆🏆 ALL CRITERIA ACHIEVED! 🏆🏆🏆")
                            self.show_victory()
                            return child
                    
                    new_agents.append(child)
                
                # Brief pause for readability
                time.sleep(0.5)
            
            # Update archive
            self.archive.extend(new_agents)
            
            status_update(f"GENERATION {self.generation} COMPLETE")
            live_print(f"🎯 Best score: {self.best_score}/4", "SUMMARY")
            live_print(f"📊 Best CAGR: {self.best_agent.performance['cagr']*100:.1f}%", "SUMMARY")
            live_print(f"📁 Archive size: {len(self.archive)}", "SUMMARY")
    
    def show_victory(self):
        """Show victory details"""
        runtime = (time.time() - self.start_time) / 3600
        
        print("\n" + "🏆" * 60)
        status_update("ALL CRITERIA ACHIEVED!")
        print("🏆" * 60)
        
        agent = self.best_agent
        live_print(f"📊 FINAL CAGR: {agent.performance['cagr']*100:.2f}% ✅", "VICTORY")
        live_print(f"📈 FINAL SHARPE: {agent.performance['sharpe']:.3f} ✅", "VICTORY")
        live_print(f"📉 FINAL MAX DD: {agent.performance['drawdown']*100:.1f}% ✅", "VICTORY")
        live_print(f"💰 FINAL AVG PROFIT: {agent.performance['avg_profit']*100:.3f}% ✅", "VICTORY")
        live_print(f"⏱️ Runtime: {runtime:.2f} hours", "VICTORY")
        live_print(f"🧬 Generations: {self.generation}", "VICTORY")
        live_print(f"🏆 Evolution: {' → '.join(agent.mutations)}", "VICTORY")


def main():
    """Launch live terminal DGM"""
    try:
        dgm = LiveTerminalDGM()
        winner = dgm.evolve_live()
        
        if winner:
            status_update("SUCCESS! ALL CRITERIA MET!")
        
    except KeyboardInterrupt:
        status_update("Evolution stopped by user")
    except Exception as e:
        status_update(f"Error: {e}")


if __name__ == "__main__":
    main()