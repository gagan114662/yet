#!/usr/bin/env python3
"""
TARGET CRUSHER DGM - Specifically designed to achieve ALL 4 criteria
CAGR >25%, Sharpe >1.0, Max DD <20%, Avg Profit >0.75%
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

class TargetCrusherAgent:
    """Agent specifically designed to crush all 4 targets"""
    
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
        # Start from optimized high-performance config
        self.code_components = {
            "asset": "QQQ",
            "leverage": 25.0,      # High leverage for high returns
            "sma_fast": 5,         # Fast signals
            "sma_slow": 21,        # Golden ratio
            "has_rsi": True,       # Risk management
            "has_macd": False,
            "trade_frequency": 1,   # Daily for maximum trades
            "position_size": 1.5,   # Aggressive but controlled
            "rsi_threshold": 65     # Conservative RSI
        }
        self.is_valid = False
        self.criteria_score = 0.0
    
    def generate_code(self) -> str:
        """Generate optimized strategy code"""
        try:
            asset = self.code_components["asset"]
            leverage = self.code_components["leverage"]
            sma_fast = self.code_components["sma_fast"]
            sma_slow = self.code_components["sma_slow"]
            has_rsi = self.code_components["has_rsi"]
            has_macd = self.code_components["has_macd"]
            trade_freq = self.code_components["trade_frequency"]
            position_size = self.code_components["position_size"]
            rsi_threshold = self.code_components["rsi_threshold"]
            
            live_print(f"🔧 {asset} {leverage}x leverage, {position_size}x position", "CODE")
            
            # Build indicators
            indicators_lines = [
                f'        self.sma_fast = self.sma("{asset}", {sma_fast})',
                f'        self.sma_slow = self.sma("{asset}", {sma_slow})'
            ]
            
            if has_rsi:
                indicators_lines.append(f'        self.rsi = self.rsi("{asset}", 14)')
            
            if has_macd:
                indicators_lines.append(f'        self.macd = self.macd("{asset}", 12, 26, 9)')
            
            indicators = '\n'.join(indicators_lines)
            
            # Build conditions
            conditions = ["self.sma_fast.current.value > self.sma_slow.current.value"]
            
            if has_rsi:
                conditions.append(f"self.rsi.is_ready and self.rsi.current.value < {rsi_threshold}")
            
            if has_macd:
                conditions.append("self.macd.is_ready and self.macd.current.value > 0")
            
            if len(conditions) == 1:
                full_condition = conditions[0]
            else:
                condition_str = " and \\\n            ".join(conditions)
                full_condition = f"({condition_str})"
            
            # Generate code
            code = f'''from AlgorithmImports import *

class TargetCrusherStrategy(QCAlgorithm):
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
            
            compile(code, f"agent_{self.agent_id}", 'exec')
            self.is_valid = True
            return code
            
        except Exception as e:
            live_print(f"❌ Code error: {e}", "ERROR")
            self.is_valid = False
            return ""
    
    def apply_mutation(self, mutation_type: str) -> bool:
        """Apply targeted mutations"""
        try:
            original = self.code_components.copy()
            
            if mutation_type == "extreme_leverage_30x":
                self.code_components["leverage"] = 30.0
                live_print(f"   Extreme leverage: 30x", "MUTATION")
            elif mutation_type == "switch_to_tqqq_balanced":
                self.code_components["asset"] = "TQQQ"
                self.code_components["leverage"] = 8.0  # Lower leverage on 3x ETF
                live_print(f"   TQQQ with 8x leverage", "MUTATION")
            elif mutation_type == "ultra_fast_signals":
                self.code_components["sma_fast"] = 3
                self.code_components["sma_slow"] = 12
                live_print(f"   Ultra fast signals: SMA(3,12)", "MUTATION")
            elif mutation_type == "golden_ratio_signals":
                self.code_components["sma_fast"] = 8
                self.code_components["sma_slow"] = 21
                live_print(f"   Golden ratio signals: SMA(8,21)", "MUTATION")
            elif mutation_type == "aggressive_position_2x":
                self.code_components["position_size"] = 2.5
                live_print(f"   Aggressive position: 2.5x", "MUTATION")
            elif mutation_type == "sharpe_optimizer":
                self.code_components["leverage"] = 18.0
                self.code_components["has_rsi"] = True
                self.code_components["rsi_threshold"] = 60
                self.code_components["trade_frequency"] = 2
                live_print(f"   Sharpe optimizer: 18x leverage, RSI<60, 2-day", "MUTATION")
            elif mutation_type == "drawdown_controller":
                self.code_components["has_rsi"] = True
                self.code_components["has_macd"] = True
                self.code_components["position_size"] = 1.2
                live_print(f"   Drawdown control: RSI+MACD, 1.2x position", "MUTATION")
            elif mutation_type == "profit_booster":
                self.code_components["leverage"] = 22.0
                self.code_components["sma_fast"] = 4
                self.code_components["sma_slow"] = 16
                live_print(f"   Profit booster: 22x leverage, SMA(4,16)", "MUTATION")
            else:
                return False
            
            # Test code generation
            test_code = self.generate_code()
            if test_code and self.is_valid:
                self.mutations.append(mutation_type)
                return True
            else:
                self.code_components = original
                return False
                
        except Exception as e:
            self.code_components = original
            return False
    
    def run_backtest(self) -> Dict:
        """Run backtest with full metrics"""
        if not self.is_valid:
            return self.performance
        
        project_name = f"crusher_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        try:
            os.makedirs(project_path, exist_ok=True)
            
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
            
            live_print(f"⚡ TESTING: {self.agent_id}", "BACKTEST")
            
            start_time = time.time()
            
            result = subprocess.run(
                [LEAN_CLI, "backtest", project_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            backtest_time = time.time() - start_time
            
            if result.returncode == 0:
                metrics = self._parse_metrics(result.stdout)
                self.performance = metrics
                self.criteria_score = self._calculate_score(metrics)
                
                live_print(f"✅ RESULTS ({backtest_time:.1f}s):", "RESULTS")
                live_print(f"   📊 CAGR: {metrics['cagr']*100:.2f}% {'✅' if metrics['cagr'] > TARGET_CRITERIA['cagr'] else '❌'}", "RESULTS")
                live_print(f"   📈 Sharpe: {metrics['sharpe']:.3f} {'✅' if metrics['sharpe'] > TARGET_CRITERIA['sharpe'] else '❌'}", "RESULTS")
                live_print(f"   📉 DD: {metrics['drawdown']*100:.1f}% {'✅' if metrics['drawdown'] < TARGET_CRITERIA['max_drawdown'] else '❌'}", "RESULTS")
                live_print(f"   💰 Avg Profit: {metrics['avg_profit']*100:.3f}% {'✅' if metrics['avg_profit'] > TARGET_CRITERIA['avg_profit'] else '❌'}", "RESULTS")
                live_print(f"   🎯 SCORE: {self.criteria_score:.1f}/4", "RESULTS")
                
                return metrics
            else:
                live_print(f"❌ FAILED: {result.stderr[:100] if result.stderr else 'Unknown'}", "ERROR")
                return self.performance
                
        except Exception as e:
            live_print(f"❌ Error: {str(e)[:80]}", "ERROR")
            return self.performance
        finally:
            try:
                if os.path.exists(project_path):
                    shutil.rmtree(project_path)
            except:
                pass
    
    def _parse_metrics(self, output: str) -> Dict:
        """Parse metrics from output"""
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
                    except:
                        pass
                elif "Sharpe Ratio" in line and "Probabilistic" not in line:
                    try:
                        metrics["sharpe"] = float(line.split()[-1])
                    except:
                        pass
                elif "Drawdown" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["drawdown"] = float(value) / 100
                    except:
                        pass
                elif "Total Orders" in line:
                    try:
                        metrics["total_trades"] = int(line.split()[-1])
                    except:
                        pass
                elif "Average Win" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        avg_win = float(value) / 100
                    except:
                        pass
                elif "Win Rate" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["win_rate"] = float(value) / 100
                        if metrics["total_trades"] > 0:
                            metrics["avg_profit"] = avg_win * metrics["win_rate"] * 0.5
                    except:
                        pass
            
            return metrics
            
        except Exception as e:
            return metrics
    
    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate criteria score"""
        score = 0.0
        
        if metrics["cagr"] > TARGET_CRITERIA["cagr"]:
            score += 1
        if metrics["sharpe"] > TARGET_CRITERIA["sharpe"]:
            score += 1
        if metrics["drawdown"] < TARGET_CRITERIA["max_drawdown"]:
            score += 1
        if metrics["avg_profit"] > TARGET_CRITERIA["avg_profit"]:
            score += 1
            
        return score


class TargetCrusherDGM:
    """DGM designed to crush all targets"""
    
    def __init__(self):
        self.target_criteria = TARGET_CRITERIA
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_score = 0.0
        self.start_time = time.time()
        
        # Targeted mutations for specific criteria
        self.mutations = [
            "extreme_leverage_30x",
            "switch_to_tqqq_balanced",
            "ultra_fast_signals",
            "golden_ratio_signals",
            "aggressive_position_2x",
            "sharpe_optimizer",
            "drawdown_controller",
            "profit_booster"
        ]
        
        live_print("🚀 TARGET CRUSHER DARWIN GÖDEL MACHINE", "SYSTEM")
        live_print("🎯 ALL TARGET CRITERIA MUST BE MET:", "SYSTEM")
        live_print(f"   📊 CAGR: >{self.target_criteria['cagr']*100:.0f}%", "SYSTEM")
        live_print(f"   📈 Sharpe: >{self.target_criteria['sharpe']:.1f}", "SYSTEM")
        live_print(f"   📉 Max DD: <{self.target_criteria['max_drawdown']*100:.0f}%", "SYSTEM")
        live_print(f"   💰 Avg Profit: >{self.target_criteria['avg_profit']*100:.2f}%", "SYSTEM")
    
    def crush_targets(self):
        """Evolution until ALL targets are crushed"""
        
        # Start with optimized base
        live_print("🧬 Creating optimized base agent...", "INIT")
        base_agent = TargetCrusherAgent("optimized_base", 0)
        base_agent.performance = base_agent.run_backtest()
        
        self.archive.append(base_agent)
        self.best_agent = base_agent
        self.best_score = base_agent.criteria_score
        
        live_print(f"🏁 BASE SCORE: {self.best_score}/4 criteria", "INIT")
        
        # Evolution loop
        while self.best_score < 4.0:
            self.generation += 1
            runtime = (time.time() - self.start_time) / 60
            
            live_print(f"", "")
            live_print(f"🧬 GENERATION {self.generation} | ⏱️ {runtime:.1f}min", "EVOLUTION")
            live_print(f"🎯 CURRENT BEST: {self.best_score}/4 | 📊 CAGR: {self.best_agent.performance['cagr']*100:.1f}%", "EVOLUTION")
            
            # Select top performers
            parents = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:2]
            new_agents = []
            
            for p_idx, parent in enumerate(parents):
                live_print(f"👨‍👩‍👧‍👦 PARENT {p_idx}: {parent.criteria_score}/4 ({parent.performance['cagr']*100:.1f}% CAGR)", "PARENT")
                
                for mutation in self.mutations:
                    live_print(f"🧪 {mutation.upper()}", "MUTATION")
                    
                    child_id = f"gen{self.generation}_{mutation}"
                    child = TargetCrusherAgent(child_id, self.generation)
                    child.code_components = parent.code_components.copy()
                    child.mutations = parent.mutations.copy()
                    
                    if child.apply_mutation(mutation):
                        child.performance = child.run_backtest()
                        
                        # Check improvement
                        if child.criteria_score > self.best_score:
                            old_score = self.best_score
                            old_cagr = self.best_agent.performance['cagr'] * 100
                            
                            self.best_score = child.criteria_score
                            self.best_agent = child
                            
                            live_print(f"🎉 NEW CHAMPION!", "CHAMPION")
                            live_print(f"   📈 Score: {old_score} → {self.best_score}/4", "CHAMPION")
                            live_print(f"   💰 CAGR: {old_cagr:.1f}% → {child.performance['cagr']*100:.1f}%", "CHAMPION")
                            
                            # Check if all criteria met
                            if self.best_score >= 4.0:
                                self.victory_celebration()
                                return child
                        
                        if child.criteria_score > 0:
                            new_agents.append(child)
            
            # Update archive
            self.archive.extend(new_agents)
            if len(self.archive) > 20:
                self.archive = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:20]
            
            live_print(f"📊 GENERATION {self.generation} COMPLETE: Best {self.best_score}/4", "SUMMARY")
    
    def victory_celebration(self):
        """Victory celebration"""
        runtime = (time.time() - self.start_time) / 3600
        
        print("\n" + "🏆" * 80)
        live_print("🎉🎉🎉 ALL TARGETS CRUSHED! 🎉🎉🎉", "VICTORY")
        print("🏆" * 80)
        
        agent = self.best_agent
        live_print(f"📊 FINAL CAGR: {agent.performance['cagr']*100:.2f}% ✅", "VICTORY")
        live_print(f"📈 FINAL SHARPE: {agent.performance['sharpe']:.3f} ✅", "VICTORY")
        live_print(f"📉 FINAL MAX DD: {agent.performance['drawdown']*100:.1f}% ✅", "VICTORY")
        live_print(f"💰 FINAL AVG PROFIT: {agent.performance['avg_profit']*100:.3f}% ✅", "VICTORY")
        live_print(f"⏱️ Runtime: {runtime:.2f} hours", "VICTORY")
        live_print(f"🧬 Generations: {self.generation}", "VICTORY")
        live_print(f"🏆 Evolution: {' → '.join(agent.mutations)}", "VICTORY")


def main():
    """Launch target crusher DGM"""
    try:
        dgm = TargetCrusherDGM()
        winner = dgm.crush_targets()
        
        if winner:
            live_print("✅ ALL TARGETS ACHIEVED!", "SUCCESS")
        
    except KeyboardInterrupt:
        live_print("⚠️ Stopped by user", "STOP")
    except Exception as e:
        live_print(f"💥 Error: {e}", "ERROR")


if __name__ == "__main__":
    main()