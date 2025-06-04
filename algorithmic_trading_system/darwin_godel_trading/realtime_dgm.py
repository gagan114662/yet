#!/usr/bin/env python3
"""
REAL-TIME Darwin Gödel Machine with Live Progress Display
Shows ACTUAL backtest results as they happen - NO MOCK DATA
"""

import os
import json
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

# Configuration
LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
LEAN_CLI = "/home/vandan/.local/bin/lean"
TARGET_CAGR = 0.25

class RealTimeTradingAgent:
    """Trading agent with real-time result display"""
    
    def __init__(self, agent_id: str, generation: int = 0):
        self.agent_id = agent_id
        self.generation = generation
        self.mutations = []
        self.performance = {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
        self.code_components = {
            "asset": "SPY",
            "leverage": 2.0,
            "sma_fast": 10,
            "sma_slow": 30,
            "has_rsi": False,
            "has_macd": False,
            "trade_frequency": 7,
            "position_size": 1.0,
            "rsi_threshold": 70
        }
        self.is_valid = False
    
    def generate_code(self) -> str:
        """Generate strategy code"""
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
            
            # Build trading condition
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

class RealTimeStrategy(QCAlgorithm):
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
            print(f"    ❌ Code generation error: {e}")
            self.is_valid = False
            return ""
    
    def apply_mutation(self, mutation_type: str) -> bool:
        """Apply mutation"""
        try:
            original = self.code_components.copy()
            
            if mutation_type == "switch_to_qqq":
                self.code_components["asset"] = "QQQ"
            elif mutation_type == "switch_to_tqqq":
                self.code_components["asset"] = "TQQQ"
            elif mutation_type == "double_leverage":
                self.code_components["leverage"] = min(self.code_components["leverage"] * 2, 10.0)
            elif mutation_type == "max_leverage":
                self.code_components["leverage"] = 10.0
            elif mutation_type == "add_rsi":
                self.code_components["has_rsi"] = True
            elif mutation_type == "add_macd":
                self.code_components["has_macd"] = True
            elif mutation_type == "daily_trading":
                self.code_components["trade_frequency"] = 1
            elif mutation_type == "faster_sma":
                self.code_components["sma_fast"] = max(3, self.code_components["sma_fast"] // 2)
                self.code_components["sma_slow"] = max(5, self.code_components["sma_slow"] // 2)
            elif mutation_type == "aggressive_position":
                self.code_components["position_size"] = 2.0
            elif mutation_type == "ultra_aggressive":
                self.code_components["asset"] = "TQQQ"
                self.code_components["leverage"] = 10.0
                self.code_components["position_size"] = 2.0
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
        """Run REAL backtest and show live results"""
        if not self.is_valid:
            return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
        
        project_name = f"realtime_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        try:
            os.makedirs(project_path, exist_ok=True)
            
            # Write code
            code = self.generate_code()
            with open(os.path.join(project_path, "main.py"), 'w') as f:
                f.write(code)
            
            # Write config
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "local-id": abs(hash(self.agent_id)) % 1000000
            }
            with open(os.path.join(project_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"    ⚡ RUNNING REAL BACKTEST: {self.agent_id}")
            print(f"       Asset: {self.code_components['asset']}, Leverage: {self.code_components['leverage']}x")
            
            start_time = time.time()
            
            # Run REAL backtest
            result = subprocess.run(
                [LEAN_CLI, "backtest", project_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            backtest_time = time.time() - start_time
            
            if result.returncode == 0:
                metrics = self._parse_results(result.stdout)
                
                # REAL-TIME RESULT DISPLAY
                print(f"    ✅ REAL RESULT ({backtest_time:.1f}s):")
                print(f"       📊 CAGR: {metrics['cagr']*100:.2f}%")
                print(f"       📈 Sharpe: {metrics['sharpe']:.3f}")
                print(f"       📉 Max DD: {metrics['drawdown']*100:.1f}%")
                
                return metrics
            else:
                print(f"    ❌ BACKTEST FAILED: {result.stderr[:150] if result.stderr else 'Unknown error'}")
                return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:100]}")
            return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
        finally:
            try:
                if os.path.exists(project_path):
                    shutil.rmtree(project_path)
            except:
                pass
    
    def _parse_results(self, output: str) -> Dict:
        """Parse REAL backtest output"""
        metrics = {"cagr": 0.0, "sharpe": 0.0, "drawdown": 1.0}
        
        try:
            lines = output.split('\n')
            
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
            
            self.performance = metrics
            return metrics
            
        except Exception as e:
            return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}


class RealTimeDGM:
    """Real-time Darwin Gödel Machine with live progress tracking"""
    
    def __init__(self):
        self.target_cagr = TARGET_CAGR
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_cagr = -999.0
        self.start_time = time.time()
        self.improvement_history = []
        
        # Strategic mutations
        self.mutations = [
            "switch_to_qqq",
            "switch_to_tqqq", 
            "double_leverage",
            "max_leverage",
            "add_rsi",
            "add_macd",
            "daily_trading",
            "faster_sma",
            "aggressive_position",
            "ultra_aggressive"
        ]
        
        print("🚀 REAL-TIME DARWIN GÖDEL MACHINE")
        print(f"🎯 TARGET: {self.target_cagr*100:.0f}% CAGR")
        print("📊 LIVE RESULTS - NO MOCK DATA!")
        print("=" * 80)
    
    def show_real_time_progress(self):
        """Display live progress with REAL numbers"""
        runtime = (time.time() - self.start_time) / 60
        progress_pct = (self.best_cagr / self.target_cagr) * 100 if self.best_cagr > 0 else 0
        gap = (self.target_cagr - self.best_cagr) * 100
        
        print(f"\n{'='*80}")
        print(f"🧬 GENERATION {self.generation} | ⏱️ {runtime:.1f}min")
        print(f"🏆 BEST: {self.best_cagr*100:.2f}% CAGR | 🎯 TARGET: {self.target_cagr*100:.0f}%")
        print(f"📊 PROGRESS: {progress_pct:.1f}% | 📈 GAP: {gap:.1f}%")
        
        if len(self.improvement_history) > 1:
            recent_improvements = self.improvement_history[-3:]
            print(f"🔥 RECENT GAINS: {[f'+{imp:.1f}%' for imp in recent_improvements]}")
        
        print("=" * 80)
    
    def evolve_until_target(self):
        """Main evolution with real-time tracking"""
        
        # Create base
        print("\n🧬 Creating base agent...")
        base_agent = RealTimeTradingAgent("base", 0)
        base_agent.code_components = {
            "asset": "SPY", "leverage": 2.0, "sma_fast": 10, "sma_slow": 30,
            "has_rsi": False, "has_macd": False, "trade_frequency": 7,
            "position_size": 1.0, "rsi_threshold": 70
        }
        base_agent.generate_code()
        base_agent.performance = base_agent.run_backtest()
        
        self.archive.append(base_agent)
        self.best_agent = base_agent
        self.best_cagr = base_agent.performance["cagr"]
        
        print(f"\n🏁 BASE STRATEGY PERFORMANCE:")
        print(f"   📊 CAGR: {self.best_cagr*100:.2f}%")
        print(f"   📈 Sharpe: {base_agent.performance['sharpe']:.3f}")
        print(f"   📉 Drawdown: {base_agent.performance['drawdown']*100:.1f}%")
        
        # Evolution loop with real-time updates
        while self.best_cagr < self.target_cagr:
            self.generation += 1
            self.show_real_time_progress()
            
            # Select parents
            parents = sorted(self.archive, key=lambda a: a.performance["cagr"], reverse=True)[:2]
            new_agents = []
            
            for p_idx, parent in enumerate(parents):
                print(f"\n👨‍👩‍👧‍👦 PARENT {p_idx}: {parent.agent_id} ({parent.performance['cagr']*100:.2f}%)")
                
                for mutation in self.mutations:
                    child_id = f"gen{self.generation}_p{p_idx}_{mutation}"
                    child = RealTimeTradingAgent(child_id, self.generation)
                    child.code_components = parent.code_components.copy()
                    child.mutations = parent.mutations.copy()
                    
                    print(f"\n  🧪 TESTING MUTATION: {mutation.upper()}")
                    
                    if child.apply_mutation(mutation):
                        child.performance = child.run_backtest()
                        
                        # Check for improvement
                        if child.performance["cagr"] > self.best_cagr:
                            improvement = (child.performance["cagr"] - self.best_cagr) * 100
                            old_best = self.best_cagr * 100
                            
                            self.best_cagr = child.performance["cagr"]
                            self.best_agent = child
                            self.improvement_history.append(improvement)
                            
                            print(f"  🎉 NEW CHAMPION FOUND!")
                            print(f"     📈 IMPROVEMENT: +{improvement:.2f}% ({old_best:.2f}% → {self.best_cagr*100:.2f}%)")
                            print(f"     🧬 MUTATIONS: {' → '.join(child.mutations)}")
                            print(f"     📊 CONFIG: {child.code_components['asset']} {child.code_components['leverage']}x")
                            
                            # Save winner
                            self.save_winner(child)
                            
                            # Check target
                            if self.best_cagr >= self.target_cagr:
                                self.victory_celebration()
                                return child
                        
                        if child.performance["cagr"] > -900:
                            new_agents.append(child)
                    else:
                        print(f"  ❌ Mutation failed")
            
            # Update archive
            self.archive.extend(new_agents)
            if len(self.archive) > 20:
                self.archive = sorted(self.archive, key=lambda a: a.performance["cagr"], reverse=True)[:20]
            
            print(f"\n📊 GENERATION {self.generation} COMPLETE:")
            print(f"   🔬 New agents tested: {len(new_agents)}")
            print(f"   📁 Archive size: {len(self.archive)}")
            print(f"   🏆 Best CAGR: {self.best_cagr*100:.2f}%")
            
            # Show leaderboard
            top_3 = sorted(self.archive, key=lambda a: a.performance["cagr"], reverse=True)[:3]
            print(f"   🥇 TOP 3:")
            for i, agent in enumerate(top_3):
                print(f"      {i+1}. {agent.agent_id}: {agent.performance['cagr']*100:.2f}%")
    
    def save_winner(self, agent):
        """Save winning strategy"""
        try:
            strategy_name = f"realtime_winner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            strategy_path = os.path.join(LEAN_WORKSPACE, strategy_name)
            os.makedirs(strategy_path, exist_ok=True)
            
            code = agent.generate_code()
            with open(os.path.join(strategy_path, "main.py"), 'w') as f:
                f.write(code)
            
            config = {"algorithm-language": "Python", "parameters": {}}
            with open(os.path.join(strategy_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            metadata = {
                "performance": agent.performance,
                "mutations": agent.mutations,
                "components": agent.code_components,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(strategy_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"     💾 Strategy saved: {strategy_name}")
            
        except Exception as e:
            print(f"     ⚠️ Save failed: {e}")
    
    def victory_celebration(self):
        """Celebrate achieving 25% CAGR"""
        runtime = (time.time() - self.start_time) / 3600
        
        print("\n" + "🏆" * 80)
        print("🎉🎉🎉 25% CAGR TARGET ACHIEVED! 🎉🎉🎉")
        print("🏆" * 80)
        print(f"\n🎯 FINAL RESULT: {self.best_cagr*100:.2f}% CAGR")
        print(f"⏱️ RUNTIME: {runtime:.2f} hours")
        print(f"🧬 GENERATIONS: {self.generation}")
        print(f"🔬 TOTAL IMPROVEMENTS: {len(self.improvement_history)}")
        print(f"\n🏆 WINNING CONFIGURATION:")
        print(f"   Asset: {self.best_agent.code_components['asset']}")
        print(f"   Leverage: {self.best_agent.code_components['leverage']}x")
        print(f"   Position Size: {self.best_agent.code_components['position_size']}x")
        print(f"   Trading Frequency: Every {self.best_agent.code_components['trade_frequency']} days")
        print(f"   Technical Indicators: RSI={self.best_agent.code_components['has_rsi']}, MACD={self.best_agent.code_components['has_macd']}")
        print(f"   Evolution Path: {' → '.join(self.best_agent.mutations)}")
        print(f"\n🤖 AI SUCCESSFULLY ACHIEVED 25% CAGR WITH REAL BACKTESTS! 🤖")


def main():
    """Launch real-time DGM"""
    try:
        dgm = RealTimeDGM()
        winner = dgm.evolve_until_target()
        
        if winner:
            print(f"\n✅ SUCCESS! Achieved {winner.performance['cagr']*100:.2f}% CAGR")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evolution stopped by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()