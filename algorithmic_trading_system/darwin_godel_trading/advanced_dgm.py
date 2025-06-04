#!/usr/bin/env python3
"""
Advanced Darwin Gödel Machine for Trading
Incorporates patterns from jennyzzt/dgm with robust code generation
"""

import os
import json
import shutil
import subprocess
import time
import traceback
import pickle
from datetime import datetime
from typing import Dict, List, Optional
import textwrap

# Configuration
LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
LEAN_CLI = "/home/vandan/.local/bin/lean"
TARGET_CAGR = 0.25
CHECKPOINT_FILE = "advanced_dgm_checkpoint.pkl"

class TradingAgent:
    """Self-modifying trading agent with robust code generation"""
    
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
            "trade_frequency": 7,  # days
            "position_size": 1.0,
            "rsi_threshold": 70
        }
        self.is_valid = False
    
    def generate_code(self) -> str:
        """Generate complete strategy code from components"""
        try:
            # Extract components
            asset = self.code_components["asset"]
            leverage = self.code_components["leverage"]
            sma_fast = self.code_components["sma_fast"]
            sma_slow = self.code_components["sma_slow"]
            has_rsi = self.code_components["has_rsi"]
            has_macd = self.code_components["has_macd"]
            trade_freq = self.code_components["trade_frequency"]
            position_size = self.code_components["position_size"]
            rsi_threshold = self.code_components["rsi_threshold"]
            
            # Build indicators section
            indicators = f"""        self.sma_fast = self.sma("{asset}", {sma_fast})
        self.sma_slow = self.sma("{asset}", {sma_slow})"""
            
            if has_rsi:
                indicators += f"""
        self.rsi = self.rsi("{asset}", 14)"""
            
            if has_macd:
                indicators += f"""
        self.macd = self.macd("{asset}", 12, 26, 9)"""
            
            # Build trading condition
            base_condition = "self.sma_fast.current.value > self.sma_slow.current.value"
            conditions = [base_condition]
            
            if has_rsi:
                conditions.append(f"self.rsi.is_ready and self.rsi.current.value < {rsi_threshold}")
            
            if has_macd:
                conditions.append("self.macd.is_ready and self.macd.current.value > 0")
            
            full_condition = " and \n            ".join(conditions)
            
            # Generate complete code
            code = f'''from AlgorithmImports import *

class AdvancedStrategy(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Asset and leverage
        self.symbol = self.add_equity("{asset}", Resolution.DAILY)
        self.symbol.set_leverage({leverage})
        
        # Technical indicators
{indicators}
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        # Trade frequency control
        if (self.time - self.last_trade).days < {trade_freq}:
            return
            
        self.last_trade = self.time
        
        # Trading logic
        if {full_condition}:
            self.set_holdings("{asset}", {position_size})
        else:
            self.liquidate()
'''
            
            # Validate code
            compile(code, f"agent_{self.agent_id}", 'exec')
            self.is_valid = True
            return code
            
        except Exception as e:
            print(f"    Code generation error: {e}")
            self.is_valid = False
            return ""
    
    def apply_mutation(self, mutation_type: str) -> bool:
        """Apply mutation to code components"""
        try:
            original_components = self.code_components.copy()
            
            if mutation_type == "switch_to_qqq":
                self.code_components["asset"] = "QQQ"
                
            elif mutation_type == "switch_to_tqqq":
                self.code_components["asset"] = "TQQQ"
                
            elif mutation_type == "double_leverage":
                current = self.code_components["leverage"]
                self.code_components["leverage"] = min(current * 2, 10.0)
                
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
                
            elif mutation_type == "relaxed_rsi":
                self.code_components["rsi_threshold"] = 80
                
            else:
                return False
            
            # Test if code generates successfully
            test_code = self.generate_code()
            if test_code and self.is_valid:
                self.mutations.append(mutation_type)
                return True
            else:
                # Revert on failure
                self.code_components = original_components
                return False
                
        except Exception as e:
            # Revert on error
            self.code_components = original_components
            print(f"    Mutation error: {e}")
            return False
    
    def run_backtest(self) -> Dict:
        """Run backtest with generated code"""
        if not self.is_valid:
            return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
        
        project_name = f"adv_dgm_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        try:
            # Create project
            os.makedirs(project_path, exist_ok=True)
            
            # Generate and write code
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
            
            print(f"    Backtesting: {self.agent_id}")
            
            # Run backtest
            result = subprocess.run(
                [LEAN_CLI, "backtest", project_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=240
            )
            
            if result.returncode == 0:
                return self._parse_results(result.stdout)
            else:
                return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
                
        except Exception as e:
            print(f"    Backtest error: {str(e)[:100]}")
            return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
        finally:
            # Cleanup
            try:
                if os.path.exists(project_path):
                    shutil.rmtree(project_path)
            except:
                pass
    
    def _parse_results(self, output: str) -> Dict:
        """Parse backtest results"""
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


class AdvancedDGM:
    """Advanced Darwin Gödel Machine with robust evolution"""
    
    def __init__(self):
        self.target_cagr = TARGET_CAGR
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_cagr = -999.0
        self.start_time = time.time()
        
        # Advanced mutation strategies
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
            "relaxed_rsi"
        ]
        
        self.log_file = open("advanced_dgm_log.txt", 'w')
        self.log("🚀 ADVANCED DARWIN GÖDEL MACHINE")
        self.log(f"Target: {self.target_cagr*100:.0f}% CAGR")
        self.log("=" * 60)
    
    def log(self, message: str):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.log_file.write(log_msg + "\n")
        self.log_file.flush()
    
    def create_base_agent(self) -> TradingAgent:
        """Create base trading agent"""
        agent = TradingAgent("base", 0)
        # Start with working configuration
        agent.code_components = {
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
        agent.generate_code()
        return agent
    
    def evolve_until_target(self):
        """Main evolution loop"""
        
        # Create base agent
        self.log("\n🧬 Creating base agent...")
        base_agent = self.create_base_agent()
        base_agent.performance = base_agent.run_backtest()
        
        self.archive.append(base_agent)
        self.best_agent = base_agent
        self.best_cagr = base_agent.performance["cagr"]
        
        self.log(f"Base CAGR: {self.best_cagr*100:.1f}%")
        
        # Evolution loop
        while self.best_cagr < self.target_cagr:
            self.generation += 1
            self.log(f"\n{'='*60}")
            self.log(f"🧬 GENERATION {self.generation}")
            self.log(f"Best: {self.best_cagr*100:.1f}% | Target: {self.target_cagr*100:.0f}%")
            
            # Select top parents
            parents = sorted(self.archive, key=lambda a: a.performance["cagr"], reverse=True)[:3]
            new_agents = []
            
            for p_idx, parent in enumerate(parents):
                self.log(f"\n  Parent {p_idx}: {parent.agent_id} ({parent.performance['cagr']*100:.1f}%)")
                
                for mutation in self.mutations:
                    # Create child
                    child_id = f"gen{self.generation}_p{p_idx}_{mutation}"
                    child = TradingAgent(child_id, self.generation)
                    child.code_components = parent.code_components.copy()
                    child.mutations = parent.mutations.copy()
                    
                    self.log(f"    Testing: {mutation}")
                    
                    if child.apply_mutation(mutation):
                        child.performance = child.run_backtest()
                        
                        # Check for improvement
                        if child.performance["cagr"] > self.best_cagr:
                            improvement = (child.performance["cagr"] - self.best_cagr) * 100
                            self.best_cagr = child.performance["cagr"]
                            self.best_agent = child
                            
                            self.log(f"    🎯 NEW BEST! +{improvement:.1f}% → {self.best_cagr*100:.1f}%")
                            
                            # Save winning strategy
                            self.save_winner(child)
                            
                            if self.best_cagr >= self.target_cagr:
                                self.log(f"\n🏆 TARGET ACHIEVED! {self.best_cagr*100:.1f}% CAGR!")
                                self.final_report()
                                return child
                        
                        if child.performance["cagr"] > -900:
                            new_agents.append(child)
                    else:
                        self.log(f"    ❌ Mutation failed")
            
            # Update archive
            self.archive.extend(new_agents)
            if len(self.archive) > 30:
                self.archive = sorted(self.archive, key=lambda a: a.performance["cagr"], reverse=True)[:30]
            
            self.log(f"\nGeneration {self.generation} complete")
            self.log(f"Archive: {len(self.archive)} agents")
            self.log(f"Best: {self.best_cagr*100:.1f}% CAGR")
            
            # Save checkpoint
            if self.generation % 10 == 0:
                self.save_checkpoint()
    
    def save_winner(self, agent: TradingAgent):
        """Save winning strategy"""
        try:
            strategy_name = f"winner_25pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            strategy_path = os.path.join(LEAN_WORKSPACE, strategy_name)
            os.makedirs(strategy_path, exist_ok=True)
            
            # Save strategy
            code = agent.generate_code()
            with open(os.path.join(strategy_path, "main.py"), 'w') as f:
                f.write(code)
            
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "local-id": abs(hash(agent.agent_id)) % 1000000
            }
            with open(os.path.join(strategy_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save metadata
            metadata = {
                "agent_id": agent.agent_id,
                "generation": agent.generation,
                "performance": agent.performance,
                "mutations": agent.mutations,
                "components": agent.code_components,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(strategy_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.log(f"💾 Winner saved: {strategy_path}")
            
        except Exception as e:
            self.log(f"⚠️ Save failed: {e}")
    
    def save_checkpoint(self):
        """Save evolution state"""
        try:
            checkpoint = {
                'generation': self.generation,
                'best_cagr': self.best_cagr,
                'archive_size': len(self.archive),
                'runtime': time.time() - self.start_time
            }
            with open(CHECKPOINT_FILE, 'wb') as f:
                pickle.dump(checkpoint, f)
            self.log(f"💾 Checkpoint saved: Gen {self.generation}")
        except Exception as e:
            self.log(f"⚠️ Checkpoint failed: {e}")
    
    def final_report(self):
        """Generate final report"""
        runtime = (time.time() - self.start_time) / 3600
        
        self.log("\n" + "🏆" * 60)
        self.log("🏆 MISSION ACCOMPLISHED! 🏆")
        self.log("🏆" * 60)
        self.log(f"\n🎯 ACHIEVED: {self.best_cagr*100:.1f}% CAGR")
        self.log(f"⏱️ Runtime: {runtime:.1f} hours")
        self.log(f"🧬 Generations: {self.generation}")
        self.log(f"\n🏆 WINNING AGENT: {self.best_agent.agent_id}")
        self.log(f"   Asset: {self.best_agent.code_components['asset']}")
        self.log(f"   Leverage: {self.best_agent.code_components['leverage']}x")
        self.log(f"   Indicators: RSI={self.best_agent.code_components['has_rsi']}, MACD={self.best_agent.code_components['has_macd']}")
        self.log(f"   Trading: Every {self.best_agent.code_components['trade_frequency']} days")
        self.log(f"   Position: {self.best_agent.code_components['position_size']}x")
        self.log(f"   Mutations: {' → '.join(self.best_agent.mutations)}")
        self.log(f"\n🎉 AI ACHIEVED 25% CAGR TARGET! 🎉")


def main():
    """Launch advanced DGM"""
    try:
        dgm = AdvancedDGM()
        winner = dgm.evolve_until_target()
        
        if winner:
            print(f"\n🎉 SUCCESS! {winner.performance['cagr']*100:.1f}% CAGR achieved!")
        else:
            print("\n❌ Evolution incomplete")
            
    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()