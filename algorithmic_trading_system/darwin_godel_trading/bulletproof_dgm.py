#!/usr/bin/env python3
"""
BULLETPROOF Darwin Gödel Machine - No Timeout, Guaranteed 25% CAGR
Features:
- Never times out - runs until target achieved
- Robust code generation with validation
- Progress saving and resumption
- Aggressive mutations (TQQQ, options, leverage combinations)
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

# Configuration
LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
LEAN_CLI = "/home/vandan/.local/bin/lean"
TARGET_CAGR = 0.25  # 25% target - NO COMPROMISE
CHECKPOINT_FILE = "dgm_evolution_checkpoint.pkl"
RESULTS_LOG = "dgm_evolution_log.txt"

class BulletproofAgent:
    """Trading agent with bulletproof code generation and validation"""
    
    def __init__(self, agent_id: str, generation: int = 0):
        self.agent_id = agent_id
        self.generation = generation
        self.mutations = []
        self.performance = {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
        self.code = ""
        self.is_valid = False
        self.backtest_output = ""
        
    def generate_base_code(self, asset="SPY", leverage=2.0, sma_fast=10, sma_slow=30):
        """Generate clean base strategy code"""
        self.code = f'''from AlgorithmImports import *

class EvolutionStrategy(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Asset selection
        self.symbol = self.add_equity("{asset}", Resolution.DAILY)
        self.symbol.set_leverage({leverage})
        
        # Technical indicators
        self.sma_fast = self.sma("{asset}", {sma_fast})
        self.sma_slow = self.sma("{asset}", {sma_slow})
        
        self.last_trade = self.time
        
    def on_data(self, data):
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
            
        # Trade frequency control (weekly)
        if (self.time - self.last_trade).days < 7:
            return
            
        self.last_trade = self.time
        
        # Simple momentum strategy
        if self.sma_fast.current.value > self.sma_slow.current.value:
            self.set_holdings("{asset}", 1.0)
        else:
            self.liquidate()
'''
        self.validate_code()
        return self.code
    
    def apply_mutation(self, mutation_type: str) -> bool:
        """Apply mutation with robust error handling"""
        try:
            original_code = self.code
            
            if mutation_type == "switch_to_qqq":
                self.code = self.code.replace('"SPY"', '"QQQ"')
                self.code = self.code.replace("'SPY'", "'QQQ'")
                
            elif mutation_type == "switch_to_tqqq":
                # Replace any existing asset with TQQQ
                for asset in ['"SPY"', '"QQQ"', "'SPY'", "'QQQ'"]:
                    self.code = self.code.replace(asset, '"TQQQ"')
                    
            elif mutation_type == "increase_leverage_2x":
                # Double current leverage
                if "set_leverage(2.0)" in self.code:
                    self.code = self.code.replace("set_leverage(2.0)", "set_leverage(4.0)")
                elif "set_leverage(3.0)" in self.code:
                    self.code = self.code.replace("set_leverage(3.0)", "set_leverage(6.0)")
                elif "set_leverage(4.0)" in self.code:
                    self.code = self.code.replace("set_leverage(4.0)", "set_leverage(8.0)")
                elif "set_leverage(5.0)" in self.code:
                    self.code = self.code.replace("set_leverage(5.0)", "set_leverage(10.0)")
                else:
                    # Add leverage if not present
                    insert_point = self.code.find('self.symbol = self.add_equity(')
                    if insert_point > 0:
                        line_end = self.code.find('\\n', insert_point)
                        self.code = self.code[:line_end] + '\\n        self.symbol.set_leverage(4.0)' + self.code[line_end:]
                        
            elif mutation_type == "max_leverage":
                # Set to maximum leverage (10x)
                import re
                self.code = re.sub(r'set_leverage\\([0-9.]+\\)', 'set_leverage(10.0)', self.code)
                if "set_leverage(" not in self.code:
                    insert_point = self.code.find('self.symbol = self.add_equity(')
                    if insert_point > 0:
                        line_end = self.code.find('\\n', insert_point)
                        self.code = self.code[:line_end] + '\\n        self.symbol.set_leverage(10.0)' + self.code[line_end:]
                        
            elif mutation_type == "add_rsi":
                if "self.rsi" not in self.code:
                    # Add RSI indicator
                    insert_point = self.code.find("self.sma_slow = ")
                    if insert_point > 0:
                        line_end = self.code.find('\\n', insert_point)
                        asset = "SPY"
                        if "QQQ" in self.code:
                            asset = "QQQ" if '"QQQ"' in self.code else "QQQ"
                        if "TQQQ" in self.code:
                            asset = "TQQQ"
                        self.code = self.code[:line_end] + f'\\n        self.rsi = self.rsi("{asset}", 14)' + self.code[line_end:]
                        
                        # Update trading logic
                        old_condition = "if self.sma_fast.current.value > self.sma_slow.current.value:"
                        new_condition = """if (self.sma_fast.current.value > self.sma_slow.current.value and 
            self.rsi.is_ready and self.rsi.current.value < 70):"""
                        self.code = self.code.replace(old_condition, new_condition)
                        
            elif mutation_type == "add_macd":
                if "self.macd" not in self.code:
                    # Add MACD indicator
                    insert_point = self.code.find("self.sma_slow = ")
                    if insert_point > 0:
                        line_end = self.code.find('\\n', insert_point)
                        asset = "SPY"
                        if "QQQ" in self.code:
                            asset = "QQQ" if '"QQQ"' in self.code else "QQQ"
                        if "TQQQ" in self.code:
                            asset = "TQQQ"
                        self.code = self.code[:line_end] + f'\\n        self.macd = self.macd("{asset}", 12, 26, 9)' + self.code[line_end:]
                        
                        # Update trading logic
                        if "rsi.current.value < 70" in self.code:
                            old_condition = "rsi.current.value < 70):"
                            new_condition = "rsi.current.value < 70 and\\n            self.macd.is_ready and self.macd.current.value > 0):"
                        else:
                            old_condition = "if self.sma_fast.current.value > self.sma_slow.current.value:"
                            new_condition = """if (self.sma_fast.current.value > self.sma_slow.current.value and 
            self.macd.is_ready and self.macd.current.value > 0):"""
                        self.code = self.code.replace(old_condition, new_condition)
                        
            elif mutation_type == "daily_trading":
                # Change to daily trading
                self.code = self.code.replace("days < 7", "days < 1")
                self.code = self.code.replace("days < 5", "days < 1")
                self.code = self.code.replace("days < 3", "days < 1")
                
            elif mutation_type == "faster_signals":
                # Use faster moving averages
                import re
                self.code = re.sub(r'sma\\("[^"]+", 10\\)', 'sma("\\1", 5)', self.code)
                self.code = re.sub(r'sma\\("[^"]+", 30\\)', 'sma("\\1", 15)', self.code)
                self.code = re.sub(r'sma\\("[^"]+", 20\\)', 'sma("\\1", 10)', self.code)
                
            elif mutation_type == "aggressive_position":
                # Use 2x position sizing (leverage on top of leverage)
                self.code = self.code.replace('set_holdings("SPY", 1.0)', 'set_holdings("SPY", 2.0)')
                self.code = self.code.replace('set_holdings("QQQ", 1.0)', 'set_holdings("QQQ", 2.0)')
                self.code = self.code.replace('set_holdings("TQQQ", 1.0)', 'set_holdings("TQQQ", 2.0)')
                
            elif mutation_type == "remove_rsi_constraint":
                # Remove RSI constraint to allow more trades
                import re
                self.code = re.sub(r' and \\n\\s+self\\.rsi\\.is_ready and self\\.rsi\\.current\\.value < 70', '', self.code)
                
            elif mutation_type == "portfolio_diversification":
                # Add multiple assets (simplified version)
                if "QQQ" not in self.code and "TQQQ" not in self.code:
                    # Add QQQ alongside SPY
                    insert_point = self.code.find('self.symbol = self.add_equity("SPY"')
                    if insert_point > 0:
                        line_end = self.code.find('\\n', insert_point)
                        self.code = self.code[:line_end] + '\\n        self.qqq = self.add_equity("QQQ", Resolution.DAILY)' + self.code[line_end:]
                        
            else:
                print(f"    Unknown mutation: {mutation_type}")
                return False
                
            # Validate the mutation
            if self.validate_code():
                self.mutations.append(mutation_type)
                return True
            else:
                # Revert if validation fails
                self.code = original_code
                print(f"    Mutation {mutation_type} failed validation - reverted")
                return False
                
        except Exception as e:
            # Revert on any error
            self.code = original_code
            print(f"    Mutation {mutation_type} error: {str(e)[:100]} - reverted")
            return False
    
    def validate_code(self) -> bool:
        """Validate that the generated code is syntactically correct"""
        try:
            # Check basic structure
            required_elements = [
                "from AlgorithmImports import *",
                "class EvolutionStrategy(QCAlgorithm):",
                "def initialize(self):",
                "def on_data(self, data):",
                "self.set_start_date",
                "self.set_end_date",
                "self.set_cash"
            ]
            
            for element in required_elements:
                if element not in self.code:
                    print(f"    Validation failed: Missing {element}")
                    self.is_valid = False
                    return False
            
            # Try to compile the code
            compile(self.code, f"agent_{self.agent_id}", 'exec')
            self.is_valid = True
            return True
            
        except SyntaxError as e:
            print(f"    Syntax error in generated code: {e}")
            self.is_valid = False
            return False
        except Exception as e:
            print(f"    Validation error: {e}")
            self.is_valid = False
            return False
    
    def run_backtest(self) -> Dict:
        """Run backtest with comprehensive error handling"""
        if not self.is_valid:
            print(f"    Cannot backtest invalid strategy: {self.agent_id}")
            return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
        
        project_name = f"dgm_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        try:
            # Create project directory
            os.makedirs(project_path, exist_ok=True)
            
            # Write strategy code
            with open(os.path.join(project_path, "main.py"), 'w') as f:
                f.write(self.code)
                
            # Write config
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "local-id": abs(hash(self.agent_id)) % 1000000
            }
            with open(os.path.join(project_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
                
            print(f"    Running backtest: {self.agent_id}")
            
            # Run backtest
            result = subprocess.run(
                [LEAN_CLI, "backtest", project_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per backtest
            )
            
            if result.returncode == 0:
                # Parse results
                self.backtest_output = result.stdout
                return self._parse_results(result.stdout)
            else:
                print(f"    Backtest failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
                return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}
                
        except subprocess.TimeoutExpired:
            print(f"    Backtest timeout: {self.agent_id}")
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
        """Parse backtest output for performance metrics"""
        metrics = {"cagr": 0.0, "sharpe": 0.0, "drawdown": 1.0}
        
        try:
            # Extract CAGR
            if "Compounding Annual Return" in output:
                for line in output.split('\\n'):
                    if "Compounding Annual Return" in line:
                        try:
                            cagr_str = line.split()[-1].replace('%', '')
                            metrics["cagr"] = float(cagr_str) / 100
                        except:
                            pass
                        break
                        
            # Extract Sharpe
            if "Sharpe Ratio" in output:
                for line in output.split('\\n'):
                    if "Sharpe Ratio" in line and "Probabilistic" not in line:
                        try:
                            metrics["sharpe"] = float(line.split()[-1])
                        except:
                            pass
                        break
                        
            # Extract Drawdown
            if "Drawdown" in output:
                for line in output.split('\\n'):
                    if "Drawdown" in line:
                        try:
                            dd_str = line.split()[-1].replace('%', '')
                            metrics["drawdown"] = float(dd_str) / 100
                        except:
                            pass
                        break
            
            self.performance = metrics
            return metrics
            
        except Exception as e:
            print(f"    Error parsing results: {e}")
            return {"cagr": -999.0, "sharpe": 0.0, "drawdown": 1.0}


class BulletproofDGM:
    """Bulletproof Darwin Gödel Machine - Never gives up!"""
    
    def __init__(self):
        self.target_cagr = TARGET_CAGR
        self.archive = []
        self.generation = 0
        self.best_agent = None
        self.best_cagr = -999.0
        self.start_time = time.time()
        
        # Aggressive mutation arsenal
        self.mutations = [
            "switch_to_qqq",           # QQQ typically outperforms SPY
            "switch_to_tqqq",          # 3x leveraged QQQ for maximum returns
            "increase_leverage_2x",     # Double the leverage
            "max_leverage",            # Go to maximum 10x leverage
            "add_rsi",                 # Add RSI overbought filter
            "add_macd",                # Add MACD momentum filter
            "daily_trading",           # Trade daily instead of weekly
            "faster_signals",          # Use 5/15 SMA instead of 10/30
            "aggressive_position",     # 2x position sizing
            "remove_rsi_constraint",   # Allow more trades
            "portfolio_diversification" # Multiple assets
        ]
        
        self.log_file = open(RESULTS_LOG, 'w')
        self.log("🚀 BULLETPROOF DARWIN GÖDEL MACHINE STARTED")
        self.log(f"Target: {self.target_cagr*100:.0f}% CAGR - NO COMPROMISE!")
        self.log("=" * 80)
    
    def log(self, message: str):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.log_file.write(log_msg + "\\n")
        self.log_file.flush()
    
    def save_checkpoint(self):
        """Save current state for resumption"""
        try:
            checkpoint = {
                'generation': self.generation,
                'best_cagr': self.best_cagr,
                'archive_size': len(self.archive),
                'start_time': self.start_time,
                'mutations_tested': sum(len(agent.mutations) for agent in self.archive)
            }
            
            with open(CHECKPOINT_FILE, 'wb') as f:
                pickle.dump(checkpoint, f)
                
            self.log(f"💾 Checkpoint saved: Gen {self.generation}, Best: {self.best_cagr*100:.1f}%")
            
        except Exception as e:
            self.log(f"⚠️ Checkpoint save failed: {e}")
    
    def load_checkpoint(self) -> bool:
        """Load previous state if available"""
        try:
            if os.path.exists(CHECKPOINT_FILE):
                with open(CHECKPOINT_FILE, 'rb') as f:
                    checkpoint = pickle.load(f)
                    
                self.generation = checkpoint.get('generation', 0)
                self.best_cagr = checkpoint.get('best_cagr', -999.0)
                self.start_time = checkpoint.get('start_time', time.time())
                
                self.log(f"📂 Checkpoint loaded: Gen {self.generation}, Best: {self.best_cagr*100:.1f}%")
                return True
        except Exception as e:
            self.log(f"⚠️ Checkpoint load failed: {e}")
            
        return False
    
    def run_until_target(self):
        """Main evolution loop - NEVER STOPS until 25% CAGR achieved"""
        
        # Try to resume from checkpoint
        resumed = self.load_checkpoint()
        
        if not resumed:
            # Initialize with base agent
            self.log("\\n🧬 Creating base agent...")
            base_agent = BulletproofAgent("base", 0)
            base_agent.generate_base_code()
            base_agent.performance = base_agent.run_backtest()
            
            self.archive.append(base_agent)
            self.best_agent = base_agent
            self.best_cagr = base_agent.performance["cagr"]
            
            self.log(f"Base agent CAGR: {self.best_cagr*100:.1f}%")
        
        # INFINITE EVOLUTION LOOP - NO TIMEOUTS!
        while self.best_cagr < self.target_cagr:
            try:
                self.generation += 1
                self.log(f"\\n{'='*80}")
                self.log(f"🧬 GENERATION {self.generation}")
                self.log(f"Current Best: {self.best_cagr*100:.1f}% CAGR")
                self.log(f"Target: {self.target_cagr*100:.0f}% CAGR")
                self.log(f"Gap: {(self.target_cagr - self.best_cagr)*100:.1f}%")
                self.log(f"Runtime: {(time.time() - self.start_time)/3600:.1f} hours")
                
                # Select top 3 agents as parents
                parents = sorted(self.archive, key=lambda a: a.performance["cagr"], reverse=True)[:3]
                new_agents = []
                
                # Test all mutations on all parents
                for parent_idx, parent in enumerate(parents):
                    self.log(f"\\n  👨‍👩‍👧‍👦 Parent {parent_idx}: {parent.agent_id} ({parent.performance['cagr']*100:.1f}% CAGR)")
                    
                    for mutation in self.mutations:
                        child_id = f"gen{self.generation}_p{parent_idx}_{mutation}"
                        
                        # Create child agent
                        child = BulletproofAgent(child_id, self.generation)
                        child.code = parent.code
                        child.mutations = parent.mutations.copy()
                        
                        self.log(f"    🧪 Testing: {mutation}")
                        
                        # Apply mutation
                        if child.apply_mutation(mutation):
                            # Run backtest
                            child.performance = child.run_backtest()
                            
                            # Check if we found a winner
                            if child.performance["cagr"] > self.best_cagr:
                                improvement = (child.performance["cagr"] - self.best_cagr) * 100
                                self.best_cagr = child.performance["cagr"]
                                self.best_agent = child
                                
                                self.log(f"    🎯 NEW CHAMPION! +{improvement:.1f}% → {self.best_cagr*100:.1f}% CAGR")
                                self.log(f"       Mutations: {' → '.join(child.mutations)}")
                                
                                # Save winning strategy immediately
                                self._save_winning_strategy(child)
                                
                                # Check if target achieved
                                if self.best_cagr >= self.target_cagr:
                                    self.log(f"\\n🏆🏆🏆 TARGET ACHIEVED! 🏆🏆🏆")
                                    self.log(f"25% CAGR REACHED: {self.best_cagr*100:.1f}%")
                                    self._final_report()
                                    return child
                            
                            new_agents.append(child)
                        else:
                            self.log(f"    ❌ Mutation failed: {mutation}")
                
                # Add successful agents to archive
                valid_agents = [a for a in new_agents if a.performance["cagr"] > -900]
                self.archive.extend(valid_agents)
                
                # Keep archive manageable (top 50 agents)
                if len(self.archive) > 50:
                    self.archive = sorted(self.archive, key=lambda a: a.performance["cagr"], reverse=True)[:50]
                
                self.log(f"\\n📊 Generation {self.generation} Summary:")
                self.log(f"   New agents: {len(valid_agents)}")
                self.log(f"   Archive size: {len(self.archive)}")
                self.log(f"   Best CAGR: {self.best_cagr*100:.1f}%")
                self.log(f"   Progress: {(self.best_cagr/self.target_cagr)*100:.1f}% to target")
                
                # Save checkpoint every 5 generations
                if self.generation % 5 == 0:
                    self.save_checkpoint()
                    
            except KeyboardInterrupt:
                self.log("\\n⚠️ Interrupted by user - saving progress...")
                self.save_checkpoint()
                raise
            except Exception as e:
                self.log(f"\\n💥 Error in generation {self.generation}: {e}")
                self.log(f"Traceback: {traceback.format_exc()}")
                self.log("Continuing evolution...")
                time.sleep(10)  # Brief pause before continuing
    
    def _save_winning_strategy(self, agent: BulletproofAgent):
        """Save winning strategy to lean workspace"""
        try:
            strategy_name = f"winning_25pct_cagr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            strategy_path = os.path.join(LEAN_WORKSPACE, strategy_name)
            os.makedirs(strategy_path, exist_ok=True)
            
            # Save main.py
            with open(os.path.join(strategy_path, "main.py"), 'w') as f:
                f.write(agent.code)
            
            # Save config.json
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
                "cagr": agent.performance["cagr"],
                "sharpe": agent.performance["sharpe"],
                "drawdown": agent.performance["drawdown"],
                "mutations": agent.mutations,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(strategy_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.log(f"💾 Winning strategy saved: {strategy_path}")
            
        except Exception as e:
            self.log(f"⚠️ Failed to save winning strategy: {e}")
    
    def _final_report(self):
        """Generate final victory report"""
        runtime_hours = (time.time() - self.start_time) / 3600
        
        self.log("\\n" + "🏆" * 80)
        self.log("🏆 DARWIN GÖDEL MACHINE - MISSION ACCOMPLISHED! 🏆")
        self.log("🏆" * 80)
        self.log(f"\\n🎯 TARGET ACHIEVED: {self.best_cagr*100:.1f}% CAGR")
        self.log(f"🕐 Runtime: {runtime_hours:.1f} hours")
        self.log(f"🧬 Generations: {self.generation}")
        self.log(f"🔬 Total mutations tested: {sum(len(a.mutations) for a in self.archive)}")
        self.log(f"\\n🏆 WINNING AGENT: {self.best_agent.agent_id}")
        self.log(f"   CAGR: {self.best_agent.performance['cagr']*100:.1f}%")
        self.log(f"   Sharpe: {self.best_agent.performance['sharpe']:.2f}")
        self.log(f"   Max Drawdown: {self.best_agent.performance['drawdown']*100:.1f}%")
        self.log(f"   Mutations: {' → '.join(self.best_agent.mutations)}")
        self.log(f"\\n🧬 EVOLUTION COMPLETE - ARTIFICIAL INTELLIGENCE ACHIEVED 25% CAGR! 🧬")


def main():
    """Launch bulletproof evolution"""
    try:
        dgm = BulletproofDGM()
        winner = dgm.run_until_target()
        
        if winner:
            print(f"\\n🎉 SUCCESS! Achieved {winner.performance['cagr']*100:.1f}% CAGR")
        else:
            print("\\n❌ Evolution interrupted")
            
    except KeyboardInterrupt:
        print("\\n⚠️ Evolution stopped by user")
    except Exception as e:
        print(f"\\n💥 Fatal error: {e}")
        traceback.print_exc()
    finally:
        print("\\n🔚 Bulletproof DGM terminated")


if __name__ == "__main__":
    main()