#!/usr/bin/env python3
"""
Enhanced Working DGM - Building from 14.82% CAGR Success
Adding incremental improvements to reach ALL criteria
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

# ALL TARGET CRITERIA
TARGET_CRITERIA = {
    "cagr": 0.25,           # >25%
    "sharpe": 1.0,          # >1.0
    "max_drawdown": 0.20,   # <20%
    "avg_profit": 0.0075    # >0.75%
}

class EnhancedWorkingAgent:
    """Enhanced version of the working 14.82% system"""
    
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
        # Start from KNOWN WORKING configuration (14.82% CAGR)
        self.code_components = {
            "asset": "QQQ",
            "leverage": 10.0,
            "sma_fast": 10,
            "sma_slow": 30,
            "has_rsi": False,
            "has_macd": False,
            "trade_frequency": 1,  # Daily trading
            "position_size": 2.0   # Aggressive position
        }
        self.is_valid = False
        self.criteria_score = 0.0
    
    def generate_code(self) -> str:
        """Generate PROVEN working code"""
        try:
            asset = self.code_components["asset"]
            leverage = self.code_components["leverage"]
            sma_fast = self.code_components["sma_fast"]
            sma_slow = self.code_components["sma_slow"]
            has_rsi = self.code_components["has_rsi"]
            has_macd = self.code_components["has_macd"]
            trade_freq = self.code_components["trade_frequency"]
            position_size = self.code_components["position_size"]
            
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
                conditions.append("self.rsi.is_ready and self.rsi.current.value < 70")
            
            if has_macd:
                conditions.append("self.macd.is_ready and self.macd.current.value > 0")
            
            if len(conditions) == 1:
                full_condition = conditions[0]
            else:
                condition_str = " and \\\n            ".join(conditions)
                full_condition = f"({condition_str})"
            
            # Generate PROVEN working code
            code = f'''from AlgorithmImports import *

class EnhancedWorkingStrategy(QCAlgorithm):
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
            print(f"    ❌ Code error: {e}")
            self.is_valid = False
            return ""
    
    def apply_mutation(self, mutation_type: str) -> bool:
        """Apply incremental improvements to working system"""
        try:
            original = self.code_components.copy()
            
            # Asset mutations
            if mutation_type == "try_spy":
                self.code_components["asset"] = "SPY"
            elif mutation_type == "try_tqqq":
                self.code_components["asset"] = "TQQQ"
                
            # Leverage optimizations
            elif mutation_type == "reduce_leverage_for_sharpe":
                self.code_components["leverage"] = 5.0  # Better Sharpe
            elif mutation_type == "moderate_leverage":
                self.code_components["leverage"] = 7.0
            elif mutation_type == "conservative_leverage":
                self.code_components["leverage"] = 3.0
                
            # Position size for drawdown control
            elif mutation_type == "reduce_position_for_dd":
                self.code_components["position_size"] = 1.5  # Reduce drawdown
            elif mutation_type == "conservative_position":
                self.code_components["position_size"] = 1.0
            elif mutation_type == "moderate_position":
                self.code_components["position_size"] = 1.8
                
            # Technical indicators for better entries
            elif mutation_type == "add_rsi_filter":
                self.code_components["has_rsi"] = True
            elif mutation_type == "add_macd_filter":
                self.code_components["has_macd"] = True
            elif mutation_type == "add_both_filters":
                self.code_components["has_rsi"] = True
                self.code_components["has_macd"] = True
                
            # Signal timing
            elif mutation_type == "faster_signals":
                self.code_components["sma_fast"] = 5
                self.code_components["sma_slow"] = 15
            elif mutation_type == "slower_signals":
                self.code_components["sma_fast"] = 15
                self.code_components["sma_slow"] = 45
            elif mutation_type == "medium_signals":
                self.code_components["sma_fast"] = 8
                self.code_components["sma_slow"] = 25
                
            # Trading frequency
            elif mutation_type == "weekly_trading":
                self.code_components["trade_frequency"] = 7  # Reduce frequency for better Sharpe
            elif mutation_type == "bi_daily_trading":
                self.code_components["trade_frequency"] = 2
                
            # Combination strategies
            elif mutation_type == "sharpe_optimized":
                # Optimize for Sharpe >1.0
                self.code_components["leverage"] = 5.0
                self.code_components["position_size"] = 1.5
                self.code_components["has_rsi"] = True
                self.code_components["trade_frequency"] = 7
                
            elif mutation_type == "drawdown_optimized":
                # Optimize for DD <20%
                self.code_components["leverage"] = 6.0
                self.code_components["position_size"] = 1.2
                self.code_components["has_rsi"] = True
                self.code_components["has_macd"] = True
                
            elif mutation_type == "balanced_optimization":
                # Balance all criteria
                self.code_components["leverage"] = 8.0
                self.code_components["position_size"] = 1.7
                self.code_components["has_rsi"] = True
                self.code_components["trade_frequency"] = 3
                
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
        
        project_name = f"enhanced_{self.agent_id}_{int(time.time())}"
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
            
            print(f"    ⚡ TESTING: {self.agent_id}")
            print(f"       {self.code_components['asset']} {self.code_components['leverage']}x, Pos: {self.code_components['position_size']}x")
            
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
                
                print(f"    ✅ RESULTS ({backtest_time:.1f}s):")
                print(f"       📊 CAGR: {metrics['cagr']*100:.2f}% {'✅' if metrics['cagr'] > TARGET_CRITERIA['cagr'] else '❌'}")
                print(f"       📈 Sharpe: {metrics['sharpe']:.3f} {'✅' if metrics['sharpe'] > TARGET_CRITERIA['sharpe'] else '❌'}")
                print(f"       📉 DD: {metrics['drawdown']*100:.1f}% {'✅' if metrics['drawdown'] < TARGET_CRITERIA['max_drawdown'] else '❌'}")
                print(f"       💰 Trades: {metrics['total_trades']}")
                print(f"       🎯 CRITERIA: {self.criteria_score:.1f}/4")
                
                return metrics
            else:
                print(f"    ❌ FAILED: {result.stderr[:100] if result.stderr else 'Unknown'}")
                return self.performance
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:80]}")
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


class EnhancedWorkingDGM:
    """Enhanced version building from 14.82% success"""
    
    def __init__(self):
        self.target_criteria = TARGET_CRITERIA
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_score = 0.0
        self.start_time = time.time()
        
        # Incremental improvement mutations
        self.mutations = [
            "try_spy",
            "try_tqqq",
            "reduce_leverage_for_sharpe",
            "moderate_leverage",
            "conservative_leverage",
            "reduce_position_for_dd",
            "conservative_position",
            "moderate_position",
            "add_rsi_filter",
            "add_macd_filter",
            "add_both_filters",
            "faster_signals",
            "slower_signals",
            "medium_signals",
            "weekly_trading",
            "bi_daily_trading",
            "sharpe_optimized",
            "drawdown_optimized",
            "balanced_optimization"
        ]
        
        print("🚀 ENHANCED WORKING DGM")
        print("📈 Building from 14.82% CAGR success")
        print("🎯 TARGET ALL CRITERIA:")
        print(f"   📊 CAGR: >{self.target_criteria['cagr']*100:.0f}%")
        print(f"   📈 Sharpe: >{self.target_criteria['sharpe']:.1f}")
        print(f"   📉 Max DD: <{self.target_criteria['max_drawdown']*100:.0f}%")
        print(f"   💰 Avg Profit: >{self.target_criteria['avg_profit']*100:.2f}%")
        print("=" * 80)
    
    def evolve_from_working_base(self):
        """Evolution from proven 14.82% base"""
        
        # Start with KNOWN WORKING strategy
        print("\n🧬 Starting from 14.82% CAGR base...")
        base_agent = EnhancedWorkingAgent("working_base", 0)
        base_agent.generate_code()
        base_agent.performance = base_agent.run_backtest()
        
        self.archive.append(base_agent)
        self.best_agent = base_agent
        self.best_score = base_agent.criteria_score
        
        print(f"\n🏁 WORKING BASE PERFORMANCE:")
        print(f"   🎯 Criteria Score: {self.best_score}/4")
        
        # Evolution loop
        while self.best_score < 4.0:
            self.generation += 1
            runtime = (time.time() - self.start_time) / 60
            
            print(f"\n{'='*80}")
            print(f"🧬 GENERATION {self.generation} | ⏱️ {runtime:.1f}min")
            print(f"🎯 SCORE: {self.best_score}/4 | 📊 CAGR: {self.best_agent.performance['cagr']*100:.1f}%")
            print("=" * 80)
            
            # Select top performers
            parents = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:2]
            new_agents = []
            
            for p_idx, parent in enumerate(parents):
                print(f"\n👨‍👩‍👧‍👦 PARENT {p_idx}: {parent.criteria_score}/4 ({parent.performance['cagr']*100:.1f}% CAGR)")
                
                for mutation in self.mutations:
                    child_id = f"gen{self.generation}_p{p_idx}_{mutation}"
                    child = EnhancedWorkingAgent(child_id, self.generation)
                    child.code_components = parent.code_components.copy()
                    child.mutations = parent.mutations.copy()
                    
                    print(f"\n  🧪 {mutation.upper()}")
                    
                    if child.apply_mutation(mutation):
                        child.performance = child.run_backtest()
                        
                        # Check improvement
                        if child.criteria_score > self.best_score or \
                           (child.criteria_score == self.best_score and child.performance['cagr'] > self.best_agent.performance['cagr']):
                            
                            old_score = self.best_score
                            old_cagr = self.best_agent.performance['cagr'] * 100
                            
                            self.best_score = child.criteria_score
                            self.best_agent = child
                            
                            print(f"  🎉 IMPROVEMENT!")
                            print(f"     📈 Score: {old_score} → {self.best_score}/4")
                            print(f"     💰 CAGR: {old_cagr:.1f}% → {child.performance['cagr']*100:.1f}%")
                            print(f"     🧬 Path: {' → '.join(child.mutations)}")
                            
                            # Save winner
                            self.save_winner(child)
                            
                            # Check if all criteria met
                            if self.best_score >= 4.0:
                                self.victory_celebration()
                                return child
                        
                        if child.criteria_score > 0:
                            new_agents.append(child)
                    else:
                        print(f"  ❌ Failed")
            
            # Update archive
            self.archive.extend(new_agents)
            if len(self.archive) > 20:
                self.archive = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:20]
            
            print(f"\n📊 GENERATION {self.generation} SUMMARY:")
            print(f"   🎯 Best Score: {self.best_score}/4")
            print(f"   📊 Best CAGR: {self.best_agent.performance['cagr']*100:.1f}%")
            print(f"   📁 Archive: {len(self.archive)} agents")
    
    def save_winner(self, agent):
        """Save winning strategy"""
        try:
            strategy_name = f"enhanced_winner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                "criteria_score": agent.criteria_score,
                "mutations": agent.mutations,
                "components": agent.code_components,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(strategy_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"     💾 Saved: {strategy_name}")
            
        except Exception as e:
            print(f"     ⚠️ Save failed: {e}")
    
    def victory_celebration(self):
        """Victory celebration"""
        runtime = (time.time() - self.start_time) / 3600
        
        print("\n" + "🏆" * 80)
        print("🎉🎉🎉 ALL CRITERIA ACHIEVED! 🎉🎉🎉")
        print("🏆" * 80)
        
        agent = self.best_agent
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   📊 CAGR: {agent.performance['cagr']*100:.2f}%")
        print(f"   📈 Sharpe: {agent.performance['sharpe']:.3f}")
        print(f"   📉 Max DD: {agent.performance['drawdown']*100:.1f}%")
        print(f"   💰 Total Trades: {agent.performance['total_trades']}")
        print(f"   🎯 Score: {agent.criteria_score}/4")
        
        print(f"\n⏱️ Runtime: {runtime:.2f}h | 🧬 Generations: {self.generation}")
        print(f"🏆 Evolution: {' → '.join(agent.mutations)}")
        print(f"🤖 ENHANCED WORKING SYSTEM SUCCESS! 🤖")


def main():
    """Launch enhanced working DGM"""
    try:
        dgm = EnhancedWorkingDGM()
        winner = dgm.evolve_from_working_base()
        
        if winner:
            print(f"\n✅ SUCCESS! Score: {winner.criteria_score}/4")
        
    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()