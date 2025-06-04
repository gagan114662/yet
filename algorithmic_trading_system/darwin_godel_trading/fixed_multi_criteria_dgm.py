#!/usr/bin/env python3
"""
FIXED Multi-Criteria Darwin Gödel Machine
Targets: CAGR >25%, Sharpe >1.0, Max DD <20%, Avg Profit >0.75%
FIXED CODE GENERATION ISSUES
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

# TARGET CRITERIA
TARGET_CRITERIA = {
    "cagr": 0.25,           # >25%
    "sharpe": 1.0,          # >1.0
    "max_drawdown": 0.20,   # <20%
    "avg_profit": 0.0075    # >0.75%
}

class FixedMultiCriteriaTradingAgent:
    """FIXED Trading agent with working code generation"""
    
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
        self.code_components = {
            "asset": "SPY",
            "leverage": 2.0,
            "sma_fast": 10,
            "sma_slow": 30,
            "has_rsi": False,
            "has_macd": False,
            "trade_frequency": 7,
            "position_size": 1.0,
            "rsi_threshold": 70,
            "has_stop_loss": False,
            "stop_loss_pct": 0.05
        }
        self.is_valid = False
        self.criteria_score = 0.0
    
    def generate_code(self) -> str:
        """Generate WORKING strategy code with FIXED syntax"""
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
            has_stop_loss = self.code_components["has_stop_loss"]
            stop_loss_pct = self.code_components["stop_loss_pct"]
            
            # Build indicators (FIXED)
            indicators_lines = [
                f'        self.sma_fast = self.sma("{asset}", {sma_fast})',
                f'        self.sma_slow = self.sma("{asset}", {sma_slow})'
            ]
            
            if has_rsi:
                indicators_lines.append(f'        self.rsi = self.rsi("{asset}", 14)')
            
            if has_macd:
                indicators_lines.append(f'        self.macd = self.macd("{asset}", 12, 26, 9)')
            
            indicators = '\n'.join(indicators_lines)
            
            # Build trading conditions (FIXED)
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
            
            # Build stop loss logic (FIXED)
            stop_loss_logic = ""
            if has_stop_loss:
                stop_loss_logic = f"""
        # Stop loss risk management
        if self.portfolio.invested:
            current_price = self.securities[self.symbol].price
            holdings = self.portfolio[self.symbol]
            if holdings.quantity > 0:  # Long position
                entry_price = holdings.average_price
                loss_pct = (entry_price - current_price) / entry_price
                if loss_pct > {stop_loss_pct}:
                    self.liquidate("{asset}")
                    return"""
            
            # Generate complete WORKING code
            code = f'''from AlgorithmImports import *

class FixedMultiCriteriaStrategy(QCAlgorithm):
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
            return{stop_loss_logic}
            
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
            return code
            
        except Exception as e:
            print(f"    ❌ Code generation error: {e}")
            self.is_valid = False
            return ""
    
    def apply_mutation(self, mutation_type: str) -> bool:
        """Apply WORKING mutations"""
        try:
            original = self.code_components.copy()
            
            if mutation_type == "switch_to_qqq":
                self.code_components["asset"] = "QQQ"
            elif mutation_type == "switch_to_spy":
                self.code_components["asset"] = "SPY"
            elif mutation_type == "moderate_leverage":
                self.code_components["leverage"] = 3.0
            elif mutation_type == "balanced_leverage":
                self.code_components["leverage"] = 5.0
            elif mutation_type == "conservative_leverage":
                self.code_components["leverage"] = 2.0
            elif mutation_type == "add_rsi":
                self.code_components["has_rsi"] = True
                self.code_components["rsi_threshold"] = 70
            elif mutation_type == "add_rsi_conservative":
                self.code_components["has_rsi"] = True
                self.code_components["rsi_threshold"] = 60
            elif mutation_type == "add_macd":
                self.code_components["has_macd"] = True
            elif mutation_type == "add_stop_loss":
                self.code_components["has_stop_loss"] = True
                self.code_components["stop_loss_pct"] = 0.05
            elif mutation_type == "tighter_stop_loss":
                self.code_components["has_stop_loss"] = True
                self.code_components["stop_loss_pct"] = 0.03
            elif mutation_type == "conservative_position":
                self.code_components["position_size"] = 0.8
            elif mutation_type == "aggressive_position":
                self.code_components["position_size"] = 1.5
            elif mutation_type == "daily_trading":
                self.code_components["trade_frequency"] = 1
            elif mutation_type == "weekly_trading":
                self.code_components["trade_frequency"] = 7
            elif mutation_type == "faster_signals":
                self.code_components["sma_fast"] = 5
                self.code_components["sma_slow"] = 15
            elif mutation_type == "slower_signals":
                self.code_components["sma_fast"] = 15
                self.code_components["sma_slow"] = 45
            elif mutation_type == "sharpe_optimizer":
                # Optimize for better Sharpe ratio
                self.code_components["has_rsi"] = True
                self.code_components["has_macd"] = True
                self.code_components["rsi_threshold"] = 65
                self.code_components["trade_frequency"] = 7
                self.code_components["position_size"] = 0.9
            elif mutation_type == "drawdown_reducer":
                # Reduce max drawdown
                self.code_components["has_stop_loss"] = True
                self.code_components["stop_loss_pct"] = 0.04
                self.code_components["position_size"] = 0.7
                self.code_components["leverage"] = 3.0
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
        """Run backtest with ALL metrics extraction"""
        if not self.is_valid:
            return self.performance
        
        project_name = f"fixed_multi_{self.agent_id}_{int(time.time())}"
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
            print(f"       {self.code_components['asset']} {self.code_components['leverage']}x")
            
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
                metrics = self._parse_all_metrics(result.stdout)
                self.performance = metrics
                self.criteria_score = self._calculate_criteria_score(metrics)
                
                print(f"    ✅ RESULTS ({backtest_time:.1f}s):")
                print(f"       📊 CAGR: {metrics['cagr']*100:.2f}% {'✅' if metrics['cagr'] > TARGET_CRITERIA['cagr'] else '❌'}")
                print(f"       📈 Sharpe: {metrics['sharpe']:.3f} {'✅' if metrics['sharpe'] > TARGET_CRITERIA['sharpe'] else '❌'}")
                print(f"       📉 Max DD: {metrics['drawdown']*100:.1f}% {'✅' if metrics['drawdown'] < TARGET_CRITERIA['max_drawdown'] else '❌'}")
                print(f"       💰 Trades: {metrics['total_trades']}, Win Rate: {metrics['win_rate']*100:.1f}%")
                print(f"       🎯 SCORE: {self.criteria_score:.1f}/4 criteria met")
                
                return metrics
            else:
                print(f"    ❌ BACKTEST FAILED: {result.stderr[:150] if result.stderr else 'Unknown error'}")
                return self.performance
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:100]}")
            return self.performance
        finally:
            try:
                if os.path.exists(project_path):
                    shutil.rmtree(project_path)
            except:
                pass
    
    def _parse_all_metrics(self, output: str) -> Dict:
        """Parse ALL metrics from backtest output"""
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
                        
                        # Estimate average profit per trade
                        if metrics["total_trades"] > 0 and avg_win > 0:
                            metrics["avg_profit"] = avg_win * metrics["win_rate"] * 0.6  # Conservative estimate
                    except:
                        pass
            
            return metrics
            
        except Exception as e:
            return metrics
    
    def _calculate_criteria_score(self, metrics: Dict) -> float:
        """Calculate criteria score (0-4)"""
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


class FixedMultiCriteriaDGM:
    """FIXED Multi-Criteria DGM"""
    
    def __init__(self):
        self.target_criteria = TARGET_CRITERIA
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_score = 0.0
        self.start_time = time.time()
        
        # WORKING mutations only
        self.mutations = [
            "switch_to_qqq",
            "switch_to_spy",
            "moderate_leverage",
            "balanced_leverage", 
            "conservative_leverage",
            "add_rsi",
            "add_rsi_conservative",
            "add_macd",
            "add_stop_loss",
            "tighter_stop_loss",
            "conservative_position",
            "aggressive_position",
            "daily_trading",
            "weekly_trading",
            "faster_signals",
            "slower_signals",
            "sharpe_optimizer",
            "drawdown_reducer"
        ]
        
        print("🚀 FIXED MULTI-CRITERIA DARWIN GÖDEL MACHINE")
        print("🎯 TARGET CRITERIA (ALL MUST BE MET):")
        print(f"   📊 CAGR: >{self.target_criteria['cagr']*100:.0f}%")
        print(f"   📈 Sharpe: >{self.target_criteria['sharpe']:.1f}")
        print(f"   📉 Max DD: <{self.target_criteria['max_drawdown']*100:.0f}%")
        print(f"   💰 Avg Profit: >{self.target_criteria['avg_profit']*100:.2f}%")
        print("=" * 80)
    
    def evolve_until_all_criteria_met(self):
        """Evolution until ALL criteria satisfied"""
        
        # Start with WORKING base strategy
        print("\n🧬 Creating working base strategy...")
        base_agent = FixedMultiCriteriaTradingAgent("base", 0)
        base_agent.code_components = {
            "asset": "SPY",
            "leverage": 2.0,
            "sma_fast": 10,
            "sma_slow": 30,
            "has_rsi": False,
            "has_macd": False,
            "trade_frequency": 7,
            "position_size": 1.0,
            "rsi_threshold": 70,
            "has_stop_loss": False,
            "stop_loss_pct": 0.05
        }
        base_agent.generate_code()
        base_agent.performance = base_agent.run_backtest()
        
        self.archive.append(base_agent)
        self.best_agent = base_agent
        self.best_score = base_agent.criteria_score
        
        print(f"\n🏁 BASE PERFORMANCE:")
        print(f"   🎯 Criteria Score: {self.best_score}/4")
        
        # Evolution loop
        while self.best_score < 4.0:
            self.generation += 1
            runtime = (time.time() - self.start_time) / 60
            
            print(f"\n{'='*80}")
            print(f"🧬 GENERATION {self.generation} | ⏱️ {runtime:.1f}min")
            print(f"🎯 BEST SCORE: {self.best_score}/4 criteria met")
            print("=" * 80)
            
            # Select parents
            parents = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:2]
            new_agents = []
            
            for p_idx, parent in enumerate(parents):
                print(f"\n👨‍👩‍👧‍👦 PARENT {p_idx}: Score {parent.criteria_score}/4")
                
                for mutation in self.mutations:
                    child_id = f"gen{self.generation}_p{p_idx}_{mutation}"
                    child = FixedMultiCriteriaTradingAgent(child_id, self.generation)
                    child.code_components = parent.code_components.copy()
                    child.mutations = parent.mutations.copy()
                    
                    print(f"\n  🧪 MUTATION: {mutation.upper()}")
                    
                    if child.apply_mutation(mutation):
                        child.performance = child.run_backtest()
                        
                        # Check improvement
                        if child.criteria_score > self.best_score:
                            improvement = child.criteria_score - self.best_score
                            self.best_score = child.criteria_score
                            self.best_agent = child
                            
                            print(f"  🎉 NEW CHAMPION!")
                            print(f"     📈 SCORE: +{improvement:.1f} → {self.best_score}/4")
                            print(f"     🧬 PATH: {' → '.join(child.mutations)}")
                            
                            # Save winner
                            self.save_winner(child)
                            
                            # Check if all criteria met
                            if self.best_score >= 4.0:
                                self.victory_celebration()
                                return child
                        
                        if child.criteria_score > 0:
                            new_agents.append(child)
                    else:
                        print(f"  ❌ Mutation failed")
            
            # Update archive
            self.archive.extend(new_agents)
            if len(self.archive) > 30:
                self.archive = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:30]
            
            print(f"\n📊 GENERATION {self.generation} COMPLETE:")
            print(f"   🎯 Best Score: {self.best_score}/4")
            print(f"   📁 Archive: {len(self.archive)} agents")
    
    def save_winner(self, agent):
        """Save winning strategy"""
        try:
            strategy_name = f"fixed_multi_winner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            
            print(f"     💾 Winner saved: {strategy_name}")
            
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
        print(f"   💰 Avg Profit: {agent.performance['avg_profit']*100:.2f}%")
        print(f"   📊 Win Rate: {agent.performance['win_rate']*100:.1f}%")
        print(f"   🎯 Score: {agent.criteria_score}/4")
        
        print(f"\n⏱️ RUNTIME: {runtime:.2f} hours")
        print(f"🧬 GENERATIONS: {self.generation}")
        print(f"🏆 CONFIGURATION: {agent.code_components}")
        print(f"🧬 EVOLUTION: {' → '.join(agent.mutations)}")
        print(f"\n🤖 ALL TRADING CRITERIA ACHIEVED! 🤖")


def main():
    """Launch FIXED multi-criteria DGM"""
    try:
        dgm = FixedMultiCriteriaDGM()
        winner = dgm.evolve_until_all_criteria_met()
        
        if winner:
            print(f"\n✅ ALL CRITERIA MET! Score: {winner.criteria_score}/4")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evolution stopped by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()