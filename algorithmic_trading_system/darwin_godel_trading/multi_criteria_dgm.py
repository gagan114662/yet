#!/usr/bin/env python3
"""
Multi-Criteria Darwin Gödel Machine
Targets: CAGR >25%, Sharpe >1.0, Max DD <20%, Avg Profit >0.75%
"""

import os
import json
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional
import re

# Configuration
LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
LEAN_CLI = "/home/vandan/.local/bin/lean"

# TARGET CRITERIA - ALL MUST BE MET
TARGET_CRITERIA = {
    "cagr": 0.25,           # >25%
    "sharpe": 1.0,          # >1.0 (with 5% risk-free rate)
    "max_drawdown": 0.20,   # <20%
    "avg_profit": 0.0075    # >0.75%
}

class MultiCriteriaTradingAgent:
    """Trading agent optimizing for multiple criteria"""
    
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
            "has_bollinger": False,
            "trade_frequency": 7,
            "position_size": 1.0,
            "rsi_threshold": 70,
            "stop_loss": None,
            "take_profit": None,
            "volatility_filter": False
        }
        self.is_valid = False
        self.criteria_score = 0.0
    
    def generate_code(self) -> str:
        """Generate strategy code with risk management"""
        try:
            asset = self.code_components["asset"]
            leverage = self.code_components["leverage"]
            sma_fast = self.code_components["sma_fast"]
            sma_slow = self.code_components["sma_slow"]
            has_rsi = self.code_components["has_rsi"]
            has_macd = self.code_components["has_macd"]
            has_bollinger = self.code_components["has_bollinger"]
            trade_freq = self.code_components["trade_frequency"]
            position_size = self.code_components["position_size"]
            rsi_threshold = self.code_components["rsi_threshold"]
            stop_loss = self.code_components["stop_loss"]
            take_profit = self.code_components["take_profit"]
            volatility_filter = self.code_components["volatility_filter"]
            
            # Build indicators
            indicators_lines = [
                f'        self.sma_fast = self.sma("{asset}", {sma_fast})',
                f'        self.sma_slow = self.sma("{asset}", {sma_slow})'
            ]
            
            if has_rsi:
                indicators_lines.append(f'        self.rsi = self.rsi("{asset}", 14)')
            
            if has_macd:
                indicators_lines.append(f'        self.macd = self.macd("{asset}", 12, 26, 9)')
                
            if has_bollinger:
                indicators_lines.append(f'        self.bb = self.bb("{asset}", 20, 2)')
                
            if volatility_filter:
                indicators_lines.append(f'        self.atr = self.atr("{asset}", 14)')
            
            indicators = '\n'.join(indicators_lines)
            
            # Build trading conditions
            conditions = ["self.sma_fast.current.value > self.sma_slow.current.value"]
            
            if has_rsi:
                conditions.append(f"self.rsi.is_ready and self.rsi.current.value < {rsi_threshold}")
            
            if has_macd:
                conditions.append("self.macd.is_ready and self.macd.current.value > 0")
                
            if has_bollinger:
                conditions.append("self.bb.is_ready and self.securities[self.symbol].price < self.bb.upper_band.current.value")
                
            if volatility_filter:
                conditions.append("self.atr.is_ready and self.atr.current.value < self.securities[self.symbol].price * 0.02")
            
            if len(conditions) == 1:
                full_condition = conditions[0]
            else:
                condition_str = " and \\\n            ".join(conditions)
                full_condition = f"({condition_str})"
            
            # Build risk management
            risk_management = ""
            if stop_loss or take_profit:
                risk_management = f"""
        # Risk management
        if self.portfolio.invested:
            current_price = self.securities[self.symbol].price
            entry_price = self.portfolio[self.symbol].average_price
            
            profit_pct = (current_price - entry_price) / entry_price"""
                
                if stop_loss:
                    risk_management += f"""
            if profit_pct < -{stop_loss}:
                self.liquidate("{asset}")
                return"""
                    
                if take_profit:
                    risk_management += f"""
            if profit_pct > {take_profit}:
                self.liquidate("{asset}")
                return"""
            
            # Generate complete code
            code = f'''from AlgorithmImports import *

class MultiCriteriaStrategy(QCAlgorithm):
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
            {risk_management}
            
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
        """Apply mutation targeting multi-criteria optimization"""
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
            elif mutation_type == "add_rsi_conservative":
                self.code_components["has_rsi"] = True
                self.code_components["rsi_threshold"] = 60  # More conservative
            elif mutation_type == "add_macd":
                self.code_components["has_macd"] = True
            elif mutation_type == "add_bollinger":
                self.code_components["has_bollinger"] = True
            elif mutation_type == "add_stop_loss":
                self.code_components["stop_loss"] = 0.05  # 5% stop loss
            elif mutation_type == "add_take_profit":
                self.code_components["take_profit"] = 0.10  # 10% take profit
            elif mutation_type == "add_volatility_filter":
                self.code_components["volatility_filter"] = True
            elif mutation_type == "conservative_position":
                self.code_components["position_size"] = 0.8  # Reduce position
            elif mutation_type == "weekly_trading":
                self.code_components["trade_frequency"] = 7
            elif mutation_type == "bi_weekly_trading":
                self.code_components["trade_frequency"] = 14
            elif mutation_type == "faster_signals":
                self.code_components["sma_fast"] = 5
                self.code_components["sma_slow"] = 15
            elif mutation_type == "risk_managed_aggressive":
                # Balanced approach: higher leverage + risk management
                self.code_components["leverage"] = 6.0
                self.code_components["stop_loss"] = 0.08
                self.code_components["has_rsi"] = True
            elif mutation_type == "sharpe_optimizer":
                # Optimize for Sharpe ratio
                self.code_components["has_rsi"] = True
                self.code_components["has_macd"] = True
                self.code_components["rsi_threshold"] = 65
                self.code_components["trade_frequency"] = 14
            elif mutation_type == "drawdown_reducer":
                # Reduce drawdown
                self.code_components["stop_loss"] = 0.06
                self.code_components["position_size"] = 0.7
                self.code_components["volatility_filter"] = True
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
        """Run backtest and extract all metrics"""
        if not self.is_valid:
            return self.performance
        
        project_name = f"multi_crit_{self.agent_id}_{int(time.time())}"
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
            
            print(f"    ⚡ BACKTESTING: {self.agent_id}")
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
                
                print(f"    ✅ RESULT ({backtest_time:.1f}s):")
                print(f"       📊 CAGR: {metrics['cagr']*100:.2f}% {'✅' if metrics['cagr'] > TARGET_CRITERIA['cagr'] else '❌'}")
                print(f"       📈 Sharpe: {metrics['sharpe']:.3f} {'✅' if metrics['sharpe'] > TARGET_CRITERIA['sharpe'] else '❌'}")
                print(f"       📉 Max DD: {metrics['drawdown']*100:.1f}% {'✅' if metrics['drawdown'] < TARGET_CRITERIA['max_drawdown'] else '❌'}")
                print(f"       💰 Avg Profit: {metrics['avg_profit']*100:.2f}% {'✅' if metrics['avg_profit'] > TARGET_CRITERIA['avg_profit'] else '❌'}")
                print(f"       🎯 Score: {self.criteria_score:.2f}/4")
                
                return metrics
            else:
                print(f"    ❌ BACKTEST FAILED")
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
        """Parse all required metrics from backtest output"""
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
                        avg_win = 0
                elif "Win Rate" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["win_rate"] = float(value) / 100
                        
                        # Calculate average profit per trade
                        if metrics["total_trades"] > 0:
                            # Estimate average profit based on win rate and average win
                            metrics["avg_profit"] = avg_win * metrics["win_rate"] * 0.5  # Conservative estimate
                    except:
                        pass
            
            return metrics
            
        except Exception as e:
            return metrics
    
    def _calculate_criteria_score(self, metrics: Dict) -> float:
        """Calculate how many criteria are met (0-4)"""
        score = 0
        
        if metrics["cagr"] > TARGET_CRITERIA["cagr"]:
            score += 1
        if metrics["sharpe"] > TARGET_CRITERIA["sharpe"]:
            score += 1
        if metrics["drawdown"] < TARGET_CRITERIA["max_drawdown"]:
            score += 1
        if metrics["avg_profit"] > TARGET_CRITERIA["avg_profit"]:
            score += 1
            
        return score


class MultiCriteriaDGM:
    """Darwin Gödel Machine optimizing for ALL criteria"""
    
    def __init__(self):
        self.target_criteria = TARGET_CRITERIA
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_score = 0.0
        self.start_time = time.time()
        
        # Mutations targeting multiple criteria
        self.mutations = [
            "switch_to_qqq",
            "switch_to_spy", 
            "moderate_leverage",
            "balanced_leverage",
            "add_rsi_conservative",
            "add_macd",
            "add_bollinger",
            "add_stop_loss",
            "add_take_profit",
            "add_volatility_filter",
            "conservative_position",
            "weekly_trading",
            "bi_weekly_trading",
            "faster_signals",
            "risk_managed_aggressive",
            "sharpe_optimizer",
            "drawdown_reducer"
        ]
        
        print("🚀 MULTI-CRITERIA DARWIN GÖDEL MACHINE")
        print("🎯 TARGET CRITERIA (ALL MUST BE MET):")
        print(f"   📊 CAGR: >{self.target_criteria['cagr']*100:.0f}%")
        print(f"   📈 Sharpe: >{self.target_criteria['sharpe']:.1f}")
        print(f"   📉 Max DD: <{self.target_criteria['max_drawdown']*100:.0f}%")
        print(f"   💰 Avg Profit: >{self.target_criteria['avg_profit']*100:.2f}%")
        print("=" * 80)
    
    def evolve_until_all_criteria_met(self):
        """Evolution continues until ALL criteria are satisfied"""
        
        # Resume from best known strategy (QQQ daily trading)
        print("\n🧬 Starting from best known strategy...")
        best_known = MultiCriteriaTradingAgent("best_known", 0)
        best_known.code_components = {
            "asset": "QQQ",
            "leverage": 5.0,  # Reduced for better Sharpe/DD
            "sma_fast": 10,
            "sma_slow": 30,
            "has_rsi": True,  # Add RSI for better entries
            "has_macd": False,
            "has_bollinger": False,
            "trade_frequency": 7,  # Weekly for better risk mgmt
            "position_size": 0.8,  # Conservative sizing
            "rsi_threshold": 65,
            "stop_loss": 0.06,  # 6% stop loss
            "take_profit": None,
            "volatility_filter": False
        }
        best_known.generate_code()
        best_known.performance = best_known.run_backtest()
        
        self.archive.append(best_known)
        self.best_agent = best_known
        self.best_score = best_known.criteria_score
        
        print(f"\n🏁 STARTING POINT:")
        print(f"   🎯 Criteria Score: {self.best_score}/4")
        
        # Evolution loop
        while self.best_score < 4.0:  # All 4 criteria must be met
            self.generation += 1
            runtime = (time.time() - self.start_time) / 60
            
            print(f"\n{'='*80}")
            print(f"🧬 GENERATION {self.generation} | ⏱️ {runtime:.1f}min")
            print(f"🎯 BEST SCORE: {self.best_score}/4 criteria met")
            print("=" * 80)
            
            # Select top performers
            parents = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:2]
            new_agents = []
            
            for p_idx, parent in enumerate(parents):
                print(f"\n👨‍👩‍👧‍👦 PARENT {p_idx}: Score {parent.criteria_score}/4")
                
                for mutation in self.mutations:
                    child_id = f"gen{self.generation}_p{p_idx}_{mutation}"
                    child = MultiCriteriaTradingAgent(child_id, self.generation)
                    child.code_components = parent.code_components.copy()
                    child.mutations = parent.mutations.copy()
                    
                    print(f"\n  🧪 MUTATION: {mutation.upper()}")
                    
                    if child.apply_mutation(mutation):
                        child.performance = child.run_backtest()
                        
                        # Check for improvement
                        if child.criteria_score > self.best_score:
                            improvement = child.criteria_score - self.best_score
                            self.best_score = child.criteria_score
                            self.best_agent = child
                            
                            print(f"  🎉 NEW CHAMPION!")
                            print(f"     📈 SCORE IMPROVEMENT: +{improvement:.1f} → {self.best_score}/4")
                            print(f"     🧬 MUTATIONS: {' → '.join(child.mutations)}")
                            
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
        """Save winning multi-criteria strategy"""
        try:
            strategy_name = f"multicrit_winner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            
            print(f"     💾 Multi-criteria winner saved: {strategy_name}")
            
        except Exception as e:
            print(f"     ⚠️ Save failed: {e}")
    
    def victory_celebration(self):
        """Celebrate meeting ALL criteria"""
        runtime = (time.time() - self.start_time) / 3600
        
        print("\n" + "🏆" * 80)
        print("🎉🎉🎉 ALL CRITERIA ACHIEVED! 🎉🎉🎉")
        print("🏆" * 80)
        
        agent = self.best_agent
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   📊 CAGR: {agent.performance['cagr']*100:.2f}% ({'✅ >25%' if agent.performance['cagr'] > TARGET_CRITERIA['cagr'] else '❌'})")
        print(f"   📈 Sharpe: {agent.performance['sharpe']:.3f} ({'✅ >1.0' if agent.performance['sharpe'] > TARGET_CRITERIA['sharpe'] else '❌'})")
        print(f"   📉 Max DD: {agent.performance['drawdown']*100:.1f}% ({'✅ <20%' if agent.performance['drawdown'] < TARGET_CRITERIA['max_drawdown'] else '❌'})")
        print(f"   💰 Avg Profit: {agent.performance['avg_profit']*100:.2f}% ({'✅ >0.75%' if agent.performance['avg_profit'] > TARGET_CRITERIA['avg_profit'] else '❌'})")
        
        print(f"\n⏱️ RUNTIME: {runtime:.2f} hours")
        print(f"🧬 GENERATIONS: {self.generation}")
        print(f"\n🏆 WINNING CONFIGURATION:")
        print(f"   Asset: {agent.code_components['asset']}")
        print(f"   Leverage: {agent.code_components['leverage']}x")
        print(f"   Risk Management: Stop Loss={agent.code_components['stop_loss']}, Take Profit={agent.code_components['take_profit']}")
        print(f"   Indicators: RSI={agent.code_components['has_rsi']}, MACD={agent.code_components['has_macd']}, BB={agent.code_components['has_bollinger']}")
        print(f"   Evolution: {' → '.join(agent.mutations)}")
        print(f"\n🤖 AI ACHIEVED ALL TRADING CRITERIA! 🤖")


def main():
    """Launch multi-criteria DGM"""
    try:
        dgm = MultiCriteriaDGM()
        winner = dgm.evolve_until_all_criteria_met()
        
        if winner:
            print(f"\n✅ ALL CRITERIA MET!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evolution stopped by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()