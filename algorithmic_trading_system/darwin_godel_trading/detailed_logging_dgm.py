#!/usr/bin/env python3
"""
Detailed Logging Darwin Gödel Machine
Enhanced logging to show EVERYTHING happening under the hood
Real-time tracking of all 4 criteria with detailed breakdowns
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
    "sharpe": 1.0,          # >1.0 (with 5% risk-free rate)
    "max_drawdown": 0.20,   # <20%
    "avg_profit": 0.0075    # >0.75%
}

class DetailedLoggingAgent:
    """Trading agent with comprehensive logging"""
    
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
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_return": 0.0
        }
        # Start from PROVEN working configuration
        self.code_components = {
            "asset": "QQQ",
            "leverage": 10.0,
            "sma_fast": 10,
            "sma_slow": 30,
            "has_rsi": False,
            "has_macd": False,
            "trade_frequency": 1,  # Daily trading
            "position_size": 2.0,   # Aggressive position
            "rsi_threshold": 70
        }
        self.is_valid = False
        self.criteria_score = 0.0
        self.criteria_details = {}
    
    def log_detailed(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps and levels"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{level}] {message}")
    
    def generate_code(self) -> str:
        """Generate code with detailed logging"""
        self.log_detailed(f"🔧 GENERATING CODE for {self.agent_id}", "CODE")
        
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
            
            self.log_detailed(f"   📊 Config: {asset} {leverage}x leverage, {position_size}x position", "CODE")
            self.log_detailed(f"   📈 Signals: SMA({sma_fast},{sma_slow}), RSI={has_rsi}, MACD={has_macd}", "CODE")
            self.log_detailed(f"   ⏰ Frequency: Every {trade_freq} day(s)", "CODE")
            
            # Build indicators
            indicators_lines = [
                f'        self.sma_fast = self.sma("{asset}", {sma_fast})',
                f'        self.sma_slow = self.sma("{asset}", {sma_slow})'
            ]
            
            if has_rsi:
                indicators_lines.append(f'        self.rsi = self.rsi("{asset}", 14)')
                self.log_detailed(f"   🔍 Added RSI(14) with threshold < {rsi_threshold}", "CODE")
            
            if has_macd:
                indicators_lines.append(f'        self.macd = self.macd("{asset}", 12, 26, 9)')
                self.log_detailed(f"   📊 Added MACD(12,26,9)", "CODE")
            
            indicators = '\n'.join(indicators_lines)
            
            # Build conditions
            conditions = ["self.sma_fast.current.value > self.sma_slow.current.value"]
            
            if has_rsi:
                conditions.append(f"self.rsi.is_ready and self.rsi.current.value < {rsi_threshold}")
            
            if has_macd:
                conditions.append("self.macd.is_ready and self.macd.current.value > 0")
            
            if len(conditions) == 1:
                full_condition = conditions[0]
                self.log_detailed(f"   ✅ Trading Logic: Simple SMA crossover", "CODE")
            else:
                condition_str = " and \\\n            ".join(conditions)
                full_condition = f"({condition_str})"
                self.log_detailed(f"   ✅ Trading Logic: SMA + {len(conditions)-1} additional filters", "CODE")
            
            # Generate code
            code = f'''from AlgorithmImports import *

class DetailedLoggingStrategy(QCAlgorithm):
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
            self.log_detailed(f"   ✅ Code compilation successful - {len(code)} characters", "CODE")
            return code
            
        except Exception as e:
            self.log_detailed(f"   ❌ Code generation FAILED: {e}", "ERROR")
            self.is_valid = False
            return ""
    
    def apply_mutation(self, mutation_type: str) -> bool:
        """Apply mutation with detailed logging"""
        self.log_detailed(f"🧬 APPLYING MUTATION: {mutation_type}", "MUTATION")
        
        try:
            original = self.code_components.copy()
            self.log_detailed(f"   📋 Original config: {original}", "MUTATION")
            
            # Asset mutations
            if mutation_type == "try_spy":
                self.code_components["asset"] = "SPY"
                self.log_detailed(f"   🔄 Switched asset: QQQ → SPY", "MUTATION")
            elif mutation_type == "try_tqqq":
                self.code_components["asset"] = "TQQQ"
                self.log_detailed(f"   🔄 Switched asset: {original['asset']} → TQQQ (3x leveraged)", "MUTATION")
                
            # Leverage optimizations
            elif mutation_type == "reduce_leverage_for_sharpe":
                old_lev = self.code_components["leverage"]
                self.code_components["leverage"] = 5.0
                self.log_detailed(f"   📉 Reduced leverage for better Sharpe: {old_lev}x → 5.0x", "MUTATION")
            elif mutation_type == "moderate_leverage":
                old_lev = self.code_components["leverage"]
                self.code_components["leverage"] = 7.0
                self.log_detailed(f"   ⚖️ Moderate leverage: {old_lev}x → 7.0x", "MUTATION")
            elif mutation_type == "conservative_leverage":
                old_lev = self.code_components["leverage"]
                self.code_components["leverage"] = 3.0
                self.log_detailed(f"   🛡️ Conservative leverage: {old_lev}x → 3.0x", "MUTATION")
                
            # Position size for drawdown control
            elif mutation_type == "reduce_position_for_dd":
                old_pos = self.code_components["position_size"]
                self.code_components["position_size"] = 1.5
                self.log_detailed(f"   📉 Reduced position for DD control: {old_pos}x → 1.5x", "MUTATION")
            elif mutation_type == "conservative_position":
                old_pos = self.code_components["position_size"]
                self.code_components["position_size"] = 1.0
                self.log_detailed(f"   🛡️ Conservative position: {old_pos}x → 1.0x", "MUTATION")
                
            # Technical indicators
            elif mutation_type == "add_rsi_filter":
                self.code_components["has_rsi"] = True
                self.log_detailed(f"   📊 Added RSI filter (overbought protection)", "MUTATION")
            elif mutation_type == "add_macd_filter":
                self.code_components["has_macd"] = True
                self.log_detailed(f"   📈 Added MACD filter (momentum confirmation)", "MUTATION")
            elif mutation_type == "add_both_filters":
                self.code_components["has_rsi"] = True
                self.code_components["has_macd"] = True
                self.log_detailed(f"   🔬 Added BOTH RSI + MACD filters", "MUTATION")
                
            # Signal timing
            elif mutation_type == "faster_signals":
                old_fast, old_slow = self.code_components["sma_fast"], self.code_components["sma_slow"]
                self.code_components["sma_fast"] = 5
                self.code_components["sma_slow"] = 15
                self.log_detailed(f"   ⚡ Faster signals: SMA({old_fast},{old_slow}) → SMA(5,15)", "MUTATION")
            elif mutation_type == "slower_signals":
                old_fast, old_slow = self.code_components["sma_fast"], self.code_components["sma_slow"]
                self.code_components["sma_fast"] = 15
                self.code_components["sma_slow"] = 45
                self.log_detailed(f"   🐌 Slower signals: SMA({old_fast},{old_slow}) → SMA(15,45)", "MUTATION")
                
            # Trading frequency
            elif mutation_type == "weekly_trading":
                old_freq = self.code_components["trade_frequency"]
                self.code_components["trade_frequency"] = 7
                self.log_detailed(f"   📅 Weekly trading: Every {old_freq} day(s) → Every 7 days", "MUTATION")
                
            # Combination strategies
            elif mutation_type == "sharpe_optimized":
                self.log_detailed(f"   🎯 SHARPE OPTIMIZATION combo applied:", "MUTATION")
                self.code_components["leverage"] = 5.0
                self.code_components["position_size"] = 1.5
                self.code_components["has_rsi"] = True
                self.code_components["trade_frequency"] = 7
                self.log_detailed(f"      - Leverage: 5.0x (moderate)", "MUTATION")
                self.log_detailed(f"      - Position: 1.5x (conservative)", "MUTATION")
                self.log_detailed(f"      - RSI filter: ON", "MUTATION")
                self.log_detailed(f"      - Frequency: Weekly", "MUTATION")
                
            elif mutation_type == "drawdown_optimized":
                self.log_detailed(f"   🛡️ DRAWDOWN OPTIMIZATION combo applied:", "MUTATION")
                self.code_components["leverage"] = 6.0
                self.code_components["position_size"] = 1.2
                self.code_components["has_rsi"] = True
                self.code_components["has_macd"] = True
                self.log_detailed(f"      - Leverage: 6.0x (balanced)", "MUTATION")
                self.log_detailed(f"      - Position: 1.2x (conservative)", "MUTATION")
                self.log_detailed(f"      - Both RSI + MACD filters: ON", "MUTATION")
                
            else:
                self.log_detailed(f"   ❌ Unknown mutation type: {mutation_type}", "ERROR")
                return False
            
            self.log_detailed(f"   📋 New config: {self.code_components}", "MUTATION")
            
            # Test code generation
            test_code = self.generate_code()
            if test_code and self.is_valid:
                self.mutations.append(mutation_type)
                self.log_detailed(f"   ✅ Mutation SUCCESS - Code valid", "MUTATION")
                return True
            else:
                self.code_components = original
                self.log_detailed(f"   ❌ Mutation FAILED - Reverted to original", "ERROR")
                return False
                
        except Exception as e:
            self.code_components = original
            self.log_detailed(f"   ❌ Mutation ERROR: {e}", "ERROR")
            return False
    
    def run_backtest(self) -> Dict:
        """Run backtest with detailed logging and metrics parsing"""
        if not self.is_valid:
            self.log_detailed(f"❌ Cannot backtest - invalid strategy", "ERROR")
            return self.performance
        
        project_name = f"detailed_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        self.log_detailed(f"⚡ STARTING BACKTEST: {project_name}", "BACKTEST")
        
        try:
            # Create project
            os.makedirs(project_path, exist_ok=True)
            self.log_detailed(f"   📁 Created project directory: {project_path}", "BACKTEST")
            
            # Write code
            code = self.generate_code()
            with open(os.path.join(project_path, "main.py"), 'w') as f:
                f.write(code)
            self.log_detailed(f"   📝 Written main.py ({len(code)} chars)", "BACKTEST")
            
            # Write config
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "local-id": abs(hash(self.agent_id)) % 1000000
            }
            with open(os.path.join(project_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            self.log_detailed(f"   ⚙️ Written config.json (ID: {config['local-id']})", "BACKTEST")
            
            self.log_detailed(f"   🚀 Executing: {LEAN_CLI} backtest {project_name}", "BACKTEST")
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
            self.log_detailed(f"   ⏱️ Backtest completed in {backtest_time:.1f}s", "BACKTEST")
            
            if result.returncode == 0:
                self.log_detailed(f"   ✅ Backtest SUCCESS - parsing results...", "BACKTEST")
                
                # Show raw output sample
                output_lines = result.stdout.split('\n')
                stats_lines = [line for line in output_lines if "STATISTICS::" in line]
                self.log_detailed(f"   📊 Found {len(stats_lines)} statistics lines", "PARSING")
                
                metrics = self._parse_metrics_detailed(result.stdout)
                self.performance = metrics
                self.criteria_score = self._calculate_score_detailed(metrics)
                
                return metrics
            else:
                self.log_detailed(f"   ❌ Backtest FAILED (code {result.returncode})", "ERROR")
                if result.stderr:
                    self.log_detailed(f"   💬 Error: {result.stderr[:200]}", "ERROR")
                return self.performance
                
        except Exception as e:
            self.log_detailed(f"   💥 EXCEPTION: {str(e)[:150]}", "ERROR")
            return self.performance
        finally:
            # Cleanup
            try:
                if os.path.exists(project_path):
                    shutil.rmtree(project_path)
                    self.log_detailed(f"   🧹 Cleaned up project directory", "BACKTEST")
            except:
                pass
    
    def _parse_metrics_detailed(self, output: str) -> Dict:
        """Parse metrics with detailed logging"""
        self.log_detailed(f"📊 PARSING BACKTEST RESULTS", "PARSING")
        
        metrics = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "drawdown": 1.0,
            "avg_profit": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_return": 0.0
        }
        
        try:
            lines = output.split('\n')
            self.log_detailed(f"   📄 Processing {len(lines)} output lines", "PARSING")
            
            for line in lines:
                if "Compounding Annual Return" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["cagr"] = float(value) / 100
                        self.log_detailed(f"   📈 Found CAGR: {metrics['cagr']*100:.2f}%", "PARSING")
                    except Exception as e:
                        self.log_detailed(f"   ⚠️ CAGR parse error: {e}", "PARSING")
                        
                elif "Sharpe Ratio" in line and "Probabilistic" not in line:
                    try:
                        metrics["sharpe"] = float(line.split()[-1])
                        self.log_detailed(f"   📊 Found Sharpe: {metrics['sharpe']:.3f}", "PARSING")
                    except Exception as e:
                        self.log_detailed(f"   ⚠️ Sharpe parse error: {e}", "PARSING")
                        
                elif "Drawdown" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["drawdown"] = float(value) / 100
                        self.log_detailed(f"   📉 Found Drawdown: {metrics['drawdown']*100:.1f}%", "PARSING")
                    except Exception as e:
                        self.log_detailed(f"   ⚠️ Drawdown parse error: {e}", "PARSING")
                        
                elif "Total Orders" in line:
                    try:
                        metrics["total_trades"] = int(line.split()[-1])
                        self.log_detailed(f"   📋 Found Total Trades: {metrics['total_trades']}", "PARSING")
                    except Exception as e:
                        self.log_detailed(f"   ⚠️ Trades parse error: {e}", "PARSING")
                        
                elif "Average Win" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["avg_win"] = float(value) / 100
                        self.log_detailed(f"   📈 Found Avg Win: {metrics['avg_win']*100:.2f}%", "PARSING")
                    except Exception as e:
                        self.log_detailed(f"   ⚠️ Avg Win parse error: {e}", "PARSING")
                        
                elif "Average Loss" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["avg_loss"] = float(value) / 100
                        self.log_detailed(f"   📉 Found Avg Loss: {metrics['avg_loss']*100:.2f}%", "PARSING")
                    except Exception as e:
                        self.log_detailed(f"   ⚠️ Avg Loss parse error: {e}", "PARSING")
                        
                elif "Win Rate" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["win_rate"] = float(value) / 100
                        self.log_detailed(f"   🎯 Found Win Rate: {metrics['win_rate']*100:.1f}%", "PARSING")
                    except Exception as e:
                        self.log_detailed(f"   ⚠️ Win Rate parse error: {e}", "PARSING")
                        
                elif "Net Profit" in line:
                    try:
                        value = line.split()[-1].replace('%', '')
                        metrics["total_return"] = float(value) / 100
                        self.log_detailed(f"   💰 Found Total Return: {metrics['total_return']*100:.1f}%", "PARSING")
                    except Exception as e:
                        self.log_detailed(f"   ⚠️ Total Return parse error: {e}", "PARSING")
            
            # Calculate average profit per trade
            if metrics["total_trades"] > 0 and metrics["avg_win"] > 0:
                metrics["avg_profit"] = metrics["avg_win"] * metrics["win_rate"] * 0.5
                self.log_detailed(f"   💰 Calculated Avg Profit: {metrics['avg_profit']*100:.3f}%", "PARSING")
            
            self.log_detailed(f"   ✅ Parsing complete - {len([k for k,v in metrics.items() if v > 0])} metrics found", "PARSING")
            return metrics
            
        except Exception as e:
            self.log_detailed(f"   💥 PARSING ERROR: {e}", "ERROR")
            return metrics
    
    def _calculate_score_detailed(self, metrics: Dict) -> float:
        """Calculate criteria score with detailed logging"""
        self.log_detailed(f"🎯 CALCULATING CRITERIA SCORE", "CRITERIA")
        
        score = 0.0
        self.criteria_details = {}
        
        # CAGR Check
        cagr_met = metrics["cagr"] > TARGET_CRITERIA["cagr"]
        if cagr_met:
            score += 1
            self.log_detailed(f"   ✅ CAGR: {metrics['cagr']*100:.2f}% > {TARGET_CRITERIA['cagr']*100:.0f}% ✓", "CRITERIA")
        else:
            gap = (TARGET_CRITERIA["cagr"] - metrics["cagr"]) * 100
            self.log_detailed(f"   ❌ CAGR: {metrics['cagr']*100:.2f}% < {TARGET_CRITERIA['cagr']*100:.0f}% (need +{gap:.1f}%)", "CRITERIA")
        self.criteria_details["cagr"] = cagr_met
        
        # Sharpe Check
        sharpe_met = metrics["sharpe"] > TARGET_CRITERIA["sharpe"]
        if sharpe_met:
            score += 1
            self.log_detailed(f"   ✅ SHARPE: {metrics['sharpe']:.3f} > {TARGET_CRITERIA['sharpe']:.1f} ✓", "CRITERIA")
        else:
            gap = TARGET_CRITERIA["sharpe"] - metrics["sharpe"]
            self.log_detailed(f"   ❌ SHARPE: {metrics['sharpe']:.3f} < {TARGET_CRITERIA['sharpe']:.1f} (need +{gap:.3f})", "CRITERIA")
        self.criteria_details["sharpe"] = sharpe_met
        
        # Drawdown Check
        dd_met = metrics["drawdown"] < TARGET_CRITERIA["max_drawdown"]
        if dd_met:
            score += 1
            self.log_detailed(f"   ✅ DRAWDOWN: {metrics['drawdown']*100:.1f}% < {TARGET_CRITERIA['max_drawdown']*100:.0f}% ✓", "CRITERIA")
        else:
            excess = (metrics["drawdown"] - TARGET_CRITERIA["max_drawdown"]) * 100
            self.log_detailed(f"   ❌ DRAWDOWN: {metrics['drawdown']*100:.1f}% > {TARGET_CRITERIA['max_drawdown']*100:.0f}% (reduce by {excess:.1f}%)", "CRITERIA")
        self.criteria_details["drawdown"] = dd_met
        
        # Avg Profit Check
        profit_met = metrics["avg_profit"] > TARGET_CRITERIA["avg_profit"]
        if profit_met:
            score += 1
            self.log_detailed(f"   ✅ AVG PROFIT: {metrics['avg_profit']*100:.3f}% > {TARGET_CRITERIA['avg_profit']*100:.2f}% ✓", "CRITERIA")
        else:
            gap = (TARGET_CRITERIA["avg_profit"] - metrics["avg_profit"]) * 100
            self.log_detailed(f"   ❌ AVG PROFIT: {metrics['avg_profit']*100:.3f}% < {TARGET_CRITERIA['avg_profit']*100:.2f}% (need +{gap:.3f}%)", "CRITERIA")
        self.criteria_details["avg_profit"] = profit_met
        
        self.log_detailed(f"   🎯 FINAL SCORE: {score}/4 criteria met", "CRITERIA")
        return score


class DetailedLoggingDGM:
    """DGM with comprehensive under-the-hood logging"""
    
    def __init__(self):
        self.target_criteria = TARGET_CRITERIA
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_score = 0.0
        self.start_time = time.time()
        
        # Enhanced mutation set
        self.mutations = [
            "try_spy",
            "try_tqqq",
            "reduce_leverage_for_sharpe",
            "moderate_leverage",
            "conservative_leverage",
            "reduce_position_for_dd",
            "conservative_position",
            "add_rsi_filter",
            "add_macd_filter",
            "add_both_filters",
            "faster_signals",
            "slower_signals",
            "weekly_trading",
            "sharpe_optimized",
            "drawdown_optimized"
        ]
        
        print("🚀 DETAILED LOGGING DARWIN GÖDEL MACHINE")
        print("📊 COMPREHENSIVE UNDER-THE-HOOD MONITORING")
        print("🎯 ALL TARGET CRITERIA:")
        print(f"   📊 CAGR: >{self.target_criteria['cagr']*100:.0f}%")
        print(f"   📈 Sharpe: >{self.target_criteria['sharpe']:.1f}")
        print(f"   📉 Max DD: <{self.target_criteria['max_drawdown']*100:.0f}%")
        print(f"   💰 Avg Profit: >{self.target_criteria['avg_profit']*100:.2f}%")
        print("=" * 80)
    
    def log_system(self, message: str, level: str = "SYSTEM"):
        """System-level logging"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{level}] {message}")
    
    def evolve_with_detailed_logging(self):
        """Evolution with comprehensive logging"""
        
        # Start from working base
        self.log_system("🧬 INITIALIZING BASE AGENT", "INIT")
        base_agent = DetailedLoggingAgent("working_base", 0)
        self.log_system(f"   📊 Base config: {base_agent.code_components}", "INIT")
        
        base_agent.performance = base_agent.run_backtest()
        
        self.archive.append(base_agent)
        self.best_agent = base_agent
        self.best_score = base_agent.criteria_score
        
        self.log_system(f"🏁 BASE RESULTS:", "RESULTS")
        self.log_system(f"   🎯 Score: {self.best_score}/4 criteria met", "RESULTS")
        self.log_system(f"   📊 CAGR: {base_agent.performance['cagr']*100:.2f}%", "RESULTS")
        self.log_system(f"   📈 Sharpe: {base_agent.performance['sharpe']:.3f}", "RESULTS")
        self.log_system(f"   📉 DD: {base_agent.performance['drawdown']*100:.1f}%", "RESULTS")
        
        # Evolution loop
        while self.best_score < 4.0:
            self.generation += 1
            runtime = (time.time() - self.start_time) / 60
            
            self.log_system(f"🧬 STARTING GENERATION {self.generation}", "EVOLUTION")
            self.log_system(f"   ⏱️ Runtime: {runtime:.1f}min", "EVOLUTION")
            self.log_system(f"   🎯 Current best: {self.best_score}/4 criteria", "EVOLUTION")
            self.log_system(f"   📊 Current CAGR: {self.best_agent.performance['cagr']*100:.1f}%", "EVOLUTION")
            
            # Select parents
            parents = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:2]
            self.log_system(f"   👨‍👩‍👧‍👦 Selected {len(parents)} parents", "EVOLUTION")
            
            new_agents = []
            
            for p_idx, parent in enumerate(parents):
                self.log_system(f"\n👨‍👩‍👧‍👦 PARENT {p_idx}: {parent.agent_id}", "PARENT")
                self.log_system(f"   🎯 Score: {parent.criteria_score}/4", "PARENT")
                self.log_system(f"   📊 CAGR: {parent.performance['cagr']*100:.2f}%", "PARENT")
                self.log_system(f"   🧬 Mutations: {parent.mutations}", "PARENT")
                
                for mutation in self.mutations:
                    self.log_system(f"\n🧪 TESTING MUTATION: {mutation}", "MUTATION")
                    
                    child_id = f"gen{self.generation}_p{p_idx}_{mutation}"
                    child = DetailedLoggingAgent(child_id, self.generation)
                    child.code_components = parent.code_components.copy()
                    child.mutations = parent.mutations.copy()
                    
                    if child.apply_mutation(mutation):
                        child.performance = child.run_backtest()
                        
                        # Detailed comparison
                        if child.criteria_score > self.best_score:
                            old_score = self.best_score
                            old_cagr = self.best_agent.performance['cagr'] * 100
                            
                            self.best_score = child.criteria_score
                            self.best_agent = child
                            
                            self.log_system(f"🎉 NEW CHAMPION FOUND!", "CHAMPION")
                            self.log_system(f"   📈 Score: {old_score} → {self.best_score}/4", "CHAMPION")
                            self.log_system(f"   💰 CAGR: {old_cagr:.1f}% → {child.performance['cagr']*100:.1f}%", "CHAMPION")
                            self.log_system(f"   🧬 Evolution: {' → '.join(child.mutations)}", "CHAMPION")
                            
                            # Save winner
                            self.save_winner(child)
                            
                            # Check if all criteria met
                            if self.best_score >= 4.0:
                                self.victory_celebration()
                                return child
                        
                        if child.criteria_score > 0:
                            new_agents.append(child)
                    
                    # Brief pause for readability
                    time.sleep(0.1)
            
            # Update archive
            self.archive.extend(new_agents)
            if len(self.archive) > 20:
                self.archive = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:20]
            
            self.log_system(f"\n📊 GENERATION {self.generation} COMPLETE", "SUMMARY")
            self.log_system(f"   🎯 Best Score: {self.best_score}/4", "SUMMARY")
            self.log_system(f"   📊 Best CAGR: {self.best_agent.performance['cagr']*100:.1f}%", "SUMMARY")
            self.log_system(f"   📁 Archive: {len(self.archive)} agents", "SUMMARY")
            self.log_system(f"   🧬 Best evolution: {' → '.join(self.best_agent.mutations)}", "SUMMARY")
    
    def save_winner(self, agent):
        """Save winning strategy with logging"""
        try:
            strategy_name = f"detailed_winner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                "criteria_details": agent.criteria_details,
                "mutations": agent.mutations,
                "components": agent.code_components,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(strategy_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.log_system(f"💾 Winner saved: {strategy_name}", "SAVE")
            
        except Exception as e:
            self.log_system(f"⚠️ Save failed: {e}", "ERROR")
    
    def victory_celebration(self):
        """Victory with detailed breakdown"""
        runtime = (time.time() - self.start_time) / 3600
        
        print("\n" + "🏆" * 80)
        print("🎉🎉🎉 ALL CRITERIA ACHIEVED! 🎉🎉🎉")
        print("🏆" * 80)
        
        agent = self.best_agent
        print(f"\n🎯 FINAL DETAILED RESULTS:")
        print(f"   📊 CAGR: {agent.performance['cagr']*100:.2f}% ✅")
        print(f"   📈 Sharpe: {agent.performance['sharpe']:.3f} ✅")
        print(f"   📉 Max DD: {agent.performance['drawdown']*100:.1f}% ✅")
        print(f"   💰 Avg Profit: {agent.performance['avg_profit']*100:.3f}% ✅")
        print(f"   📊 Win Rate: {agent.performance['win_rate']*100:.1f}%")
        print(f"   🎯 Score: {agent.criteria_score}/4 ✅")
        
        print(f"\n⏱️ Runtime: {runtime:.2f}h | 🧬 Generations: {self.generation}")
        print(f"🏆 Final config: {agent.code_components}")
        print(f"🧬 Evolution path: {' → '.join(agent.mutations)}")
        print(f"\n🤖 DETAILED LOGGING SUCCESS! 🤖")


def main():
    """Launch detailed logging DGM"""
    try:
        dgm = DetailedLoggingDGM()
        winner = dgm.evolve_with_detailed_logging()
        
        if winner:
            print(f"\n✅ ALL CRITERIA MET! Score: {winner.criteria_score}/4")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evolution stopped by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()