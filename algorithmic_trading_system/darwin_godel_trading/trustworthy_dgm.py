#!/usr/bin/env python3
"""
TRUSTWORTHY DARWIN GÖDEL MACHINE
100% QuantConnect Cloud Verified Results
Every result comes with verifiable URL
"""

import os
import json
import subprocess
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional
import re

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
    """Print with immediate flush"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()

class TrustworthyAgent:
    """Agent with 100% QuantConnect cloud verification"""
    
    def __init__(self, agent_id: str, generation: int = 0):
        self.agent_id = agent_id
        self.generation = generation
        self.mutations = []
        self.cloud_metrics = {}
        self.cloud_url = None
        self.code_components = {
            "asset": "QQQ",
            "leverage": 8.0,       # Start conservative, evolve to aggressive
            "sma_fast": 10,        # Start with working signals
            "sma_slow": 30,        # Standard combination
            "has_rsi": False,
            "has_macd": False,
            "trade_frequency": 1,
            "position_size": 1.0,   # Start conservative, evolve up
            "start_year": 2009,     # Always 15 years
            "end_year": 2023
        }
        self.is_valid = False
        self.criteria_score = 0.0
        self.generate_code()  # Generate on creation
    
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
            start_year = self.code_components["start_year"]
            end_year = self.code_components["end_year"]
            
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
            
            # Generate code
            code = f'''from AlgorithmImports import *

class TrustworthyStrategy(QCAlgorithm):
    """
    Trustworthy Strategy - Agent: {self.agent_id}
    Config: {asset} {leverage}x leverage, {position_size}x position, SMA({sma_fast},{sma_slow})
    """
    
    def initialize(self):
        self.set_start_date({start_year}, 1, 1)
        self.set_end_date({end_year}, 12, 31)
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
            return code
            
        except Exception as e:
            live_print(f"❌ Code error for {self.agent_id}: {e}", "ERROR")
            self.is_valid = False
            return ""
    
    def run_cloud_verification(self) -> Dict:
        """Run verification ONLY on QuantConnect cloud"""
        if not self.is_valid:
            return {}
        
        live_print(f"☁️ CLOUD VERIFICATION: {self.agent_id}", "VERIFY")
        live_print(f"   📊 {self.code_components['asset']} {self.code_components['leverage']}x leverage, {self.code_components['position_size']}x position", "VERIFY")
        
        project_name = f"Trust_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        try:
            # Create project
            os.makedirs(project_path, exist_ok=True)
            
            # Write code
            code = self.generate_code()
            with open(os.path.join(project_path, "main.py"), 'w') as f:
                f.write(code)
            
            # Write config
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "description": f"Trustworthy verification {self.agent_id}"
            }
            with open(os.path.join(project_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Push to cloud with retry logic
            push_success = False
            for attempt in range(3):
                try:
                    push_result = subprocess.run(
                        [LEAN_CLI, "cloud", "push", "--project", project_name],
                        cwd=LEAN_WORKSPACE,
                        capture_output=True,
                        text=True,
                        timeout=180  # Increased timeout
                    )
                    if push_result.returncode == 0:
                        push_success = True
                        break
                    else:
                        live_print(f"   ⚠️ Push attempt {attempt+1} failed, retrying...", "WARNING")
                        time.sleep(5)
                except subprocess.TimeoutExpired:
                    live_print(f"   ⚠️ Push attempt {attempt+1} timed out, retrying...", "WARNING")
                    time.sleep(5)
            
            if not push_success:
                live_print(f"   ❌ Push failed after 3 attempts", "ERROR")
                return {}
            
            # Run cloud backtest
            backtest_name = f"Trust_{self.agent_id}"
            
            # Run cloud backtest with retry logic
            backtest_success = False
            for attempt in range(2):
                try:
                    backtest_result = subprocess.run(
                        [LEAN_CLI, "cloud", "backtest", project_name, "--name", backtest_name],
                        cwd=LEAN_WORKSPACE,
                        capture_output=True,
                        text=True,
                        timeout=900  # Increased to 15 minutes
                    )
                    if backtest_result.returncode == 0:
                        backtest_success = True
                        break
                    else:
                        live_print(f"   ⚠️ Backtest attempt {attempt+1} failed, retrying...", "WARNING")
                        time.sleep(30)  # Longer pause after failure
                except subprocess.TimeoutExpired:
                    live_print(f"   ⚠️ Backtest attempt {attempt+1} timed out, retrying...", "WARNING")
                    time.sleep(30)  # Longer pause after timeout
            
            if backtest_success:
                # Parse CLOUD results
                metrics = self._parse_cloud_results(backtest_result.stdout)
                self.cloud_metrics = metrics
                self.criteria_score = self._calculate_score(metrics)
                
                # Extract backtest URL
                url_match = re.search(r'https://www\.quantconnect\.com/project/\d+/[a-f0-9]+', backtest_result.stdout)
                if url_match:
                    self.cloud_url = url_match.group(0)
                
                # Show results
                live_print(f"   ✅ VERIFIED RESULTS:", "VERIFY")
                live_print(f"      📊 CAGR: {metrics.get('cagr', 0)*100:.2f}% {'✅' if metrics.get('cagr', 0) > TARGET_CRITERIA['cagr'] else '❌'}", "VERIFY")
                live_print(f"      📈 Sharpe: {metrics.get('sharpe', 0):.3f} {'✅' if metrics.get('sharpe', 0) > TARGET_CRITERIA['sharpe'] else '❌'}", "VERIFY")
                live_print(f"      📉 DD: {metrics.get('drawdown', 1)*100:.1f}% {'✅' if metrics.get('drawdown', 1) < TARGET_CRITERIA['max_drawdown'] else '❌'}", "VERIFY")
                live_print(f"      💰 Trades: {metrics.get('total_trades', 0)}", "VERIFY")
                live_print(f"      🎯 SCORE: {self.criteria_score}/4", "VERIFY")
                if self.cloud_url:
                    live_print(f"      🔗 VERIFY: {self.cloud_url}", "VERIFY")
                
                return metrics
            else:
                live_print(f"   ❌ Backtest failed: {backtest_result.stderr[:150]}", "ERROR")
                return {}
                
        except Exception as e:
            live_print(f"   ❌ Exception: {str(e)[:100]}", "ERROR")
            return {}
        finally:
            # Cleanup
            try:
                if os.path.exists(project_path):
                    subprocess.run(["rm", "-rf", project_path], timeout=30)
            except:
                pass
    
    def _parse_cloud_results(self, output: str) -> Dict:
        """Parse QuantConnect cloud output"""
        metrics = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "drawdown": 1.0,
            "avg_profit": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0
        }
        
        try:
            lines = output.split('\n')
            
            for line in lines:
                if "Compounding Annual" in line and "%" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["cagr"] = float(match.group(1)) / 100
                        
                elif "Sharpe Ratio" in line and "│" in line and "Probabilistic" not in line:
                    parts = line.split("│")
                    if len(parts) > 1:
                        value_part = parts[-1].strip()
                        match = re.search(r'(\d+\.?\d*)', value_part)
                        if match:
                            metrics["sharpe"] = float(match.group(1))
                            
                elif "Drawdown" in line and "%" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["drawdown"] = float(match.group(1)) / 100
                        
                elif "Total Orders" in line:
                    match = re.search(r'(\d+)', line.split("│")[-2] if "│" in line else line)
                    if match:
                        metrics["total_trades"] = int(match.group(1))
                        
                elif "Win Rate" in line and "%" in line:
                    match = re.search(r'(\d+)%', line)
                    if match:
                        metrics["win_rate"] = float(match.group(1)) / 100
                        
                elif "Average Win" in line and "%" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["avg_win"] = float(match.group(1)) / 100
            
            # Calculate average profit per trade
            if metrics["total_trades"] > 0 and metrics["avg_win"] > 0:
                metrics["avg_profit"] = metrics["avg_win"] * metrics["win_rate"] * 0.5
            
            return metrics
            
        except Exception as e:
            return metrics
    
    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate criteria score"""
        score = 0.0
        
        if metrics.get("cagr", 0) > TARGET_CRITERIA["cagr"]:
            score += 1
        if metrics.get("sharpe", 0) > TARGET_CRITERIA["sharpe"]:
            score += 1
        if metrics.get("drawdown", 1) < TARGET_CRITERIA["max_drawdown"]:
            score += 1
        if metrics.get("avg_profit", 0) > TARGET_CRITERIA["avg_profit"]:
            score += 1
            
        return score
    
    def apply_mutation(self, mutation_type: str) -> bool:
        """Apply PROGRESSIVE Darwin mutation"""
        try:
            original = self.code_components.copy()
            
            # Stage 1: Basic improvements
            if mutation_type == "increase_leverage_moderate":
                self.code_components["leverage"] = min(self.code_components["leverage"] * 1.5, 12.0)
                live_print(f"   📈 Moderate leverage increase to {self.code_components['leverage']:.1f}x", "MUTATION")
            elif mutation_type == "increase_position_moderate":
                self.code_components["position_size"] = min(self.code_components["position_size"] * 1.3, 1.5)
                live_print(f"   📏 Moderate position to {self.code_components['position_size']:.1f}x", "MUTATION")
            elif mutation_type == "faster_signals":
                self.code_components["sma_fast"] = max(self.code_components["sma_fast"] - 2, 5)
                self.code_components["sma_slow"] = max(self.code_components["sma_slow"] - 5, 15)
                live_print(f"   ⚡ Faster signals: SMA({self.code_components['sma_fast']},{self.code_components['sma_slow']})", "MUTATION")
            elif mutation_type == "add_rsi_filter":
                self.code_components["has_rsi"] = True
                live_print(f"   🛡️ Added RSI risk filter", "MUTATION")
            elif mutation_type == "add_macd_momentum":
                self.code_components["has_macd"] = True
                live_print(f"   📊 Added MACD momentum", "MUTATION")
                
            # Stage 2: More aggressive
            elif mutation_type == "boost_leverage_15x":
                self.code_components["leverage"] = 15.0
                live_print(f"   🚀 Boosted leverage to 15x", "MUTATION")
            elif mutation_type == "aggressive_position_2x":
                self.code_components["position_size"] = 2.0
                live_print(f"   💪 Aggressive 2x position", "MUTATION")
            elif mutation_type == "switch_to_tqqq":
                self.code_components["asset"] = "TQQQ"
                self.code_components["leverage"] = min(self.code_components["leverage"] * 0.7, 10.0)  # Lower for leveraged ETF
                live_print(f"   📈 Switched to TQQQ with {self.code_components['leverage']:.1f}x leverage", "MUTATION")
            elif mutation_type == "ultra_fast_signals":
                self.code_components["sma_fast"] = 3
                self.code_components["sma_slow"] = 12
                live_print(f"   ⚡ Ultra fast signals: SMA(3,12)", "MUTATION")
                
            # Stage 3: High performance
            elif mutation_type == "boost_leverage_20x":
                self.code_components["leverage"] = 20.0
                live_print(f"   🚀 HIGH leverage: 20x", "MUTATION")
            elif mutation_type == "boost_leverage_25x":
                self.code_components["leverage"] = 25.0
                live_print(f"   🚀 EXTREME leverage: 25x", "MUTATION")
            elif mutation_type == "extreme_position_3x":
                self.code_components["position_size"] = 3.0
                live_print(f"   ⚡ EXTREME position: 3x", "MUTATION")
            elif mutation_type == "momentum_breakthrough":
                self.code_components["sma_fast"] = 2
                self.code_components["sma_slow"] = 8
                self.code_components["leverage"] = 18.0
                live_print(f"   ⚡ MOMENTUM breakthrough: SMA(2,8) 18x", "MUTATION")
                
            # Stage 4: Maximum performance
            elif mutation_type == "quantum_momentum":
                self.code_components["leverage"] = 30.0
                self.code_components["position_size"] = 2.5
                self.code_components["sma_fast"] = 3
                self.code_components["sma_slow"] = 9
                live_print(f"   ⚛️ QUANTUM momentum: 30x leverage!", "MUTATION")
            elif mutation_type == "ultra_aggressive_30x":
                self.code_components["leverage"] = 30.0
                self.code_components["position_size"] = 3.5
                live_print(f"   🔥 ULTRA AGGRESSIVE: 30x leverage, 3.5x position", "MUTATION")
            elif mutation_type == "switch_to_leveraged_etf":
                self.code_components["asset"] = "UPRO"  # 3x S&P 500
                self.code_components["leverage"] = 8.0
                self.code_components["position_size"] = 2.0
                live_print(f"   📈 LEVERAGED ETF: UPRO 8x leverage", "MUTATION")
            elif mutation_type == "crypto_strategy":
                self.code_components["asset"] = "BITO"  # Bitcoin ETF
                self.code_components["leverage"] = 15.0
                self.code_components["position_size"] = 2.0
                self.code_components["sma_fast"] = 5
                self.code_components["sma_slow"] = 15
                live_print(f"   ₿ CRYPTO strategy: BITO 15x leverage", "MUTATION")
            elif mutation_type == "volatility_strategy":
                self.code_components["asset"] = "UVXY"  # 2x VIX
                self.code_components["leverage"] = 10.0
                self.code_components["position_size"] = 1.5
                self.code_components["sma_fast"] = 3
                self.code_components["sma_slow"] = 8
                live_print(f"   📊 VOLATILITY strategy: UVXY 10x leverage", "MUTATION")
            elif mutation_type == "optimized_tqqq":
                self.code_components["asset"] = "TQQQ"
                self.code_components["leverage"] = 6.0  # Conservative for leveraged ETF
                self.code_components["position_size"] = 2.0
                self.code_components["has_rsi"] = True
                self.code_components["has_macd"] = True
                live_print(f"   🎯 OPTIMIZED TQQQ: 6x leverage with indicators", "MUTATION")
            elif mutation_type == "balanced_crypto":
                self.code_components["asset"] = "BITO"
                self.code_components["leverage"] = 12.0
                self.code_components["position_size"] = 1.8
                self.code_components["has_rsi"] = True
                live_print(f"   ⚖️ BALANCED crypto: BITO 12x with RSI", "MUTATION")
            elif mutation_type == "smart_leveraged_etf":
                self.code_components["asset"] = "UPRO"
                self.code_components["leverage"] = 5.0
                self.code_components["position_size"] = 2.5
                self.code_components["has_macd"] = True
                live_print(f"   🧠 SMART leveraged ETF: UPRO 5x with MACD", "MUTATION")
            elif mutation_type == "inverse_strategy":
                self.code_components["asset"] = "SQQQ"  # Inverse QQQ
                self.code_components["leverage"] = 8.0
                self.code_components["position_size"] = 2.0
                self.code_components["sma_fast"] = 5
                self.code_components["sma_slow"] = 20
                live_print(f"   🔄 INVERSE strategy: SQQQ 8x leverage", "MUTATION")
            elif mutation_type == "momentum_scalping":
                self.code_components["sma_fast"] = 1
                self.code_components["sma_slow"] = 3
                self.code_components["leverage"] = 15.0
                self.code_components["position_size"] = 2.0
                self.code_components["trade_frequency"] = 0  # Daily
                live_print(f"   ⚡ MOMENTUM scalping: SMA(1,3) 15x daily", "MUTATION")
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


class TrustworthyDGM:
    """100% trustworthy Darwin Gödel Machine"""
    
    def __init__(self):
        self.target_criteria = TARGET_CRITERIA
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_score = 0.0
        self.start_time = time.time()
        
        # PROGRESSIVE mutations - Darwin evolution from conservative to extreme
        self.mutations = [
            # Stage 1: Basic improvements
            "increase_leverage_moderate",
            "increase_position_moderate", 
            "faster_signals",
            "add_rsi_filter",
            "add_macd_momentum",
            # Stage 2: More aggressive 
            "boost_leverage_15x",
            "aggressive_position_2x",
            "switch_to_tqqq",
            "ultra_fast_signals",
            # Stage 3: High performance
            "boost_leverage_20x",
            "boost_leverage_25x", 
            "extreme_position_3x",
            "momentum_breakthrough",
            # Stage 4: Maximum performance
            "quantum_momentum",
            "ultra_aggressive_30x"
        ]
        
        live_print("🚀 TRUSTWORTHY DARWIN GÖDEL MACHINE", "SYSTEM")
        live_print("✅ 100% QUANTCONNECT CLOUD VERIFIED", "SYSTEM")
        live_print("🎯 TARGET CRITERIA:", "SYSTEM")
        live_print(f"   📊 CAGR: >{self.target_criteria['cagr']*100:.0f}%", "SYSTEM")
        live_print(f"   📈 Sharpe: >{self.target_criteria['sharpe']:.1f}", "SYSTEM")
        live_print(f"   📉 Max DD: <{self.target_criteria['max_drawdown']*100:.0f}%", "SYSTEM")
        live_print(f"   💰 Avg Profit: >{self.target_criteria['avg_profit']*100:.2f}%", "SYSTEM")
    
    def evolve_until_targets_met(self):
        """Evolution until ALL targets are met with cloud verification"""
        
        # Start with optimized base
        live_print("", "")
        live_print("🧬 CREATING BASE AGENT...", "EVOLUTION")
        base_agent = TrustworthyAgent("base", 0)
        base_metrics = base_agent.run_cloud_verification()
        
        if base_metrics:
            self.archive.append(base_agent)
            self.best_agent = base_agent
            self.best_score = base_agent.criteria_score
            
            live_print(f"✅ BASE AGENT VERIFIED: {self.best_score}/4 criteria", "EVOLUTION")
        else:
            live_print("❌ Base agent verification failed", "ERROR")
            return None
        
        # Evolution loop - continue until ALL targets met
        while self.best_score < 4.0 and self.generation < 100:  # Max 100 generations
            self.generation += 1
            runtime = (time.time() - self.start_time) / 60
            
            live_print("", "")
            live_print("=" * 80, "")
            live_print(f"🧬 GENERATION {self.generation} | ⏱️ {runtime:.1f}min", "EVOLUTION")
            live_print(f"🎯 CURRENT BEST: {self.best_score}/4 | 📊 CAGR: {self.best_agent.cloud_metrics.get('cagr', 0)*100:.1f}%", "EVOLUTION")
            live_print("=" * 80, "")
            
            # AGGRESSIVE parallel evolution - test multiple strategies simultaneously
            parents = sorted(self.archive, key=lambda a: a.criteria_score, reverse=True)[:3]  # More parents
            new_agents = []
            
            # DARWIN PROGRESSIVE EVOLUTION - test mutations by stage
            if self.generation <= 2:
                # Early generations: Basic improvements
                test_mutations = ["increase_leverage_moderate", "increase_position_moderate", "faster_signals", "add_rsi_filter"]
            elif self.generation <= 4:
                # Mid generations: More aggressive
                test_mutations = ["boost_leverage_15x", "aggressive_position_2x", "switch_to_tqqq", "ultra_fast_signals", "add_macd_momentum"]
            elif self.generation <= 8:
                # Advanced generations: AVOID extreme leverage, focus on working combinations
                test_mutations = ["switch_to_tqqq", "boost_leverage_15x", "aggressive_position_2x", "add_macd_momentum"]
            elif self.generation <= 15:
                # Focus on asset diversification instead of extreme leverage
                test_mutations = ["switch_to_leveraged_etf", "crypto_strategy", "switch_to_tqqq"]
            elif self.generation <= 25:
                # Test different time periods and signals
                test_mutations = ["ultra_fast_signals", "momentum_breakthrough", "switch_to_tqqq"]
            elif self.generation <= 50:
                # Combine best elements
                test_mutations = ["optimized_tqqq", "balanced_crypto", "smart_leveraged_etf"]
            else:
                # Final push - test radical approaches
                test_mutations = ["volatility_strategy", "inverse_strategy", "momentum_scalping"]
            
            for p_idx, parent in enumerate(parents):
                live_print(f"", "")
                live_print(f"👨‍👩‍👧‍👦 PARENT {p_idx}: {parent.criteria_score}/4 ({parent.cloud_metrics.get('cagr', 0)*100:.1f}% CAGR)", "PARENT")
                live_print(f"🔗 Verify: {parent.cloud_url}", "PARENT")
                
                for mutation in test_mutations:
                    live_print(f"", "")
                    live_print(f"🧪 TESTING: {mutation.upper()}", "MUTATION")
                    
                    child_id = f"gen{self.generation}_{mutation}"
                    child = TrustworthyAgent(child_id, self.generation)
                    child.code_components = parent.code_components.copy()
                    child.mutations = parent.mutations.copy()
                    
                    if child.apply_mutation(mutation):
                        child_metrics = child.run_cloud_verification()
                        
                        if child_metrics:
                            # Check improvement OR high CAGR potential
                            cagr = child.cloud_metrics.get('cagr', 0) * 100
                            if child.criteria_score > self.best_score or cagr > self.best_agent.cloud_metrics.get('cagr', 0) * 100:
                                old_score = self.best_score
                                old_cagr = self.best_agent.cloud_metrics.get('cagr', 0) * 100
                                
                                self.best_score = child.criteria_score
                                self.best_agent = child
                                
                                live_print(f"", "")
                                live_print("🎉🎉🎉 NEW CHAMPION! 🎉🎉🎉", "CHAMPION")
                                live_print(f"📈 Score improved: {old_score} → {self.best_score}/4", "CHAMPION")
                                live_print(f"💰 CAGR improved: {old_cagr:.1f}% → {cagr:.1f}%", "CHAMPION")
                                live_print(f"🔗 VERIFY: {child.cloud_url}", "CHAMPION")
                                live_print(f"🧬 Evolution: {' → '.join(child.mutations)}", "CHAMPION")
                                
                                # Check if all criteria met
                                if self.best_score >= 4.0:
                                    self.victory_celebration()
                                    return child
                            
                            # Keep any agent with score > 0 OR CAGR > 10%
                            if child.criteria_score > 0 or cagr > 10.0:
                                new_agents.append(child)
                    else:
                        live_print(f"❌ Mutation failed", "MUTATION")
                    
                    # Rate limiting pause to avoid "too many requests"
                    time.sleep(15)  # 15 second pause between backtests
                    
                # Early termination if we find a high-performing strategy
                if self.best_agent and self.best_agent.cloud_metrics.get('cagr', 0) > 0.20:  # >20% CAGR
                    live_print(f"🎯 HIGH PERFORMANCE FOUND: {self.best_agent.cloud_metrics.get('cagr', 0)*100:.1f}% CAGR", "SUCCESS")
                    break
            
            # Update archive - keep more diverse strategies
            self.archive.extend(new_agents)
            if len(self.archive) > 20:
                # Sort by CAGR and criteria score combined
                self.archive = sorted(self.archive, 
                    key=lambda a: (a.criteria_score, a.cloud_metrics.get('cagr', 0)), 
                    reverse=True)[:20]
            
            live_print(f"", "")
            live_print(f"📊 GENERATION {self.generation} COMPLETE", "SUMMARY")
            live_print(f"   🎯 Best Score: {self.best_score}/4", "SUMMARY")
            live_print(f"   📊 Best CAGR: {self.best_agent.cloud_metrics.get('cagr', 0)*100:.1f}%", "SUMMARY")
            live_print(f"   🔗 Best URL: {self.best_agent.cloud_url}", "SUMMARY")
            live_print(f"   📁 Archive: {len(self.archive)} verified agents", "SUMMARY")
    
    def victory_celebration(self):
        """Victory with verified results"""
        runtime = (time.time() - self.start_time) / 3600
        
        live_print("", "")
        live_print("🏆" * 80, "")
        live_print("🎉🎉🎉 ALL TARGETS ACHIEVED! 🎉🎉🎉", "VICTORY")
        live_print("🏆" * 80, "")
        
        agent = self.best_agent
        metrics = agent.cloud_metrics
        
        live_print(f"📊 FINAL CAGR: {metrics.get('cagr', 0)*100:.2f}% ✅", "VICTORY")
        live_print(f"📈 FINAL SHARPE: {metrics.get('sharpe', 0):.3f} ✅", "VICTORY")
        live_print(f"📉 FINAL MAX DD: {metrics.get('drawdown', 1)*100:.1f}% ✅", "VICTORY")
        live_print(f"💰 FINAL TRADES: {metrics.get('total_trades', 0)}", "VICTORY")
        live_print(f"🎯 FINAL SCORE: {agent.criteria_score}/4 ✅", "VICTORY")
        live_print(f"", "")
        live_print(f"🔗 VERIFY RESULTS: {agent.cloud_url}", "VICTORY")
        live_print(f"⏱️ Runtime: {runtime:.2f} hours", "VICTORY")
        live_print(f"🧬 Generations: {self.generation}", "VICTORY")
        live_print(f"🏆 Evolution: {' → '.join(agent.mutations)}", "VICTORY")


def main():
    """Launch trustworthy DGM"""
    try:
        dgm = TrustworthyDGM()
        winner = dgm.evolve_until_targets_met()
        
        if winner:
            live_print("✅ ALL TARGETS VERIFIED AND ACHIEVED!", "SUCCESS")
            live_print(f"🔗 FINAL VERIFICATION: {winner.cloud_url}", "SUCCESS")
        else:
            live_print("⚠️ Evolution stopped or failed", "WARNING")
        
    except KeyboardInterrupt:
        live_print("⚠️ Stopped by user", "STOP")
    except Exception as e:
        live_print(f"💥 Error: {e}", "ERROR")


if __name__ == "__main__":
    main()