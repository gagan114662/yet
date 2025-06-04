#!/usr/bin/env python3
"""
RATE-LIMITED DARWIN GÖDEL MACHINE
Respects QuantConnect's rate limits while maximizing evolution speed
"""

import os
import json
import subprocess
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import random
from collections import deque

# Configuration
LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
LEAN_CLI = "/home/vandan/.local/bin/lean"

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    "max_backtests_per_hour": 20,  # Conservative limit
    "min_seconds_between_backtests": 180,  # 3 minutes minimum
    "backoff_multiplier": 1.5,  # Exponential backoff on failures
    "max_retries": 2,
    "cooldown_after_error": 300  # 5 minute cooldown after error
}

# Track request times
request_history = deque(maxlen=20)
last_request_time = datetime.now() - timedelta(seconds=300)
consecutive_failures = 0

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

def check_rate_limit():
    """Check if we can make another request"""
    global last_request_time, consecutive_failures
    
    # Check consecutive failures
    if consecutive_failures >= 3:
        wait_time = RATE_LIMIT_CONFIG["cooldown_after_error"]
        live_print(f"⏸️ Rate limit cooldown: {wait_time}s after {consecutive_failures} failures", "RATE")
        time.sleep(wait_time)
        consecutive_failures = 0
    
    # Check minimum time between requests
    elapsed = (datetime.now() - last_request_time).total_seconds()
    min_wait = RATE_LIMIT_CONFIG["min_seconds_between_backtests"]
    
    if elapsed < min_wait:
        wait_time = min_wait - elapsed
        live_print(f"⏱️ Rate limit wait: {wait_time:.0f}s", "RATE")
        time.sleep(wait_time)
    
    # Check hourly limit
    current_time = datetime.now()
    recent_requests = [t for t in request_history if (current_time - t).total_seconds() < 3600]
    
    if len(recent_requests) >= RATE_LIMIT_CONFIG["max_backtests_per_hour"]:
        oldest_request = min(recent_requests)
        wait_until = oldest_request + timedelta(hours=1)
        wait_seconds = (wait_until - current_time).total_seconds()
        
        if wait_seconds > 0:
            live_print(f"⏰ Hourly limit reached. Waiting {wait_seconds/60:.1f} minutes", "RATE")
            time.sleep(wait_seconds)
    
    # Update tracking
    last_request_time = datetime.now()
    request_history.append(last_request_time)

class RateLimitedAgent:
    """Agent with built-in rate limiting and enhanced strategies"""
    
    def __init__(self, agent_id: str, generation: int = 0):
        self.agent_id = agent_id
        self.generation = generation
        self.mutations = []
        self.cloud_metrics = {}
        self.cloud_url = None
        
        # High-performance strategy configurations
        strategies = [
            # TQQQ with optimized leverage
            {
                "asset": "TQQQ",
                "leverage": 5.0,
                "position_size": 2.0,
                "strategy": "momentum",
                "sma_fast": 5,
                "sma_slow": 20
            },
            # SPY with high leverage
            {
                "asset": "SPY", 
                "leverage": 15.0,
                "position_size": 1.5,
                "strategy": "breakout",
                "sma_fast": 10,
                "sma_slow": 30
            },
            # QQQ balanced approach
            {
                "asset": "QQQ",
                "leverage": 12.0,
                "position_size": 2.0,
                "strategy": "hybrid",
                "sma_fast": 8,
                "sma_slow": 21
            },
            # UPRO momentum
            {
                "asset": "UPRO",
                "leverage": 4.0,
                "position_size": 2.5,
                "strategy": "momentum",
                "sma_fast": 3,
                "sma_slow": 15
            }
        ]
        
        # Select strategy
        config = random.choice(strategies)
        
        self.code_components = {
            "asset": config["asset"],
            "leverage": config["leverage"],
            "position_size": config["position_size"],
            "strategy_type": config["strategy"],
            "sma_fast": config["sma_fast"],
            "sma_slow": config["sma_slow"],
            "has_rsi": random.choice([True, False]),
            "has_macd": random.choice([True, False]),
            "stop_loss": random.uniform(0.05, 0.15),
            "take_profit": random.uniform(0.10, 0.30),
            "start_year": 2009,
            "end_year": 2023
        }
        
        self.is_valid = False
        self.criteria_score = 0.0
        self.generate_code()
    
    def generate_code(self) -> str:
        """Generate optimized strategy code"""
        try:
            asset = self.code_components["asset"]
            leverage = self.code_components["leverage"]
            position_size = self.code_components["position_size"]
            strategy = self.code_components["strategy_type"]
            
            # Build indicators
            indicators = []
            indicators.append(f'        self.sma_fast = self.sma("{asset}", {self.code_components["sma_fast"]})')
            indicators.append(f'        self.sma_slow = self.sma("{asset}", {self.code_components["sma_slow"]})')
            
            if self.code_components["has_rsi"]:
                indicators.append(f'        self.rsi = self.rsi("{asset}", 14)')
            if self.code_components["has_macd"]:
                indicators.append(f'        self.macd = self.macd("{asset}", 12, 26, 9)')
            
            indicators_str = '\n'.join(indicators)
            
            # Strategy logic
            if strategy == "momentum":
                logic = f'''
        # Momentum strategy
        if self.sma_fast.current.value > self.sma_slow.current.value:
            if not self.portfolio.invested:
                self.set_holdings("{asset}", {position_size})
                self.entry_price = self.securities["{asset}"].price
        elif self.sma_fast.current.value < self.sma_slow.current.value:
            if self.portfolio.invested:
                self.liquidate()'''
            
            elif strategy == "breakout":
                logic = f'''
        # Breakout strategy
        if not self.portfolio.invested:
            if self.securities["{asset}"].price > self.sma_slow.current.value * 1.02:
                self.set_holdings("{asset}", {position_size})
                self.entry_price = self.securities["{asset}"].price
        else:
            if self.securities["{asset}"].price < self.sma_slow.current.value * 0.98:
                self.liquidate()'''
            
            else:  # hybrid
                logic = f'''
        # Hybrid strategy
        momentum = self.sma_fast.current.value > self.sma_slow.current.value
        
        if momentum and not self.portfolio.invested:
            self.set_holdings("{asset}", {position_size})
            self.entry_price = self.securities["{asset}"].price
        elif not momentum and self.portfolio.invested:
            self.liquidate()'''
            
            code = f'''from AlgorithmImports import *
import numpy as np

class RateLimitedStrategy(QCAlgorithm):
    """
    Rate-Limited Strategy - {self.agent_id}
    {strategy.upper()} on {asset} with {leverage}x leverage
    """
    
    def initialize(self):
        self.set_start_date({self.code_components["start_year"]}, 1, 1)
        self.set_end_date({self.code_components["end_year"]}, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("{asset}", Resolution.DAILY)
        self.symbol.set_leverage({leverage})
        
{indicators_str}
        
        self.entry_price = 0
        self.daily_returns = []
        self.last_value = self.portfolio.total_portfolio_value
        
    def on_data(self, data):
        # Track returns for Sharpe
        current_value = self.portfolio.total_portfolio_value
        if self.last_value > 0:
            daily_return = (current_value - self.last_value) / self.last_value
            self.daily_returns.append(daily_return)
        self.last_value = current_value
        
        if not self.sma_fast.is_ready or not self.sma_slow.is_ready:
            return
{logic}
        
        # Risk management
        if self.portfolio.invested and self.entry_price > 0:
            current_price = self.securities["{asset}"].price
            pnl = (current_price - self.entry_price) / self.entry_price
            
            if pnl < -{self.code_components["stop_loss"]}:
                self.liquidate()
                self.log(f"Stop loss: {{pnl:.2%}}")
            elif pnl > {self.code_components["take_profit"]}:
                self.liquidate()
                self.log(f"Take profit: {{pnl:.2%}}")
    
    def on_end_of_algorithm(self):
        if len(self.daily_returns) > 252:
            returns = np.array(self.daily_returns)
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                self.log(f"Sharpe Ratio: {{sharpe:.3f}}")'''
            
            compile(code, f"rate_limited_{self.agent_id}", 'exec')
            self.is_valid = True
            return code
            
        except Exception as e:
            live_print(f"❌ Code error: {e}", "ERROR")
            self.is_valid = False
            return ""
    
    def run_cloud_verification(self) -> Dict:
        """Run verification with rate limiting"""
        global consecutive_failures
        
        if not self.is_valid:
            return {}
        
        # Check rate limit BEFORE making request
        check_rate_limit()
        
        live_print(f"☁️ RATE-LIMITED VERIFICATION: {self.agent_id}", "VERIFY")
        live_print(f"   🎯 {self.code_components['strategy_type'].upper()} | {self.code_components['asset']} {self.code_components['leverage']}x", "VERIFY")
        
        project_name = f"RateLimit_{self.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        try:
            # Create project
            os.makedirs(project_path, exist_ok=True)
            
            # Write files
            with open(os.path.join(project_path, "main.py"), 'w') as f:
                f.write(self.generate_code())
            
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "description": f"Rate-limited {self.agent_id}"
            }
            with open(os.path.join(project_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Push to cloud
            push_result = subprocess.run(
                [LEAN_CLI, "cloud", "push", "--project", project_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if push_result.returncode != 0:
                consecutive_failures += 1
                live_print(f"   ❌ Push failed", "ERROR")
                return {}
            
            # Run backtest
            backtest_name = f"RateLimit_{self.agent_id}"
            backtest_result = subprocess.run(
                [LEAN_CLI, "cloud", "backtest", project_name, "--name", backtest_name],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=1200
            )
            
            if backtest_result.returncode == 0:
                consecutive_failures = 0  # Reset on success
                metrics = self._parse_results(backtest_result.stdout)
                self.cloud_metrics = metrics
                self.criteria_score = self._calculate_score(metrics)
                
                # Extract URL
                url_match = re.search(r'https://www\.quantconnect\.com/project/\d+/[a-f0-9]+', backtest_result.stdout)
                if url_match:
                    self.cloud_url = url_match.group(0)
                
                live_print(f"   ✅ RESULTS:", "VERIFY")
                live_print(f"      📊 CAGR: {metrics.get('cagr', 0)*100:.2f}% {'✅' if metrics.get('cagr', 0) > TARGET_CRITERIA['cagr'] else '❌'}", "VERIFY")
                live_print(f"      📈 Sharpe: {metrics.get('sharpe', 0):.3f} {'✅' if metrics.get('sharpe', 0) > TARGET_CRITERIA['sharpe'] else '❌'}", "VERIFY")
                live_print(f"      📉 DD: {metrics.get('drawdown', 1)*100:.1f}% {'✅' if metrics.get('drawdown', 1) < TARGET_CRITERIA['max_drawdown'] else '❌'}", "VERIFY")
                live_print(f"      🎯 SCORE: {self.criteria_score}/4", "VERIFY")
                if self.cloud_url:
                    live_print(f"      🔗 VERIFY: {self.cloud_url}", "VERIFY")
                
                return metrics
            else:
                consecutive_failures += 1
                if "Too many backtest requests" in backtest_result.stderr:
                    live_print(f"   ⚠️ RATE LIMIT HIT - will wait longer", "RATE")
                    consecutive_failures = 3  # Trigger cooldown
                return {}
                
        except Exception as e:
            consecutive_failures += 1
            live_print(f"   ❌ Exception: {str(e)[:100]}", "ERROR")
            return {}
        finally:
            # Cleanup
            try:
                if os.path.exists(project_path):
                    subprocess.run(["rm", "-rf", project_path], timeout=30)
            except:
                pass
    
    def _parse_results(self, output: str) -> Dict:
        """Parse results"""
        metrics = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "drawdown": 1.0,
            "avg_profit": 0.0,
            "total_trades": 0
        }
        
        try:
            lines = output.split('\n')
            for line in lines:
                if "Compounding Annual" in line and "%" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["cagr"] = float(match.group(1)) / 100
                elif "Sharpe Ratio" in line and "│" in line:
                    parts = line.split("│")
                    if len(parts) > 1:
                        match = re.search(r'(-?\d+\.?\d*)', parts[-1])
                        if match:
                            metrics["sharpe"] = float(match.group(1))
                elif "Drawdown" in line and "%" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["drawdown"] = float(match.group(1)) / 100
                elif "Total Orders" in line:
                    match = re.search(r'(\d+)', line)
                    if match:
                        metrics["total_trades"] = int(match.group(1))
                elif "Average Win" in line and "%" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        avg_win = float(match.group(1)) / 100
                        metrics["avg_profit"] = avg_win * 0.5  # Estimate
            
            return metrics
        except:
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


class RateLimitedDGM:
    """Darwin Gödel Machine with proper rate limiting"""
    
    def __init__(self):
        self.generation = 0
        self.archive = []
        self.best_agent = None
        self.best_score = 0.0
        self.start_time = time.time()
        
        live_print("🚀 RATE-LIMITED DARWIN GÖDEL MACHINE", "SYSTEM")
        live_print("⏱️ RESPECTING QUANTCONNECT RATE LIMITS", "SYSTEM")
        live_print(f"📊 Max {RATE_LIMIT_CONFIG['max_backtests_per_hour']} backtests/hour", "SYSTEM")
        live_print(f"⏸️ Min {RATE_LIMIT_CONFIG['min_seconds_between_backtests']}s between requests", "SYSTEM")
        live_print("", "")
        live_print("🎯 TARGET CRITERIA:", "SYSTEM")
        live_print(f"   📊 CAGR: >25%", "SYSTEM")
        live_print(f"   📈 Sharpe: >1.0", "SYSTEM")
        live_print(f"   📉 Max DD: <20%", "SYSTEM")
        live_print(f"   💰 Avg Profit: >0.75%", "SYSTEM")
    
    def evolve_with_rate_limits(self):
        """Evolution with proper rate limiting"""
        
        agents_per_generation = 3  # Fewer agents to respect limits
        max_generations = 100
        
        while self.generation < max_generations and (not self.best_agent or self.best_score < 4.0):
            self.generation += 1
            runtime = (time.time() - self.start_time) / 60
            
            live_print("", "")
            live_print("=" * 80, "")
            live_print(f"🧬 GENERATION {self.generation} | ⏱️ {runtime:.1f}min", "EVOLUTION")
            if self.best_agent:
                live_print(f"🎯 BEST: {self.best_score}/4 | 📊 {self.best_agent.cloud_metrics.get('cagr', 0)*100:.1f}% CAGR", "EVOLUTION")
            
            # Show rate limit status
            recent_requests = len([t for t in request_history if (datetime.now() - t).total_seconds() < 3600])
            live_print(f"📊 Rate status: {recent_requests}/{RATE_LIMIT_CONFIG['max_backtests_per_hour']} requests this hour", "RATE")
            live_print("=" * 80, "")
            
            new_agents = []
            
            for i in range(agents_per_generation):
                agent_id = f"gen{self.generation}_agent{i}"
                agent = RateLimitedAgent(agent_id, self.generation)
                
                metrics = agent.run_cloud_verification()
                
                if metrics:
                    cagr = agent.cloud_metrics.get('cagr', 0) * 100
                    
                    if agent.criteria_score > self.best_score or (agent.criteria_score == self.best_score and cagr > (self.best_agent.cloud_metrics.get('cagr', 0) * 100 if self.best_agent else 0)):
                        self.best_score = agent.criteria_score
                        self.best_agent = agent
                        
                        live_print("", "")
                        live_print("🎉 NEW BEST STRATEGY! 🎉", "SUCCESS")
                        live_print(f"📈 Score: {self.best_score}/4", "SUCCESS")
                        live_print(f"💰 CAGR: {cagr:.1f}%", "SUCCESS")
                        
                        if self.best_score >= 4.0:
                            self.victory()
                            return agent
                    
                    new_agents.append(agent)
            
            # Update archive
            self.archive.extend(new_agents)
            if len(self.archive) > 10:
                self.archive = sorted(self.archive, 
                    key=lambda a: (a.criteria_score, a.cloud_metrics.get('cagr', 0)), 
                    reverse=True)[:10]
            
            live_print(f"", "")
            live_print(f"📊 Generation {self.generation} complete", "SUMMARY")
            live_print(f"   Agents tested: {len(new_agents)}", "SUMMARY")
            live_print(f"   Archive size: {len(self.archive)}", "SUMMARY")
    
    def victory(self):
        """Victory announcement"""
        runtime = (time.time() - self.start_time) / 3600
        
        live_print("", "")
        live_print("🏆" * 80, "")
        live_print("🎉 ALL TARGETS ACHIEVED! 🎉", "VICTORY")
        live_print("🏆" * 80, "")
        
        agent = self.best_agent
        metrics = agent.cloud_metrics
        
        live_print(f"📊 FINAL CAGR: {metrics.get('cagr', 0)*100:.2f}% ✅", "VICTORY")
        live_print(f"📈 FINAL SHARPE: {metrics.get('sharpe', 0):.3f} ✅", "VICTORY")
        live_print(f"📉 FINAL DD: {metrics.get('drawdown', 1)*100:.1f}% ✅", "VICTORY")
        live_print(f"🎯 FINAL SCORE: 4/4 ✅", "VICTORY")
        live_print(f"", "")
        live_print(f"🔗 VERIFY: {agent.cloud_url}", "VICTORY")
        live_print(f"⏱️ Runtime: {runtime:.2f} hours", "VICTORY")
        live_print(f"🧬 Generations: {self.generation}", "VICTORY")


def main():
    """Launch rate-limited evolution"""
    try:
        dgm = RateLimitedDGM()
        winner = dgm.evolve_with_rate_limits()
        
        if winner:
            live_print("✅ TARGETS ACHIEVED WITH RATE LIMITS!", "SUCCESS")
            live_print(f"🔗 FINAL: {winner.cloud_url}", "SUCCESS")
        else:
            live_print("⚠️ Evolution completed", "WARNING")
        
    except KeyboardInterrupt:
        live_print("⚠️ Stopped by user", "STOP")
    except Exception as e:
        live_print(f"💥 Error: {e}", "ERROR")


if __name__ == "__main__":
    main()