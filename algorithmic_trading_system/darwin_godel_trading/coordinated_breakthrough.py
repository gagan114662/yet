#!/usr/bin/env python3
"""
COORDINATED BREAKTHROUGH SYSTEM
Revolutionary exploration across high-return strategy spaces
No more 6% ceiling - targeting 25%+ CAGR through specialized evolution
"""

import os
import json
import subprocess
import time
import sys
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
import re

# Configuration
LEAN_WORKSPACE = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
LEAN_CLI = "/home/vandan/.local/bin/lean"

# Rate limiting with shared pool
RATE_LIMIT = {
    "max_per_hour": 18,  # Conservative shared limit
    "min_between": 200,  # 3.3 minutes between ANY request
    "cooldown": 600      # 10 min cooldown on errors
}

request_queue = deque(maxlen=18)
last_request = datetime.now() - timedelta(seconds=300)

# Target criteria
TARGET_CRITERIA = {
    "cagr": 0.25,           # >25%
    "sharpe": 1.0,          # >1.0
    "max_drawdown": 0.20,   # <20%
    "avg_profit": 0.0075    # >0.75%
}

def live_print(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()

def wait_for_rate_limit():
    """Shared rate limiter for all strategies"""
    global last_request
    
    # Check hourly limit
    now = datetime.now()
    recent = [t for t in request_queue if (now - t).total_seconds() < 3600]
    
    if len(recent) >= RATE_LIMIT["max_per_hour"]:
        wait_time = (recent[0] + timedelta(hours=1) - now).total_seconds()
        if wait_time > 0:
            live_print(f"⏰ Rate limit: waiting {wait_time/60:.1f} min", "RATE")
            time.sleep(wait_time)
    
    # Minimum time between requests
    elapsed = (now - last_request).total_seconds()
    if elapsed < RATE_LIMIT["min_between"]:
        wait = RATE_LIMIT["min_between"] - elapsed
        live_print(f"⏱️ Rate wait: {wait:.0f}s", "RATE")
        time.sleep(wait)
    
    last_request = datetime.now()
    request_queue.append(last_request)


class OptionsHarvester:
    """Specialized evolution for options strategies"""
    
    def __init__(self, agent_id: str):
        self.agent_id = f"Options_{agent_id}"
        self.strategy_config = self._generate_options_strategy()
        
    def _generate_options_strategy(self) -> Dict:
        """Generate options-like strategies using leveraged positions"""
        strategies = [
            {
                "name": "covered_call_simulator",
                "asset": "SPY",
                "leverage": 8.0,
                "strategy": """
        # Simulate covered call by selling upside
        current_price = self.securities["SPY"].price
        strike = current_price * 1.02  # 2% OTM
        
        if not self.portfolio.invested:
            # Buy underlying
            self.set_holdings("SPY", 0.8)
            self.strike_price = strike
        elif current_price > self.strike_price:
            # "Assignment" - close position
            self.liquidate()
            self.log(f"Call assigned at {self.strike_price:.2f}")""",
                "expected_cagr": 0.15
            },
            {
                "name": "put_selling_simulator", 
                "asset": "QQQ",
                "leverage": 10.0,
                "strategy": """
        # Simulate cash-secured puts
        current_price = self.securities["QQQ"].price
        put_strike = current_price * 0.95  # 5% OTM
        
        if not self.portfolio.invested:
            if self.rsi.current.value < 40:  # Oversold
                # "Sell put" by preparing to buy
                self.pending_strike = put_strike
        elif hasattr(self, 'pending_strike'):
            if current_price <= self.pending_strike:
                # "Put assigned" - buy shares
                self.set_holdings("QQQ", 1.5)
            elif current_price > self.pending_strike * 1.05:
                # Take profit
                self.liquidate()""",
                "expected_cagr": 0.18
            },
            {
                "name": "iron_condor_simulator",
                "asset": "SPY",
                "leverage": 15.0,
                "strategy": """
        # Simulate iron condor with range trading
        current_price = self.securities["SPY"].price
        upper_bound = self.bb.upper_band.current.value
        lower_bound = self.bb.lower_band.current.value
        
        if not self.portfolio.invested:
            # Enter when price is mid-range
            if lower_bound < current_price < upper_bound:
                range_width = upper_bound - lower_bound
                if range_width / current_price > 0.02:  # 2% range
                    self.set_holdings("SPY", 0.5)  # Conservative
                    self.upper_exit = upper_bound
                    self.lower_exit = lower_bound
        else:
            # Exit at boundaries
            if current_price >= self.upper_exit or current_price <= self.lower_exit:
                self.liquidate()
                self.log("Iron condor boundary hit")""",
                "expected_cagr": 0.20
            }
        ]
        
        return random.choice(strategies)
    
    def generate_code(self) -> str:
        config = self.strategy_config
        
        return f'''from AlgorithmImports import *
import numpy as np

class OptionsStrategy(QCAlgorithm):
    """Options Harvesting Strategy - {config['name']}"""
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("{config['asset']}", Resolution.DAILY)
        self.symbol.set_leverage({config['leverage']})
        
        # Options-like indicators
        self.rsi = self.rsi("{config['asset']}", 14)
        self.bb = self.bb("{config['asset']}", 20, 2)
        self.sma = self.sma("{config['asset']}", 50)
        
        # Track performance
        self.daily_returns = []
        self.last_value = self.portfolio.total_portfolio_value
        
    def on_data(self, data):
        # Calculate daily returns
        current_value = self.portfolio.total_portfolio_value
        if self.last_value > 0:
            ret = (current_value - self.last_value) / self.last_value
            self.daily_returns.append(ret)
        self.last_value = current_value
        
        if not self.rsi.is_ready or not self.bb.is_ready:
            return
            
{config['strategy']}
    
    def on_end_of_algorithm(self):
        if len(self.daily_returns) > 252:
            returns = np.array(self.daily_returns)
            sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
            self.log(f"Final Sharpe: {{sharpe:.3f}}")'''


class CryptoMomentum:
    """Specialized evolution for crypto strategies"""
    
    def __init__(self, agent_id: str):
        self.agent_id = f"Crypto_{agent_id}"
        self.strategy_config = self._generate_crypto_strategy()
        
    def _generate_crypto_strategy(self) -> Dict:
        """Generate crypto momentum strategies"""
        strategies = [
            {
                "name": "btc_trend_follower",
                "asset": "BITO",  # Bitcoin ETF
                "leverage": random.uniform(8, 15),
                "fast_ma": random.randint(3, 10),
                "slow_ma": random.randint(15, 30),
                "atr_multiplier": random.uniform(1.5, 3.0),
                "expected_cagr": 0.30
            },
            {
                "name": "crypto_breakout",
                "asset": "BITO",
                "leverage": random.uniform(10, 18),
                "breakout_periods": random.randint(10, 20),
                "volume_threshold": random.uniform(1.2, 2.0),
                "expected_cagr": 0.35
            }
        ]
        
        return random.choice(strategies)
    
    def generate_code(self) -> str:
        config = self.strategy_config
        
        if "trend_follower" in config['name']:
            strategy = f"""
        # Crypto trend following
        fast_ma = self.sma_fast.current.value
        slow_ma = self.sma_slow.current.value
        atr = self.atr.current.value
        
        if fast_ma > slow_ma and not self.portfolio.invested:
            # Strong uptrend
            volatility_position = min(2.0, 1.0 + (atr / self.securities["{config['asset']}"].price * 100))
            self.set_holdings("{config['asset']}", volatility_position)
            self.stop_loss = self.securities["{config['asset']}"].price - (atr * {config['atr_multiplier']})
        elif self.portfolio.invested:
            current_price = self.securities["{config['asset']}"].price
            if current_price < self.stop_loss or fast_ma < slow_ma * 0.98:
                self.liquidate()"""
        else:
            strategy = f"""
        # Crypto breakout strategy
        high_{config['breakout_periods']} = max([bar.high for bar in self.history(self.symbol, {config['breakout_periods']}, Resolution.DAILY)])
        current_price = self.securities["{config['asset']}"].price
        volume = self.securities["{config['asset']}"].volume
        avg_volume = np.mean([bar.volume for bar in self.history(self.symbol, 20, Resolution.DAILY)])
        
        if not self.portfolio.invested:
            # Breakout with volume confirmation
            if current_price > high_{config['breakout_periods']} * 1.01:
                if volume > avg_volume * {config['volume_threshold']}:
                    self.set_holdings("{config['asset']}", 2.0)
                    self.entry = current_price
        else:
            # Trailing stop
            if current_price < self.entry * 0.92:  # 8% stop
                self.liquidate()"""
        
        return f'''from AlgorithmImports import *
import numpy as np

class CryptoStrategy(QCAlgorithm):
    """Crypto Momentum - {config['name']}"""
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("{config['asset']}", Resolution.DAILY)
        self.symbol.set_leverage({config['leverage']})
        
        self.sma_fast = self.sma("{config['asset']}", {config.get('fast_ma', 5)})
        self.sma_slow = self.sma("{config['asset']}", {config.get('slow_ma', 20)})
        self.atr = self.atr("{config['asset']}", 14)
        
        self.daily_returns = []
        self.last_value = self.portfolio.total_portfolio_value
        
    def on_data(self, data):
        current_value = self.portfolio.total_portfolio_value
        if self.last_value > 0:
            ret = (current_value - self.last_value) / self.last_value
            self.daily_returns.append(ret)
        self.last_value = current_value
        
        if not self.sma_fast.is_ready or not self.atr.is_ready:
            return
            
{strategy}
    
    def on_end_of_algorithm(self):
        if len(self.daily_returns) > 252:
            returns = np.array(self.daily_returns)
            sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
            self.log(f"Final Sharpe: {{sharpe:.3f}}")'''


class VolatilityTrader:
    """Specialized evolution for volatility strategies"""
    
    def __init__(self, agent_id: str):
        self.agent_id = f"Vol_{agent_id}"
        self.strategy_config = self._generate_vol_strategy()
        
    def _generate_vol_strategy(self) -> Dict:
        """Generate volatility harvesting strategies"""
        return {
            "name": "vix_spike_trader",
            "asset": random.choice(["UVXY", "VXX"]),  # VIX ETFs
            "leverage": random.uniform(5, 12),
            "vix_threshold": random.uniform(18, 25),
            "holding_days": random.randint(2, 7),
            "expected_cagr": 0.25
        }
    
    def generate_code(self) -> str:
        config = self.strategy_config
        
        return f'''from AlgorithmImports import *
import numpy as np

class VolatilityStrategy(QCAlgorithm):
    """Volatility Harvesting - {config['name']}"""
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.vol_symbol = self.add_equity("{config['asset']}", Resolution.DAILY)
        self.vol_symbol.set_leverage({config['leverage']})
        
        # Add VIX for reference
        self.vix = self.add_index("VIX", Resolution.DAILY)
        
        # Track spy for inverse correlation
        self.spy = self.add_equity("SPY", Resolution.DAILY)
        self.spy_sma = self.sma("SPY", 20)
        
        self.holding_days = 0
        self.daily_returns = []
        self.last_value = self.portfolio.total_portfolio_value
        
    def on_data(self, data):
        current_value = self.portfolio.total_portfolio_value
        if self.last_value > 0:
            ret = (current_value - self.last_value) / self.last_value
            self.daily_returns.append(ret)
        self.last_value = current_value
        
        if not self.spy_sma.is_ready:
            return
            
        # Get VIX value
        vix_value = self.securities["VIX"].price if "VIX" in self.securities else 20
        
        if not self.portfolio.invested:
            # Enter on VIX spike
            if vix_value > {config['vix_threshold']}:
                spy_below_ma = self.securities["SPY"].price < self.spy_sma.current.value
                if spy_below_ma:  # Market stress confirmed
                    self.set_holdings("{config['asset']}", 1.5)
                    self.holding_days = 0
                    self.entry_vix = vix_value
        else:
            self.holding_days += 1
            
            # Exit conditions
            if self.holding_days >= {config['holding_days']}:
                self.liquidate()
                self.log(f"Time exit after {{self.holding_days}} days")
            elif vix_value < self.entry_vix * 0.8:  # VIX dropped 20%
                self.liquidate()
                self.log("VIX normalization exit")
    
    def on_end_of_algorithm(self):
        if len(self.daily_returns) > 252:
            returns = np.array(self.daily_returns)
            sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
            self.log(f"Final Sharpe: {{sharpe:.3f}}")'''


class LeveragedETFMaster:
    """Specialized evolution for leveraged ETF strategies"""
    
    def __init__(self, agent_id: str):
        self.agent_id = f"LevETF_{agent_id}"
        self.strategy_config = self._generate_lev_etf_strategy()
        
    def _generate_lev_etf_strategy(self) -> Dict:
        """Generate leveraged ETF strategies"""
        etfs = ["TQQQ", "UPRO", "SOXL", "SPXL"]
        return {
            "name": "momentum_leveraged_etf",
            "asset": random.choice(etfs),
            "leverage": random.uniform(3, 8),  # Lower since ETF already leveraged
            "momentum_periods": random.randint(10, 30),
            "regime_sma": random.randint(50, 200),
            "expected_cagr": 0.40
        }
    
    def generate_code(self) -> str:
        config = self.strategy_config
        
        return f'''from AlgorithmImports import *
import numpy as np

class LeveragedETFStrategy(QCAlgorithm):
    """Leveraged ETF Master - {config['name']}"""
    
    def initialize(self):
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.symbol = self.add_equity("{config['asset']}", Resolution.DAILY)
        self.symbol.set_leverage({config['leverage']})
        
        # Momentum indicators
        self.momentum = self.momp("{config['asset']}", {config['momentum_periods']})
        self.regime_filter = self.sma("{config['asset']}", {config['regime_sma']})
        self.rsi = self.rsi("{config['asset']}", 14)
        
        # Risk management
        self.max_drawdown = 0.15
        self.peak_value = self.portfolio.total_portfolio_value
        
        self.daily_returns = []
        self.last_value = self.portfolio.total_portfolio_value
        
    def on_data(self, data):
        # Track returns and drawdown
        current_value = self.portfolio.total_portfolio_value
        if self.last_value > 0:
            ret = (current_value - self.last_value) / self.last_value
            self.daily_returns.append(ret)
        self.last_value = current_value
        
        # Update peak for drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        current_dd = (self.peak_value - current_value) / self.peak_value
        
        if not self.momentum.is_ready or not self.regime_filter.is_ready:
            return
            
        current_price = self.securities["{config['asset']}"].price
        
        if not self.portfolio.invested:
            # Enter on strong momentum in uptrend
            if (self.momentum.current.value > 0 and 
                current_price > self.regime_filter.current.value and
                self.rsi.current.value < 70):
                
                # Dynamic position sizing based on momentum strength
                momentum_strength = abs(self.momentum.current.value) / current_price
                position = min(2.0, 1.0 + momentum_strength * 10)
                self.set_holdings("{config['asset']}", position)
                
        else:
            # Exit conditions
            if (self.momentum.current.value < 0 or 
                current_price < self.regime_filter.current.value * 0.98 or
                current_dd > self.max_drawdown):
                self.liquidate()
                if current_dd > self.max_drawdown:
                    self.log(f"Max drawdown exit: {{current_dd:.2%}}")
    
    def on_end_of_algorithm(self):
        if len(self.daily_returns) > 252:
            returns = np.array(self.daily_returns)
            sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
            self.log(f"Final Sharpe: {{sharpe:.3f}}")'''


class CoordinatedBreakthrough:
    """Master coordinator for specialized strategies"""
    
    def __init__(self):
        self.specialists = {
            "options": OptionsHarvester,
            "crypto": CryptoMomentum,
            "volatility": VolatilityTrader,
            "leveraged_etf": LeveragedETFMaster
        }
        self.results = []
        self.best_strategy = None
        self.best_score = 0
        self.start_time = time.time()
        
    def run_coordinated_evolution(self):
        """Run coordinated evolution across all strategy types"""
        
        live_print("🚀 COORDINATED BREAKTHROUGH SYSTEM", "SYSTEM")
        live_print("🎯 REVOLUTIONARY EXPLORATION: 25%+ CAGR", "SYSTEM")
        live_print("", "")
        
        generation = 0
        max_generations = 20
        
        while generation < max_generations and self.best_score < 4:
            generation += 1
            runtime = (time.time() - self.start_time) / 60
            
            live_print("=" * 80, "")
            live_print(f"🧬 COORDINATED GENERATION {generation} | ⏱️ {runtime:.1f}min", "GEN")
            if self.best_strategy:
                live_print(f"🏆 BEST: {self.best_score}/4 criteria | {self.best_strategy['cagr']*100:.1f}% CAGR", "GEN")
            live_print("=" * 80, "")
            
            # Test one of each specialist per generation
            for specialist_name, specialist_class in self.specialists.items():
                agent_id = f"gen{generation}"
                specialist = specialist_class(agent_id)
                
                live_print(f"", "")
                live_print(f"🧪 TESTING: {specialist_name.upper()} SPECIALIST", "TEST")
                
                # Rate limit check
                wait_for_rate_limit()
                
                # Run backtest
                result = self.run_backtest(specialist)
                
                if result:
                    self.results.append(result)
                    
                    # Check for breakthrough
                    if result['score'] > self.best_score or (result['score'] == self.best_score and result['cagr'] > self.best_strategy.get('cagr', 0)):
                        self.best_strategy = result
                        self.best_score = result['score']
                        
                        live_print("", "")
                        live_print("🎉🎉🎉 BREAKTHROUGH! 🎉🎉🎉", "BREAKTHROUGH")
                        live_print(f"Strategy: {result['strategy_type']}", "BREAKTHROUGH")
                        live_print(f"CAGR: {result['cagr']*100:.1f}%", "BREAKTHROUGH")
                        live_print(f"Sharpe: {result['sharpe']:.2f}", "BREAKTHROUGH")
                        live_print(f"Score: {result['score']}/4", "BREAKTHROUGH")
                        live_print(f"URL: {result['url']}", "BREAKTHROUGH")
                        
                        if self.best_score >= 4:
                            self.victory()
                            return
            
            live_print("", "")
            live_print(f"Generation {generation} Summary:", "SUMMARY")
            live_print(f"  Strategies tested: {len(self.specialists)}", "SUMMARY")
            live_print(f"  Best CAGR: {self.best_strategy['cagr']*100:.1f}%" if self.best_strategy else "  No valid results yet", "SUMMARY")
    
    def run_backtest(self, specialist) -> Optional[Dict]:
        """Run backtest for a specialist"""
        project_name = f"Coord_{specialist.agent_id}_{int(time.time())}"
        project_path = os.path.join(LEAN_WORKSPACE, project_name)
        
        try:
            # Create project
            os.makedirs(project_path, exist_ok=True)
            
            # Generate and write code
            code = specialist.generate_code()
            with open(os.path.join(project_path, "main.py"), 'w') as f:
                f.write(code)
            
            config = {
                "algorithm-language": "Python",
                "parameters": {},
                "description": f"Coordinated {specialist.agent_id}"
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
                live_print(f"  ❌ Push failed", "ERROR")
                return None
            
            # Run backtest
            backtest_result = subprocess.run(
                [LEAN_CLI, "cloud", "backtest", project_name, "--name", f"Coord_{specialist.agent_id}"],
                cwd=LEAN_WORKSPACE,
                capture_output=True,
                text=True,
                timeout=1200
            )
            
            if backtest_result.returncode == 0:
                # Parse results
                metrics = self._parse_results(backtest_result.stdout)
                
                # Get URL
                url_match = re.search(r'https://www\.quantconnect\.com/project/\d+/[a-f0-9]+', backtest_result.stdout)
                url = url_match.group(0) if url_match else ""
                
                live_print(f"  ✅ Results: CAGR={metrics['cagr']*100:.1f}%, Sharpe={metrics['sharpe']:.2f}", "RESULT")
                live_print(f"  🔗 {url}", "RESULT")
                
                return {
                    "strategy_type": specialist.agent_id,
                    "cagr": metrics['cagr'],
                    "sharpe": metrics['sharpe'],
                    "drawdown": metrics['drawdown'],
                    "score": self._calculate_score(metrics),
                    "url": url
                }
            else:
                if "Too many backtest requests" in backtest_result.stderr:
                    live_print(f"  ⚠️ Rate limit hit - will wait", "RATE")
                    time.sleep(300)  # 5 min cooldown
                return None
                
        except Exception as e:
            live_print(f"  ❌ Error: {str(e)[:100]}", "ERROR")
            return None
        finally:
            try:
                if os.path.exists(project_path):
                    subprocess.run(["rm", "-rf", project_path], timeout=30)
            except:
                pass
    
    def _parse_results(self, output: str) -> Dict:
        """Parse backtest results"""
        metrics = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "drawdown": 1.0,
            "avg_profit": 0.0
        }
        
        try:
            for line in output.split('\n'):
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
                elif "Average Win" in line and "%" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["avg_profit"] = float(match.group(1)) / 100 * 0.5
            
            return metrics
        except:
            return metrics
    
    def _calculate_score(self, metrics: Dict) -> int:
        """Calculate criteria score"""
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
    
    def victory(self):
        """Victory announcement"""
        runtime = (time.time() - self.start_time) / 60
        
        live_print("", "")
        live_print("🏆" * 80, "")
        live_print("🎉🎉🎉 COORDINATED BREAKTHROUGH ACHIEVED! 🎉🎉🎉", "VICTORY")
        live_print("🏆" * 80, "")
        live_print(f"", "")
        live_print(f"Strategy Type: {self.best_strategy['strategy_type']}", "VICTORY")
        live_print(f"CAGR: {self.best_strategy['cagr']*100:.1f}% ✅", "VICTORY")
        live_print(f"Sharpe: {self.best_strategy['sharpe']:.2f} ✅", "VICTORY")
        live_print(f"Drawdown: {self.best_strategy['drawdown']*100:.1f}% ✅", "VICTORY")
        live_print(f"Score: 4/4 ✅", "VICTORY")
        live_print(f"", "")
        live_print(f"🔗 VERIFY: {self.best_strategy['url']}", "VICTORY")
        live_print(f"Runtime: {runtime:.1f} minutes", "VICTORY")


def main():
    """Launch coordinated breakthrough"""
    try:
        coordinator = CoordinatedBreakthrough()
        coordinator.run_coordinated_evolution()
    except KeyboardInterrupt:
        live_print("\n⚠️ Stopped by user", "STOP")
    except Exception as e:
        live_print(f"💥 Error: {e}", "ERROR")


if __name__ == "__main__":
    main()