#!/usr/bin/env python3
"""
Optimized Pipeline: Real Trading Conditions + Min 100 Trades/Year

This version optimizes for actual market conditions and ensures strategies generate enough trades.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from rd_agent_qc_bridge import QuantConnectIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedQuantConnectIntegration(QuantConnectIntegration):
    """Enhanced integration with realistic trading parameters"""
    
    def __init__(self):
        super().__init__()
        # Updated targets including minimum trades
        self.target_metrics = {
            "cagr": 0.25,  # 25% CAGR
            "sharpe_ratio": 1.0,  # Sharpe > 1.0
            "max_drawdown": 0.20,  # Max DD < 20%
            "avg_profit": 0.0075,  # Average profit > 0.75%
            "min_trades_per_year": 100  # NEW: Minimum 100 trades per year
        }
        
    def evaluate_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Enhanced evaluation including trade frequency"""
        
        evaluation = {
            "meets_criteria": True,
            "details": {}
        }
        
        # Check traditional metrics
        if metrics["cagr"] < self.target_metrics["cagr"]:
            evaluation["meets_criteria"] = False
            evaluation["details"]["cagr"] = f"Target: {self.target_metrics['cagr']:.2%}, Actual: {metrics['cagr']:.2%}"
            
        if metrics["sharpe_ratio"] < self.target_metrics["sharpe_ratio"]:
            evaluation["meets_criteria"] = False
            evaluation["details"]["sharpe_ratio"] = f"Target: {self.target_metrics['sharpe_ratio']:.2f}, Actual: {metrics['sharpe_ratio']:.2f}"
            
        if metrics["max_drawdown"] > self.target_metrics["max_drawdown"]:
            evaluation["meets_criteria"] = False
            evaluation["details"]["max_drawdown"] = f"Target: <{self.target_metrics['max_drawdown']:.2%}, Actual: {metrics['max_drawdown']:.2%}"
            
        if metrics["avg_profit"] < self.target_metrics["avg_profit"]:
            evaluation["meets_criteria"] = False
            evaluation["details"]["avg_profit"] = f"Target: >{self.target_metrics['avg_profit']:.2%}, Actual: {metrics['avg_profit']:.2%}"
            
        # NEW: Check minimum trades per year
        if metrics.get("trades_per_year", 0) < self.target_metrics["min_trades_per_year"]:
            evaluation["meets_criteria"] = False
            evaluation["details"]["trades_per_year"] = f"Target: >{self.target_metrics['min_trades_per_year']}, Actual: {metrics.get('trades_per_year', 0):.0f}"
            
        return evaluation
        
    def parse_backtest_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced parsing with trade frequency calculation"""
        
        statistics = results.get("Statistics", {})
        
        # Calculate CAGR
        total_return = float(statistics.get("Total Return", "0").rstrip('%')) / 100
        years = float(statistics.get("Total Trading Days", "252")) / 252
        if years <= 0:
            years = 1
        cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
        
        # Extract metrics
        total_trades = int(statistics.get("Total Trades", "0"))
        trades_per_year = total_trades / years if years > 0 else 0
        
        metrics = {
            "cagr": cagr,
            "sharpe_ratio": float(statistics.get("Sharpe Ratio", "0")),
            "max_drawdown": abs(float(statistics.get("Drawdown", "0").rstrip('%')) / 100),
            "total_return": total_return,
            "win_rate": float(statistics.get("Win Rate", "0").rstrip('%')) / 100,
            "profit_loss_ratio": float(statistics.get("Profit-Loss Ratio", "0")),
            "avg_profit": float(statistics.get("Average Win", "0").rstrip('%')) / 100,
            "total_trades": total_trades,
            "trades_per_year": trades_per_year,  # NEW metric
            "annual_volatility": float(statistics.get("Annual Standard Deviation", "0")),
            "years_tested": years
        }
        
        return metrics

class OptimizedStrategyPipeline:
    """Pipeline optimized for real trading conditions"""
    
    def __init__(self):
        self.qc_integration = OptimizedQuantConnectIntegration()
        self.results = []
        
    def get_optimized_strategies(self):
        """Return strategies optimized for actual trading conditions"""
        
        return [
            {
                "name": "Aggressive_SPY_Momentum",
                "description": "High-frequency SPY momentum with tight stops",
                "start_date": "2020,1,1",
                "end_date": "2023,12,31",
                "strategy_code": '''
from AlgorithmImports import *

class AggressiveSPYMomentum(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Trade SPY only for reliable data
        self.spy = self.AddEquity("SPY", Resolution.Hour)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
        
        # Indicators for high-frequency signals
        self.momentum_fast = self.MOMP("SPY", 5)
        self.momentum_slow = self.MOMP("SPY", 15) 
        self.rsi = self.RSI("SPY", 14)
        self.bb = self.BB("SPY", 20, 2)
        
        # Track for trade frequency
        self.trade_count = 0
        self.last_trade_time = self.StartDate
        
    def OnData(self, data):
        if not (self.momentum_fast.IsReady and self.momentum_slow.IsReady and self.rsi.IsReady):
            return
            
        # Get current values
        fast_mom = self.momentum_fast.Current.Value
        slow_mom = self.momentum_slow.Current.Value
        rsi_val = self.rsi.Current.Value
        price = self.Securities["SPY"].Price
        
        # Current position
        holdings = self.Portfolio["SPY"].Quantity
        
        # AGGRESSIVE ENTRY CONDITIONS (for more trades)
        # Long entry: Fast momentum > slow momentum and RSI not overbought
        if fast_mom > slow_mom and fast_mom > 0.002 and rsi_val < 80 and holdings <= 0:
            self.SetHoldings("SPY", 0.95)
            self.trade_count += 1
            self.last_trade_time = self.Time
            
        # Short entry: Fast momentum < slow momentum and RSI not oversold  
        elif fast_mom < slow_mom and fast_mom < -0.002 and rsi_val > 20 and holdings >= 0:
            self.SetHoldings("SPY", -0.95)
            self.trade_count += 1
            self.last_trade_time = self.Time
            
        # TIGHT STOP LOSSES (for more frequent trades)
        elif holdings > 0 and (fast_mom < -0.005 or rsi_val > 85):
            self.Liquidate("SPY")
            self.trade_count += 1
            
        elif holdings < 0 and (fast_mom > 0.005 or rsi_val < 15):
            self.Liquidate("SPY")
            self.trade_count += 1
            
        # MEAN REVERSION TRADES (additional trade opportunities)
        elif abs(holdings) < 0.1:  # No position
            bb_upper = self.bb.UpperBand.Current.Value
            bb_lower = self.bb.LowerBand.Current.Value
            bb_middle = self.bb.MiddleBand.Current.Value
            
            # Buy oversold bounces
            if price < bb_lower and rsi_val < 30:
                self.SetHoldings("SPY", 0.5)
                self.trade_count += 1
                
            # Sell overbought reversals  
            elif price > bb_upper and rsi_val > 70:
                self.SetHoldings("SPY", -0.5)
                self.trade_count += 1
                
    def OnEndOfAlgorithm(self):
        years = (self.EndDate - self.StartDate).days / 365.25
        trades_per_year = self.trade_count / years
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
'''
            },
            {
                "name": "Multi_Timeframe_Reversal", 
                "description": "Multi-timeframe mean reversion with frequent rebalancing",
                "start_date": "2020,1,1",
                "end_date": "2023,12,31", 
                "strategy_code": '''
from AlgorithmImports import *

class MultiTimeframeReversal(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Trade liquid ETFs for reliable data
        self.symbols = ["SPY", "QQQ", "IWM"]
        self.securities = {}
        self.indicators = {}
        
        for symbol in self.symbols:
            security = self.AddEquity(symbol, Resolution.Hour)
            security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            self.securities[symbol] = security
            
            # Multiple timeframe indicators
            self.indicators[symbol] = {
                "rsi_short": self.RSI(symbol, 6),   # Very short-term
                "rsi_medium": self.RSI(symbol, 14), # Medium-term
                "bb": self.BB(symbol, 20, 2),
                "momentum": self.MOMP(symbol, 10),
                "ema_fast": self.EMA(symbol, 8),
                "ema_slow": self.EMA(symbol, 21)
            }
            
        self.trade_count = 0
        self.rebalance_frequency = 4  # Rebalance every 4 hours for more trades
        self.last_rebalance = self.StartDate
        
    def OnData(self, data):
        # Rebalance frequently for more trade opportunities
        hours_since_rebalance = (self.Time - self.last_rebalance).total_seconds() / 3600
        if hours_since_rebalance < self.rebalance_frequency:
            return
            
        self.last_rebalance = self.Time
        
        # Check if all indicators are ready
        ready_symbols = []
        for symbol in self.symbols:
            if all(indicator.IsReady for indicator in self.indicators[symbol].values()):
                ready_symbols.append(symbol)
                
        if not ready_symbols:
            return
            
        # Generate signals for each symbol
        signals = {}
        for symbol in ready_symbols:
            signals[symbol] = self.GenerateSignal(symbol)
            
        # Execute trades based on signals
        self.ExecuteTrades(signals)
        
    def GenerateSignal(self, symbol):
        indicators = self.indicators[symbol]
        price = self.Securities[symbol].Price
        
        rsi_short = indicators["rsi_short"].Current.Value
        rsi_medium = indicators["rsi_medium"].Current.Value
        momentum = indicators["momentum"].Current.Value
        ema_fast = indicators["ema_fast"].Current.Value
        ema_slow = indicators["ema_slow"].Current.Value
        bb_upper = indicators["bb"].UpperBand.Current.Value
        bb_lower = indicators["bb"].LowerBand.Current.Value
        
        # AGGRESSIVE MEAN REVERSION SIGNALS
        signal_strength = 0
        
        # RSI oversold/overbought
        if rsi_short < 25:
            signal_strength += 2  # Strong buy
        elif rsi_short < 35:
            signal_strength += 1  # Moderate buy
        elif rsi_short > 75:
            signal_strength -= 2  # Strong sell
        elif rsi_short > 65:
            signal_strength -= 1  # Moderate sell
            
        # Bollinger Band reversals
        if price < bb_lower:
            signal_strength += 1
        elif price > bb_upper:
            signal_strength -= 1
            
        # EMA crossover for momentum
        if ema_fast > ema_slow and momentum > 0:
            signal_strength += 1
        elif ema_fast < ema_slow and momentum < 0:
            signal_strength -= 1
            
        # Return normalized signal (-1 to 1)
        return max(-1, min(1, signal_strength / 3))
        
    def ExecuteTrades(self, signals):
        # Equal weight allocation with frequent rebalancing
        total_weight = sum(abs(signal) for signal in signals.values())
        
        if total_weight > 0:
            for symbol, signal in signals.items():
                if abs(signal) > 0.2:  # Only trade if signal is strong enough
                    weight = (signal / total_weight) * 0.9  # Use 90% of capital
                    current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                    
                    # Only trade if position change is significant
                    if abs(weight - current_weight) > 0.05:
                        self.SetHoldings(symbol, weight)
                        self.trade_count += 1
                        
        else:
            # No strong signals, reduce positions
            for symbol in self.symbols:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    
    def OnEndOfAlgorithm(self):
        years = (self.EndDate - self.StartDate).days / 365.25
        trades_per_year = self.trade_count / years
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
'''
            },
            {
                "name": "Breakout_Scalping_Strategy",
                "description": "High-frequency breakout trading with tight risk management", 
                "start_date": "2020,1,1",
                "end_date": "2023,12,31",
                "strategy_code": '''
from AlgorithmImports import *

class BreakoutScalpingStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Trade multiple liquid ETFs for more opportunities
        self.symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        self.securities = {}
        self.indicators = {}
        
        for symbol in self.symbols:
            security = self.AddEquity(symbol, Resolution.Hour)
            security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            self.securities[symbol] = security
            
            # Breakout detection indicators
            self.indicators[symbol] = {
                "sma_short": self.SMA(symbol, 5),
                "sma_long": self.SMA(symbol, 20),
                "atr": self.ATR(symbol, 14),
                "bb": self.BB(symbol, 20, 1.5),  # Tighter bands
                "rsi": self.RSI(symbol, 10),     # Faster RSI
                "momentum": self.MOMP(symbol, 5)  # Very short momentum
            }
            
        self.trade_count = 0
        self.positions = {}
        self.stop_losses = {}
        
        # Schedule frequent checks
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(timedelta(hours=2)),
                        self.CheckBreakouts)
                        
    def CheckBreakouts(self):
        for symbol in self.symbols:
            if all(indicator.IsReady for indicator in self.indicators[symbol].values()):
                self.ProcessSymbol(symbol)
                
    def ProcessSymbol(self, symbol):
        indicators = self.indicators[symbol]
        price = self.Securities[symbol].Price
        
        sma_short = indicators["sma_short"].Current.Value
        sma_long = indicators["sma_long"].Current.Value
        atr = indicators["atr"].Current.Value
        bb_upper = indicators["bb"].UpperBand.Current.Value
        bb_lower = indicators["bb"].LowerBand.Current.Value
        bb_middle = indicators["bb"].MiddleBand.Current.Value
        rsi = indicators["rsi"].Current.Value
        momentum = indicators["momentum"].Current.Value
        
        current_holdings = self.Portfolio[symbol].Quantity
        
        # UPSIDE BREAKOUT CONDITIONS
        if (price > bb_upper and 
            sma_short > sma_long and 
            momentum > 0.005 and 
            rsi > 50 and rsi < 80 and
            current_holdings <= 0):
            
            # Enter long position
            position_size = 0.15  # Smaller positions for more trades
            self.SetHoldings(symbol, position_size)
            self.stop_losses[symbol] = price - (2 * atr)  # 2 ATR stop
            self.trade_count += 1
            
        # DOWNSIDE BREAKOUT CONDITIONS  
        elif (price < bb_lower and
              sma_short < sma_long and
              momentum < -0.005 and
              rsi < 50 and rsi > 20 and
              current_holdings >= 0):
              
            # Enter short position
            position_size = -0.15
            self.SetHoldings(symbol, position_size)
            self.stop_losses[symbol] = price + (2 * atr)  # 2 ATR stop
            self.trade_count += 1
            
        # PROFIT TAKING / STOP MANAGEMENT
        elif current_holdings != 0:
            # Long position management
            if current_holdings > 0:
                # Take profit at middle band or stop loss
                if price <= bb_middle or price <= self.stop_losses.get(symbol, 0):
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    if symbol in self.stop_losses:
                        del self.stop_losses[symbol]
                        
            # Short position management  
            elif current_holdings < 0:
                # Take profit at middle band or stop loss
                if price >= bb_middle or price >= self.stop_losses.get(symbol, float('inf')):
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    if symbol in self.stop_losses:
                        del self.stop_losses[symbol]
                        
    def OnData(self, data):
        # Additional intraday momentum trades
        for symbol in self.symbols:
            if symbol in data and self.indicators[symbol]["momentum"].IsReady:
                momentum = self.indicators[symbol]["momentum"].Current.Value
                current_holdings = self.Portfolio[symbol].Quantity
                
                # Quick momentum scalps
                if abs(momentum) > 0.01 and abs(current_holdings) < 0.05:
                    quick_position = 0.1 if momentum > 0 else -0.1
                    self.SetHoldings(symbol, quick_position)
                    self.trade_count += 1
                    
    def OnEndOfAlgorithm(self):
        years = (self.EndDate - self.StartDate).days / 365.25
        trades_per_year = self.trade_count / years
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
'''
            }
        ]
        
    def create_strategy_file(self, strategy_name: str, strategy_code: str) -> str:
        """Create a QuantConnect project with optimized strategy"""
        
        # Create new integration for unique project name
        qc = OptimizedQuantConnectIntegration()
        project_path = qc.create_lean_project()
        
        # Write the strategy code directly
        with open(f"{project_path}/main.py", "w") as f:
            f.write(strategy_code)
            
        return project_path
        
    def run_optimized_tests(self):
        """Run tests with strategies optimized for real trading"""
        
        print("🚀 OPTIMIZED STRATEGY PIPELINE")
        print("Requirements:")
        print("• CAGR > 25%")
        print("• Sharpe Ratio > 1.0") 
        print("• Max Drawdown < 20%")
        print("• Average Profit > 0.75%")
        print("• MINIMUM 100 TRADES PER YEAR")
        print("="*60)
        
        strategies = self.get_optimized_strategies()
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n📊 Testing Strategy {i}/{len(strategies)}: {strategy['name']}")
            print(f"Description: {strategy['description']}")
            print("-" * 60)
            
            try:
                # Create project with strategy
                project_path = self.create_strategy_file(
                    strategy['name'], 
                    strategy['strategy_code']
                )
                
                print(f"✅ Created project: {project_path}")
                print("🔄 Running backtest (this may take 2-3 minutes)...")
                
                # Run backtest using the lean workspace
                result = self.run_backtest_in_workspace(project_path)
                
                if "error" in result:
                    print(f"❌ Backtest failed: {result['error']}")
                    self.results.append({
                        "strategy": strategy['name'],
                        "status": "failed",
                        "error": result['error']
                    })
                    continue
                    
                # Evaluate results
                evaluation = self.qc_integration.evaluate_performance(result)
                
                # Print results
                print(f"\n📈 RESULTS for {strategy['name']}:")
                print(f"  CAGR: {result['cagr']:.2%}")
                print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
                print(f"  Total Trades: {result['total_trades']}")
                print(f"  Trades/Year: {result['trades_per_year']:.1f}")
                print(f"  Win Rate: {result['win_rate']:.2%}")
                
                if evaluation["meets_criteria"]:
                    print("🎉 ✅ MEETS ALL CRITERIA!")
                else:
                    print("❌ Does not meet criteria:")
                    for criterion, detail in evaluation['details'].items():
                        print(f"    • {criterion}: {detail}")
                        
                self.results.append({
                    "strategy": strategy['name'],
                    "metrics": result,
                    "evaluation": evaluation,
                    "status": "success" if evaluation['meets_criteria'] else "below_target",
                    "project_path": project_path
                })
                
            except Exception as e:
                print(f"❌ Error testing {strategy['name']}: {str(e)}")
                self.results.append({
                    "strategy": strategy['name'],
                    "status": "error", 
                    "error": str(e)
                })
                
            print("=" * 60)
            time.sleep(2)  # Brief pause between tests
            
        # Final summary
        self.print_final_summary()
        
    def run_backtest_in_workspace(self, project_path: str) -> Dict[str, Any]:
        """Run backtest in the configured lean workspace"""
        
        import subprocess
        import json
        from pathlib import Path
        
        # Copy strategy to lean workspace
        workspace_path = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
        project_name = Path(project_path).name
        
        # Copy the project to workspace
        subprocess.run(f"cp -r '{project_path}' '{workspace_path}/'", shell=True)
        
        # Run backtest
        cmd = f"cd '{workspace_path}' && lean backtest '{project_name}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"error": result.stderr}
            
        # Find results file
        results_pattern = f"{workspace_path}/{project_name}/backtests/*/1*-summary.json"
        import glob
        results_files = glob.glob(results_pattern)
        
        if results_files:
            with open(results_files[-1], 'r') as f:  # Get latest results
                backtest_results = json.load(f)
                
            # Parse the results
            stats = backtest_results.get("statistics", {})
            portfolio_stats = backtest_results.get("totalPerformance", {}).get("portfolioStatistics", {})
            
            # Calculate metrics
            total_return = float(portfolio_stats.get("totalNetProfit", "0")) / 100
            years = 4.0  # 2020-2023
            cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
            
            total_trades = int(stats.get("Total Orders", "0"))
            trades_per_year = total_trades / years
            
            metrics = {
                "cagr": cagr,
                "sharpe_ratio": float(stats.get("Sharpe Ratio", "0")),
                "max_drawdown": abs(float(stats.get("Drawdown", "0").rstrip('%')) / 100),
                "total_return": total_return,
                "win_rate": float(stats.get("Win Rate", "0").rstrip('%')) / 100,
                "total_trades": total_trades,
                "trades_per_year": trades_per_year,
                "avg_profit": float(stats.get("Average Win", "0").rstrip('%')) / 100
            }
            
            return metrics
        else:
            return {"error": "No results file found"}
            
    def print_final_summary(self):
        """Print final summary with trade frequency focus"""
        
        print("\n" + "="*80)
        print("🏆 FINAL OPTIMIZED STRATEGY RESULTS")
        print("="*80)
        
        successful = [r for r in self.results if r.get('status') == 'success']
        below_target = [r for r in self.results if r.get('status') == 'below_target']
        failed = [r for r in self.results if r.get('status') in ['failed', 'error']]
        
        print(f"\nTotal Strategies Tested: {len(self.results)}")
        print(f"✅ Meeting ALL Criteria: {len(successful)}")
        print(f"⚠️  Below Target: {len(below_target)}")  
        print(f"❌ Failed/Error: {len(failed)}")
        
        if successful:
            print("\n🎉 SUCCESSFUL STRATEGIES (Meeting All Criteria):")
            for r in successful:
                metrics = r['metrics']
                print(f"\n🏆 {r['strategy']}:")
                print(f"  📈 CAGR: {metrics['cagr']:.2%}")
                print(f"  📊 Sharpe: {metrics['sharpe_ratio']:.2f}")
                print(f"  📉 Max DD: {metrics['max_drawdown']:.2%}")
                print(f"  🔄 Trades/Year: {metrics['trades_per_year']:.1f}")
                print(f"  📁 Project: {r['project_path']}")
                
        if below_target:
            print(f"\n⚠️  BELOW TARGET STRATEGIES:")
            for r in below_target:
                if 'metrics' in r:
                    metrics = r['metrics']
                    print(f"\n{r['strategy']}:")
                    print(f"  CAGR: {metrics['cagr']:.2%} | Trades/Year: {metrics['trades_per_year']:.1f}")
                    
        # Save detailed results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tested": len(self.results),
            "successful": len(successful),
            "criteria": self.qc_integration.target_metrics,
            "results": self.results
        }
        
        with open("optimized_strategy_results.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n💾 Detailed results saved to: optimized_strategy_results.json")
        print("="*80)


def main():
    """Run optimized pipeline"""
    
    pipeline = OptimizedStrategyPipeline()
    pipeline.run_optimized_tests()
    

if __name__ == "__main__":
    main()