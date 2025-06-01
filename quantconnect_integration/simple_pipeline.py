#!/usr/bin/env python3
"""
Simplified Pipeline: Direct QuantConnect Strategy Development
This version runs without the full RD-Agent framework
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

class SimpleStrategyPipeline:
    """Simplified pipeline for strategy development"""
    
    def __init__(self):
        self.qc_integration = QuantConnectIntegration()
        self.results = []
        
    def run_demo(self):
        """Run a demonstration with pre-defined strategies"""
        
        logger.info("Starting simplified strategy development pipeline")
        logger.info(f"Target metrics: CAGR>25%, Sharpe>1.0, MaxDD<20%, AvgProfit>0.75%")
        
        # Define test strategies
        test_strategies = [
            {
                "name": "Momentum_Cross_Strategy",
                "description": "Dual momentum crossover with risk management",
                "type": "momentum",
                "start_date": "2021,1,1",
                "end_date": "2023,12,31",
                "lookback_period": 20,
                "rebalance_frequency": 5,
                "position_size": 0.1,
                "universe_size": 30,
                "min_price": 10,
                "min_volume": 5000000,
                "indicator_setup": '''
                "fast_momentum": self.MOMP(symbol, 10),
                "slow_momentum": self.MOMP(symbol, 30),
                "rsi": self.RSI(symbol, 14),
                "atr": self.ATR(symbol, 14)
                ''',
                "signal_generation_logic": '''
        indicators = self.indicators[symbol]
        fast_mom = indicators["fast_momentum"].Current.Value
        slow_mom = indicators["slow_momentum"].Current.Value
        rsi = indicators["rsi"].Current.Value
        atr = indicators["atr"].Current.Value
        
        # Momentum crossover with RSI filter
        if fast_mom > slow_mom and fast_mom > 0.02 and rsi < 70:
            signal = 1  # Buy signal
        elif fast_mom < slow_mom or rsi > 80:
            signal = -1  # Sell signal
        else:
            signal = 0  # Hold
        '''
            },
            {
                "name": "Mean_Reversion_BB",
                "description": "Bollinger Bands mean reversion strategy",
                "type": "mean_reversion",
                "start_date": "2021,1,1",
                "end_date": "2023,12,31",
                "lookback_period": 20,
                "rebalance_frequency": 3,
                "position_size": 0.05,
                "universe_size": 50,
                "min_price": 5,
                "min_volume": 2000000,
                "indicator_setup": '''
                "bb": self.BB(symbol, 20, 2),
                "volume": self.V(symbol, 20),
                "rsi": self.RSI(symbol, 14)
                ''',
                "signal_generation_logic": '''
        indicators = self.indicators[symbol]
        price = self.Securities[symbol].Price
        upper_band = indicators["bb"].UpperBand.Current.Value
        lower_band = indicators["bb"].LowerBand.Current.Value
        middle_band = indicators["bb"].MiddleBand.Current.Value
        rsi = indicators["rsi"].Current.Value
        
        # Mean reversion with RSI confirmation
        if price < lower_band and rsi < 30:
            signal = 1  # Oversold - Buy
        elif price > upper_band and rsi > 70:
            signal = -1  # Overbought - Sell
        elif abs(price - middle_band) / middle_band < 0.01:
            signal = 0  # Near mean - Exit
        else:
            signal = 0  # Hold current position
        '''
            },
            {
                "name": "Quality_Factor_Strategy",
                "description": "Quality stocks with momentum filter",
                "type": "factor",
                "start_date": "2021,1,1", 
                "end_date": "2023,12,31",
                "lookback_period": 60,
                "rebalance_frequency": 20,
                "position_size": 0.02,
                "universe_size": 100,
                "min_price": 15,
                "min_volume": 10000000,
                "indicator_setup": '''
                "momentum": self.MOMP(symbol, 60),
                "volatility": self.STD(symbol, 60),
                "volume_ratio": self.V(symbol, 5)
                ''',
                "signal_generation_logic": '''
        # Quality factor with momentum overlay
        fundamentals = self.Securities[symbol].Fundamentals
        indicators = self.indicators[symbol]
        
        if fundamentals:
            # Quality metrics
            roe = fundamentals.OperationRatios.ROE.Value if hasattr(fundamentals.OperationRatios, 'ROE') else 0
            debt_to_equity = fundamentals.OperationRatios.DebttoEquityRatio.Value if hasattr(fundamentals.OperationRatios, 'DebttoEquityRatio') else 999
            profit_margin = fundamentals.OperationRatios.NetMargin.Value if hasattr(fundamentals.OperationRatios, 'NetMargin') else 0
            
            # Technical overlay
            momentum = indicators["momentum"].Current.Value
            
            # Quality score
            quality_score = 0
            if roe > 0.15:  # ROE > 15%
                quality_score += 1
            if debt_to_equity < 0.5 and debt_to_equity > 0:  # Low debt
                quality_score += 1
            if profit_margin > 0.10:  # Profit margin > 10%
                quality_score += 1
                
            # Final signal combining quality and momentum
            if quality_score >= 2 and momentum > 0:
                signal = 1
            else:
                signal = 0
        else:
            signal = 0
        '''
            }
        ]
        
        # Test each strategy
        for i, strategy in enumerate(test_strategies):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing strategy {i+1}/{len(test_strategies)}: {strategy['name']}")
            logger.info(f"{'='*60}")
            
            try:
                # Create new integration instance for each strategy to avoid name conflicts
                qc = QuantConnectIntegration()
                
                # Create project
                project_path = qc.create_lean_project()
                logger.info(f"Created project: {project_path}")
                
                # Generate strategy code
                strategy_code = qc.generate_strategy_code(strategy)
                
                # Save strategy code
                with open(f"{project_path}/strategy_code.txt", 'w') as f:
                    f.write(strategy_code)
                
                logger.info("Generated strategy code")
                logger.info("Running backtest... (this may take a few minutes)")
                
                # Run backtest
                metrics = qc.run_backtest(strategy_code, project_path)
                
                if "error" in metrics:
                    logger.error(f"Backtest failed: {metrics['error']}")
                    self.results.append({
                        "strategy": strategy['name'],
                        "status": "failed",
                        "error": metrics['error']
                    })
                    continue
                
                # Evaluate performance
                evaluation = qc.evaluate_performance(metrics)
                
                # Generate report
                report = qc.generate_report(strategy, metrics, evaluation)
                
                # Save report
                with open(f"{project_path}/report.md", 'w') as f:
                    f.write(report)
                
                # Log results
                logger.info(f"\nResults for {strategy['name']}:")
                logger.info(f"CAGR: {metrics['cagr']:.2%}")
                logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
                logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
                logger.info(f"Meets Criteria: {'✅ YES' if evaluation['meets_criteria'] else '❌ NO'}")
                
                # Store results
                self.results.append({
                    "strategy": strategy['name'],
                    "metrics": metrics,
                    "evaluation": evaluation,
                    "status": "success" if evaluation['meets_criteria'] else "below_target",
                    "project_path": project_path
                })
                
                # Print report
                print("\n" + "="*60)
                print(report)
                print("="*60 + "\n")
                
            except Exception as e:
                logger.error(f"Error testing {strategy['name']}: {str(e)}")
                self.results.append({
                    "strategy": strategy['name'],
                    "status": "error",
                    "error": str(e)
                })
            
            # Small delay between strategies
            time.sleep(2)
        
        # Final summary
        self.print_final_summary()
        
    def print_final_summary(self):
        """Print final summary of all tested strategies"""
        
        print("\n" + "="*80)
        print("FINAL SUMMARY - STRATEGY DEVELOPMENT RESULTS")
        print("="*80)
        
        successful = [r for r in self.results if r.get('status') == 'success']
        below_target = [r for r in self.results if r.get('status') == 'below_target']
        failed = [r for r in self.results if r.get('status') in ['failed', 'error']]
        
        print(f"\nTotal Strategies Tested: {len(self.results)}")
        print(f"✅ Successful (meets all criteria): {len(successful)}")
        print(f"⚠️  Below Target: {len(below_target)}")
        print(f"❌ Failed/Error: {len(failed)}")
        
        if successful:
            print("\n🏆 SUCCESSFUL STRATEGIES:")
            for r in successful:
                metrics = r['metrics']
                print(f"\n{r['strategy']}:")
                print(f"  - CAGR: {metrics['cagr']:.2%}")
                print(f"  - Sharpe: {metrics['sharpe_ratio']:.2f}")
                print(f"  - Max DD: {metrics['max_drawdown']:.2%}")
                print(f"  - Win Rate: {metrics['win_rate']:.2%}")
                print(f"  - Project: {r['project_path']}")
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tested": len(self.results),
            "successful": len(successful),
            "results": self.results
        }
        
        with open("strategy_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nDetailed results saved to: strategy_summary.json")
        print("="*80)


def main():
    """Main entry point"""
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     QuantConnect Automated Strategy Development Pipeline       ║
║                                                               ║
║  This will test several pre-defined trading strategies        ║
║  against your performance targets:                            ║
║  • CAGR > 25%                                                ║
║  • Sharpe Ratio > 1.0                                        ║
║  • Max Drawdown < 20%                                        ║
║  • Average Profit > 0.75%                                     ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    pipeline = SimpleStrategyPipeline()
    pipeline.run_demo()


if __name__ == "__main__":
    main()