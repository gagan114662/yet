#!/usr/bin/env python3
"""
Batch Deployment Script for QuantConnect Cloud
Deploys multiple strategies in parallel and manages the deployment pipeline
"""

import asyncio
import json
import time
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import argparse

from cloud_deployment import QuantConnectCloudDeployer, StrategyConfig, BacktestResult
from strategy_converter import StrategyConverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DeploymentPlan:
    """Deployment plan for batch processing"""
    strategy_name: str
    variations: List[str]
    priority: int
    expected_performance: Dict[str, float]
    
@dataclass
class BatchResults:
    """Results from batch deployment"""
    total_strategies: int
    successful_deployments: int
    failed_deployments: int
    best_performer: Optional[BacktestResult]
    target_achievers: List[BacktestResult]
    all_results: List[BacktestResult]
    deployment_time: float

class BatchDeployer:
    """Manages batch deployment of multiple strategies"""
    
    def __init__(self, user_id: str, api_token: str, organization_id: str = None):
        self.deployer = QuantConnectCloudDeployer(user_id, api_token, organization_id)
        self.converter = StrategyConverter()
        self.results = []
        self.failed_deployments = []
        
    def create_deployment_plans(self) -> List[DeploymentPlan]:
        """Create deployment plans for all strategies"""
        plans = [
            DeploymentPlan(
                strategy_name="AggressiveSPYMomentum",
                variations=["Conservative", "Aggressive", "UltraAggressive"],
                priority=1,  # Highest priority - proven performer
                expected_performance={
                    "Conservative": {"cagr": 0.15, "sharpe": 0.8, "max_dd": 0.15},
                    "Aggressive": {"cagr": 0.25, "sharpe": 1.0, "max_dd": 0.18},
                    "UltraAggressive": {"cagr": 0.35, "sharpe": 1.2, "max_dd": 0.20}
                }
            ),
            DeploymentPlan(
                strategy_name="SuperAggressiveMomentum",
                variations=["Conservative", "Aggressive", "UltraAggressive"],
                priority=2,
                expected_performance={
                    "Conservative": {"cagr": 0.20, "sharpe": 0.9, "max_dd": 0.16},
                    "Aggressive": {"cagr": 0.30, "sharpe": 1.1, "max_dd": 0.19},
                    "UltraAggressive": {"cagr": 0.40, "sharpe": 1.3, "max_dd": 0.22}
                }
            ),
            DeploymentPlan(
                strategy_name="MasterStrategyRotator",
                variations=["Aggressive", "UltraAggressive"],
                priority=3,
                expected_performance={
                    "Aggressive": {"cagr": 0.35, "sharpe": 1.5, "max_dd": 0.18},
                    "UltraAggressive": {"cagr": 0.50, "sharpe": 1.8, "max_dd": 0.20}
                }
            ),
            DeploymentPlan(
                strategy_name="CrisisAlphaHarvester",
                variations=["Aggressive", "UltraAggressive"],
                priority=4,
                expected_performance={
                    "Aggressive": {"cagr": 0.30, "sharpe": 1.2, "max_dd": 0.20},
                    "UltraAggressive": {"cagr": 0.45, "sharpe": 1.5, "max_dd": 0.22}
                }
            ),
            DeploymentPlan(
                strategy_name="GammaScalperPro",
                variations=["Aggressive", "UltraAggressive"],
                priority=5,
                expected_performance={
                    "Aggressive": {"cagr": 0.40, "sharpe": 1.6, "max_dd": 0.18},
                    "UltraAggressive": {"cagr": 0.55, "sharpe": 2.0, "max_dd": 0.20}
                }
            )
        ]
        
        return plans
        
    def prepare_strategies(self, plans: List[DeploymentPlan]) -> Dict[str, StrategyConfig]:
        """Prepare all strategy variations for deployment"""
        prepared_strategies = {}
        
        for plan in plans:
            base_path = f"/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
            
            # Find base strategy file
            potential_paths = [
                f"{base_path}/{plan.strategy_name.lower()}/main.py",
                f"{base_path}/rd_agent_strategy_20250530_171329_076469/main.py" if "AggressiveSPY" in plan.strategy_name else None,
                f"{base_path}/super_aggressive_momentum/main.py",
                f"{base_path}/master_strategy_rotator/main.py",
                f"{base_path}/crisis_alpha_harvester/main.py",
                f"{base_path}/gamma_scalper_pro/main.py"
            ]
            
            base_file = None
            for path in potential_paths:
                if path and Path(path).exists():
                    base_file = path
                    break
                    
            if not base_file:
                # Use the best performing strategy as fallback
                base_file = f"{base_path}/rd_agent_strategy_20250530_171329_076469/main.py"
                logger.warning(f"Using fallback strategy for {plan.strategy_name}")
                
            # Create variations
            for variation in plan.variations:
                strategy_name = f"{plan.strategy_name}_{variation}"
                output_path = f"/tmp/cloud_strategies/{strategy_name}.py"
                
                # Convert and optimize strategy
                content = self.converter.convert_strategy(base_file, output_path, strategy_name)
                
                # Apply variation-specific optimizations
                if variation == "Conservative":
                    content = self.converter._create_conservative_version(content)
                elif variation == "Aggressive":
                    content = self.converter._create_aggressive_version(content)
                elif variation == "UltraAggressive":
                    content = self.converter._create_ultra_aggressive_version(content)
                    
                # Save optimized version
                with open(output_path, 'w') as f:
                    f.write(content)
                    
                # Create strategy config
                strategy_config = StrategyConfig(
                    name=strategy_name,
                    path=output_path,
                    description=f"{plan.strategy_name} - {variation} variation targeting 25%+ CAGR",
                    parameters={
                        "startDate": "2018-01-01",
                        "endDate": "2023-12-31",
                        "cashAmount": 100000,
                        "dataResolution": "Minute",
                        "brokerage": "InteractiveBrokersBrokerage"
                    }
                )
                
                prepared_strategies[strategy_name] = strategy_config
                
        return prepared_strategies
        
    def deploy_strategy_batch(self, strategies: Dict[str, StrategyConfig], 
                            max_concurrent: int = 5) -> List[BacktestResult]:
        """Deploy strategies in batches with concurrency control"""
        results = []
        strategy_items = list(strategies.items())
        
        # Deploy in batches
        for i in range(0, len(strategy_items), max_concurrent):
            batch = strategy_items[i:i + max_concurrent]
            batch_results = self._deploy_concurrent_batch(batch)
            results.extend(batch_results)
            
            # Wait between batches to avoid rate limiting
            if i + max_concurrent < len(strategy_items):
                logger.info(f"Waiting 30 seconds before next batch...")
                time.sleep(30)
                
        return results
        
    def _deploy_concurrent_batch(self, batch: List[Tuple[str, StrategyConfig]]) -> List[BacktestResult]:
        """Deploy a batch of strategies concurrently"""
        batch_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            # Submit all deployments
            future_to_strategy = {
                executor.submit(self._deploy_single_strategy, name, config): name 
                for name, config in batch
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        logger.info(f"✅ {strategy_name}: CAGR={result.cagr:.1%}, Sharpe={result.sharpe_ratio:.2f}")
                    else:
                        logger.error(f"❌ {strategy_name}: Deployment failed")
                        self.failed_deployments.append(strategy_name)
                except Exception as e:
                    logger.error(f"❌ {strategy_name}: Exception during deployment: {str(e)}")
                    self.failed_deployments.append(strategy_name)
                    
        return batch_results
        
    def _deploy_single_strategy(self, name: str, config: StrategyConfig) -> Optional[BacktestResult]:
        """Deploy a single strategy and return results"""
        try:
            logger.info(f"🚀 Deploying {name}...")
            
            # Deploy strategy
            project_id, backtest_id = self.deployer.deploy_strategy(config)
            
            # Wait for completion with timeout
            self.deployer.wait_for_backtest(project_id, backtest_id, timeout=900)  # 15 minutes
            
            # Get results
            result = self.deployer.analyze_results(project_id, backtest_id)
            return result
            
        except Exception as e:
            logger.error(f"Failed to deploy {name}: {str(e)}")
            return None
            
    def analyze_batch_results(self, results: List[BacktestResult]) -> BatchResults:
        """Analyze results from batch deployment"""
        if not results:
            return BatchResults(
                total_strategies=0,
                successful_deployments=0,
                failed_deployments=len(self.failed_deployments),
                best_performer=None,
                target_achievers=[],
                all_results=[],
                deployment_time=0
            )
            
        # Find best performers
        best_cagr = max(results, key=lambda r: r.cagr)
        best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
        
        # Find strategies meeting targets
        target_achievers = [
            r for r in results 
            if r.cagr >= 0.25 and r.sharpe_ratio >= 1.0 and r.max_drawdown <= 0.20
        ]
        
        # Sort target achievers by performance score
        target_achievers.sort(key=lambda r: r.cagr * r.sharpe_ratio, reverse=True)
        
        batch_results = BatchResults(
            total_strategies=len(results) + len(self.failed_deployments),
            successful_deployments=len(results),
            failed_deployments=len(self.failed_deployments),
            best_performer=best_cagr if best_cagr.cagr > best_sharpe.cagr else best_sharpe,
            target_achievers=target_achievers,
            all_results=results,
            deployment_time=0  # Will be set by caller
        )
        
        return batch_results
        
    def generate_report(self, batch_results: BatchResults) -> str:
        """Generate comprehensive deployment report"""
        report = f"""
# QuantConnect Cloud Deployment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Strategies Deployed: {batch_results.total_strategies}
- Successful Deployments: {batch_results.successful_deployments}
- Failed Deployments: {batch_results.failed_deployments}
- Success Rate: {batch_results.successful_deployments/batch_results.total_strategies*100:.1f}%

## 🎯 STRATEGIES MEETING ALL TARGETS (CAGR>25%, Sharpe>1.0, MaxDD<20%)
"""
        
        if batch_results.target_achievers:
            report += f"Found {len(batch_results.target_achievers)} strategies meeting all targets!\n\n"
            for i, result in enumerate(batch_results.target_achievers[:5], 1):
                report += f"{i}. **{result.strategy_name}**\n"
                report += f"   - CAGR: {result.cagr:.1%}\n"
                report += f"   - Sharpe Ratio: {result.sharpe_ratio:.2f}\n"
                report += f"   - Max Drawdown: {result.max_drawdown:.1%}\n"
                report += f"   - Win Rate: {result.win_rate:.1%}\n"
                report += f"   - Total Trades: {result.total_trades}\n\n"
        else:
            report += "❌ No strategies met all targets. Best performers:\n\n"
            
        # Top performers by different metrics
        if batch_results.all_results:
            top_cagr = sorted(batch_results.all_results, key=lambda r: r.cagr, reverse=True)[:3]
            top_sharpe = sorted(batch_results.all_results, key=lambda r: r.sharpe_ratio, reverse=True)[:3]
            
            report += "## 📈 Top CAGR Performers\n"
            for i, result in enumerate(top_cagr, 1):
                report += f"{i}. {result.strategy_name}: {result.cagr:.1%}\n"
                
            report += "\n## 📊 Top Sharpe Ratio Performers\n"
            for i, result in enumerate(top_sharpe, 1):
                report += f"{i}. {result.strategy_name}: {result.sharpe_ratio:.2f}\n"
                
        # Failed deployments
        if self.failed_deployments:
            report += f"\n## ❌ Failed Deployments ({len(self.failed_deployments)})\n"
            for failed in self.failed_deployments:
                report += f"- {failed}\n"
                
        # Recommendations
        report += "\n## 🚀 Next Steps\n"
        if batch_results.target_achievers:
            report += f"1. **Deploy the top {min(3, len(batch_results.target_achievers))} target-achieving strategies to live trading**\n"
            report += "2. Monitor performance closely with 5% daily loss limits\n"
            report += "3. Scale up capital allocation gradually\n"
        else:
            report += "1. **Optimize the best performing strategies further**\n"
            report += "2. **Consider ensemble methods combining top performers**\n"
            report += "3. **Test with higher leverage/frequency in controlled manner**\n"
            
        return report

def main():
    """Main batch deployment function"""
    parser = argparse.ArgumentParser(description="Batch deploy strategies to QuantConnect Cloud")
    parser.add_argument("--user-id", required=True, help="QuantConnect User ID")
    parser.add_argument("--api-token", required=True, help="QuantConnect API Token")
    parser.add_argument("--organization-id", help="Organization ID")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent deployments")
    parser.add_argument("--priority-only", action="store_true", help="Deploy only priority 1-2 strategies")
    
    args = parser.parse_args()
    
    # Initialize batch deployer
    batch_deployer = BatchDeployer(args.user_id, args.api_token, args.organization_id)
    
    # Create deployment plans
    plans = batch_deployer.create_deployment_plans()
    
    if args.priority_only:
        plans = [p for p in plans if p.priority <= 2]
        logger.info("Deploying only priority 1-2 strategies")
    
    # Prepare strategies
    logger.info("Preparing strategies for deployment...")
    strategies = batch_deployer.prepare_strategies(plans)
    logger.info(f"Prepared {len(strategies)} strategy variations")
    
    # Deploy in batches
    start_time = time.time()
    logger.info(f"Starting batch deployment of {len(strategies)} strategies...")
    
    results = batch_deployer.deploy_strategy_batch(strategies, args.max_concurrent)
    
    deployment_time = time.time() - start_time
    
    # Analyze results
    batch_results = batch_deployer.analyze_batch_results(results)
    batch_results.deployment_time = deployment_time
    
    # Generate report
    report = batch_deployer.generate_report(batch_results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = f"batch_deployment_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'batch_results': asdict(batch_results),
            'individual_results': [asdict(r) for r in results]
        }, f, indent=2)
        
    # Save report
    report_file = f"deployment_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report)
        
    # Print summary
    print(report)
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    # Print immediate action items
    if batch_results.target_achievers:
        print(f"\n🎉 SUCCESS! {len(batch_results.target_achievers)} strategies met your targets!")
        print("Top performer:", batch_results.target_achievers[0].strategy_name)
        print(f"CAGR: {batch_results.target_achievers[0].cagr:.1%}")
        print(f"Sharpe: {batch_results.target_achievers[0].sharpe_ratio:.2f}")
    else:
        print("\n⚠️  No strategies met all targets. Consider optimization or ensemble methods.")

if __name__ == "__main__":
    main()