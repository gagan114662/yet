#!/usr/bin/env python3
"""
QuantConnect Cloud Deployment Script
Deploys strategies from local Lean environment to QuantConnect Cloud for professional backtesting
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Configuration for a strategy deployment"""
    name: str
    path: str
    description: str
    parameters: Dict
    
@dataclass
class BacktestResult:
    """Results from a cloud backtest"""
    strategy_name: str
    backtest_id: str
    status: str
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    net_profit: float

class QuantConnectCloudDeployer:
    """Handles deployment of strategies to QuantConnect Cloud"""
    
    def __init__(self, user_id: str, api_token: str, organization_id: str = None):
        self.user_id = user_id
        self.api_token = api_token
        self.organization_id = organization_id or user_id
        self.api_base = "https://www.quantconnect.com/api/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
    def create_project(self, name: str, description: str = "") -> str:
        """Create a new project in QuantConnect Cloud"""
        endpoint = f"{self.api_base}/projects/create"
        
        data = {
            "name": name,
            "description": description,
            "language": "Python"
        }
        
        response = requests.post(endpoint, json=data, headers=self.headers)
        response.raise_for_status()
        
        result = response.json()
        project_id = result["projects"][0]["projectId"]
        logger.info(f"Created project '{name}' with ID: {project_id}")
        return project_id
        
    def upload_file(self, project_id: str, file_name: str, content: str) -> bool:
        """Upload a file to the project"""
        endpoint = f"{self.api_base}/files/create"
        
        data = {
            "projectId": project_id,
            "name": file_name,
            "content": content
        }
        
        response = requests.post(endpoint, json=data, headers=self.headers)
        response.raise_for_status()
        
        logger.info(f"Uploaded file '{file_name}' to project {project_id}")
        return True
        
    def create_backtest(self, project_id: str, name: str, parameters: Dict = None) -> str:
        """Create and run a backtest"""
        endpoint = f"{self.api_base}/backtests/create"
        
        # Default parameters for aggressive performance
        default_params = {
            "startDate": "2018-01-01",
            "endDate": "2023-12-31",
            "cashAmount": 100000,
            "periodFinish": "Completed",
            "dataResolution": "Hour",
            "brokerage": "InteractiveBrokersBrokerage"
        }
        
        if parameters:
            default_params.update(parameters)
            
        data = {
            "projectId": project_id,
            "compileId": "",
            "backtestName": name,
            **default_params
        }
        
        response = requests.post(endpoint, json=data, headers=self.headers)
        response.raise_for_status()
        
        result = response.json()
        backtest_id = result["backtestId"]
        logger.info(f"Started backtest '{name}' with ID: {backtest_id}")
        return backtest_id
        
    def get_backtest_status(self, project_id: str, backtest_id: str) -> Dict:
        """Get the status of a backtest"""
        endpoint = f"{self.api_base}/backtests/read"
        
        params = {
            "projectId": project_id,
            "backtestId": backtest_id
        }
        
        response = requests.get(endpoint, params=params, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
        
    def wait_for_backtest(self, project_id: str, backtest_id: str, timeout: int = 600) -> Dict:
        """Wait for a backtest to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_backtest_status(project_id, backtest_id)
            
            if status["completed"]:
                logger.info(f"Backtest {backtest_id} completed")
                return status
                
            logger.info(f"Backtest {backtest_id} progress: {status.get('progress', 0)}%")
            time.sleep(10)
            
        raise TimeoutError(f"Backtest {backtest_id} timed out after {timeout} seconds")
        
    def deploy_strategy(self, strategy: StrategyConfig) -> Tuple[str, str]:
        """Deploy a single strategy to the cloud"""
        logger.info(f"Deploying strategy: {strategy.name}")
        
        # Create project
        project_name = f"{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_id = self.create_project(project_name, strategy.description)
        
        # Read strategy file
        with open(strategy.path, 'r') as f:
            strategy_content = f.read()
            
        # Convert strategy to cloud format
        cloud_content = self.convert_to_cloud_format(strategy_content, strategy.name)
        
        # Upload main.py
        self.upload_file(project_id, "main.py", cloud_content)
        
        # Create and run backtest
        backtest_id = self.create_backtest(project_id, strategy.name, strategy.parameters)
        
        return project_id, backtest_id
        
    def convert_to_cloud_format(self, content: str, class_name: str) -> str:
        """Convert local Lean strategy to QuantConnect cloud format"""
        # Add cloud-specific imports if needed
        cloud_imports = """# QuantConnect Cloud Version
from AlgorithmImports import *
import numpy as np
from datetime import timedelta

"""
        
        # Replace local-specific code patterns
        content = content.replace("from AlgorithmImports import *", "")
        content = cloud_imports + content
        
        # Ensure proper class naming
        if f"class {class_name}" not in content:
            # Find the class definition and rename it
            import re
            content = re.sub(r'class \w+\(QCAlgorithm\):', f'class {class_name}(QCAlgorithm):', content)
            
        return content
        
    def analyze_results(self, project_id: str, backtest_id: str) -> BacktestResult:
        """Analyze backtest results"""
        status = self.get_backtest_status(project_id, backtest_id)
        
        # Extract key metrics
        stats = status.get("statistics", {})
        
        result = BacktestResult(
            strategy_name=status.get("name", "Unknown"),
            backtest_id=backtest_id,
            status="Completed" if status["completed"] else "Running",
            sharpe_ratio=float(stats.get("SharpeRatio", 0)),
            cagr=float(stats.get("CompoundingAnnualReturn", 0)),
            max_drawdown=float(stats.get("Drawdown", 0)),
            total_trades=int(stats.get("TotalOrders", 0)),
            win_rate=float(stats.get("WinRate", 0)),
            net_profit=float(stats.get("NetProfit", 0))
        )
        
        return result

def load_strategies_config() -> List[StrategyConfig]:
    """Load strategy configurations for deployment"""
    strategies = [
        StrategyConfig(
            name="SuperAggressiveMomentum",
            path="/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/super_aggressive_momentum/main.py",
            description="Enhanced momentum strategy with dynamic leverage targeting 25%+ CAGR",
            parameters={
                "startDate": "2018-01-01",
                "endDate": "2023-12-31",
                "cashAmount": 100000
            }
        ),
        StrategyConfig(
            name="AggressiveSPYMomentum",
            path="/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/rd_agent_strategy_20250530_171329_076469/main.py",
            description="Best performing local strategy with 33% returns",
            parameters={
                "startDate": "2020-01-01",
                "endDate": "2023-12-31",
                "cashAmount": 100000
            }
        ),
        StrategyConfig(
            name="TargetCrusherUltimate",
            path="/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/target_crusher_ultimate/main.py",
            description="Multi-timeframe momentum and mean reversion with 3x leveraged ETFs",
            parameters={
                "startDate": "2015-01-01",
                "endDate": "2023-12-31",
                "cashAmount": 100000
            }
        ),
        StrategyConfig(
            name="ExtremePerformance2025",
            path="/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/extreme_performance_2025/main.py",
            description="Aggressive momentum and mean reversion with 3x leveraged ETFs",
            parameters={
                "startDate": "2015-01-01",
                "endDate": "2023-12-31",
                "cashAmount": 100000
            }
        ),
        StrategyConfig(
            name="MasterStrategyRotator",
            path="/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/master_strategy_rotator/main.py",
            description="Meta-strategy with intelligent rotation between 5 sub-strategies",
            parameters={
                "startDate": "2015-01-01",
                "endDate": "2023-12-31",
                "cashAmount": 100000
            }
        )
    ]
    
    # Check which strategies exist
    available_strategies = []
    for strategy in strategies:
        if Path(strategy.path).exists():
            available_strategies.append(strategy)
        else:
            logger.warning(f"Strategy file not found: {strategy.path}")
            
    return available_strategies

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy strategies to QuantConnect Cloud")
    parser.add_argument("--user-id", required=True, help="QuantConnect User ID")
    parser.add_argument("--api-token", required=True, help="QuantConnect API Token")
    parser.add_argument("--organization-id", help="Organization ID (defaults to user ID)")
    parser.add_argument("--strategy", help="Deploy specific strategy by name")
    parser.add_argument("--all", action="store_true", help="Deploy all available strategies")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = QuantConnectCloudDeployer(
        user_id=args.user_id,
        api_token=args.api_token,
        organization_id=args.organization_id
    )
    
    # Load strategies
    strategies = load_strategies_config()
    
    if args.strategy:
        # Deploy specific strategy
        strategies = [s for s in strategies if s.name == args.strategy]
        if not strategies:
            logger.error(f"Strategy '{args.strategy}' not found")
            return
    elif not args.all:
        # Deploy only the best performing strategies
        strategies = strategies[:3]
        
    # Deploy strategies
    results = []
    for strategy in strategies:
        try:
            project_id, backtest_id = deployer.deploy_strategy(strategy)
            
            # Wait for completion
            deployer.wait_for_backtest(project_id, backtest_id)
            
            # Analyze results
            result = deployer.analyze_results(project_id, backtest_id)
            results.append(result)
            
            logger.info(f"Strategy: {result.strategy_name}")
            logger.info(f"  CAGR: {result.cagr:.2%}")
            logger.info(f"  Sharpe: {result.sharpe_ratio:.2f}")
            logger.info(f"  Max DD: {result.max_drawdown:.2%}")
            logger.info(f"  Win Rate: {result.win_rate:.2%}")
            logger.info(f"  Net Profit: {result.net_profit:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to deploy {strategy.name}: {str(e)}")
            
    # Save results
    results_file = f"cloud_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
        
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n=== DEPLOYMENT SUMMARY ===")
    print(f"Deployed {len(results)} strategies")
    
    if results:
        best_cagr = max(results, key=lambda r: r.cagr)
        best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
        
        print(f"\nBest CAGR: {best_cagr.strategy_name} ({best_cagr.cagr:.2%})")
        print(f"Best Sharpe: {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.2f})")
        
        # Check if any met targets
        target_met = [r for r in results if r.cagr > 0.25 and r.sharpe_ratio > 1.0 and r.max_drawdown < 0.20]
        if target_met:
            print(f"\n🎯 STRATEGIES MEETING ALL TARGETS:")
            for r in target_met:
                print(f"  - {r.strategy_name}: CAGR={r.cagr:.2%}, Sharpe={r.sharpe_ratio:.2f}")

if __name__ == "__main__":
    main()