#!/usr/bin/env python3
"""
VERIFIED METRICS PARSER
Ensures local and cloud results are parsed identically
Cross-validates against QuantConnect cloud results
"""

import re
import json
from typing import Dict
from dataclasses import dataclass

@dataclass
class StrategyMetrics:
    cagr: float
    sharpe: float
    drawdown: float
    total_orders: int
    win_rate: float
    avg_win: float
    avg_loss: float
    net_profit: float
    
    def meets_criteria(self, targets: Dict) -> Dict:
        """Check which criteria are met"""
        return {
            "cagr": self.cagr > targets["cagr"],
            "sharpe": self.sharpe > targets["sharpe"], 
            "drawdown": self.drawdown < targets["max_drawdown"],
            "avg_profit": self.calculate_avg_profit() > targets["avg_profit"]
        }
    
    def calculate_avg_profit(self) -> float:
        """Calculate average profit per trade"""
        if self.total_orders == 0:
            return 0.0
        return (self.avg_win * self.win_rate + self.avg_loss * (1 - self.win_rate))
    
    def criteria_score(self, targets: Dict) -> int:
        """Count how many criteria are met"""
        criteria = self.meets_criteria(targets)
        return sum(criteria.values())

class VerifiedMetricsParser:
    """Parser that extracts metrics identically from local and cloud"""
    
    def parse_local_output(self, output: str) -> StrategyMetrics:
        """Parse local Lean CLI output"""
        metrics = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "total_orders": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "net_profit": 0.0
        }
        
        for line in output.split('\n'):
            if "STATISTICS::" in line:
                # Parse each statistic line
                if "Compounding Annual Return" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["cagr"] = float(match.group(1)) / 100
                        
                elif "Sharpe Ratio" in line and "Probabilistic" not in line:
                    match = re.search(r'(\d+\.?\d*)', line.split('::')[-1])
                    if match:
                        metrics["sharpe"] = float(match.group(1))
                        
                elif "Drawdown" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["drawdown"] = float(match.group(1)) / 100
                        
                elif "Total Orders" in line:
                    match = re.search(r'(\d+)', line.split('::')[-1])
                    if match:
                        metrics["total_orders"] = int(match.group(1))
                        
                elif "Win Rate" in line:
                    match = re.search(r'(\d+)%', line)
                    if match:
                        metrics["win_rate"] = float(match.group(1)) / 100
                        
                elif "Average Win" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["avg_win"] = float(match.group(1)) / 100
                        
                elif "Average Loss" in line:
                    match = re.search(r'-(\d+\.?\d*)%', line)
                    if match:
                        metrics["avg_loss"] = -float(match.group(1)) / 100
                        
                elif "Net Profit" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        metrics["net_profit"] = float(match.group(1)) / 100
        
        return StrategyMetrics(**metrics)
    
    def parse_cloud_output(self, output: str) -> StrategyMetrics:
        """Parse QuantConnect cloud output (same format as local)"""
        # Cloud and local use same STATISTICS format
        return self.parse_local_output(output)
    
    def verify_consistency(self, local_metrics: StrategyMetrics, cloud_metrics: StrategyMetrics) -> Dict:
        """Verify local and cloud results are consistent"""
        tolerance = 0.05  # 5% tolerance for minor differences
        
        checks = {
            "cagr_consistent": abs(local_metrics.cagr - cloud_metrics.cagr) < tolerance,
            "sharpe_consistent": abs(local_metrics.sharpe - cloud_metrics.sharpe) < tolerance,
            "drawdown_consistent": abs(local_metrics.drawdown - cloud_metrics.drawdown) < tolerance,
            "orders_consistent": abs(local_metrics.total_orders - cloud_metrics.total_orders) < (local_metrics.total_orders * tolerance)
        }
        
        return {
            "all_consistent": all(checks.values()),
            "details": checks,
            "local": local_metrics,
            "cloud": cloud_metrics
        }

# Target criteria
TARGET_CRITERIA = {
    "cagr": 0.25,           # >25%
    "sharpe": 1.0,          # >1.0
    "max_drawdown": 0.20,   # <20%
    "avg_profit": 0.0075    # >0.75%
}

def main():
    """Test the verified parser"""
    # Read the local output we just captured
    with open("/tmp/local_backtest_output.txt", 'r') as f:
        local_output = f.read()
    
    parser = VerifiedMetricsParser()
    local_metrics = parser.parse_local_output(local_output)
    
    print("=== VERIFIED LOCAL METRICS ===")
    print(f"CAGR: {local_metrics.cagr*100:.3f}%")
    print(f"Sharpe: {local_metrics.sharpe:.3f}")
    print(f"Drawdown: {local_metrics.drawdown*100:.3f}%")
    print(f"Total Orders: {local_metrics.total_orders}")
    print(f"Win Rate: {local_metrics.win_rate*100:.1f}%")
    print(f"Avg Win: {local_metrics.avg_win*100:.3f}%")
    print(f"Avg Loss: {local_metrics.avg_loss*100:.3f}%")
    print(f"Avg Profit: {local_metrics.calculate_avg_profit()*100:.3f}%")
    
    print("\n=== CRITERIA CHECK ===")
    criteria = local_metrics.meets_criteria(TARGET_CRITERIA)
    for criterion, met in criteria.items():
        status = "✅" if met else "❌"
        print(f"{criterion.upper()}: {status}")
    
    print(f"\nSCORE: {local_metrics.criteria_score(TARGET_CRITERIA)}/4")

if __name__ == "__main__":
    main()