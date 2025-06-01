#!/usr/bin/env python3
"""
QuantConnect Cloud Results Analyzer
Advanced analysis and monitoring of cloud backtest results
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    strategy_name: str
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    volatility: float
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk 95%
    target_score: float  # Composite score for target achievement

@dataclass
class RiskMetrics:
    """Risk analysis metrics"""
    strategy_name: str
    var_99: float
    cvar_95: float  # Conditional VaR
    maximum_consecutive_losses: int
    tail_ratio: float
    downside_deviation: float
    beta: float
    tracking_error: float
    information_ratio: float

class CloudResultsAnalyzer:
    """Analyzes QuantConnect cloud backtest results"""
    
    def __init__(self):
        self.results_data = []
        self.performance_metrics = []
        self.risk_metrics = []
        
    def load_results(self, results_file: str) -> None:
        """Load results from JSON file"""
        with open(results_file, 'r') as f:
            data = json.load(f)
            
        if 'individual_results' in data:
            self.results_data = data['individual_results']
        else:
            self.results_data = data
            
        logger.info(f"Loaded {len(self.results_data)} strategy results")
        
    def calculate_performance_metrics(self) -> List[PerformanceMetrics]:
        """Calculate comprehensive performance metrics"""
        metrics = []
        
        for result in self.results_data:
            # Basic metrics from backtest
            cagr = result.get('cagr', 0)
            sharpe = result.get('sharpe_ratio', 0)
            max_dd = result.get('max_drawdown', 0)
            win_rate = result.get('win_rate', 0)
            total_trades = result.get('total_trades', 0)
            
            # Calculate additional metrics
            sortino_ratio = self._calculate_sortino_ratio(result)
            calmar_ratio = cagr / abs(max_dd) if max_dd != 0 else 0
            profit_factor = self._calculate_profit_factor(result)
            volatility = self._estimate_volatility(cagr, sharpe)
            
            # Risk metrics
            var_95 = self._calculate_var(result, 0.95)
            
            # Target achievement score
            target_score = self._calculate_target_score(cagr, sharpe, max_dd, win_rate)
            
            metric = PerformanceMetrics(
                strategy_name=result.get('strategy_name', 'Unknown'),
                cagr=cagr,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_dd,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                avg_trade_duration=self._estimate_trade_duration(total_trades),
                volatility=volatility,
                skewness=self._estimate_skewness(result),
                kurtosis=self._estimate_kurtosis(result),
                var_95=var_95,
                target_score=target_score
            )
            
            metrics.append(metric)
            
        self.performance_metrics = metrics
        return metrics
        
    def calculate_risk_metrics(self) -> List[RiskMetrics]:
        """Calculate detailed risk metrics"""
        risk_metrics = []
        
        for result in self.results_data:
            var_99 = self._calculate_var(result, 0.99)
            cvar_95 = self._calculate_cvar(result, 0.95)
            max_consecutive_losses = self._estimate_max_consecutive_losses(result)
            tail_ratio = self._calculate_tail_ratio(result)
            downside_deviation = self._calculate_downside_deviation(result)
            
            risk_metric = RiskMetrics(
                strategy_name=result.get('strategy_name', 'Unknown'),
                var_99=var_99,
                cvar_95=cvar_95,
                maximum_consecutive_losses=max_consecutive_losses,
                tail_ratio=tail_ratio,
                downside_deviation=downside_deviation,
                beta=self._estimate_beta(result),
                tracking_error=self._estimate_tracking_error(result),
                information_ratio=self._calculate_information_ratio(result)
            )
            
            risk_metrics.append(risk_metric)
            
        self.risk_metrics = risk_metrics
        return risk_metrics
        
    def find_target_achievers(self, min_cagr: float = 0.25, min_sharpe: float = 1.0, 
                            max_drawdown: float = 0.20) -> List[PerformanceMetrics]:
        """Find strategies that meet all targets"""
        if not self.performance_metrics:
            self.calculate_performance_metrics()
            
        target_achievers = [
            metric for metric in self.performance_metrics
            if (metric.cagr >= min_cagr and 
                metric.sharpe_ratio >= min_sharpe and 
                abs(metric.max_drawdown) <= max_drawdown)
        ]
        
        # Sort by target score (combination of all metrics)
        target_achievers.sort(key=lambda x: x.target_score, reverse=True)
        
        return target_achievers
        
    def analyze_strategy_clusters(self) -> Dict[str, List[str]]:
        """Group strategies by performance characteristics"""
        if not self.performance_metrics:
            self.calculate_performance_metrics()
            
        clusters = {
            'High_Performance': [],  # CAGR > 30%, Sharpe > 1.5
            'Target_Achievers': [],  # Meet all basic targets
            'High_Sharpe': [],       # Sharpe > 1.5, regardless of returns
            'High_Returns': [],      # CAGR > 25%, regardless of risk
            'Conservative': [],      # Low drawdown < 10%
            'Aggressive': [],        # High returns but high risk
            'Underperformers': []    # Below targets
        }
        
        for metric in self.performance_metrics:
            # High performance cluster
            if metric.cagr > 0.30 and metric.sharpe_ratio > 1.5:
                clusters['High_Performance'].append(metric.strategy_name)
                
            # Target achievers
            elif metric.cagr >= 0.25 and metric.sharpe_ratio >= 1.0 and abs(metric.max_drawdown) <= 0.20:
                clusters['Target_Achievers'].append(metric.strategy_name)
                
            # High Sharpe
            elif metric.sharpe_ratio > 1.5:
                clusters['High_Sharpe'].append(metric.strategy_name)
                
            # High returns
            elif metric.cagr > 0.25:
                clusters['High_Returns'].append(metric.strategy_name)
                
            # Conservative
            elif abs(metric.max_drawdown) < 0.10:
                clusters['Conservative'].append(metric.strategy_name)
                
            # Aggressive
            elif metric.cagr > 0.20 and abs(metric.max_drawdown) > 0.15:
                clusters['Aggressive'].append(metric.strategy_name)
                
            # Underperformers
            else:
                clusters['Underperformers'].append(metric.strategy_name)
                
        return clusters
        
    def generate_optimization_recommendations(self) -> Dict[str, List[str]]:
        """Generate specific optimization recommendations"""
        if not self.performance_metrics:
            self.calculate_performance_metrics()
            
        recommendations = {
            'Ready_for_Live_Trading': [],
            'Needs_Risk_Reduction': [],
            'Needs_Return_Enhancement': [],
            'Needs_Complete_Redesign': [],
            'Ensemble_Candidates': []
        }
        
        for metric in self.performance_metrics:
            # Ready for live trading
            if (metric.cagr >= 0.25 and metric.sharpe_ratio >= 1.0 and 
                abs(metric.max_drawdown) <= 0.20 and metric.win_rate >= 0.4):
                recommendations['Ready_for_Live_Trading'].append(metric.strategy_name)
                
            # Good returns but too risky
            elif metric.cagr >= 0.25 and abs(metric.max_drawdown) > 0.20:
                recommendations['Needs_Risk_Reduction'].append(metric.strategy_name)
                
            # Low risk but poor returns
            elif metric.cagr < 0.25 and abs(metric.max_drawdown) <= 0.15:
                recommendations['Needs_Return_Enhancement'].append(metric.strategy_name)
                
            # Poor on all metrics
            elif metric.cagr < 0.15 and metric.sharpe_ratio < 0.8:
                recommendations['Needs_Complete_Redesign'].append(metric.strategy_name)
                
            # Good for ensemble
            elif metric.sharpe_ratio > 1.0 and metric.win_rate > 0.45:
                recommendations['Ensemble_Candidates'].append(metric.strategy_name)
                
        return recommendations
        
    def create_performance_dashboard(self, output_dir: str = ".") -> None:
        """Create visual performance dashboard"""
        if not self.performance_metrics:
            self.calculate_performance_metrics()
            
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('QuantConnect Strategy Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([
            {
                'Strategy': m.strategy_name,
                'CAGR': m.cagr,
                'Sharpe': m.sharpe_ratio,
                'Max_DD': abs(m.max_drawdown),
                'Win_Rate': m.win_rate,
                'Total_Trades': m.total_trades,
                'Target_Score': m.target_score
            }
            for m in self.performance_metrics
        ])
        
        # 1. CAGR vs Sharpe Ratio scatter
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df['Sharpe'], df['CAGR'], c=df['Max_DD'], 
                            cmap='RdYlGn_r', s=100, alpha=0.7)
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_ylabel('CAGR')
        ax1.set_title('CAGR vs Sharpe Ratio (Color = Max Drawdown)')
        
        # Add target lines
        ax1.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='CAGR Target')
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe Target')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Max Drawdown')
        
        # 2. Top performers bar chart
        ax2 = axes[0, 1]
        top_performers = df.nlargest(10, 'Target_Score')
        bars = ax2.barh(range(len(top_performers)), top_performers['Target_Score'])
        ax2.set_yticks(range(len(top_performers)))
        ax2.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                           for name in top_performers['Strategy']])
        ax2.set_xlabel('Target Achievement Score')
        ax2.set_title('Top 10 Strategies by Target Score')
        
        # Color bars based on whether they meet targets
        for i, (_, row) in enumerate(top_performers.iterrows()):
            color = 'green' if (row['CAGR'] >= 0.25 and row['Sharpe'] >= 1.0 and row['Max_DD'] <= 0.20) else 'orange'
            bars[i].set_color(color)
        
        # 3. Risk-Return profile
        ax3 = axes[0, 2]
        ax3.scatter(df['Max_DD'], df['CAGR'], c=df['Sharpe'], 
                   cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Max Drawdown')
        ax3.set_ylabel('CAGR')
        ax3.set_title('Risk-Return Profile')
        ax3.axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=0.20, color='red', linestyle='--', alpha=0.5)
        
        # 4. Win Rate distribution
        ax4 = axes[1, 0]
        ax4.hist(df['Win_Rate'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Win Rate')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Win Rate Distribution')
        ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% Target')
        ax4.legend()
        
        # 5. Trading frequency vs performance
        ax5 = axes[1, 1]
        ax5.scatter(df['Total_Trades'], df['CAGR'], c=df['Sharpe'], 
                   cmap='plasma', s=100, alpha=0.7)
        ax5.set_xlabel('Total Trades')
        ax5.set_ylabel('CAGR')
        ax5.set_title('Trading Frequency vs Performance')
        
        # 6. Target achievement summary
        ax6 = axes[1, 2]
        target_met = len([m for m in self.performance_metrics 
                         if m.cagr >= 0.25 and m.sharpe_ratio >= 1.0 and abs(m.max_drawdown) <= 0.20])
        target_missed = len(self.performance_metrics) - target_met
        
        ax6.pie([target_met, target_missed], 
               labels=[f'Met Targets\n({target_met})', f'Missed Targets\n({target_missed})'],
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%',
               startangle=90)
        ax6.set_title('Target Achievement Summary')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_dashboard.png", dpi=300, bbox_inches='tight')
        logger.info(f"Performance dashboard saved to {output_dir}/performance_dashboard.png")
        
    def generate_detailed_report(self, output_file: str = "detailed_analysis_report.md") -> str:
        """Generate comprehensive analysis report"""
        if not self.performance_metrics:
            self.calculate_performance_metrics()
            
        target_achievers = self.find_target_achievers()
        clusters = self.analyze_strategy_clusters()
        recommendations = self.generate_optimization_recommendations()
        
        report = f"""# QuantConnect Cloud Strategy Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Strategies Analyzed**: {len(self.performance_metrics)}
- **Strategies Meeting All Targets**: {len(target_achievers)}
- **Success Rate**: {len(target_achievers)/len(self.performance_metrics)*100:.1f}%

"""
        
        # Target achievers section
        if target_achievers:
            report += "## 🏆 TARGET ACHIEVERS (CAGR≥25%, Sharpe≥1.0, MaxDD≤20%)\n\n"
            for i, strategy in enumerate(target_achievers[:5], 1):
                report += f"### {i}. {strategy.strategy_name}\n"
                report += f"- **CAGR**: {strategy.cagr:.1%}\n"
                report += f"- **Sharpe Ratio**: {strategy.sharpe_ratio:.2f}\n"
                report += f"- **Max Drawdown**: {abs(strategy.max_drawdown):.1%}\n"
                report += f"- **Win Rate**: {strategy.win_rate:.1%}\n"
                report += f"- **Total Trades**: {strategy.total_trades}\n"
                report += f"- **Calmar Ratio**: {strategy.calmar_ratio:.2f}\n"
                report += f"- **Target Score**: {strategy.target_score:.2f}\n\n"
        else:
            report += "## ❌ NO STRATEGIES MET ALL TARGETS\n\n"
            
        # Performance clusters
        report += "## 📊 STRATEGY CLUSTERS\n\n"
        for cluster_name, strategies in clusters.items():
            if strategies:
                report += f"### {cluster_name.replace('_', ' ')} ({len(strategies)} strategies)\n"
                for strategy in strategies[:5]:  # Show top 5 per cluster
                    report += f"- {strategy}\n"
                if len(strategies) > 5:
                    report += f"- ... and {len(strategies)-5} more\n"
                report += "\n"
                
        # Optimization recommendations
        report += "## 🚀 OPTIMIZATION RECOMMENDATIONS\n\n"
        for category, strategies in recommendations.items():
            if strategies:
                report += f"### {category.replace('_', ' ')}\n"
                for strategy in strategies:
                    report += f"- {strategy}\n"
                report += "\n"
                
        # Statistical analysis
        if self.performance_metrics:
            cagrs = [m.cagr for m in self.performance_metrics]
            sharpes = [m.sharpe_ratio for m in self.performance_metrics]
            drawdowns = [abs(m.max_drawdown) for m in self.performance_metrics]
            
            report += "## 📈 STATISTICAL SUMMARY\n\n"
            report += f"**CAGR Statistics**\n"
            report += f"- Mean: {np.mean(cagrs):.1%}\n"
            report += f"- Median: {np.median(cagrs):.1%}\n"
            report += f"- Std Dev: {np.std(cagrs):.1%}\n"
            report += f"- Max: {np.max(cagrs):.1%}\n"
            report += f"- Min: {np.min(cagrs):.1%}\n\n"
            
            report += f"**Sharpe Ratio Statistics**\n"
            report += f"- Mean: {np.mean(sharpes):.2f}\n"
            report += f"- Median: {np.median(sharpes):.2f}\n"
            report += f"- Std Dev: {np.std(sharpes):.2f}\n"
            report += f"- Max: {np.max(sharpes):.2f}\n"
            report += f"- Min: {np.min(sharpes):.2f}\n\n"
            
        # Next steps
        report += "## 🎯 IMMEDIATE ACTION ITEMS\n\n"
        if target_achievers:
            report += "### For Target Achievers:\n"
            report += "1. **Immediate Deployment**: Deploy top 3 target achievers to live trading\n"
            report += "2. **Risk Management**: Implement 5% daily loss limits\n"
            report += "3. **Position Sizing**: Start with 10% of target capital per strategy\n"
            report += "4. **Monitoring**: Daily performance review for first month\n\n"
            
        report += "### For Non-Target Achievers:\n"
        report += "1. **Optimization**: Focus on top performers in 'Needs Risk Reduction' category\n"
        report += "2. **Ensemble Methods**: Combine strategies from 'Ensemble Candidates'\n"
        report += "3. **Parameter Tuning**: Optimize the 'High Returns' strategies for lower drawdown\n"
        report += "4. **Alternative Approaches**: Consider regime-based switching for volatile strategies\n\n"
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
            
        logger.info(f"Detailed report saved to {output_file}")
        return report
        
    def _calculate_target_score(self, cagr: float, sharpe: float, max_dd: float, win_rate: float) -> float:
        """Calculate composite target achievement score"""
        # Normalize each metric to 0-100 scale
        cagr_score = min(100, (cagr / 0.25) * 25)  # 25% target = 25 points
        sharpe_score = min(100, (sharpe / 1.0) * 25)  # 1.0 target = 25 points
        dd_score = min(100, max(0, 25 - (abs(max_dd) / 0.20) * 25))  # 20% target = 25 points
        win_score = min(100, (win_rate / 0.5) * 25)  # 50% target = 25 points
        
        return cagr_score + sharpe_score + dd_score + win_score
        
    def _calculate_sortino_ratio(self, result: dict) -> float:
        """Estimate Sortino ratio"""
        sharpe = result.get('sharpe_ratio', 0)
        # Rough approximation: Sortino is typically 1.2-1.5x Sharpe for good strategies
        return sharpe * 1.3 if sharpe > 0 else 0
        
    def _calculate_profit_factor(self, result: dict) -> float:
        """Estimate profit factor"""
        win_rate = result.get('win_rate', 0)
        # Rough approximation based on win rate
        if win_rate > 0.6:
            return 2.0 + (win_rate - 0.6) * 5
        elif win_rate > 0.4:
            return 1.2 + (win_rate - 0.4) * 4
        else:
            return max(0.8, 0.5 + win_rate * 1.4)
            
    def _estimate_volatility(self, cagr: float, sharpe: float) -> float:
        """Estimate annualized volatility"""
        if sharpe != 0:
            return abs(cagr) / abs(sharpe)
        return 0.15  # Default assumption
        
    def _calculate_var(self, result: dict, confidence: float) -> float:
        """Estimate Value at Risk"""
        volatility = self._estimate_volatility(result.get('cagr', 0), result.get('sharpe_ratio', 0))
        # Assuming normal distribution (simplified)
        if confidence == 0.95:
            return -1.645 * volatility / np.sqrt(252)  # Daily VaR
        elif confidence == 0.99:
            return -2.326 * volatility / np.sqrt(252)
        return 0
        
    def _calculate_cvar(self, result: dict, confidence: float) -> float:
        """Estimate Conditional Value at Risk"""
        var = self._calculate_var(result, confidence)
        # CVaR is typically 1.2-1.5x VaR for normal distributions
        return var * 1.3
        
    def _estimate_max_consecutive_losses(self, result: dict) -> int:
        """Estimate maximum consecutive losses"""
        win_rate = result.get('win_rate', 0)
        total_trades = result.get('total_trades', 0)
        
        if win_rate > 0 and total_trades > 0:
            # Statistical estimation
            lose_rate = 1 - win_rate
            # Expected max consecutive losses in a series
            return max(1, int(np.log(0.01) / np.log(lose_rate)))
        return 5  # Default assumption
        
    def _calculate_tail_ratio(self, result: dict) -> float:
        """Estimate tail ratio (right tail / left tail)"""
        sharpe = result.get('sharpe_ratio', 0)
        # Positive sharpe suggests positive skew
        return 1.2 + max(0, sharpe * 0.1)
        
    def _calculate_downside_deviation(self, result: dict) -> float:
        """Estimate downside deviation"""
        volatility = self._estimate_volatility(result.get('cagr', 0), result.get('sharpe_ratio', 0))
        # Downside deviation is typically 0.6-0.8x of total volatility for good strategies
        return volatility * 0.7
        
    def _estimate_beta(self, result: dict) -> float:
        """Estimate beta vs market"""
        # Rough approximation based on returns
        cagr = result.get('cagr', 0)
        market_return = 0.10  # Assume 10% market return
        return cagr / market_return if market_return != 0 else 1.0
        
    def _estimate_tracking_error(self, result: dict) -> float:
        """Estimate tracking error"""
        volatility = self._estimate_volatility(result.get('cagr', 0), result.get('sharpe_ratio', 0))
        return volatility * 0.3  # Rough approximation
        
    def _calculate_information_ratio(self, result: dict) -> float:
        """Calculate information ratio"""
        cagr = result.get('cagr', 0)
        market_return = 0.10
        tracking_error = self._estimate_tracking_error(result)
        
        if tracking_error != 0:
            return (cagr - market_return) / tracking_error
        return 0
        
    def _estimate_skewness(self, result: dict) -> float:
        """Estimate return skewness"""
        sharpe = result.get('sharpe_ratio', 0)
        # Positive sharpe often correlates with positive skew
        return max(-2, min(2, sharpe * 0.2))
        
    def _estimate_kurtosis(self, result: dict) -> float:
        """Estimate return kurtosis"""
        max_dd = abs(result.get('max_drawdown', 0))
        # Higher drawdown suggests fat tails
        return 3 + max_dd * 10  # 3 is normal distribution baseline
        
    def _estimate_trade_duration(self, total_trades: int) -> float:
        """Estimate average trade duration in days"""
        if total_trades > 0:
            # Assume 252 trading days per year over ~5 year backtest
            total_days = 252 * 5
            return total_days / total_trades
        return 10  # Default assumption

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="Analyze QuantConnect cloud backtest results")
    parser.add_argument("--results-file", required=True, help="JSON file with backtest results")
    parser.add_argument("--output-dir", default=".", help="Output directory for reports and charts")
    parser.add_argument("--create-dashboard", action="store_true", help="Create visual dashboard")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CloudResultsAnalyzer()
    
    # Load and analyze results
    analyzer.load_results(args.results_file)
    analyzer.calculate_performance_metrics()
    analyzer.calculate_risk_metrics()
    
    # Find target achievers
    target_achievers = analyzer.find_target_achievers()
    
    # Generate detailed report
    report_file = f"{args.output_dir}/analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    analyzer.generate_detailed_report(report_file)
    
    # Create dashboard if requested
    if args.create_dashboard:
        analyzer.create_performance_dashboard(args.output_dir)
        
    # Print summary
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Analyzed {len(analyzer.performance_metrics)} strategies")
    print(f"Target achievers: {len(target_achievers)}")
    print(f"Report saved to: {report_file}")
    
    if target_achievers:
        print(f"\n🎉 TOP TARGET ACHIEVER:")
        top = target_achievers[0]
        print(f"  Strategy: {top.strategy_name}")
        print(f"  CAGR: {top.cagr:.1%}")
        print(f"  Sharpe: {top.sharpe_ratio:.2f}")
        print(f"  Max DD: {abs(top.max_drawdown):.1%}")

if __name__ == "__main__":
    main()