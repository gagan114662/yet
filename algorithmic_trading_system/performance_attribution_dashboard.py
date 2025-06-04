"""
Performance Attribution Dashboard
Priority 4: Comprehensive analytics to understand strategy performance drivers
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Core performance metrics"""
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    avg_trade_duration: float
    
    
@dataclass
class AttributionFactors:
    """Performance attribution factors"""
    market_timing: float
    stock_selection: float
    factor_exposure: float
    risk_management: float
    transaction_costs: float
    slippage: float
    regime_alignment: float
    
    
class PerformanceAttributionDashboard:
    """
    Comprehensive dashboard for understanding strategy performance
    - Factor attribution
    - Regime analysis
    - Trade analytics
    - Risk decomposition
    - Cost analysis
    """
    
    def __init__(self):
        self.performance_data = []
        self.trade_history = []
        self.factor_exposures = {}
        self.regime_history = []
        self.benchmarks = {}
        
    def analyze_strategy_performance(self, 
                                   strategy_results: Dict,
                                   trade_data: Optional[List] = None,
                                   market_data: Optional[Dict] = None) -> Dict:
        """
        Comprehensive performance analysis
        
        Args:
            strategy_results: Backtest results
            trade_data: Individual trade records
            market_data: Market conditions during backtest
            
        Returns:
            Complete performance attribution
        """
        analysis = {
            'summary': self._generate_summary(strategy_results),
            'attribution': self._calculate_attribution(strategy_results, trade_data, market_data),
            'regime_analysis': self._analyze_regime_performance(strategy_results, market_data),
            'trade_analysis': self._analyze_trades(trade_data) if trade_data else {},
            'risk_decomposition': self._decompose_risk(strategy_results, trade_data),
            'cost_analysis': self._analyze_costs(strategy_results, trade_data),
            'failure_analysis': self._analyze_failures(strategy_results, trade_data),
            'recommendations': self._generate_recommendations(strategy_results)
        }
        
        # Store for historical analysis
        self.performance_data.append({
            'timestamp': datetime.now(),
            'strategy': strategy_results.get('strategy_name', 'Unknown'),
            'analysis': analysis
        })
        
        return analysis
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate performance summary"""
        metrics = PerformanceMetrics(
            total_return=results.get('total_return', 0),
            cagr=results.get('cagr', 0),
            sharpe_ratio=results.get('sharpe_ratio', 0),
            sortino_ratio=results.get('sortino_ratio', results.get('sharpe_ratio', 0) * 1.2),
            max_drawdown=results.get('max_drawdown', 0),
            win_rate=results.get('win_rate', 0.5),
            profit_factor=results.get('profit_factor', 1.0),
            avg_win=results.get('avg_win', 0),
            avg_loss=results.get('avg_loss', 0),
            total_trades=results.get('total_trades', 0),
            avg_trade_duration=results.get('avg_trade_duration', 1)
        )
        
        # Performance rating
        rating = self._calculate_performance_rating(metrics)
        
        return {
            'metrics': metrics.__dict__,
            'rating': rating,
            'strengths': self._identify_strengths(metrics),
            'weaknesses': self._identify_weaknesses(metrics)
        }
    
    def _calculate_attribution(self, results: Dict, trade_data: List, market_data: Dict) -> Dict:
        """Calculate performance attribution"""
        total_return = results.get('cagr', 0)
        
        # Initialize attribution factors
        attribution = AttributionFactors(
            market_timing=0,
            stock_selection=0,
            factor_exposure=0,
            risk_management=0,
            transaction_costs=0,
            slippage=0,
            regime_alignment=0
        )
        
        # Market timing attribution
        if market_data:
            market_return = market_data.get('spy_return', 0.10)  # Baseline SPY return
            attribution.market_timing = self._calculate_timing_contribution(results, market_data)
        
        # Stock selection attribution
        if trade_data:
            attribution.stock_selection = self._calculate_selection_contribution(trade_data)
        
        # Factor exposure attribution
        attribution.factor_exposure = self._calculate_factor_contribution(results)
        
        # Risk management attribution
        attribution.risk_management = self._calculate_risk_contribution(results)
        
        # Cost attribution
        attribution.transaction_costs = results.get('total_fees', 0) / results.get('total_value', 100000)
        attribution.slippage = results.get('total_slippage', 0) / results.get('total_value', 100000)
        
        # Regime alignment
        attribution.regime_alignment = self._calculate_regime_alignment(results, market_data)
        
        # Normalize attributions
        total_attribution = sum([
            abs(attribution.market_timing),
            abs(attribution.stock_selection),
            abs(attribution.factor_exposure),
            abs(attribution.risk_management),
            abs(attribution.regime_alignment)
        ])
        
        if total_attribution > 0:
            scale_factor = total_return / total_attribution
            attribution.market_timing *= scale_factor
            attribution.stock_selection *= scale_factor
            attribution.factor_exposure *= scale_factor
            attribution.risk_management *= scale_factor
            attribution.regime_alignment *= scale_factor
        
        return {
            'factors': attribution.__dict__,
            'total_explained': sum(attribution.__dict__.values()),
            'unexplained': total_return - sum(attribution.__dict__.values()),
            'breakdown_chart': self._create_attribution_chart(attribution)
        }
    
    def _analyze_regime_performance(self, results: Dict, market_data: Dict) -> Dict:
        """Analyze performance across different market regimes"""
        regime_performance = {
            'bull': {'trades': 0, 'return': 0, 'win_rate': 0},
            'bear': {'trades': 0, 'return': 0, 'win_rate': 0},
            'sideways': {'trades': 0, 'return': 0, 'win_rate': 0},
            'high_volatility': {'trades': 0, 'return': 0, 'win_rate': 0},
            'low_volatility': {'trades': 0, 'return': 0, 'win_rate': 0}
        }
        
        # Simulate regime performance (in production, would use actual regime data)
        total_return = results.get('cagr', 0)
        
        # Distribute performance across regimes based on strategy type
        strategy_type = results.get('strategy_type', 'balanced')
        
        if strategy_type == 'momentum':
            regime_performance['bull']['return'] = total_return * 0.6
            regime_performance['bear']['return'] = total_return * -0.2
            regime_performance['sideways']['return'] = total_return * 0.1
        elif strategy_type == 'mean_reversion':
            regime_performance['sideways']['return'] = total_return * 0.5
            regime_performance['high_volatility']['return'] = total_return * 0.3
        else:
            # Balanced distribution
            for regime in regime_performance:
                regime_performance[regime]['return'] = total_return * 0.2
        
        # Calculate best/worst regimes
        best_regime = max(regime_performance.items(), key=lambda x: x[1]['return'])
        worst_regime = min(regime_performance.items(), key=lambda x: x[1]['return'])
        
        return {
            'regime_breakdown': regime_performance,
            'best_regime': best_regime[0],
            'worst_regime': worst_regime[0],
            'regime_sensitivity': self._calculate_regime_sensitivity(regime_performance),
            'recommendations': self._generate_regime_recommendations(regime_performance)
        }
    
    def _analyze_trades(self, trade_data: List) -> Dict:
        """Detailed trade analysis"""
        if not trade_data:
            return {}
        
        df = pd.DataFrame(trade_data)
        
        analysis = {
            'summary_stats': {
                'total_trades': len(df),
                'avg_profit': df['profit'].mean() if 'profit' in df else 0,
                'win_rate': (df['profit'] > 0).mean() if 'profit' in df else 0,
                'avg_duration': df['duration'].mean() if 'duration' in df else 0,
                'max_win': df['profit'].max() if 'profit' in df else 0,
                'max_loss': df['profit'].min() if 'profit' in df else 0
            },
            'time_analysis': self._analyze_trade_timing(df),
            'symbol_analysis': self._analyze_symbol_performance(df),
            'pattern_analysis': self._analyze_trade_patterns(df),
            'streak_analysis': self._analyze_winning_streaks(df)
        }
        
        return analysis
    
    def _decompose_risk(self, results: Dict, trade_data: List) -> Dict:
        """Decompose risk into components"""
        total_risk = results.get('max_drawdown', 0.15)
        
        risk_components = {
            'market_risk': total_risk * 0.4,  # Market beta risk
            'specific_risk': total_risk * 0.3,  # Stock-specific risk
            'timing_risk': total_risk * 0.2,   # Entry/exit timing risk
            'concentration_risk': total_risk * 0.1  # Position concentration risk
        }
        
        # Calculate risk-adjusted metrics
        risk_metrics = {
            'value_at_risk': self._calculate_var(results),
            'conditional_var': self._calculate_cvar(results),
            'downside_deviation': self._calculate_downside_deviation(results),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(trade_data)
        }
        
        return {
            'risk_decomposition': risk_components,
            'risk_metrics': risk_metrics,
            'risk_efficiency': results.get('sharpe_ratio', 0) / max(total_risk, 0.01),
            'recommendations': self._generate_risk_recommendations(risk_components, risk_metrics)
        }
    
    def _analyze_costs(self, results: Dict, trade_data: List) -> Dict:
        """Analyze all trading costs"""
        total_trades = results.get('total_trades', 0)
        total_volume = results.get('total_volume', 0)
        
        costs = {
            'commission': {
                'total': results.get('total_commission', 0),
                'per_trade': results.get('total_commission', 0) / max(total_trades, 1),
                'impact_on_return': results.get('total_commission', 0) / results.get('total_value', 100000)
            },
            'slippage': {
                'total': results.get('total_slippage', 0),
                'per_trade': results.get('total_slippage', 0) / max(total_trades, 1),
                'impact_on_return': results.get('total_slippage', 0) / results.get('total_value', 100000)
            },
            'spread': {
                'avg_spread': results.get('avg_spread', 0.0001),
                'total_cost': total_volume * results.get('avg_spread', 0.0001),
                'impact_on_return': (total_volume * results.get('avg_spread', 0.0001)) / results.get('total_value', 100000)
            }
        }
        
        # Total cost impact
        total_cost_impact = sum(c['impact_on_return'] for c in costs.values())
        
        return {
            'cost_breakdown': costs,
            'total_cost_impact': total_cost_impact,
            'cost_efficiency': results.get('cagr', 0) / max(total_cost_impact, 0.0001),
            'recommendations': self._generate_cost_recommendations(costs)
        }
    
    def _analyze_failures(self, results: Dict, trade_data: List) -> Dict:
        """Analyze why strategies fail to meet targets"""
        target_cagr = 0.25
        target_sharpe = 1.0
        target_drawdown = 0.15
        
        failures = {
            'return_failure': {
                'failed': results.get('cagr', 0) < target_cagr,
                'gap': target_cagr - results.get('cagr', 0),
                'reasons': []
            },
            'risk_failure': {
                'failed': results.get('sharpe_ratio', 0) < target_sharpe,
                'gap': target_sharpe - results.get('sharpe_ratio', 0),
                'reasons': []
            },
            'drawdown_failure': {
                'failed': results.get('max_drawdown', 1) > target_drawdown,
                'gap': results.get('max_drawdown', 1) - target_drawdown,
                'reasons': []
            }
        }
        
        # Analyze return failures
        if failures['return_failure']['failed']:
            failures['return_failure']['reasons'] = [
                'Low win rate' if results.get('win_rate', 0) < 0.5 else None,
                'Small average wins' if results.get('avg_win', 0) < 0.01 else None,
                'High costs' if self._analyze_costs(results, trade_data)['total_cost_impact'] > 0.05 else None,
                'Poor market timing' if results.get('timing_score', 0) < 0.3 else None
            ]
            failures['return_failure']['reasons'] = [r for r in failures['return_failure']['reasons'] if r]
        
        # Analyze risk failures
        if failures['risk_failure']['failed']:
            failures['risk_failure']['reasons'] = [
                'High volatility' if results.get('volatility', 0.20) > 0.25 else None,
                'Inconsistent returns' if results.get('return_consistency', 0) < 0.5 else None,
                'Poor risk management' if results.get('max_drawdown', 1) > 0.20 else None
            ]
            failures['risk_failure']['reasons'] = [r for r in failures['risk_failure']['reasons'] if r]
        
        # Generate improvement suggestions
        improvements = self._generate_improvement_suggestions(failures, results)
        
        return {
            'failure_analysis': failures,
            'root_causes': self._identify_root_causes(failures, results),
            'improvement_suggestions': improvements,
            'success_probability': self._estimate_success_probability(results, improvements)
        }
    
    def _calculate_performance_rating(self, metrics: PerformanceMetrics) -> Dict:
        """Calculate overall performance rating"""
        scores = {
            'return_score': min(metrics.cagr / 0.25, 1.0) * 100,  # Target 25% CAGR
            'risk_score': min(metrics.sharpe_ratio / 1.0, 1.0) * 100,  # Target 1.0 Sharpe
            'drawdown_score': min(0.15 / max(metrics.max_drawdown, 0.01), 1.0) * 100,  # Target <15% DD
            'consistency_score': metrics.win_rate * 100,
            'efficiency_score': min(metrics.profit_factor / 1.5, 1.0) * 100  # Target 1.5 profit factor
        }
        
        overall_score = np.mean(list(scores.values()))
        
        rating = {
            'scores': scores,
            'overall': overall_score,
            'grade': self._score_to_grade(overall_score),
            'percentile': self._calculate_percentile(overall_score)
        }
        
        return rating
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'F'
    
    def _identify_strengths(self, metrics: PerformanceMetrics) -> List[str]:
        """Identify strategy strengths"""
        strengths = []
        
        if metrics.sharpe_ratio > 1.5:
            strengths.append("Excellent risk-adjusted returns")
        if metrics.cagr > 0.30:
            strengths.append("Strong absolute returns")
        if metrics.max_drawdown < 0.10:
            strengths.append("Low drawdown risk")
        if metrics.win_rate > 0.60:
            strengths.append("High win rate")
        if metrics.profit_factor > 2.0:
            strengths.append("Strong profit factor")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: PerformanceMetrics) -> List[str]:
        """Identify strategy weaknesses"""
        weaknesses = []
        
        if metrics.sharpe_ratio < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
        if metrics.cagr < 0.10:
            weaknesses.append("Low absolute returns")
        if metrics.max_drawdown > 0.25:
            weaknesses.append("High drawdown risk")
        if metrics.win_rate < 0.40:
            weaknesses.append("Low win rate")
        if metrics.profit_factor < 1.0:
            weaknesses.append("Negative profit factor")
        
        return weaknesses
    
    def _calculate_timing_contribution(self, results: Dict, market_data: Dict) -> float:
        """Calculate market timing contribution to returns"""
        # Simplified calculation
        market_return = market_data.get('market_return', 0.10)
        strategy_return = results.get('cagr', 0)
        
        # If strategy outperforms in good markets and preserves in bad, good timing
        timing_score = (strategy_return - market_return) * 0.3
        
        return timing_score
    
    def _calculate_selection_contribution(self, trade_data: List) -> float:
        """Calculate stock selection contribution"""
        if not trade_data:
            return 0
        
        # Average outperformance of selected stocks
        avg_alpha = np.mean([t.get('alpha', 0) for t in trade_data if 'alpha' in t])
        return avg_alpha * 0.4
    
    def _calculate_factor_contribution(self, results: Dict) -> float:
        """Calculate factor exposure contribution"""
        # Simplified - based on strategy type
        factor_returns = {
            'momentum': 0.05,
            'value': 0.03,
            'quality': 0.04,
            'low_volatility': 0.02
        }
        
        strategy_type = results.get('strategy_type', 'balanced')
        return factor_returns.get(strategy_type, 0.03)
    
    def _calculate_risk_contribution(self, results: Dict) -> float:
        """Calculate risk management contribution"""
        # Good risk management preserves capital
        max_dd = results.get('max_drawdown', 0.15)
        if max_dd < 0.10:
            return 0.03  # 3% contribution for excellent risk management
        elif max_dd < 0.15:
            return 0.01
        else:
            return -0.02  # Negative contribution for poor risk management
    
    def _calculate_regime_alignment(self, results: Dict, market_data: Dict) -> float:
        """Calculate how well strategy aligned with market regimes"""
        # Simplified calculation
        return 0.02 if results.get('regime_aligned', True) else -0.02
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Return recommendations
        if results.get('cagr', 0) < 0.20:
            recommendations.append("Consider increasing position sizes or leverage within risk limits")
            recommendations.append("Look for higher conviction trade signals")
        
        # Risk recommendations
        if results.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("Improve entry/exit timing to reduce volatility")
            recommendations.append("Consider adding filters to avoid choppy markets")
        
        # Drawdown recommendations
        if results.get('max_drawdown', 1) > 0.15:
            recommendations.append("Implement tighter stop losses")
            recommendations.append("Reduce position sizes during high volatility")
            recommendations.append("Add portfolio-level risk limits")
        
        # Cost recommendations
        if results.get('total_trades', 0) > 500:
            recommendations.append("Reduce trading frequency to lower costs")
            recommendations.append("Focus on higher-quality signals")
        
        return recommendations
    
    def generate_html_report(self, analysis: Dict) -> str:
        """Generate HTML performance report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Performance Attribution Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Strategy Performance Attribution Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Summary</h2>
            <div>
                <div class="metric">
                    <strong>Overall Grade:</strong> {analysis['summary']['rating']['grade']}
                </div>
                <div class="metric">
                    <strong>Score:</strong> {analysis['summary']['rating']['overall']:.1f}/100
                </div>
            </div>
            
            <h2>Key Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Target</th>
                    <th>Status</th>
                </tr>
        """
        
        # Add metrics to table
        metrics = analysis['summary']['metrics']
        targets = {'cagr': 0.25, 'sharpe_ratio': 1.0, 'max_drawdown': 0.15}
        
        for key, value in metrics.items():
            if key in targets:
                target = targets[key]
                if key == 'max_drawdown':
                    status = 'positive' if value <= target else 'negative'
                else:
                    status = 'positive' if value >= target else 'negative'
                
                html += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{value:.3f}</td>
                    <td>{target:.3f}</td>
                    <td class="{status}">{'✓' if status == 'positive' else '✗'}</td>
                </tr>
                """
        
        html += """
            </table>
            
            <h2>Performance Attribution</h2>
        """
        
        if 'attribution' in analysis:
            html += "<table><tr><th>Factor</th><th>Contribution</th></tr>"
            for factor, value in analysis['attribution']['factors'].items():
                html += f"<tr><td>{factor.replace('_', ' ').title()}</td><td>{value:.3%}</td></tr>"
            html += "</table>"
        
        html += """
            <h2>Recommendations</h2>
            <ul>
        """
        
        for rec in analysis.get('recommendations', []):
            html += f"<li>{rec}</li>"
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html
    
    def _create_attribution_chart(self, attribution: AttributionFactors) -> Dict:
        """Create attribution chart data"""
        return {
            'type': 'waterfall',
            'categories': list(attribution.__dict__.keys()),
            'values': list(attribution.__dict__.values()),
            'title': 'Performance Attribution Breakdown'
        }
    
    def _calculate_var(self, results: Dict) -> float:
        """Calculate Value at Risk"""
        # Simplified VaR calculation
        return results.get('volatility', 0.20) * 1.645  # 95% VaR
    
    def _calculate_cvar(self, results: Dict) -> float:
        """Calculate Conditional Value at Risk"""
        # Simplified CVaR
        return self._calculate_var(results) * 1.2
    
    def _calculate_downside_deviation(self, results: Dict) -> float:
        """Calculate downside deviation"""
        return results.get('downside_volatility', results.get('volatility', 0.20) * 0.7)
    
    def _calculate_max_consecutive_losses(self, trade_data: List) -> int:
        """Calculate maximum consecutive losses"""
        if not trade_data:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for trade in trade_data:
            if trade.get('profit', 0) < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_percentile(self, score: float) -> float:
        """Calculate performance percentile"""
        # Simplified - in production would compare against all strategies
        if score >= 85:
            return 95
        elif score >= 75:
            return 80
        elif score >= 65:
            return 60
        elif score >= 55:
            return 40
        else:
            return 20


# Example usage
if __name__ == '__main__':
    # Initialize dashboard
    dashboard = PerformanceAttributionDashboard()
    
    # Example strategy results
    strategy_results = {
        'strategy_name': 'MomentumAlpha_v2',
        'strategy_type': 'momentum',
        'cagr': 0.22,
        'sharpe_ratio': 0.85,
        'max_drawdown': 0.18,
        'total_return': 0.88,
        'win_rate': 0.52,
        'profit_factor': 1.4,
        'avg_win': 0.015,
        'avg_loss': -0.011,
        'total_trades': 342,
        'total_commission': 1500,
        'total_slippage': 800,
        'total_value': 100000,
        'volatility': 0.22
    }
    
    # Example trade data
    trade_data = [
        {'symbol': 'AAPL', 'profit': 0.02, 'duration': 5, 'alpha': 0.01},
        {'symbol': 'GOOGL', 'profit': -0.01, 'duration': 3, 'alpha': -0.005},
        {'symbol': 'MSFT', 'profit': 0.015, 'duration': 7, 'alpha': 0.008},
        # ... more trades
    ]
    
    # Example market data
    market_data = {
        'spy_return': 0.12,
        'market_volatility': 0.16,
        'regime': 'bull'
    }
    
    # Run analysis
    analysis = dashboard.analyze_strategy_performance(strategy_results, trade_data, market_data)
    
    # Display results
    print("=== Performance Attribution Dashboard ===\n")
    
    print(f"Overall Grade: {analysis['summary']['rating']['grade']}")
    print(f"Score: {analysis['summary']['rating']['overall']:.1f}/100")
    print(f"Percentile: {analysis['summary']['rating']['percentile']}th\n")
    
    print("Key Metrics:")
    for metric, value in analysis['summary']['metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\nPerformance Attribution:")
    for factor, contribution in analysis['attribution']['factors'].items():
        print(f"  {factor}: {contribution:+.3%}")
    
    print(f"\nTotal Explained: {analysis['attribution']['total_explained']:.3%}")
    print(f"Unexplained: {analysis['attribution']['unexplained']:.3%}")
    
    print("\nStrengths:")
    for strength in analysis['summary']['strengths']:
        print(f"  ✓ {strength}")
    
    print("\nWeaknesses:")
    for weakness in analysis['summary']['weaknesses']:
        print(f"  ✗ {weakness}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Generate HTML report
    html_report = dashboard.generate_html_report(analysis)
    # Save to file in production
    print("\n[HTML Report Generated]")