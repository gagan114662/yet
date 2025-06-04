"""
Enhanced Controller using all priority fixes
Replaces the original controller with cloud-first, regime-aware approach
"""

import time
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import enhanced components
from enhanced_trading_system import EnhancedTradingSystem
import config

logger = logging.getLogger(__name__)


class EnhancedTargetSeekingController:
    """
    Enhanced controller that uses:
    1. QuantConnect cloud data exclusively
    2. Market regime detection
    3. Ensemble strategy generation
    4. Performance attribution
    5. Walk-forward testing
    """
    
    def __init__(self):
        """Initialize enhanced controller"""
        self.targets = config.TARGET_METRICS
        self.required_successful_strategies = config.REQUIRED_SUCCESSFUL_STRATEGIES
        
        # Use enhanced trading system
        self.trading_system = EnhancedTradingSystem()
        
        # Controller state
        self.successful_strategies = []
        self.iteration_count = 0
        self.start_time = time.time()
        self.last_progress_update_time = time.time()
        
        # Enhanced tracking
        self.performance_by_regime = {}
        self.failure_patterns = {}
        self.ensemble_strategies = []
        
        logger.info("✅ Enhanced Target-Seeking Controller initialized")
        logger.info(f"Targets: CAGR≥{self.targets['cagr']:.1%}, Sharpe≥{self.targets['sharpe_ratio']}, DD≤{self.targets['max_drawdown']:.1%}")
        logger.info(f"Required successful strategies: {self.required_successful_strategies}")
        logger.info("Using QuantConnect cloud data exclusively")
    
    def run_until_success(self):
        """
        Run enhanced strategy search until targets are met
        """
        logger.info(f"\n{'='*80}")
        logger.info("🚀 Starting Enhanced Target-Seeking System")
        logger.info(f"{'='*80}\n")
        
        try:
            # Run enhanced search
            results = self.trading_system.run_enhanced_strategy_search(
                target_strategies=self.required_successful_strategies,
                max_iterations=100
            )
            
            # Extract successful strategies
            self.successful_strategies = self.trading_system.successful_strategies
            
            # Display final results
            self._display_final_results(results)
            
            # Save comprehensive report
            self._save_comprehensive_report(results)
            
        except KeyboardInterrupt:
            logger.info("\n⚠️ Search interrupted by user")
        except Exception as e:
            logger.error(f"\n❌ Error during search: {e}")
            raise
        finally:
            elapsed_time = time.time() - self.start_time
            logger.info(f"\n⏱️ Total runtime: {elapsed_time/60:.1f} minutes")
    
    def _display_final_results(self, results: Dict):
        """Display comprehensive final results"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 FINAL RESULTS SUMMARY")
        logger.info(f"{'='*80}\n")
        
        # Summary statistics
        logger.info(f"Total Strategies Tested: {results['summary']['total_tested']}")
        logger.info(f"Successful Strategies: {results['summary']['successful']}")
        logger.info(f"Success Rate: {results['summary']['success_rate']:.1%}")
        logger.info(f"Market Regime: {results['summary']['market_regime']}")
        
        # Performance metrics
        if results.get('performance_metrics'):
            logger.info("\n📈 Aggregate Performance Metrics:")
            logger.info(f"  Average CAGR: {results['performance_metrics']['avg_cagr']:.1%}")
            logger.info(f"  Best CAGR: {results['performance_metrics']['best_cagr']:.1%}")
            logger.info(f"  Average Sharpe: {results['performance_metrics']['avg_sharpe']:.2f}")
            logger.info(f"  Best Sharpe: {results['performance_metrics']['best_sharpe']:.2f}")
        
        # Successful strategies
        if results['successful_strategies']:
            logger.info("\n✅ Successful Strategies:")
            for i, strategy in enumerate(results['successful_strategies'], 1):
                logger.info(f"\n  {i}. {strategy['name']}")
                logger.info(f"     CAGR: {strategy['cagr']:.1%}")
                logger.info(f"     Sharpe: {strategy['sharpe']:.2f}")
                logger.info(f"     Max DD: {strategy['drawdown']:.1%}")
                logger.info(f"     Grade: {strategy['grade']}")
        
        # Ensemble recommendation
        if results.get('ensemble_recommendation'):
            ensemble = results['ensemble_recommendation']
            logger.info(f"\n🎯 Recommended Ensemble Strategy: {ensemble['name']}")
            logger.info(f"   Method: {ensemble['method']}")
            logger.info(f"   Components: {len(ensemble['components'])}")
            logger.info(f"   Expected CAGR: {ensemble['expected_metrics']['expected_cagr']:.1%}")
            logger.info(f"   Expected Sharpe: {ensemble['expected_metrics']['expected_sharpe_ratio']:.2f}")
            logger.info(f"   Diversification Ratio: {ensemble['expected_metrics']['diversification_ratio']:.1f}")
            
            logger.info("\n   Component Weights:")
            for comp in ensemble['components']:
                logger.info(f"     - {comp['name']}: {comp['weight']:.1%}")
        
        # Recommendations
        if results.get('recommendations'):
            logger.info("\n💡 System Recommendations:")
            for rec in results['recommendations']:
                logger.info(f"  {rec}")
    
    def _save_comprehensive_report(self, results: Dict):
        """Save detailed report with all analytics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_configuration': {
                'targets': self.targets,
                'required_strategies': self.required_successful_strategies,
                'data_source': 'QuantConnect Cloud',
                'features': [
                    'Market Regime Detection',
                    'Ensemble Strategy Generation',
                    'Walk-Forward Testing',
                    'Performance Attribution',
                    'Multi-Asset Support'
                ]
            },
            'results': results,
            'successful_strategies': [
                {
                    'strategy': s['strategy'],
                    'results': s['results'],
                    'analysis': s['analysis']['summary'] if 'analysis' in s else {},
                    'attribution': s['analysis']['attribution'] if 'analysis' in s else {}
                }
                for s in self.successful_strategies
            ],
            'market_conditions': {
                'regime': self.trading_system.current_market_regime.value if self.trading_system.current_market_regime else 'unknown',
                'regime_history': self.trading_system.regime_detector.regime_history[-10:] if hasattr(self.trading_system, 'regime_detector') else []
            },
            'ensemble_strategies': self.trading_system.ensemble_generator.strategy_pool if hasattr(self.trading_system, 'ensemble_generator') else [],
            'runtime_minutes': (time.time() - self.start_time) / 60
        }
        
        # Save main report
        report_path = f"enhanced_results_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n💾 Comprehensive report saved to: {report_path}")
        
        # Generate HTML dashboard if we have successful strategies
        if self.successful_strategies:
            self._generate_html_dashboard(report, timestamp)
    
    def _generate_html_dashboard(self, report: Dict, timestamp: str):
        """Generate interactive HTML dashboard"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Trading System Results - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metric-card {{
            display: inline-block;
            background: #ecf0f1;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            min-width: 200px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .success {{
            color: #27ae60;
        }}
        .warning {{
            color: #f39c12;
        }}
        .danger {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .grade-A {{
            color: #27ae60;
            font-weight: bold;
        }}
        .grade-B {{
            color: #3498db;
            font-weight: bold;
        }}
        .grade-C {{
            color: #f39c12;
        }}
        .grade-F {{
            color: #e74c3c;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Enhanced Trading System Results</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Summary Metrics</h2>
        <div>
            <div class="metric-card">
                <div class="metric-value">{report['results']['summary']['total_tested']}</div>
                <div class="metric-label">Strategies Tested</div>
            </div>
            <div class="metric-card">
                <div class="metric-value success">{report['results']['summary']['successful']}</div>
                <div class="metric-label">Successful Strategies</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['results']['summary']['success_rate']:.1%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['runtime_minutes']:.1f} min</div>
                <div class="metric-label">Runtime</div>
            </div>
        </div>
        """
        
        # Add performance metrics if available
        if report['results'].get('performance_metrics'):
            metrics = report['results']['performance_metrics']
            html += f"""
        <h2>📈 Performance Overview</h2>
        <div>
            <div class="metric-card">
                <div class="metric-value">{metrics['avg_cagr']:.1%}</div>
                <div class="metric-label">Average CAGR</div>
            </div>
            <div class="metric-card">
                <div class="metric-value success">{metrics['best_cagr']:.1%}</div>
                <div class="metric-label">Best CAGR</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['avg_sharpe']:.2f}</div>
                <div class="metric-label">Average Sharpe</div>
            </div>
            <div class="metric-card">
                <div class="metric-value success">{metrics['best_sharpe']:.2f}</div>
                <div class="metric-label">Best Sharpe</div>
            </div>
        </div>
            """
        
        # Add successful strategies table
        if report['results']['successful_strategies']:
            html += """
        <h2>✅ Successful Strategies</h2>
        <table>
            <tr>
                <th>Strategy Name</th>
                <th>CAGR</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Grade</th>
            </tr>
            """
            
            for strategy in report['results']['successful_strategies']:
                grade_class = f"grade-{strategy['grade'][0]}"
                html += f"""
            <tr>
                <td>{strategy['name']}</td>
                <td>{strategy['cagr']:.1%}</td>
                <td>{strategy['sharpe']:.2f}</td>
                <td>{strategy['drawdown']:.1%}</td>
                <td class="{grade_class}">{strategy['grade']}</td>
            </tr>
                """
            
            html += "</table>"
        
        # Add ensemble recommendation
        if report['results'].get('ensemble_recommendation'):
            ensemble = report['results']['ensemble_recommendation']
            html += f"""
        <h2>🎯 Recommended Ensemble Strategy</h2>
        <div style="background: #e8f6f3; padding: 20px; border-radius: 8px; margin-top: 20px;">
            <h3>{ensemble['name']}</h3>
            <p><strong>Method:</strong> {ensemble['method']}</p>
            <p><strong>Expected Performance:</strong></p>
            <ul>
                <li>CAGR: {ensemble['expected_metrics']['expected_cagr']:.1%}</li>
                <li>Sharpe Ratio: {ensemble['expected_metrics']['expected_sharpe_ratio']:.2f}</li>
                <li>Diversification Ratio: {ensemble['expected_metrics']['diversification_ratio']:.1f}</li>
            </ul>
            """
            
            if ensemble['components']:
                html += "<p><strong>Components:</strong></p><ul>"
                for comp in ensemble['components']:
                    html += f"<li>{comp['name']} ({comp['weight']:.1%})</li>"
                html += "</ul>"
            
            html += "</div>"
        
        # Add system configuration
        html += f"""
        <h2>⚙️ System Configuration</h2>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <p><strong>Target Metrics:</strong></p>
            <ul>
                <li>CAGR ≥ {self.targets['cagr']:.1%}</li>
                <li>Sharpe Ratio ≥ {self.targets['sharpe_ratio']}</li>
                <li>Max Drawdown ≤ {self.targets['max_drawdown']:.1%}</li>
            </ul>
            <p><strong>Data Source:</strong> QuantConnect Cloud (Professional Quality)</p>
            <p><strong>Market Regime:</strong> {report['market_conditions']['regime']}</p>
            <p><strong>Features Used:</strong></p>
            <ul>
        """
        
        for feature in report['system_configuration']['features']:
            html += f"<li>✓ {feature}</li>"
        
        html += """
            </ul>
        </div>
        """
        
        # Add recommendations
        if report['results'].get('recommendations'):
            html += """
        <h2>💡 Recommendations</h2>
        <div style="background: #fff3cd; padding: 20px; border-radius: 8px;">
            <ul>
            """
            for rec in report['results']['recommendations']:
                html += f"<li>{rec}</li>"
            html += """
            </ul>
        </div>
            """
        
        html += """
    </div>
</body>
</html>
        """
        
        # Save HTML dashboard
        dashboard_path = f"enhanced_dashboard_{timestamp}.html"
        with open(dashboard_path, 'w') as f:
            f.write(html)
        
        logger.info(f"📊 Interactive dashboard saved to: {dashboard_path}")


def main():
    """Main execution function"""
    # Initialize enhanced controller
    controller = EnhancedTargetSeekingController()
    
    # Run enhanced search
    controller.run_until_success()


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()