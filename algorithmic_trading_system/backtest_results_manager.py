#!/usr/bin/env python3
"""
BACKTEST RESULTS MANAGER - Fetch and store QuantConnect backtest results locally
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add QuantConnect integration path
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')
from working_qc_api import QuantConnectCloudAPI

class BacktestResultsManager:
    """Manages fetching and storing backtest results from QuantConnect Cloud API"""
    
    def __init__(self, results_dir: str = None):
        if results_dir is None:
            results_dir = "/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/backtests"
        
        self.results_dir = results_dir
        self.api = QuantConnectCloudAPI(
            "357130", 
            "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
        )
        
        # Ensure directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create subdirectories for organization
        self.subdirs = {
            'raw': os.path.join(self.results_dir, 'raw'),           # Raw API responses
            'processed': os.path.join(self.results_dir, 'processed'), # Cleaned metrics
            'summaries': os.path.join(self.results_dir, 'summaries'), # Summary reports
            'strategies': os.path.join(self.results_dir, 'strategies') # Strategy code archive
        }
        
        for subdir in self.subdirs.values():
            os.makedirs(subdir, exist_ok=True)
        
        print(f"📁 BacktestResultsManager initialized")
        print(f"   Results directory: {self.results_dir}")
        print(f"   Subdirectories: {list(self.subdirs.keys())}")
    
    def fetch_and_store_backtest(self, project_id: str, backtest_id: str, 
                                strategy_name: str = None, strategy_code: str = None,
                                metadata: Dict = None) -> Optional[Dict]:
        """
        Fetch backtest results from QuantConnect API and store locally
        
        Args:
            project_id: QuantConnect project ID
            backtest_id: QuantConnect backtest ID  
            strategy_name: Optional strategy name for identification
            strategy_code: Optional strategy source code
            metadata: Optional additional metadata
            
        Returns:
            Dictionary with results and file paths, or None if failed
        """
        
        timestamp = int(time.time())
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not strategy_name:
            strategy_name = f"strategy_{project_id}_{backtest_id[:8]}"
        
        print(f"📊 Fetching backtest results...")
        print(f"   Project ID: {project_id}")
        print(f"   Backtest ID: {backtest_id}")
        print(f"   Strategy: {strategy_name}")
        
        try:
            # Fetch results from QuantConnect API
            print("   🔄 Calling QuantConnect API...")
            raw_results = self.api.read_backtest_results(project_id, backtest_id)
            
            if not raw_results:
                print("   ❌ No results returned from API")
                return None
            
            print(f"   ✅ Results fetched: {raw_results.get('cagr', 'N/A')}% CAGR")
            
            # Create comprehensive result package
            result_package = {
                'fetch_info': {
                    'timestamp': timestamp,
                    'datetime': datetime_str,
                    'project_id': project_id,
                    'backtest_id': backtest_id,
                    'strategy_name': strategy_name,
                    'api_source': 'quantconnect_cloud'
                },
                'raw_results': raw_results,
                'processed_metrics': self._process_metrics(raw_results),
                'metadata': metadata or {},
                'files': {}
            }
            
            # Store raw results
            raw_filename = f"{strategy_name}_{datetime_str}_raw.json"
            raw_path = os.path.join(self.subdirs['raw'], raw_filename)
            
            with open(raw_path, 'w') as f:
                json.dump({
                    'project_id': project_id,
                    'backtest_id': backtest_id,
                    'strategy_name': strategy_name,
                    'timestamp': timestamp,
                    'raw_api_response': raw_results
                }, f, indent=2)
            
            result_package['files']['raw'] = raw_path
            print(f"   💾 Raw results saved: {raw_filename}")
            
            # Store processed metrics
            processed_filename = f"{strategy_name}_{datetime_str}_metrics.json"
            processed_path = os.path.join(self.subdirs['processed'], processed_filename)
            
            with open(processed_path, 'w') as f:
                json.dump(result_package['processed_metrics'], f, indent=2)
            
            result_package['files']['processed'] = processed_path
            print(f"   📈 Processed metrics saved: {processed_filename}")
            
            # Store strategy code if provided
            if strategy_code:
                code_filename = f"{strategy_name}_{datetime_str}_strategy.py"
                code_path = os.path.join(self.subdirs['strategies'], code_filename)
                
                with open(code_path, 'w') as f:
                    f.write(f"# Strategy: {strategy_name}\n")
                    f.write(f"# Project ID: {project_id}\n") 
                    f.write(f"# Backtest ID: {backtest_id}\n")
                    f.write(f"# CAGR: {raw_results.get('cagr', 'N/A')}%\n")
                    f.write(f"# Fetched: {datetime_str}\n\n")
                    f.write(strategy_code)
                
                result_package['files']['strategy_code'] = code_path
                print(f"   🧬 Strategy code saved: {code_filename}")
            
            # Create summary report
            summary_filename = f"{strategy_name}_{datetime_str}_summary.txt"
            summary_path = os.path.join(self.subdirs['summaries'], summary_filename)
            result_package['files']['summary'] = summary_path
            
            summary_content = self._generate_summary_report(result_package)
            with open(summary_path, 'w') as f:
                f.write(summary_content)
            
            print(f"   📋 Summary report saved: {summary_filename}")
            
            # Update index file
            self._update_results_index(result_package)
            
            print(f"   ✅ Complete backtest package stored successfully")
            return result_package
            
        except Exception as e:
            print(f"   ❌ Error fetching/storing results: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_metrics(self, raw_results: Dict) -> Dict:
        """Process raw API results into standardized metrics"""
        
        processed = {
            'performance': {
                'cagr': raw_results.get('cagr', 0.0),
                'sharpe_ratio': raw_results.get('sharpe', 0.0),
                'max_drawdown': raw_results.get('drawdown', 0.0),
                'net_profit': raw_results.get('net_profit', 0.0),
                'alpha': raw_results.get('alpha', 0.0),
                'beta': raw_results.get('beta', 0.0)
            },
            'trading': {
                'total_orders': raw_results.get('total_orders', 0),
                'win_rate': raw_results.get('win_rate', 0.0)
            },
            'classification': self._classify_performance(raw_results),
            'risk_metrics': self._calculate_risk_metrics(raw_results)
        }
        
        return processed
    
    def _classify_performance(self, results: Dict) -> Dict:
        """Classify performance into categories"""
        
        cagr = results.get('cagr', 0)
        sharpe = results.get('sharpe', 0)
        drawdown = results.get('drawdown', 100)
        
        # CAGR classification
        if cagr >= 15:
            cagr_class = "excellent"
        elif cagr >= 12:
            cagr_class = "good"
        elif cagr >= 8:
            cagr_class = "decent"
        elif cagr >= 4:
            cagr_class = "weak"
        else:
            cagr_class = "poor"
        
        # Sharpe classification
        if sharpe >= 1.5:
            sharpe_class = "excellent"
        elif sharpe >= 1.0:
            sharpe_class = "good"
        elif sharpe >= 0.5:
            sharpe_class = "decent"
        else:
            sharpe_class = "poor"
        
        # Drawdown classification
        if drawdown <= 10:
            dd_class = "excellent"
        elif drawdown <= 20:
            dd_class = "good"
        elif drawdown <= 30:
            dd_class = "acceptable"
        else:
            dd_class = "poor"
        
        # Overall grade
        scores = {"excellent": 4, "good": 3, "decent": 2, "weak": 1, "acceptable": 2, "poor": 0}
        avg_score = (scores[cagr_class] + scores[sharpe_class] + scores[dd_class]) / 3
        
        if avg_score >= 3.5:
            overall = "A"
        elif avg_score >= 2.5:
            overall = "B"
        elif avg_score >= 1.5:
            overall = "C"
        else:
            overall = "D"
        
        return {
            'cagr_class': cagr_class,
            'sharpe_class': sharpe_class,
            'drawdown_class': dd_class,
            'overall_grade': overall,
            'score': avg_score
        }
    
    def _calculate_risk_metrics(self, results: Dict) -> Dict:
        """Calculate additional risk metrics"""
        
        cagr = results.get('cagr', 0)
        sharpe = results.get('sharpe', 0)
        drawdown = results.get('drawdown', 0)
        
        # Risk-adjusted return
        risk_adj_return = cagr / max(drawdown, 1) if drawdown > 0 else cagr
        
        # Volatility estimate (from Sharpe)
        volatility = (cagr / sharpe) if sharpe != 0 else 0
        
        return {
            'risk_adjusted_return': risk_adj_return,
            'estimated_volatility': abs(volatility),
            'calmar_ratio': cagr / max(drawdown, 1) if drawdown > 0 else cagr
        }
    
    def _generate_summary_report(self, result_package: Dict) -> str:
        """Generate human-readable summary report"""
        
        fetch_info = result_package['fetch_info']
        metrics = result_package['processed_metrics']
        classification = metrics['classification']
        
        report = f"""
BACKTEST RESULTS SUMMARY
========================

Strategy: {fetch_info['strategy_name']}
Project ID: {fetch_info['project_id']}
Backtest ID: {fetch_info['backtest_id']}
Fetched: {fetch_info['datetime']}

PERFORMANCE METRICS
===================
CAGR: {metrics['performance']['cagr']:.2f}% ({classification['cagr_class']})
Sharpe Ratio: {metrics['performance']['sharpe_ratio']:.3f} ({classification['sharpe_class']})
Max Drawdown: {metrics['performance']['max_drawdown']:.1f}% ({classification['drawdown_class']})
Net Profit: {metrics['performance']['net_profit']:.2f}%
Alpha: {metrics['performance']['alpha']:.3f}
Beta: {metrics['performance']['beta']:.3f}

TRADING ACTIVITY
================
Total Orders: {metrics['trading']['total_orders']}
Win Rate: {metrics['trading']['win_rate']:.1f}%

RISK ANALYSIS
=============
Risk-Adjusted Return: {metrics['risk_metrics']['risk_adjusted_return']:.3f}
Estimated Volatility: {metrics['risk_metrics']['estimated_volatility']:.1f}%
Calmar Ratio: {metrics['risk_metrics']['calmar_ratio']:.3f}

OVERALL ASSESSMENT
==================
Grade: {classification['overall_grade']}
Score: {classification['score']:.1f}/4.0

FILES CREATED
=============
Raw Results: {os.path.basename(result_package['files']['raw'])}
Processed Metrics: {os.path.basename(result_package['files']['processed'])}
Summary Report: {os.path.basename(result_package['files']['summary'])}
"""
        
        if 'strategy_code' in result_package['files']:
            report += f"Strategy Code: {os.path.basename(result_package['files']['strategy_code'])}\n"
        
        return report
    
    def _update_results_index(self, result_package: Dict):
        """Update master index of all results"""
        
        index_file = os.path.join(self.results_dir, "results_index.json")
        
        # Load existing index or create new
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {
                'created': datetime.now().isoformat(),
                'total_backtests': 0,
                'backtests': []
            }
        
        # Add new entry
        entry = {
            'strategy_name': result_package['fetch_info']['strategy_name'],
            'project_id': result_package['fetch_info']['project_id'],
            'backtest_id': result_package['fetch_info']['backtest_id'],
            'timestamp': result_package['fetch_info']['timestamp'],
            'datetime': result_package['fetch_info']['datetime'],
            'cagr': result_package['processed_metrics']['performance']['cagr'],
            'sharpe': result_package['processed_metrics']['performance']['sharpe_ratio'],
            'drawdown': result_package['processed_metrics']['performance']['max_drawdown'],
            'grade': result_package['processed_metrics']['classification']['overall_grade'],
            'files': result_package['files']
        }
        
        index['backtests'].append(entry)
        index['total_backtests'] = len(index['backtests'])
        index['last_updated'] = datetime.now().isoformat()
        
        # Save updated index
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"   📇 Results index updated: {index['total_backtests']} total backtests")
    
    def list_stored_results(self, limit: int = 10) -> List[Dict]:
        """List recently stored backtest results"""
        
        index_file = os.path.join(self.results_dir, "results_index.json")
        
        if not os.path.exists(index_file):
            print("📋 No backtest results found")
            return []
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        recent_results = sorted(index['backtests'], 
                              key=lambda x: x['timestamp'], 
                              reverse=True)[:limit]
        
        print(f"📋 Recent Backtest Results ({len(recent_results)}/{index['total_backtests']})")
        print("=" * 60)
        
        for i, result in enumerate(recent_results):
            print(f"{i+1}. {result['strategy_name']} ({result['datetime']})")
            print(f"   CAGR: {result['cagr']:.2f}% | Sharpe: {result['sharpe']:.2f} | Grade: {result['grade']}")
            print(f"   Project: {result['project_id']} | Backtest: {result['backtest_id'][:8]}...")
            print()
        
        return recent_results
    
    def get_best_strategies(self, metric: str = 'cagr', limit: int = 5) -> List[Dict]:
        """Get top performing strategies by specified metric"""
        
        index_file = os.path.join(self.results_dir, "results_index.json")
        
        if not os.path.exists(index_file):
            return []
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        if metric == 'cagr':
            top_results = sorted(index['backtests'], 
                               key=lambda x: x['cagr'], 
                               reverse=True)[:limit]
        elif metric == 'sharpe':
            top_results = sorted(index['backtests'], 
                               key=lambda x: x['sharpe'], 
                               reverse=True)[:limit]
        elif metric == 'grade':
            grade_order = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
            top_results = sorted(index['backtests'], 
                               key=lambda x: grade_order.get(x['grade'], 0), 
                               reverse=True)[:limit]
        else:
            top_results = index['backtests'][:limit]
        
        print(f"🏆 Top {limit} Strategies by {metric.upper()}")
        print("=" * 50)
        
        for i, result in enumerate(top_results):
            print(f"{i+1}. {result['strategy_name']}")
            print(f"   CAGR: {result['cagr']:.2f}% | Sharpe: {result['sharpe']:.2f} | Grade: {result['grade']}")
            print()
        
        return top_results


def test_results_manager():
    """Test the backtest results manager with recent results"""
    
    print("🧪 TESTING BACKTEST RESULTS MANAGER")
    print("=" * 50)
    
    manager = BacktestResultsManager()
    
    # Test with recent backtest results we know exist
    test_cases = [
        {
            'project_id': '23358162',
            'backtest_id': 'e4d1937c4ed4b802d7d84ab1ff724ea6',
            'strategy_name': 'MinimalTest_Validation',
            'metadata': {'test': 'validation', 'target_cagr': 'none'}
        },
        {
            'project_id': '23358651',
            'backtest_id': 'cc5f4ff8f142f60d225e60b372f39e3c',
            'strategy_name': 'Realistic12Percent_Phase2',
            'metadata': {'test': 'realistic_target', 'target_cagr': 12.0}
        }
    ]
    
    print(f"\n📊 Testing with {len(test_cases)} known backtest results...")
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {test_case['strategy_name']} ---")
        
        result = manager.fetch_and_store_backtest(
            project_id=test_case['project_id'],
            backtest_id=test_case['backtest_id'],
            strategy_name=test_case['strategy_name'],
            metadata=test_case['metadata']
        )
        
        if result:
            print(f"✅ Successfully stored: {test_case['strategy_name']}")
        else:
            print(f"❌ Failed to store: {test_case['strategy_name']}")
    
    # Show stored results
    print(f"\n📋 STORED RESULTS SUMMARY:")
    manager.list_stored_results()
    
    print(f"\n🏆 BEST STRATEGIES:")
    manager.get_best_strategies('cagr', 3)


if __name__ == "__main__":
    test_results_manager()