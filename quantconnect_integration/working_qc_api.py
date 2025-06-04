#!/usr/bin/env python3
"""
Working QuantConnect Cloud API Implementation
This version successfully creates projects, compiles them, and runs backtests
"""

import requests
import hashlib
import base64
import time
import json
from typing import Dict, Any, Optional

class QuantConnectCloudAPI:
    """Working QuantConnect Cloud API implementation"""
    
    def __init__(self, user_id: str, api_token: str):
        self.user_id = user_id
        self.api_token = api_token
        self.base_url = "https://www.quantconnect.com/api/v2"
        
    def get_headers(self) -> Dict[str, str]:
        """Generate proper authentication headers"""
        timestamp = str(int(time.time()))
        time_stamped_token = f"{self.api_token}:{timestamp}".encode('utf-8')
        hashed_token = hashlib.sha256(time_stamped_token).hexdigest()
        authentication = f"{self.user_id}:{hashed_token}".encode('utf-8')
        authentication = base64.b64encode(authentication).decode('ascii')
        
        return {
            'Authorization': f'Basic {authentication}',
            'Timestamp': timestamp,
            'Content-Type': 'application/json'
        }
    
    def create_project(self, name: str, language: str = "Py") -> Optional[str]:
        """Create a new project and return project ID"""
        url = f"{self.base_url}/projects/create"
        headers = self.get_headers()
        
        data = {
            "name": name,
            "language": language
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            if result.get('success') and 'projects' in result:
                project_id = result['projects'][0]['projectId']
                print(f"✅ Project created: {name} (ID: {project_id})")
                return str(project_id)
        
        print(f"❌ Failed to create project: {response.text}")
        return None
    
    def add_file(self, project_id: str, filename: str, content: str) -> bool:
        """Add or update a file in the project"""
        # Use files/update instead of files/create to update existing main.py
        url = f"{self.base_url}/files/update"
        headers = self.get_headers()
        
        data = {
            "projectId": int(project_id),
            "name": filename,
            "content": content
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            if result.get('success'):
                print(f"✅ File updated: {filename}")
                return True
        
        print(f"❌ Failed to update file: {response.text}")
        return False
    
    def compile_project(self, project_id: str) -> Optional[str]:
        """Compile project and return compile ID"""
        url = f"{self.base_url}/compile/create"
        headers = self.get_headers()
        
        data = {
            "projectId": project_id
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            if result.get('success') and 'compileId' in result:
                compile_id = result['compileId']
                print(f"✅ Project compiled (ID: {compile_id})")
                return compile_id
        
        print(f"❌ Failed to compile: {response.text}")
        return None
    
    def create_backtest(self, project_id: str, compile_id: str, name: str = "API Backtest") -> Optional[str]:
        """Create backtest and return backtest ID"""
        url = f"{self.base_url}/backtests/create"
        
        # Wait for compile to complete and retry
        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"Waiting for compile to complete... (attempt {attempt + 1}/{max_retries})")
                time.sleep(10)  # Wait 10 seconds
            
            headers = self.get_headers()
            data = {
                "projectId": project_id,
                "compileId": compile_id,
                "backtestName": name
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = json.loads(response.text)
                if result.get('success'):
                    # Check for backtestId in different possible locations
                    backtest_id = None
                    if 'backtestId' in result:
                        backtest_id = result['backtestId']
                    elif 'backtest' in result and 'backtestId' in result['backtest']:
                        backtest_id = result['backtest']['backtestId']
                    
                    if backtest_id:
                        print(f"✅ Backtest created (ID: {backtest_id})")
                        print(f"🌐 View at: https://www.quantconnect.com/terminal/{project_id}#open/{backtest_id}")
                        return backtest_id
                    
                elif "Compile id not found" in response.text and attempt < max_retries - 1:
                    continue  # Retry
            
            print(f"❌ Failed to create backtest (attempt {attempt + 1}): {response.text}")
        
        return None
    
    def deploy_strategy(self, strategy_name: str, strategy_code: str) -> Dict[str, Any]:
        """
        Complete workflow to deploy a strategy:
        1. Create project
        2. Upload code
        3. Compile
        4. Run backtest
        """
        print(f"\n{'='*60}")
        print(f"🚀 Deploying Strategy: {strategy_name}")
        print(f"{'='*60}")
        
        # Step 1: Create project
        project_name = f"{strategy_name}_{int(time.time())}"
        project_id = self.create_project(project_name)
        if not project_id:
            return {'success': False, 'error': 'Failed to create project'}
        
        # Step 2: Upload code
        if not self.add_file(project_id, "main.py", strategy_code):
            return {'success': False, 'error': 'Failed to upload code'}
        
        # Step 3: Compile
        compile_id = self.compile_project(project_id)
        if not compile_id:
            return {'success': False, 'error': 'Failed to compile project'}
        
        # Step 4: Create backtest
        backtest_id = self.create_backtest(project_id, compile_id, f"{strategy_name} Backtest")
        if not backtest_id:
            return {'success': False, 'error': 'Failed to create backtest'}
        
        return {
            'success': True,
            'project_id': project_id,
            'compile_id': compile_id,
            'backtest_id': backtest_id,
            'url': f"https://www.quantconnect.com/terminal/{project_id}#open/{backtest_id}"
        }
    
    def read_backtest_results(self, project_id: str, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Read actual backtest results from QuantConnect"""
        url = f"{self.base_url}/backtests/read"
        headers = self.get_headers()
        
        data = {
            "projectId": int(project_id),
            "backtestId": backtest_id
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            if result.get('success') and 'backtest' in result:
                backtest_data = result['backtest']
                
                # Look for overall statistics first
                if 'statistics' in backtest_data:
                    # Try to find overall summary statistics
                    overall_stats = backtest_data['statistics']
                    print(f"Found overall statistics section")
                    
                    # Extract from overall stats
                    cagr = self._extract_stat(overall_stats, 'Compounding Annual Return')
                    sharpe = self._extract_stat(overall_stats, 'Sharpe Ratio') 
                    total_trades = self._extract_stat(overall_stats, 'Total Orders')
                    drawdown = self._extract_stat(overall_stats, 'Drawdown')
                    win_rate = self._extract_stat(overall_stats, 'Win Rate')
                    net_profit = self._extract_stat(overall_stats, 'Net Profit')
                    alpha = self._extract_stat(overall_stats, 'Alpha')
                    beta = self._extract_stat(overall_stats, 'Beta')
                    
                elif 'rollingWindow' in backtest_data:
                    # Fallback to rolling window approach (calculate cumulative)
                    print(f"Using rolling window approach")
                    rolling_windows = backtest_data['rollingWindow']
                    
                    # Calculate total trades across all periods
                    total_trades = 0
                    for period_data in rolling_windows.values():
                        if 'tradeStatistics' in period_data:
                            period_trades = self._extract_stat(period_data['tradeStatistics'], 'totalNumberOfTrades')
                            total_trades += period_trades
                    
                    # Use the latest portfolio stats for performance metrics
                    last_key = sorted(rolling_windows.keys())[-1]
                    final_stats = rolling_windows[last_key]['portfolioStatistics']
                    trade_stats = rolling_windows[last_key]['tradeStatistics']
                    
                    # For multi-period strategies, need to calculate overall CAGR differently
                    start_equity = 100000  # Initial value
                    end_equity = self._extract_stat(final_stats, 'endEquity')
                    years = 15  # 2009-2023
                    
                    if end_equity > 0 and start_equity > 0:
                        cagr = ((end_equity / start_equity) ** (1/years) - 1) * 100
                    else:
                        cagr = 0
                    
                    sharpe = self._extract_stat(final_stats, 'sharpeRatio')
                    drawdown = self._extract_stat(final_stats, 'drawdown') * 100
                    net_profit = ((end_equity / start_equity) - 1) * 100
                    alpha = self._extract_stat(final_stats, 'alpha')
                    beta = self._extract_stat(final_stats, 'beta')
                    win_rate = self._extract_stat(trade_stats, 'winRate') * 100
                else:
                    print("No statistics found")
                    return None
                
                performance_data = {
                    'cagr': cagr,
                    'sharpe': sharpe,
                    'total_orders': total_trades,
                    'drawdown': drawdown,
                    'win_rate': win_rate,
                    'net_profit': net_profit,
                    'alpha': alpha,
                    'beta': beta
                }
                
                print(f"✅ Read backtest results: {performance_data['cagr']:.3f}% CAGR")
                return performance_data
        
        print(f"❌ Failed to read backtest results: {response.text}")
        return None
    
    def _extract_stat(self, stats: Dict, key: str) -> float:
        """Extract and convert statistic value"""
        try:
            value = stats.get(key, "0")
            # Remove % signs and convert to float
            if isinstance(value, str):
                value = value.replace('%', '').replace('$', '').replace(',', '')
            return float(value)
        except:
            return 0.0

def get_high_performance_strategy_code():
    """Get a high-performance strategy targeting 25% CAGR"""
    return '''from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class HighPerformanceStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Enable margin for leverage
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # High-performance ETFs
        self.tqqq = self.AddEquity("TQQQ", Resolution.Hour).Symbol  # 3x Nasdaq
        self.spy = self.AddEquity("SPY", Resolution.Hour).Symbol    # S&P 500
        self.vxx = self.AddEquity("VXX", Resolution.Hour).Symbol    # Volatility
        
        # Technical indicators
        self.spy_sma_fast = self.SMA("SPY", 10, Resolution.Daily)
        self.spy_sma_slow = self.SMA("SPY", 30, Resolution.Daily)
        self.spy_rsi = self.RSI("SPY", 14, Resolution.Daily)
        self.vxx_sma = self.SMA("VXX", 10, Resolution.Daily)
        
        # Performance tracking
        self.last_rebalance = self.Time
        self.performance_window = RollingWindow[float](30)
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.Rebalance)
        
        self.Debug("High Performance Strategy initialized")
    
    def OnData(self, data):
        # Track daily performance
        if self.Portfolio.TotalPortfolioValue > 0:
            daily_return = (self.Portfolio.TotalPortfolioValue - self.Portfolio.TotalHoldingsValue) / self.Portfolio.TotalPortfolioValue
            self.performance_window.Add(daily_return)
    
    def Rebalance(self):
        """Execute high-performance rebalancing logic"""
        if not self.spy_sma_fast.IsReady or not self.spy_rsi.IsReady:
            return
        
        # Market conditions
        trend_up = self.spy_sma_fast.Current.Value > self.spy_sma_slow.Current.Value
        momentum_strong = self.spy_rsi.Current.Value > 50 and self.spy_rsi.Current.Value < 70
        volatility_low = self.vxx_sma.IsReady and self.Securities[self.vxx].Price < self.vxx_sma.Current.Value
        
        # High-performance allocation logic
        if trend_up and momentum_strong and volatility_low:
            # Aggressive growth mode - leverage up
            self.SetHoldings(self.tqqq, 1.5)  # 150% TQQQ
            self.SetHoldings(self.spy, -0.3)  # Small hedge
            self.Debug(f"AGGRESSIVE MODE: TQQQ 150%, SPY -30%")
            
        elif trend_up and momentum_strong:
            # Growth mode
            self.SetHoldings(self.tqqq, 0.8)  # 80% TQQQ
            self.SetHoldings(self.spy, 0.2)   # 20% SPY
            self.Debug(f"GROWTH MODE: TQQQ 80%, SPY 20%")
            
        elif not trend_up and self.spy_rsi.Current.Value < 30:
            # Oversold bounce play
            self.SetHoldings(self.spy, 1.2)   # 120% SPY
            self.SetHoldings(self.vxx, 0.1)   # 10% volatility hedge
            self.Debug(f"BOUNCE MODE: SPY 120%, VXX 10%")
            
        else:
            # Risk-off mode
            self.SetHoldings(self.spy, 0.5)   # 50% SPY
            self.SetHoldings([], 0.5)          # 50% cash
            self.Debug(f"RISK-OFF MODE: SPY 50%, Cash 50%")
    
    def OnEndOfAlgorithm(self):
        """Log final performance"""
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        total_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000 * 100
        self.Debug(f"Total Return: {total_return:.2f}%")'''

def main():
    """Test the working API implementation"""
    # Credentials
    USER_ID = "357130"
    API_TOKEN = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    # Initialize API
    api = QuantConnectCloudAPI(USER_ID, API_TOKEN)
    
    print("🚀 QUANTCONNECT CLOUD DEPLOYMENT")
    print("=" * 60)
    
    # Deploy high-performance strategy
    strategy_code = get_high_performance_strategy_code()
    result = api.deploy_strategy("HighPerformance25CAGR", strategy_code)
    
    if result['success']:
        print(f"\n✅ DEPLOYMENT SUCCESSFUL!")
        print(f"Project ID: {result['project_id']}")
        print(f"Backtest ID: {result['backtest_id']}")
        print(f"View Results: {result['url']}")
    else:
        print(f"\n❌ DEPLOYMENT FAILED: {result['error']}")

if __name__ == "__main__":
    main()