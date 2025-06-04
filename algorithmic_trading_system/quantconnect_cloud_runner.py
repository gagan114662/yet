#!/usr/bin/env python3
"""
QuantConnect Cloud Backtest Runner
==================================

Runs backtests directly on QuantConnect's cloud using your credentials.
This bypasses local data issues and uses QuantConnect's professional data.
"""

import requests
import json
import time
import sys
from datetime import datetime

class QuantConnectCloudRunner:
    def __init__(self, user_id: str, api_token: str):
        self.user_id = user_id
        self.api_token = api_token
        self.base_url = "https://www.quantconnect.com/api/v2"
        # Encode credentials for Basic auth
        import base64
        auth_string = f"{user_id}:{api_token}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        self.headers = {
            "Authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/json"
        }
    
    def create_project(self, name: str) -> str:
        """Create a new project on QuantConnect"""
        url = f"{self.base_url}/projects/create"
        data = {
            "projectName": name,
            "language": "Py"
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if "projects" in result and result["projects"]:
                return result["projects"][0].get("projectId")
            elif "projectId" in result:
                return result["projectId"]
            else:
                print(f"Unexpected response format: {result}")
                return None
        else:
            print(f"Failed to create project: Status {response.status_code}")
            print(f"Response: {response.text}")
            return None
    
    def upload_file(self, project_id: str, filename: str, content: str):
        """Upload a file to the project"""
        url = f"{self.base_url}/files/create"
        data = {
            "projectId": project_id,
            "name": filename,
            "content": content
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        return response.status_code == 200
    
    def run_backtest(self, project_id: str, name: str) -> str:
        """Run a backtest on the project"""
        url = f"{self.base_url}/backtests/create"
        data = {
            "projectId": project_id,
            "compileId": "",
            "backtestName": name
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result.get("backtestId")
        else:
            print(f"Failed to start backtest: {response.text}")
            return None
    
    def get_backtest_status(self, project_id: str, backtest_id: str):
        """Get backtest status and results"""
        url = f"{self.base_url}/backtests/read"
        params = {
            "projectId": project_id,
            "backtestId": backtest_id
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        return None

def create_aggressive_strategy_code():
    """Create the ultimate aggressive strategy for 25%+ CAGR"""
    return '''from AlgorithmImports import *

class UltimateCloudCrusher(QCAlgorithm):
    """ULTIMATE 25%+ CAGR CLOUD STRATEGY with QuantConnect Data"""
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Maximum leverage configuration
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # Core high-performance assets
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy.SetLeverage(5.0)
        
        self.tqqq = self.AddEquity("TQQQ", Resolution.Daily)
        self.tqqq.SetLeverage(3.0)  # 3x leveraged ETF
        
        # Technical indicators
        self.spy_sma_fast = self.SMA("SPY", 5)
        self.spy_sma_slow = self.SMA("SPY", 20)
        self.spy_rsi = self.RSI("SPY", 14)
        
        self.rebalance_time = self.Time
        
    def OnData(self, data):
        if not (self.spy_sma_fast.IsReady and self.spy_sma_slow.IsReady and self.spy_rsi.IsReady):
            return
            
        # Rebalance weekly for maximum alpha
        if (self.Time - self.rebalance_time).days < 5:
            return
            
        self.rebalance_time = self.Time
        
        # AGGRESSIVE MOMENTUM STRATEGY
        if self.spy_sma_fast.Current.Value > self.spy_sma_slow.Current.Value:
            if self.spy_rsi.Current.Value < 30:
                # Oversold in uptrend - MAXIMUM POSITION
                self.SetHoldings("TQQQ", 2.0)  # 200% in 3x leveraged tech
                self.Log(f"MAXIMUM BULL: 200% TQQQ @ ${data['TQQQ'].Close}")
            elif self.spy_rsi.Current.Value < 70:
                # Normal uptrend
                self.SetHoldings("SPY", 3.0)  # 300% SPY with 5x leverage
                self.Log(f"BULL: 300% SPY @ ${data['SPY'].Close}")
            else:
                # Overbought - reduce exposure
                self.SetHoldings("SPY", 1.0)
        else:
            # Downtrend - go to cash or short
            self.Liquidate()
            self.Log(f"BEAR: Liquidated @ SPY ${data['SPY'].Close}")
            
    def OnEndOfAlgorithm(self):
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value / 100000) - 1
        days = (self.EndDate - self.StartDate).days
        cagr = ((final_value / 100000) ** (365.25 / days)) - 1
        
        self.Log("=" * 60)
        self.Log("🚀 ULTIMATE CLOUD CRUSHER RESULTS")
        self.Log("=" * 60)
        self.Log(f"Final Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return*100:.1f}%")
        self.Log(f"CAGR: {cagr*100:.1f}%")
        self.Log("=" * 60)
        
        if cagr >= 0.25:
            self.Log("🏆 25% TARGET CRUSHED!")
        else:
            self.Log(f"Target: {(0.25-cagr)*100:.1f}% short")
'''

def main():
    print("🚀 QUANTCONNECT CLOUD BACKTEST RUNNER")
    print("=" * 60)
    print("Running strategies on QuantConnect's cloud with REAL professional data!")
    print()
    
    # Your QuantConnect credentials
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    runner = QuantConnectCloudRunner(user_id, api_token)
    
    # Create project
    project_name = f"AggressiveTargetCrusher_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Creating project: {project_name}")
    
    project_id = runner.create_project(project_name)
    if not project_id:
        print("❌ Failed to create project")
        return
    
    print(f"✅ Project created: {project_id}")
    
    # Upload strategy
    strategy_code = create_aggressive_strategy_code()
    if runner.upload_file(project_id, "main.py", strategy_code):
        print("✅ Strategy uploaded")
    else:
        print("❌ Failed to upload strategy")
        return
    
    # Run backtest
    backtest_id = runner.run_backtest(project_id, "AggressiveTarget25CAGR")
    if not backtest_id:
        print("❌ Failed to start backtest")
        return
    
    print(f"✅ Backtest started: {backtest_id}")
    print("⏳ Waiting for results...")
    
    # Wait for completion
    max_wait = 300  # 5 minutes
    wait_time = 0
    
    while wait_time < max_wait:
        result = runner.get_backtest_status(project_id, backtest_id)
        if result and "backtests" in result:
            backtest = result["backtests"][0]
            if backtest.get("completed"):
                print("✅ Backtest completed!")
                
                # Extract results
                statistics = backtest.get("statistics", {})
                
                print("\n🏆 CLOUD BACKTEST RESULTS:")
                print("=" * 50)
                
                for key, value in statistics.items():
                    if any(keyword in key.lower() for keyword in ["return", "sharpe", "drawdown", "trades"]):
                        print(f"{key}: {value}")
                
                # Check if we hit targets
                cagr_str = statistics.get("Compounding Annual Return", "0%")
                try:
                    cagr = float(cagr_str.replace("%", "")) / 100
                    if cagr >= 0.25:
                        print("\n🎉 25% CAGR TARGET ACHIEVED!")
                    else:
                        print(f"\n📈 Got {cagr*100:.1f}% CAGR - need {(0.25-cagr)*100:.1f}% more")
                except:
                    print(f"\n📊 CAGR: {cagr_str}")
                
                print(f"\n🔗 View full results: https://www.quantconnect.com/terminal/processCache?request={backtest_id}")
                break
        
        time.sleep(10)
        wait_time += 10
        print(f"⏳ Waiting... ({wait_time}s)")
    
    if wait_time >= max_wait:
        print("⏰ Timeout - check QuantConnect web interface for results")
        print(f"🔗 Project: https://www.quantconnect.com/terminal/index#open/{project_id}")

if __name__ == "__main__":
    main()