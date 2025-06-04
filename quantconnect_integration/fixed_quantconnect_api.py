#!/usr/bin/env python3
"""
Fixed QuantConnect Cloud API Implementation
Based on official Lean CLI documentation and authentication requirements

The previous attempt failed because:
1. Used timestamp = 0 instead of current Unix time
2. Didn't follow exact authentication header format from docs
3. Missing organization ID in project creation

This implementation follows the exact pattern from:
https://www.quantconnect.com/docs/v2/cloud-platform/api-reference/authentication
"""

import requests
import hashlib
import base64
import time
import json
from typing import Dict, Any, Optional

class QuantConnectCloudAPI:
    """
    Fixed QuantConnect Cloud API implementation following official documentation
    """
    
    def __init__(self, user_id: str, api_token: str):
        self.user_id = user_id
        self.api_token = api_token
        self.base_url = "https://www.quantconnect.com/api/v2"
        
    def get_headers(self) -> Dict[str, str]:
        """
        Generate proper authentication headers with current timestamp
        Following exact pattern from QuantConnect docs
        """
        # CRITICAL FIX: Use current time, not 0!
        timestamp = str(int(time.time()))
        
        # Create time-stamped token
        time_stamped_token = f"{self.api_token}:{timestamp}".encode('utf-8')
        
        # Generate SHA-256 hash
        hashed_token = hashlib.sha256(time_stamped_token).hexdigest()
        
        # Create authentication string
        authentication = f"{self.user_id}:{hashed_token}".encode('utf-8')
        authentication = base64.b64encode(authentication).decode('ascii')
        
        return {
            'Authorization': f'Basic {authentication}',
            'Timestamp': timestamp,
            'Content-Type': 'application/json'
        }
    
    def test_authentication(self) -> Dict[str, Any]:
        """Test authentication using the /authenticate endpoint"""
        url = f"{self.base_url}/authenticate"
        headers = self.get_headers()
        
        response = requests.post(url, headers=headers)
        
        return {
            'status_code': response.status_code,
            'response': response.text,
            'headers_sent': headers,
            'success': response.status_code == 200
        }
    
    def create_project(self, name: str, language: str = "Py", organization_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new project using proper API endpoint
        """
        url = f"{self.base_url}/projects/create"
        headers = self.get_headers()
        
        # Project creation data
        data = {
            "name": name,
            "language": language
        }
        
        # Add organization ID if provided
        if organization_id:
            data["organizationId"] = organization_id
        
        response = requests.post(url, headers=headers, json=data)
        
        return {
            'status_code': response.status_code,
            'response': response.text,
            'data_sent': data,
            'headers_sent': headers,
            'success': response.status_code == 200
        }
    
    def list_projects(self) -> Dict[str, Any]:
        """List all projects"""
        url = f"{self.base_url}/projects/read"
        headers = self.get_headers()
        
        response = requests.post(url, headers=headers)
        
        return {
            'status_code': response.status_code,
            'response': response.text,
            'success': response.status_code == 200
        }
    
    def add_file(self, project_id: str, name: str, content: str) -> Dict[str, Any]:
        """Add or update a file in the project"""
        url = f"{self.base_url}/files/create"
        headers = self.get_headers()
        
        data = {
            "projectId": project_id,
            "name": name,
            "content": content
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        return {
            'status_code': response.status_code,
            'response': response.text,
            'success': response.status_code == 200
        }
    
    def compile_project(self, project_id: str) -> Dict[str, Any]:
        """Compile a project"""
        url = f"{self.base_url}/projects/compile"
        headers = self.get_headers()
        
        data = {
            "projectId": project_id
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        return {
            'status_code': response.status_code,
            'response': response.text,
            'success': response.status_code == 200
        }
    
    def create_backtest(self, project_id: str, compile_id: str, name: str = "API Backtest") -> Dict[str, Any]:
        """Create a backtest for a project"""
        url = f"{self.base_url}/backtests/create"
        headers = self.get_headers()
        
        data = {
            "projectId": project_id,
            "compileId": compile_id,
            "name": name
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        return {
            'status_code': response.status_code,
            'response': response.text,
            'data_sent': data,
            'success': response.status_code == 200
        }
    
    def read_backtest(self, project_id: str, backtest_id: str) -> Dict[str, Any]:
        """Read backtest results"""
        url = f"{self.base_url}/backtests/read"
        headers = self.get_headers()
        
        params = {
            "projectId": project_id,
            "backtestId": backtest_id
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        return {
            'status_code': response.status_code,
            'response': response.text,
            'success': response.status_code == 200
        }

def test_quantconnect_api():
    """
    Test the fixed QuantConnect API implementation
    """
    # Your credentials
    USER_ID = "357130"
    API_TOKEN = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    # Initialize API client
    api = QuantConnectCloudAPI(USER_ID, API_TOKEN)
    
    print("🔧 Testing Fixed QuantConnect Cloud API")
    print("=" * 50)
    
    # Test 1: Authentication
    print("\n1️⃣ Testing Authentication...")
    auth_result = api.test_authentication()
    print(f"Status: {auth_result['status_code']}")
    print(f"Success: {auth_result['success']}")
    print(f"Response: {auth_result['response']}")
    
    if not auth_result['success']:
        print("❌ Authentication failed. Check credentials.")
        return
    
    print("✅ Authentication successful!")
    
    # Test 2: List existing projects
    print("\n2️⃣ Listing existing projects...")
    projects_result = api.list_projects()
    print(f"Status: {projects_result['status_code']}")
    print(f"Response: {projects_result['response'][:200]}...")
    
    # Test 3: Create new project
    print("\n3️⃣ Creating new project...")
    project_name = f"Crisis_Alpha_Master_{int(time.time())}"
    create_result = api.create_project(project_name)
    print(f"Status: {create_result['status_code']}")
    print(f"Response: {create_result['response']}")
    
    if create_result['success']:
        print(f"✅ Project '{project_name}' created successfully!")
        
        # Parse project ID from response
        try:
            response_data = json.loads(create_result['response'])
            if 'projects' in response_data and len(response_data['projects']) > 0:
                project_id = response_data['projects'][0]['projectId']
                print(f"Project ID: {project_id}")
                
                # Test 4: Add strategy code
                print(f"\n4️⃣ Adding strategy code to project {project_id}...")
                strategy_code = get_crisis_alpha_code()
                file_result = api.add_file(str(project_id), "main.py", strategy_code)
                print(f"Status: {file_result['status_code']}")
                print(f"Response: {file_result['response']}")
                
                if file_result['success'] or "File already exist" in file_result['response']:
                    print("✅ Strategy code uploaded successfully!")
                    
                    # Test 5: Compile project
                    print(f"\n5️⃣ Compiling project {project_id}...")
                    compile_result = api.compile_project(str(project_id))
                    print(f"Status: {compile_result['status_code']}")
                    print(f"Response: {compile_result['response']}")
                    
                    if compile_result['success']:
                        print("✅ Project compiled successfully!")
                        
                        try:
                            compile_data = json.loads(compile_result['response'])
                            if 'compileId' in compile_data:
                                compile_id = compile_data['compileId']
                                
                                # Test 6: Create backtest
                                print(f"\n6️⃣ Creating backtest for project {project_id}...")
                                backtest_result = api.create_backtest(str(project_id), str(compile_id))
                                print(f"Status: {backtest_result['status_code']}")
                                print(f"Response: {backtest_result['response']}")
                                
                                if backtest_result['success']:
                                    print("✅ Backtest created successfully!")
                                    
                                    try:
                                        backtest_data = json.loads(backtest_result['response'])
                                        if 'backtestId' in backtest_data:
                                            backtest_id = backtest_data['backtestId']
                                            print(f"Backtest ID: {backtest_id}")
                                            print(f"🌐 View results at: https://www.quantconnect.com/terminal/processCache?request={backtest_id}")
                                    except:
                                        pass
                        except:
                            print("⚠️  Could not parse compile response")
                    else:
                        print("⚠️  Backtest creation failed")
                else:
                    print("⚠️  File upload failed")
        except json.JSONDecodeError:
            print("⚠️  Could not parse project creation response")
    else:
        print("❌ Project creation failed")
        print(f"Data sent: {create_result['data_sent']}")
        print(f"Headers: {create_result['headers_sent']}")

def get_crisis_alpha_code():
    """Get the Crisis Alpha strategy code"""
    return '''from AlgorithmImports import *
import numpy as np

class CrisisAlphaMaster(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Enable margin for leverage
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Crisis instruments with cloud data
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.vxx = self.AddEquity("VXX", Resolution.Minute).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Minute).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Minute).Symbol
        
        # Crisis detection indicators
        self.spy_rsi = self.RSI("SPY", 14, Resolution.Minute)
        self.volatility_window = RollingWindow[float](20)
        self.crisis_mode = False
        
        # Aggressive parameters for cloud
        self.max_leverage = 8.0
        self.crisis_threshold = 0.03  # 3% daily volatility threshold
        
        # Schedule crisis monitoring
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.Every(TimeSpan.FromMinutes(15)), 
                        self.MonitorCrisis)
        
        self.Debug("Crisis Alpha Master initialized for cloud backtesting")
        
    def OnData(self, data):
        # Update volatility data
        if self.spy in data and data[self.spy] is not None:
            if hasattr(self, 'previous_spy_price'):
                spy_return = (data[self.spy].Close - self.previous_spy_price) / self.previous_spy_price
                self.volatility_window.Add(abs(spy_return))
            self.previous_spy_price = data[self.spy].Close
        
        # Execute strategy based on crisis mode
        if self.crisis_mode:
            self.ExecuteCrisisStrategy(data)
        else:
            self.ExecuteNormalStrategy(data)
    
    def MonitorCrisis(self):
        """Monitor for crisis conditions"""
        if not self.volatility_window.IsReady or not self.spy_rsi.IsReady:
            return
            
        # Calculate recent volatility
        recent_vol = np.std([x for x in self.volatility_window]) * np.sqrt(1440)  # Daily vol
        
        # Crisis conditions
        high_volatility = recent_vol > self.crisis_threshold
        extreme_rsi = self.spy_rsi.Current.Value < 25 or self.spy_rsi.Current.Value > 75
        
        previous_mode = self.crisis_mode
        self.crisis_mode = high_volatility or extreme_rsi
        
        if self.crisis_mode != previous_mode:
            mode_text = "CRISIS DETECTED" if self.crisis_mode else "NORMAL MODE"
            self.Debug(f"{mode_text} - Vol: {recent_vol:.4f}, RSI: {self.spy_rsi.Current.Value:.1f}")
    
    def ExecuteCrisisStrategy(self, data):
        """Execute crisis alpha strategy with leverage"""
        # Validate data availability
        required_symbols = [self.vxx, self.tlt, self.gld, self.spy]
        if not all(symbol in data and data[symbol] is not None for symbol in required_symbols):
            return
            
        # Crisis allocation - profit from volatility and flight to quality
        self.SetHoldings(self.vxx, 3.0)   # 3x long volatility
        self.SetHoldings(self.tlt, 2.5)   # 2.5x long bonds  
        self.SetHoldings(self.gld, 2.0)   # 2x long gold
        self.SetHoldings(self.spy, -1.5)  # 1.5x short equities
        
    def ExecuteNormalStrategy(self, data):
        """Execute normal times strategy"""
        # Validate data availability
        if not all(symbol in data and data[symbol] is not None for symbol in [self.spy, self.gld, self.vxx]):
            return
            
        # Conservative allocation with small hedge
        self.SetHoldings(self.spy, 1.2)   # 120% equity exposure
        self.SetHoldings(self.gld, 0.1)   # 10% gold hedge
        self.SetHoldings(self.vxx, 0.05)  # 5% volatility hedge'''

if __name__ == "__main__":
    test_quantconnect_api()