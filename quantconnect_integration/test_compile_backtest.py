#!/usr/bin/env python3
"""
Test QuantConnect compilation and backtesting endpoints
"""

import requests
import hashlib
import base64
import time
import json
from typing import Dict, Any

class QuantConnectAPI:
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
    
    def compile_project(self, project_id: str) -> Dict[str, Any]:
        """Try different compile endpoints"""
        headers = self.get_headers()
        
        # Try different endpoint variations
        endpoints = [
            f"{self.base_url}/compile/create",
            f"{self.base_url}/compile",
            f"{self.base_url}/projects/{project_id}/compile",
            f"{self.base_url}/compile/project"
        ]
        
        for endpoint in endpoints:
            print(f"\nTrying endpoint: {endpoint}")
            
            # Try POST with projectId in body
            response = requests.post(endpoint, headers=headers, json={"projectId": project_id})
            print(f"POST with body - Status: {response.status_code}")
            if response.status_code == 200:
                return {'success': True, 'response': response.text, 'endpoint': endpoint}
            print(f"Response: {response.text[:100]}...")
            
            # Try GET with projectId as parameter
            response = requests.get(endpoint, headers=headers, params={"projectId": project_id})
            print(f"GET with params - Status: {response.status_code}")
            if response.status_code == 200:
                return {'success': True, 'response': response.text, 'endpoint': endpoint}
            print(f"Response: {response.text[:100]}...")
        
        return {'success': False, 'message': 'No working compile endpoint found'}
    
    def create_backtest(self, project_id: str) -> Dict[str, Any]:
        """Try different backtest endpoints"""
        headers = self.get_headers()
        
        # Try different endpoint variations
        endpoints = [
            f"{self.base_url}/backtests/create",
            f"{self.base_url}/backtest/create",
            f"{self.base_url}/projects/{project_id}/backtests",
            f"{self.base_url}/backtests"
        ]
        
        for endpoint in endpoints:
            print(f"\nTrying backtest endpoint: {endpoint}")
            
            # Try POST with projectId in body
            data = {
                "projectId": project_id,
                "name": f"Test_Backtest_{int(time.time())}"
            }
            response = requests.post(endpoint, headers=headers, json=data)
            print(f"POST - Status: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                try:
                    result = json.loads(response.text)
                    if result.get('success'):
                        return {'success': True, 'response': response.text, 'endpoint': endpoint}
                except:
                    pass
        
        return {'success': False, 'message': 'No working backtest endpoint found'}

def main():
    # Your credentials
    USER_ID = "357130"
    API_TOKEN = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    api = QuantConnectAPI(USER_ID, API_TOKEN)
    
    print("🔍 QUANTCONNECT API ENDPOINT DISCOVERY")
    print("=" * 60)
    
    # Use the project we just created
    project_id = "23344795"
    
    print(f"\n📊 Testing compilation for project {project_id}...")
    compile_result = api.compile_project(project_id)
    
    if compile_result['success']:
        print(f"\n✅ Found working compile endpoint: {compile_result['endpoint']}")
        print(f"Response: {compile_result['response']}")
    else:
        print("\n❌ No working compile endpoint found")
    
    print("\n" + "=" * 60)
    print(f"\n🚀 Testing backtest creation for project {project_id}...")
    backtest_result = api.create_backtest(project_id)
    
    if backtest_result['success']:
        print(f"\n✅ Found working backtest endpoint: {backtest_result['endpoint']}")
        print(f"Response: {backtest_result['response']}")
    else:
        print("\n❌ No working backtest endpoint found")

if __name__ == "__main__":
    main()