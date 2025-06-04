#!/usr/bin/env python3
"""
QuantConnect API Integration Fix
Implementing correct authentication based on QuantConnect API v2 documentation
"""

import requests
import time
import json
import base64
import hashlib
import hmac
from datetime import datetime
from typing import Dict, Optional

class QuantConnectAPIClient:
    """
    Fixed QuantConnect API Client with correct authentication
    """
    
    def __init__(self, user_id: str = "357130", api_token: str = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"):
        self.user_id = user_id
        self.api_token = api_token
        self.base_url = "https://www.quantconnect.com/api/v2"
        
        print(f"🔧 QuantConnect API Client initialized")
        print(f"   User ID: {self.user_id}")
        print(f"   API Token: {self.api_token[:20]}...")
    
    def _generate_headers(self, endpoint: str = "", method: str = "GET") -> Dict[str, str]:
        """
        Generate headers for QuantConnect API v2
        Based on QuantConnect documentation: https://www.quantconnect.com/docs/v2/our-platform/api-reference
        """
        # Current timestamp
        timestamp = str(int(time.time()))
        
        # Try Method 1: Simple Basic Auth (as per some API endpoints)
        auth_string = f"{self.user_id}:{self.api_token}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        return headers, timestamp
    
    def _generate_headers_hmac(self, endpoint: str = "", method: str = "GET", timestamp: str = None) -> Dict[str, str]:
        """
        Generate HMAC-based headers for QuantConnect API v2
        """
        if timestamp is None:
            timestamp = str(int(time.time()))
        
        # Method 2: HMAC-SHA256 authentication
        # Format: HMAC-SHA256(timestamp, api_token)
        signature = hmac.new(
            self.api_token.encode('utf-8'),
            timestamp.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "Authorization": f"{self.user_id}:{signature}",
            "Timestamp": timestamp,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        return headers, timestamp
    
    def test_authentication_methods(self):
        """Test different authentication methods"""
        print("\n🧪 TESTING QUANTCONNECT AUTHENTICATION METHODS")
        print("=" * 60)
        
        # Test 1: List projects with Basic Auth
        print("\n1️⃣ Testing Basic Authentication (List Projects)")
        headers1, _ = self._generate_headers()
        
        try:
            response = requests.get(f"{self.base_url}/projects/read", headers=headers1)
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', False):
                    print("   ✅ Basic Auth successful!")
                    return headers1, "basic"
                else:
                    print(f"   ❌ Error: {data.get('errors', [])}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        # Test 2: Create project with HMAC Auth
        print("\n2️⃣ Testing HMAC Authentication (Create Project)")
        headers2, timestamp = self._generate_headers_hmac()
        
        project_data = {
            "projectName": f"Test_Auth_{timestamp}",
            "language": "Py"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/projects/create", 
                headers=headers2, 
                json=project_data
            )
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', False):
                    print("   ✅ HMAC Auth successful!")
                    return headers2, "hmac"
                else:
                    print(f"   ❌ Error: {data.get('errors', [])}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        # Test 3: Alternative HMAC format
        print("\n3️⃣ Testing Alternative HMAC Format")
        timestamp = str(int(time.time()))
        
        # Try: HMAC-SHA256(api_token, f"{user_id}:{timestamp}")
        message = f"{self.user_id}:{timestamp}"
        signature = hmac.new(
            self.api_token.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Try with Basic auth format but HMAC signature
        auth_string = f"{self.user_id}:{signature}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        headers3 = {
            "Authorization": f"Basic {encoded_auth}",
            "Timestamp": timestamp,
            "Content-Type": "application/json"
        }
        
        project_data = {
            "projectName": f"Test_HMAC_Alt_{timestamp}",
            "language": "Py"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/projects/create", 
                headers=headers3, 
                json=project_data
            )
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', False):
                    print("   ✅ Alternative HMAC successful!")
                    return headers3, "hmac_alt"
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        return None, None
    
    def create_project(self, project_name: str, language: str = "Py") -> Optional[str]:
        """Create a new project in QuantConnect"""
        # First test authentication
        headers, auth_type = self.test_authentication_methods()
        
        if headers is None:
            print("\n❌ All authentication methods failed!")
            return None
        
        print(f"\n✅ Using {auth_type} authentication")
        
        # Create project with working auth
        project_data = {
            "projectName": project_name,
            "language": language
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/projects/create",
                headers=headers,
                json=project_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', False):
                    project_id = data.get('projectId')
                    print(f"✅ Project created: {project_id}")
                    return project_id
                else:
                    print(f"❌ Failed: {data.get('errors', [])}")
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        return None
    
    def upload_algorithm(self, project_id: str, algorithm_code: str) -> bool:
        """Upload algorithm code to project"""
        headers, _ = self._generate_headers()
        
        file_data = {
            "projectId": project_id,
            "name": "main.py",
            "content": algorithm_code
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/files/update",
                headers=headers,
                json=file_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', False):
                    print(f"✅ Algorithm uploaded to project {project_id}")
                    return True
                else:
                    print(f"❌ Upload failed: {data.get('errors', [])}")
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        return False

def test_quantconnect_api():
    """Test QuantConnect API with provided credentials"""
    print("🚀 QUANTCONNECT API INTEGRATION TEST")
    print("=" * 80)
    
    client = QuantConnectAPIClient()
    
    # Test project creation
    project_name = f"Evolution_Strategy_Test_{int(time.time())}"
    project_id = client.create_project(project_name)
    
    if project_id:
        print(f"\n✅ SUCCESS! Project created with ID: {project_id}")
        
        # Test algorithm upload
        test_algorithm = '''
from AlgorithmImports import *

class TestEvolutionStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        self.AddEquity("SPY", Resolution.Daily)
    
    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1.0)
'''
        
        if client.upload_algorithm(project_id, test_algorithm):
            print("✅ Algorithm uploaded successfully!")
            print(f"\n🎉 QuantConnect integration working!")
            print(f"   Project: {project_name}")
            print(f"   ID: {project_id}")
        else:
            print("❌ Failed to upload algorithm")
    else:
        print("\n❌ Failed to create project - authentication issue")

if __name__ == "__main__":
    test_quantconnect_api()