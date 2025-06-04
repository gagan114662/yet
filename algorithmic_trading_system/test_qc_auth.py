#!/usr/bin/env python3
"""
QuantConnect API Authentication Test - Debug authentication issues
"""

import requests
import time
import base64
import hmac
import hashlib
import json

def test_basic_auth():
    """Test basic auth format"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    # Basic auth format
    auth_string = f"{user_id}:{api_token}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {encoded_auth}"
    }
    
    print("🧪 Testing Basic Authentication")
    print(f"User ID: {user_id}")
    print(f"Auth string: {auth_string}")
    print(f"Encoded: {encoded_auth}")
    print()
    
    # Test with projects/read endpoint
    response = requests.get("https://www.quantconnect.com/api/v2/projects/read", headers=headers)
    
    print(f"Response Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()
    
    return response.status_code == 200

def test_hmac_auth():
    """Test HMAC-SHA256 authentication"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    timestamp = str(int(time.time()))
    
    # Create HMAC hash
    message = f"{user_id}:{timestamp}"
    hash_signature = hmac.new(
        api_token.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "Content-Type": "application/json",
        "Timestamp": timestamp,
        "Authorization": f"{user_id}:{hash_signature}"
    }
    
    print("🧪 Testing HMAC Authentication")
    print(f"User ID: {user_id}")
    print(f"Timestamp: {timestamp}")
    print(f"Message: {message}")
    print(f"Hash: {hash_signature}")
    print()
    
    response = requests.get("https://www.quantconnect.com/api/v2/projects/read", headers=headers)
    
    print(f"Response Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()
    
    return response.status_code == 200

def test_list_projects():
    """Test simple project listing"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    # Basic auth format
    auth_string = f"{user_id}:{api_token}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {encoded_auth}"
    }
    
    print("🧪 Testing Project List")
    response = requests.get("https://www.quantconnect.com/api/v2/projects/read", headers=headers)
    
    print(f"Response Status: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"Projects found: {len(data.get('projects', []))}")
            if data.get('projects'):
                print("First project:", data['projects'][0].get('name', 'Unknown'))
        except:
            print("Could not parse JSON response")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("🔐 QUANTCONNECT AUTHENTICATION TESTING")
    print("=" * 60)
    
    print("Testing different authentication methods...")
    print()
    
    # Test 1: Basic Authentication
    success1 = test_basic_auth()
    
    # Test 2: HMAC Authentication  
    success2 = test_hmac_auth()
    
    # Test 3: List Projects
    success3 = test_list_projects()
    
    print("=" * 60)
    print("RESULTS:")
    print(f"Basic Auth: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print(f"HMAC Auth: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print(f"List Projects: {'✅ SUCCESS' if success3 else '❌ FAILED'}")
    
    if any([success1, success2, success3]):
        print("\n🎉 At least one authentication method worked!")
    else:
        print("\n❌ All authentication methods failed")
        print("   Check if credentials are correct or if QuantConnect API has changed")