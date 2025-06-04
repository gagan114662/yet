#!/usr/bin/env python3
"""
Test QuantConnect API with timestamp in requests
"""

import requests
import base64
import json
import time
import hashlib
import hmac

def test_basic_auth_with_timestamp():
    """Test Basic auth with timestamp header"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    timestamp = str(int(time.time()))
    
    # Create Basic auth header with original API token
    auth_string = f"{user_id}:{api_token}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/json",
        "Timestamp": timestamp
    }
    
    # Test read projects first
    print("🔍 Testing Basic Auth with Timestamp - Read Projects")
    print(f"Timestamp: {timestamp}")
    
    response = requests.get(
        "https://www.quantconnect.com/api/v2/projects/read",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print("✅ Read successful!")
            if 'projects' in data:
                print(f"Found {len(data['projects'])} projects")
        else:
            print(f"❌ Read failed: {data.get('errors', [])}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Now test project creation
    print("📝 Testing Project Creation with Timestamp")
    
    project_data = {
        "name": f"EvolutionTest_{timestamp}",
        "language": "Python"
    }
    
    response = requests.post(
        "https://www.quantconnect.com/api/v2/projects/create",
        headers=headers,
        json=project_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    return response

def test_hmac_with_different_format():
    """Test HMAC with different message formats"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    timestamp = str(int(time.time()))
    
    # Try different message formats
    formats = [
        ("userId:apiToken:timestamp", f"{user_id}:{api_token}:{timestamp}"),
        ("apiToken:timestamp", f"{api_token}:{timestamp}"),
        ("timestamp:userId", f"{timestamp}:{user_id}"),
        ("timestamp", timestamp),
    ]
    
    print("🔐 Testing Different HMAC Message Formats")
    print("=" * 60)
    
    for format_name, message in formats:
        print(f"\nTesting format: {format_name}")
        print(f"Message: {message[:50]}...")
        
        # Create HMAC hash
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
        
        # Test with read endpoint
        response = requests.get(
            "https://www.quantconnect.com/api/v2/projects/read",
            headers=headers
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:100]}...")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ SUCCESS! Found working format!")
                return format_name, message
    
    return None, None

def test_qc_cli_format():
    """Test the format used by QC CLI"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    timestamp = str(int(time.time()))
    
    print("\n🛠️ Testing QuantConnect CLI Format")
    print("=" * 60)
    
    # QC CLI might use a different approach
    # Let's try the format: HMAC(api_token, timestamp)
    hash_signature = hmac.new(
        api_token.encode('utf-8'),
        timestamp.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "Content-Type": "application/json",
        "Timestamp": timestamp,
        "Authorization": f"ApiToken {user_id}:{hash_signature}"
    }
    
    response = requests.get(
        "https://www.quantconnect.com/api/v2/projects/read",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    
    # Try another format
    print("\n🔧 Testing Bearer Token Format")
    
    headers = {
        "Content-Type": "application/json",
        "Timestamp": timestamp,
        "Authorization": f"Bearer {api_token}"
    }
    
    response = requests.get(
        "https://www.quantconnect.com/api/v2/projects/read",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}...")

if __name__ == "__main__":
    print("🚀 QUANTCONNECT API TIMESTAMP TESTING")
    print("=" * 70)
    
    # Test 1: Basic auth with timestamp
    test_basic_auth_with_timestamp()
    
    print("\n" + "=" * 70 + "\n")
    
    # Test 2: Different HMAC formats
    working_format, working_message = test_hmac_with_different_format()
    
    print("\n" + "=" * 70 + "\n")
    
    # Test 3: QC CLI format
    test_qc_cli_format()
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    if working_format:
        print(f"✅ Found working HMAC format: {working_format}")
    else:
        print("❌ No working HMAC format found")
        print("\nNOTE: The API credentials might need to be regenerated or")
        print("      QuantConnect may have changed their authentication method.")