#!/usr/bin/env python3
"""
Test QuantConnect HMAC Authentication for Project Creation
"""

import requests
import time
import base64
import hmac
import hashlib
import json

def test_hmac_project_creation():
    """Test creating a project with HMAC authentication"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    timestamp = str(int(time.time()))
    
    # HMAC authentication format
    # Hash = HMAC-SHA256(api_token, user_id:timestamp)
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
    
    # Create project data
    project_data = {
        "projectName": f"Evolution_HMAC_Test_{timestamp}",
        "language": "Py"
    }
    
    print("🧪 Testing HMAC Project Creation")
    print(f"User ID: {user_id}")
    print(f"Timestamp: {timestamp}")
    print(f"Message: {message}")
    print(f"Hash: {hash_signature}")
    print(f"Project Name: {project_data['projectName']}")
    print()
    
    response = requests.post(
        "https://www.quantconnect.com/api/v2/projects/create",
        headers=headers,
        json=project_data
    )
    
    print(f"Response Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        try:
            data = response.json()
            if data.get('success'):
                project_id = data.get('projectId')
                print(f"\n✅ Project created successfully!")
                print(f"Project ID: {project_id}")
                return project_id
            else:
                print(f"\n❌ Project creation failed: {data.get('errors', [])}")
                return None
        except Exception as e:
            print(f"\n❌ Could not parse response: {e}")
            return None
    else:
        print(f"\n❌ HTTP Error: {response.status_code}")
        return None

def test_alternate_hmac():
    """Test alternate HMAC format"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    timestamp = str(int(time.time()))
    
    # Try different HMAC format: Hash = HMAC-SHA256(timestamp, api_token)
    hash_signature = hmac.new(
        timestamp.encode('utf-8'),
        api_token.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "Content-Type": "application/json",
        "Timestamp": timestamp,
        "Authorization": f"{user_id}:{hash_signature}"
    }
    
    project_data = {
        "projectName": f"Evolution_Alt_HMAC_{timestamp}",
        "language": "Py"
    }
    
    print("🧪 Testing Alternate HMAC Format")
    print(f"Hash = HMAC-SHA256(timestamp, api_token)")
    print(f"Timestamp: {timestamp}")
    print(f"Hash: {hash_signature}")
    print()
    
    response = requests.post(
        "https://www.quantconnect.com/api/v2/projects/create",
        headers=headers,
        json=project_data
    )
    
    print(f"Response Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    return response.status_code == 200 and response.json().get('success', False)

def test_basic_with_hmac():
    """Test Basic auth with HMAC hash"""
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
    
    # Use Basic auth with the hash
    auth_string = f"{user_id}:{hash_signature}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Timestamp": timestamp,
        "Authorization": f"Basic {encoded_auth}"
    }
    
    project_data = {
        "projectName": f"Evolution_Basic_HMAC_{timestamp}",
        "language": "Py"
    }
    
    print("🧪 Testing Basic Auth with HMAC Hash")
    print(f"Auth string: {auth_string}")
    print(f"Encoded: {encoded_auth}")
    print()
    
    response = requests.post(
        "https://www.quantconnect.com/api/v2/projects/create",
        headers=headers,
        json=project_data
    )
    
    print(f"Response Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    return response.status_code == 200 and response.json().get('success', False)

if __name__ == "__main__":
    print("🔐 QUANTCONNECT HMAC AUTHENTICATION TESTING")
    print("=" * 70)
    
    # Test different HMAC approaches
    print("Testing various HMAC authentication methods for project creation...")
    print()
    
    # Test 1: Standard HMAC
    project_id1 = test_hmac_project_creation()
    print("\n" + "-" * 50 + "\n")
    
    # Test 2: Alternate HMAC
    success2 = test_alternate_hmac()
    print("\n" + "-" * 50 + "\n")
    
    # Test 3: Basic with HMAC
    success3 = test_basic_with_hmac()
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print(f"Standard HMAC: {'✅ SUCCESS' if project_id1 else '❌ FAILED'}")
    print(f"Alternate HMAC: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print(f"Basic + HMAC: {'✅ SUCCESS' if success3 else '❌ FAILED'}")
    
    if any([project_id1, success2, success3]):
        print("\n🎉 Found working HMAC authentication!")
    else:
        print("\n❌ All HMAC methods failed")
        print("   The credentials might need verification or QuantConnect API has changed")