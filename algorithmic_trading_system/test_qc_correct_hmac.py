#!/usr/bin/env python3
"""
Test QuantConnect API with correct HMAC implementation
Based on their documentation, the hash should be:
hash = HMAC-SHA256(API_TOKEN, timestamp)
Authorization header: "UserId:Hash"
"""

import requests
import time
import hmac
import hashlib
import json

def create_qc_auth_headers(user_id, api_token):
    """Create proper QuantConnect authentication headers"""
    timestamp = str(int(time.time()))
    
    # Create HMAC hash: HMAC-SHA256(API_TOKEN, timestamp)
    hash_signature = hmac.new(
        api_token.encode('utf-8'),
        timestamp.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "Timestamp": timestamp,
        "Authorization": f"{user_id}:{hash_signature}",
        "Content-Type": "application/json"
    }
    
    return headers, timestamp

def test_project_read():
    """Test reading projects"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    print("📚 Testing Project Read")
    headers, timestamp = create_qc_auth_headers(user_id, api_token)
    
    print(f"Timestamp: {timestamp}")
    print(f"Authorization: {headers['Authorization']}")
    
    response = requests.get(
        "https://www.quantconnect.com/api/v2/projects/read",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:300]}...")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print("\n✅ Authentication successful!")
            if 'projects' in data:
                print(f"Found {len(data['projects'])} projects")
                for proj in data['projects'][:3]:
                    print(f"  - {proj.get('name')} (ID: {proj.get('projectId')})")
            return True
    return False

def test_project_create():
    """Test creating a project"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    print("\n📝 Testing Project Creation")
    headers, timestamp = create_qc_auth_headers(user_id, api_token)
    
    project_data = {
        "name": f"Evolution_Test_{timestamp}",
        "language": "Python"
    }
    
    print(f"Project name: {project_data['name']}")
    
    response = requests.post(
        "https://www.quantconnect.com/api/v2/projects/create",
        headers=headers,
        json=project_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            project_id = data.get('projects', [{}])[0].get('projectId')
            print(f"\n✅ Project created! ID: {project_id}")
            return project_id
    return None

def test_alternative_endpoints():
    """Test alternative API endpoints"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    print("\n🔍 Testing Alternative Endpoints")
    headers, timestamp = create_qc_auth_headers(user_id, api_token)
    
    # Try user endpoint
    response = requests.get(
        "https://www.quantconnect.com/api/v2/user/read",
        headers=headers
    )
    
    print(f"User Read Status: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    
    # Try live/read endpoint
    response = requests.get(
        "https://www.quantconnect.com/api/v2/live/read",
        headers=headers
    )
    
    print(f"\nLive Read Status: {response.status_code}")
    print(f"Response: {response.text[:200]}...")

def test_with_api_prefix():
    """Test with different URL formats"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    print("\n🌐 Testing Different URL Formats")
    headers, timestamp = create_qc_auth_headers(user_id, api_token)
    
    # Try without /api prefix
    response = requests.get(
        "https://www.quantconnect.com/v2/projects/read",
        headers=headers
    )
    
    print(f"Without /api prefix Status: {response.status_code}")
    
    # Try with www subdomain removed
    response = requests.get(
        "https://quantconnect.com/api/v2/projects/read",
        headers=headers
    )
    
    print(f"Without www Status: {response.status_code}")

if __name__ == "__main__":
    print("🔐 QUANTCONNECT CORRECT HMAC AUTHENTICATION TEST")
    print("=" * 70)
    print("Using format: hash = HMAC-SHA256(API_TOKEN, timestamp)")
    print("=" * 70)
    
    # Test 1: Read projects
    success = test_project_read()
    
    # Test 2: Create project if read was successful
    if success:
        project_id = test_project_create()
    
    # Test 3: Try alternative endpoints
    test_alternative_endpoints()
    
    # Test 4: Different URL formats
    test_with_api_prefix()
    
    print("\n" + "=" * 70)
    print("If all tests failed, possible issues:")
    print("1. API token might be invalid or expired")
    print("2. User ID might be incorrect")
    print("3. QuantConnect API might have changed")
    print("4. Account might not have API access enabled")
    print("\nPlease verify your credentials at:")
    print("https://www.quantconnect.com/account")