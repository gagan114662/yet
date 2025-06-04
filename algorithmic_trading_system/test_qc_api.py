#!/usr/bin/env python3
"""Test QuantConnect API authentication"""

import requests
import base64
import json

# Your credentials
USER_ID = "357130"
API_TOKEN = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"

# Create Basic auth header
auth_string = f"{USER_ID}:{API_TOKEN}"
encoded_auth = base64.b64encode(auth_string.encode()).decode()

headers = {
    "Authorization": f"Basic {encoded_auth}",
    "Content-Type": "application/json"
}

# Test 1: Get existing projects
print("Testing QuantConnect API Authentication...")
print("=" * 60)

url = "https://www.quantconnect.com/api/v2/projects/read"
response = requests.get(url, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:500]}...")

if response.status_code == 200:
    print("\n✅ Authentication successful!")
    result = response.json()
    if "projects" in result:
        print(f"Found {len(result['projects'])} existing projects")
        for project in result["projects"][:3]:  # Show first 3
            print(f"  - {project.get('name', 'Unnamed')} (ID: {project.get('projectId')})")
else:
    print("\n❌ Authentication failed!")
    
# Test 2: Try to create a new project
print("\n" + "=" * 60)
print("Attempting to create a new project...")

import time
timestamp = int(time.time())

create_url = "https://www.quantconnect.com/api/v2/projects/create"
data = {
    "name": f"Test25CAGR_{timestamp}",
    "language": "Python"
}

response = requests.post(create_url, headers=headers, json=data)
print(f"Create Status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    result = response.json()
    project_id = result.get("projects", [{}])[0].get("projectId")
    if project_id:
        print(f"\n✅ Created project ID: {project_id}")
    else:
        print("\n⚠️ Project created but no ID returned")