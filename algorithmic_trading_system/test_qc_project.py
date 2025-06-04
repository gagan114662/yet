#!/usr/bin/env python3
"""
Test QuantConnect Project Creation
"""

import requests
import time
import base64
import json

def test_create_project():
    """Test creating a new project in QuantConnect"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    timestamp = str(int(time.time()))
    
    # Basic auth format
    auth_string = f"{user_id}:{api_token}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Timestamp": timestamp,
        "Authorization": f"Basic {encoded_auth}"
    }
    
    # Create project data
    project_data = {
        "projectName": f"Test_Evolution_Strategy_{timestamp}",
        "language": "Py"
    }
    
    print("🧪 Testing Project Creation")
    print(f"Project Name: {project_data['projectName']}")
    print(f"Language: {project_data['language']}")
    print(f"Timestamp: {timestamp}")
    print()
    
    response = requests.post(
        "https://www.quantconnect.com/api/v2/projects/create",
        headers=headers,
        json=project_data
    )
    
    print(f"Response Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
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

def test_list_existing_projects():
    """List existing projects to see what we have"""
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    # Basic auth format
    auth_string = f"{user_id}:{api_token}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {encoded_auth}"
    }
    
    print("📂 Listing Existing Projects")
    response = requests.get("https://www.quantconnect.com/api/v2/projects/read", headers=headers)
    
    print(f"Response Status: {response.status_code}")
    
    if response.status_code == 200:
        try:
            data = response.json()
            projects = data.get('projects', [])
            print(f"Total Projects: {len(projects)}")
            
            if projects:
                print("\nExisting Projects:")
                for i, project in enumerate(projects[:5]):  # Show first 5
                    print(f"  {i+1}. {project.get('name', 'Unknown')} (ID: {project.get('projectId', 'Unknown')})")
            else:
                print("\nNo existing projects found.")
            
            return projects
        except Exception as e:
            print(f"Could not parse response: {e}")
            return []
    else:
        print(f"Error: {response.text}")
        return []

if __name__ == "__main__":
    print("🚀 QUANTCONNECT PROJECT TESTING")
    print("=" * 60)
    
    # First, list existing projects
    existing_projects = test_list_existing_projects()
    print()
    
    # Then try to create a new project
    project_id = test_create_project()
    
    print("=" * 60)
    if project_id:
        print("✅ SUCCESS: Project creation worked!")
        print(f"   New Project ID: {project_id}")
        print("   Ready for algorithm upload and backtesting!")
    else:
        print("❌ FAILED: Could not create project")
        if existing_projects:
            print("   But we can use existing projects for testing")
        else:
            print("   Need to investigate authentication or permissions")