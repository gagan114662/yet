#!/usr/bin/env python3
"""
Simple QuantConnect API Test
Test basic authentication and API access
"""

import requests
import base64
import json

def test_qc_api():
    user_id = "357130"
    token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    # Test different authentication formats
    auth_formats = [
        ("Basic", base64.b64encode(f'{user_id}:{token}'.encode()).decode()),
        ("Bearer", token),
        ("Token", token),
    ]
    
    for auth_type, auth_value in auth_formats:
        print(f"\n🔑 Testing {auth_type} authentication:")
        
        headers = {
            "Authorization": f"{auth_type} {auth_value}",
            "Content-Type": "application/json"
        }
        
        # Test with projects list endpoint (read-only)
        try:
            response = requests.get("https://www.quantconnect.com/api/v2/projects/read", 
                                  headers=headers, timeout=10)
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                print("   ✅ SUCCESS!")
                return headers
            else:
                print("   ❌ Failed")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Test manual authentication format
    print(f"\n🔑 Testing manual format:")
    manual_headers = {
        "Authorization": f"Basic {user_id}:{token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get("https://www.quantconnect.com/api/v2/projects/read", 
                              headers=manual_headers, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    return None

if __name__ == "__main__":
    print("🚀 TESTING QUANTCONNECT API ACCESS")
    print("=" * 50)
    
    working_headers = test_qc_api()
    
    if working_headers:
        print("\n✅ Found working authentication format!")
        print(f"Headers: {working_headers}")
    else:
        print("\n❌ No working authentication format found")
        print("Check credentials or API documentation")