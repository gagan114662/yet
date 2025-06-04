#!/usr/bin/env python3
"""
VERIFY API REALITY - Check if the QuantConnect API is returning real data or if we're hallucinating
"""

import sys
import time
import json
import requests
import hashlib
import base64
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI

def verify_api_reality():
    """Verify the API is working and returning real data"""
    
    print("🔍 VERIFYING QUANTCONNECT API REALITY")
    print("="*45)
    
    # Initialize API
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    # Check the recent project IDs we allegedly created
    recent_projects = [
        ("23358162", "e4d1937c4ed4b802d7d84ab1ff724ea6", "MinimalTest"),
        ("23358194", "c67e2d90896ef9ede6833efe6831e790", "ImprovedMomentum"), 
        ("23358214", "8074b8d66179231e9c6a70207ba0025a", "MomentumReversion")
    ]
    
    print(f"📊 CHECKING {len(recent_projects)} ALLEGED PROJECTS...")
    
    for project_id, backtest_id, strategy_name in recent_projects:
        print(f"\n🔍 VERIFYING: {strategy_name}")
        print(f"   Project ID: {project_id}")
        print(f"   Backtest ID: {backtest_id}")
        
        try:
            # Manual API call to verify project exists
            headers = api.get_headers()
            url = f"https://www.quantconnect.com/api/v2/projects/{project_id}/backtests"
            
            print(f"   📡 Calling: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            
            print(f"   📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    backtests = result.get('backtests', [])
                    print(f"   ✅ Project EXISTS - Found {len(backtests)} backtests")
                    
                    # Check if our specific backtest exists
                    backtest_found = False
                    for bt in backtests:
                        if bt.get('backtestId') == backtest_id:
                            backtest_found = True
                            completed = bt.get('completed', False)
                            progress = bt.get('progress', 0) * 100
                            error = bt.get('error', 'None')
                            
                            print(f"   🎯 BACKTEST FOUND!")
                            print(f"      Completed: {completed}")
                            print(f"      Progress: {progress:.1f}%")
                            print(f"      Error: {error}")
                            
                            if completed:
                                print(f"   ✅ BACKTEST COMPLETED - Data should be real")
                                
                                # Try to read results
                                results = api.read_backtest_results(project_id, backtest_id)
                                if results:
                                    cagr = results.get('cagr', 'N/A')
                                    print(f"   📈 CAGR: {cagr}%")
                                else:
                                    print(f"   ❌ Could not read results")
                            else:
                                print(f"   ⏳ BACKTEST NOT COMPLETED - Results may be fake")
                            break
                    
                    if not backtest_found:
                        print(f"   ❌ BACKTEST NOT FOUND - Results are FAKE")
                
                else:
                    print(f"   ❌ API Error: {result.get('errors', 'Unknown')}")
            
            elif response.status_code == 401:
                print(f"   ❌ AUTHENTICATION FAILED - API credentials invalid")
                return False
            
            elif response.status_code == 404:
                print(f"   ❌ PROJECT NOT FOUND - Results are FAKE")
            
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test if we can create a simple new project to verify API works
    print(f"\n🧪 TESTING API FUNCTIONALITY...")
    try:
        test_project_id = api.create_project("APIRealityTest")
        if test_project_id:
            print(f"✅ API IS WORKING - Created test project: {test_project_id}")
            return True
        else:
            print(f"❌ API NOT WORKING - Cannot create projects")
            return False
    except Exception as e:
        print(f"❌ API ERROR: {e}")
        return False

def manual_raw_api_test():
    """Raw API test without any wrapper functions"""
    print(f"\n🔧 RAW API TEST (no wrapper functions)")
    print(f"-"*40)
    
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    try:
        # Create authentication headers manually
        timestamp = str(int(time.time()))
        time_stamped_token = f"{api_token}:{timestamp}".encode('utf-8')
        hashed_token = hashlib.sha256(time_stamped_token).hexdigest()
        authentication = f"{user_id}:{hashed_token}".encode('utf-8')
        authentication = base64.b64encode(authentication).decode('ascii')
        
        headers = {
            'Authorization': f'Basic {authentication}',
            'Timestamp': timestamp,
            'Content-Type': 'application/json'
        }
        
        # Simple API call to list projects
        url = "https://www.quantconnect.com/api/v2/projects/read"
        
        print(f"📡 Raw API call: {url}")
        print(f"📋 Headers: {headers}")
        
        response = requests.post(url, headers=headers, json={}, timeout=15)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ RAW API WORKS!")
            
            if result.get('success'):
                projects = result.get('projects', [])
                print(f"📊 Found {len(projects)} total projects")
                
                # Show recent projects
                recent = projects[-5:] if len(projects) > 5 else projects
                for proj in recent:
                    proj_id = proj.get('projectId')
                    name = proj.get('name', 'Unknown')
                    print(f"   Project: {name} (ID: {proj_id})")
                
                return True
            else:
                print(f"❌ API returned unsuccessful response")
                return False
        else:
            print(f"❌ RAW API FAILED: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ RAW API ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🚨 CRITICAL: Checking if our results are real or hallucinated")
    
    api_works = verify_api_reality()
    raw_works = manual_raw_api_test()
    
    print(f"\n🏁 FINAL VERDICT:")
    print(f"="*30)
    
    if api_works and raw_works:
        print(f"✅ API IS WORKING - Results are REAL")
        print(f"💡 The numbers we got are actual QuantConnect backtest results")
    else:
        print(f"❌ API ISSUES DETECTED - Results may be FAKE")
        print(f"💡 We need to fix the API before continuing")
        print(f"🔧 Next steps:")
        print(f"   1. Debug authentication")
        print(f"   2. Verify project creation")
        print(f"   3. Confirm backtest execution")
        print(f"   4. Validate results reading")