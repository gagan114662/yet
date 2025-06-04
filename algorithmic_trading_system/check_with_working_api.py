#!/usr/bin/env python3
"""
CHECK RESULTS WITH WORKING API - Use the correct authentication format
"""

import sys
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI
import json

def check_results_properly():
    """Use the working API to check for backtest results"""
    
    print("🔍 CHECKING RESULTS WITH WORKING API")
    print("="*45)
    
    # Initialize API with working format
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    project_id = "23349848"
    
    print(f"📊 Project ID: {project_id}")
    print(f"🔗 URL: https://www.quantconnect.com/project/{project_id}")
    
    # Method 1: Try to read backtest results directly
    print(f"\n📈 METHOD 1: Reading backtest results...")
    try:
        results = api.read_backtest_results(project_id, None)
        
        if results:
            print(f"🎉 BACKTEST RESULTS FOUND!")
            print(f"📊 Type: {type(results)}")
            
            if isinstance(results, dict):
                # Display key metrics
                key_fields = ['cagr', 'sharpe', 'total_orders', 'win_rate', 'max_drawdown']
                
                print(f"\n📊 KEY METRICS:")
                for field in key_fields:
                    if field in results:
                        print(f"   {field.upper()}: {results[field]}")
                
                print(f"\n📋 ALL AVAILABLE FIELDS:")
                for key, value in results.items():
                    print(f"   {key}: {value}")
                
                # Save to file
                with open('/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/FOUND_RESULTS.json', 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"\n💾 Results saved to: FOUND_RESULTS.json")
                return results
            else:
                print(f"📊 Results type: {type(results)}")
                print(f"📊 Results content: {results}")
        else:
            print(f"❌ No results found")
            
    except Exception as e:
        print(f"❌ Error reading results: {e}")
    
    # Method 2: Try to get project info
    print(f"\n📋 METHOD 2: Checking project status...")
    try:
        # The working API might have a method to list backtests
        # Let's check what methods are available
        api_methods = [method for method in dir(api) if not method.startswith('_')]
        print(f"📋 Available API methods: {api_methods}")
        
        # Try some basic project operations
        if hasattr(api, 'get_project'):
            project_info = api.get_project(project_id)
            print(f"📊 Project info: {project_info}")
        
    except Exception as e:
        print(f"❌ Error checking project: {e}")
    
    # Method 3: Manual API call with working auth
    print(f"\n🔧 METHOD 3: Manual API call with proper auth...")
    try:
        import requests
        
        headers = api.get_headers()
        url = f"https://www.quantconnect.com/api/v2/projects/{project_id}/backtests"
        
        print(f"📡 URL: {url}")
        print(f"📋 Headers: {headers}")
        
        response = requests.get(url, headers=headers, timeout=10)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Raw API Response:")
            print(json.dumps(result, indent=2))
            
            if result.get('success'):
                backtests = result.get('backtests', [])
                if backtests:
                    print(f"\n🎯 FOUND {len(backtests)} BACKTEST(S)!")
                    for i, bt in enumerate(backtests):
                        print(f"\nBacktest {i+1}:")
                        print(f"   ID: {bt.get('backtestId')}")
                        print(f"   Completed: {bt.get('completed')}")
                        print(f"   Error: {bt.get('error', 'None')}")
                        
                        if bt.get('completed'):
                            print(f"   🎉 COMPLETED BACKTEST FOUND!")
                            
                else:
                    print(f"❌ No backtests in project")
            else:
                print(f"❌ API error: {result.get('errors', 'Unknown')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Manual API error: {e}")
    
    print(f"\n💡 SUMMARY:")
    print(f"   If no results found = Backtest never completed successfully")
    print(f"   If results found = We can analyze actual performance")
    print(f"   Next step depends on what we found above")

if __name__ == "__main__":
    check_results_properly()