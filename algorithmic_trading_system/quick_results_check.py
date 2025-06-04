#!/usr/bin/env python3
"""
QUICK RESULTS CHECK - Single API call to check for backtest results
"""

import sys
import time
import json
import requests
import hashlib
import hmac

def quick_check():
    """Single check for backtest results"""
    
    print("🔍 QUICK BACKTEST RESULTS CHECK")
    print("="*40)
    
    # API credentials
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    project_id = "23349848"
    
    try:
        # Get list of backtests
        timestamp = str(int(time.time()))
        data = ""
        
        # Create signature
        message = f"GET\n{data}\n/projects/{project_id}/backtests\n{timestamp}"
        signature = hmac.new(
            api_token.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'Authorization': f'Basic {user_id}:{signature}',
            'Timestamp': timestamp
        }
        
        # Make request
        url = f"https://www.quantconnect.com/api/v2/projects/{project_id}/backtests"
        print(f"📡 Calling: {url}")
        
        response = requests.get(url, headers=headers, timeout=10)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Response: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                backtests = result.get('backtests', [])
                print(f"\n📈 Found {len(backtests)} backtest(s)")
                
                if backtests:
                    for i, bt in enumerate(backtests):
                        print(f"\n🎯 Backtest {i+1}:")
                        print(f"   ID: {bt.get('backtestId', 'N/A')}")
                        print(f"   Name: {bt.get('name', 'N/A')}")
                        print(f"   Completed: {bt.get('completed', False)}")
                        print(f"   Progress: {bt.get('progress', 0)*100:.1f}%")
                        print(f"   Error: {bt.get('error', 'None')}")
                        
                        if bt.get('completed'):
                            print(f"   🎉 THIS BACKTEST IS COMPLETE!")
                            return bt.get('backtestId')
                else:
                    print(f"❌ NO BACKTESTS FOUND")
                    print(f"💡 This means the backtest never started or failed immediately")
            else:
                print(f"❌ API returned unsuccessful response")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

if __name__ == "__main__":
    quick_check()