#!/usr/bin/env python3
"""
Quick QuantConnect API status check and backtest trigger
"""

import sys
import os
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI

def main():
    # Initialize API
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    project_id = "23349848"
    
    print("🔍 QUANTCONNECT PROJECT STATUS CHECK")
    print("="*50)
    
    # 1. Check project exists
    print(f"📊 Project ID: {project_id}")
    print(f"🔗 URL: https://www.quantconnect.com/project/{project_id}")
    
    # 2. List existing backtests
    print(f"\n📈 CHECKING EXISTING BACKTESTS...")
    try:
        results = api.read_backtest_results(project_id, None)
        if results:
            print(f"✅ Found completed backtest results!")
            print(f"📊 CAGR: {results.get('cagr', 'N/A'):.2f}%")
            print(f"📊 Sharpe: {results.get('sharpe', 'N/A'):.2f}")
            print(f"📊 Total Orders: {results.get('total_orders', 'N/A')}")
            print(f"📊 Win Rate: {results.get('win_rate', 'N/A'):.1f}%")
            return results
        else:
            print(f"⏳ No completed backtests found")
    except Exception as e:
        print(f"❌ Error checking backtests: {e}")
    
    # 3. Try to get project details
    print(f"\n📋 CHECKING PROJECT DETAILS...")
    try:
        # This would require a different API endpoint
        print(f"💡 Project appears to exist but no backtests completed")
        print(f"💡 This could mean:")
        print(f"   • Backtest never started successfully")
        print(f"   • Initialization error prevented completion")
        print(f"   • Strategy compilation failed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. Suggest next steps
    print(f"\n🎯 RECOMMENDED ACTIONS:")
    print(f"1. Go to: https://www.quantconnect.com/project/{project_id}")
    print(f"2. Check the 'Cloud Terminal' tab for error messages")
    print(f"3. Try running a backtest manually from the web interface")
    print(f"4. If errors persist, use our fresh strategy in a new project")
    
    return None

if __name__ == "__main__":
    main()