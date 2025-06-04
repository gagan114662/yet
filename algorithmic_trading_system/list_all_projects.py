#!/usr/bin/env python3
"""
List all QuantConnect projects and their backtest results
"""

import sys
import json
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI

def main():
    # Initialize API
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    print("🔍 SCANNING ALL QUANTCONNECT PROJECTS")
    print("="*60)
    
    # Known project IDs from config files in lean_workspace
    known_projects = []
    
    # Try to find projects with completed backtests
    test_project_ids = [
        "23349848",  # Our main project
        "23349849", "23349850", "23349851",  # Possible adjacent IDs
        "25000000", "24000000", "23000000"   # Other ranges
    ]
    
    successful_projects = []
    
    for project_id in test_project_ids:
        try:
            print(f"\n📊 Checking Project {project_id}...")
            results = api.read_backtest_results(str(project_id), None)
            
            if results and isinstance(results, dict):
                if results.get('cagr') is not None:
                    print(f"✅ FOUND RESULTS!")
                    print(f"   📈 CAGR: {results.get('cagr', 0):.2f}%")
                    print(f"   📊 Sharpe: {results.get('sharpe', 0):.2f}")
                    print(f"   🎯 Total Orders: {results.get('total_orders', 0)}")
                    print(f"   💰 Win Rate: {results.get('win_rate', 0):.1f}%")
                    print(f"   🔗 URL: https://www.quantconnect.com/project/{project_id}")
                    successful_projects.append({
                        'project_id': project_id,
                        'results': results
                    })
                else:
                    print(f"   ⏳ No results yet")
            else:
                print(f"   ❌ No access or empty")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")
    
    print(f"\n" + "="*60)
    print(f"📋 SUMMARY")
    print(f"="*60)
    
    if successful_projects:
        print(f"✅ Found {len(successful_projects)} projects with results:")
        for proj in successful_projects:
            results = proj['results']
            print(f"   🎯 Project {proj['project_id']}: {results.get('cagr', 0):.1f}% CAGR")
    else:
        print(f"❌ No projects found with completed backtest results")
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Create a new project: https://www.quantconnect.com/terminal/")
        print(f"   2. Use our minimal test strategy")
        print(f"   3. Run backtest manually from web interface")
        print(f"   4. Check for compilation errors in Cloud Terminal")
    
    return successful_projects

if __name__ == "__main__":
    main()