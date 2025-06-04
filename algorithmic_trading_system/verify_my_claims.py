#!/usr/bin/env python3
"""
VERIFY CLAUDE'S CLAIMS - Find actual QuantConnect results to validate performance claims
"""

import sys
import json
import requests
import hashlib
import hmac
import time
from datetime import datetime

sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

def verify_claims_with_real_data():
    """Find real QuantConnect results to verify Claude's performance claims"""
    
    print("🔍 VERIFYING CLAUDE'S PERFORMANCE CLAIMS")
    print("="*70)
    print("📊 Searching for ANY completed backtests in your account...")
    
    # API credentials
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    
    # Try to find projects using different methods
    
    # Method 1: Check config files for project IDs
    print("\n📁 METHOD 1: Scanning lean_workspace config files for project IDs...")
    
    import os
    import glob
    
    lean_workspace = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
    project_ids = set()
    
    # Find all config.json files
    config_files = glob.glob(f"{lean_workspace}/**/config.json", recursive=True)
    
    for config_file in config_files[:10]:  # Limit to first 10
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if 'cloud-id' in config:
                    project_ids.add(str(config['cloud-id']))
                    print(f"   📋 Found project ID: {config['cloud-id']} in {config_file}")
        except:
            continue
    
    print(f"\n📊 Found {len(project_ids)} unique project IDs in config files")
    
    # Method 2: Test these project IDs
    print("\n🧪 METHOD 2: Testing found project IDs for backtest results...")
    
    successful_results = []
    
    for project_id in list(project_ids)[:5]:  # Test first 5
        print(f"\n📊 Testing Project {project_id}...")
        
        try:
            # Prepare API request
            timestamp = str(int(time.time()))
            data = ""
            
            # Create signature
            message = f"GET\n{data}\n/projects/{project_id}/backtests\n{timestamp}"
            signature = hmac.new(
                api_token.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Make request
            headers = {
                'Authorization': f'Basic {user_id}:{signature}',
                'Timestamp': timestamp
            }
            
            url = f"https://www.quantconnect.com/api/v2/projects/{project_id}/backtests"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success') and result.get('backtests'):
                    backtests = result.get('backtests', [])
                    completed_backtests = [b for b in backtests if b.get('completed')]
                    
                    if completed_backtests:
                        print(f"   ✅ FOUND {len(completed_backtests)} COMPLETED BACKTESTS!")
                        
                        # Get the most recent one
                        latest = completed_backtests[-1]
                        backtest_id = latest.get('backtestId')
                        
                        # Try to get detailed results
                        if backtest_id:
                            results_url = f"https://www.quantconnect.com/api/v2/projects/{project_id}/backtests/{backtest_id}/read"
                            results_response = requests.get(results_url, headers=headers, timeout=10)
                            
                            if results_response.status_code == 200:
                                results_data = results_response.json()
                                if results_data.get('success'):
                                    successful_results.append({
                                        'project_id': project_id,
                                        'backtest_id': backtest_id,
                                        'data': results_data
                                    })
                                    
                                    print(f"   📈 Retrieved detailed results!")
                                    print(f"   🎯 Project URL: https://www.quantconnect.com/project/{project_id}")
                    else:
                        print(f"   ⏳ Project exists but no completed backtests")
                else:
                    print(f"   ❌ No backtests found")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")
    
    # Method 3: Analyze any results we found
    print(f"\n" + "="*70)
    print("📊 ANALYSIS OF ACTUAL RESULTS")
    print("="*70)
    
    if successful_results:
        print(f"✅ FOUND {len(successful_results)} PROJECTS WITH REAL RESULTS!")
        
        for i, result in enumerate(successful_results):
            project_id = result['project_id']
            data = result['data']
            
            print(f"\n🎯 PROJECT {i+1}: {project_id}")
            print(f"🔗 URL: https://www.quantconnect.com/project/{project_id}")
            
            # Extract key metrics if available
            try:
                statistics = data.get('result', {}).get('Statistics', {})
                if statistics:
                    total_return = statistics.get('Total Performance', {}).get('value', 'N/A')
                    sharpe_ratio = statistics.get('Sharpe Ratio', {}).get('value', 'N/A')
                    max_drawdown = statistics.get('Maximum Drawdown', {}).get('value', 'N/A')
                    
                    print(f"   📈 Total Return: {total_return}")
                    print(f"   📊 Sharpe Ratio: {sharpe_ratio}")
                    print(f"   📉 Max Drawdown: {max_drawdown}")
                    
                    # Check if this matches Claude's claims
                    print(f"\n🔍 CLAIM VERIFICATION:")
                    if isinstance(total_return, str) and '%' in str(total_return):
                        return_pct = float(str(total_return).replace('%', ''))
                        if return_pct > 10:
                            print(f"   ✅ HIGH PERFORMANCE CONFIRMED: {total_return}")
                        else:
                            print(f"   ⚠️  MODERATE PERFORMANCE: {total_return}")
                    
                    print(f"\n📋 FULL STATISTICS AVAILABLE:")
                    for key, value in list(statistics.items())[:10]:
                        print(f"   • {key}: {value.get('value', 'N/A') if isinstance(value, dict) else value}")
                        
            except Exception as e:
                print(f"   ❌ Error parsing results: {e}")
                
        return successful_results
        
    else:
        print("❌ NO COMPLETED BACKTESTS FOUND IN ANY PROJECT")
        print("\n💡 THIS MEANS:")
        print("   • Claude's performance claims CANNOT be verified")
        print("   • No strategies have successfully completed backtests")
        print("   • The 'return without exception set' error is blocking all runs")
        print("   • We need to fix the initialization issue first")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   1. Fix the initialization error before testing performance")
        print(f"   2. Start with a simple buy-and-hold SPY strategy")
        print(f"   3. Once basic strategy works, upgrade to multi-component")
        print(f"   4. Only then can we verify Claude's performance claims")
        
        return []

if __name__ == "__main__":
    verify_claims_with_real_data()