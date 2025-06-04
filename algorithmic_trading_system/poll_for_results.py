#!/usr/bin/env python3
"""
POLL FOR BACKTEST RESULTS - Keep checking until we get actual data
"""

import sys
import time
import json
import requests
import hashlib
import hmac
from datetime import datetime

sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

def poll_for_results():
    """Continuously poll QuantConnect API for backtest results"""
    
    print("🔄 POLLING FOR BACKTEST RESULTS")
    print("="*50)
    
    # API credentials
    user_id = "357130"
    api_token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    project_id = "23349848"
    
    print(f"📊 Project: {project_id}")
    print(f"🔗 URL: https://www.quantconnect.com/project/{project_id}")
    print(f"⏰ Starting poll at {datetime.now().strftime('%H:%M:%S')}")
    
    poll_count = 0
    max_polls = 60  # 5 minutes of polling
    
    while poll_count < max_polls:
        poll_count += 1
        print(f"\n🔍 Poll #{poll_count} at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Get list of backtests
            timestamp = str(int(time.time()))
            data = ""
            
            # Create signature for backtests list
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
            
            # Get backtests list
            url = f"https://www.quantconnect.com/api/v2/projects/{project_id}/backtests"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    backtests = result.get('backtests', [])
                    
                    if backtests:
                        print(f"   📈 Found {len(backtests)} backtest(s)")
                        
                        # Check each backtest
                        for i, backtest in enumerate(backtests):
                            backtest_id = backtest.get('backtestId')
                            name = backtest.get('name', 'Unnamed')
                            completed = backtest.get('completed', False)
                            progress = backtest.get('progress', 0)
                            error = backtest.get('error', '')
                            
                            print(f"   🎯 Backtest {i+1}: {name}")
                            print(f"      ID: {backtest_id}")
                            print(f"      Completed: {completed}")
                            print(f"      Progress: {progress*100:.1f}%")
                            
                            if error:
                                print(f"      ❌ Error: {error}")
                            
                            # If completed, get detailed results
                            if completed and backtest_id:
                                print(f"   ✅ GETTING DETAILED RESULTS...")
                                
                                # Create signature for results
                                result_message = f"GET\n{data}\n/projects/{project_id}/backtests/{backtest_id}/read\n{timestamp}"
                                result_signature = hmac.new(
                                    api_token.encode('utf-8'),
                                    result_message.encode('utf-8'),
                                    hashlib.sha256
                                ).hexdigest()
                                
                                result_headers = {
                                    'Authorization': f'Basic {user_id}:{result_signature}',
                                    'Timestamp': timestamp
                                }
                                
                                results_url = f"https://www.quantconnect.com/api/v2/projects/{project_id}/backtests/{backtest_id}/read"
                                results_response = requests.get(results_url, headers=result_headers, timeout=15)
                                
                                if results_response.status_code == 200:
                                    results_data = results_response.json()
                                    
                                    if results_data.get('success'):
                                        print(f"   🎉 BACKTEST RESULTS FOUND!")
                                        
                                        # Extract key metrics
                                        try:
                                            result_obj = results_data.get('result', {})
                                            statistics = result_obj.get('Statistics', {})
                                            
                                            if statistics:
                                                print(f"\n📊 ACTUAL PERFORMANCE METRICS:")
                                                print(f"="*40)
                                                
                                                # Key metrics
                                                key_metrics = [
                                                    'Total Performance',
                                                    'CAGR',
                                                    'Sharpe Ratio',
                                                    'Maximum Drawdown',
                                                    'Win Rate',
                                                    'Total Trades',
                                                    'Average Win',
                                                    'Average Loss'
                                                ]
                                                
                                                for metric in key_metrics:
                                                    if metric in statistics:
                                                        value = statistics[metric]
                                                        if isinstance(value, dict):
                                                            display_value = value.get('value', 'N/A')
                                                        else:
                                                            display_value = value
                                                        print(f"   {metric}: {display_value}")
                                                
                                                print(f"\n📈 ALL AVAILABLE STATISTICS:")
                                                print(f"-"*40)
                                                for key, value in statistics.items():
                                                    if isinstance(value, dict):
                                                        display_value = value.get('value', 'N/A')
                                                    else:
                                                        display_value = value
                                                    print(f"   {key}: {display_value}")
                                                
                                                # Save results to file
                                                with open('/mnt/VANDAN_DISK/gagan_stuff/again and again/algorithmic_trading_system/ACTUAL_BACKTEST_RESULTS.json', 'w') as f:
                                                    json.dump(results_data, f, indent=2)
                                                
                                                print(f"\n💾 Full results saved to: ACTUAL_BACKTEST_RESULTS.json")
                                                print(f"🎯 THIS IS REAL DATA - NO MORE GUESSING!")
                                                
                                                return results_data
                                            else:
                                                print(f"   ⚠️  No statistics found in results")
                                                
                                        except Exception as e:
                                            print(f"   ❌ Error parsing results: {e}")
                                    else:
                                        print(f"   ❌ Failed to get results: {results_data}")
                                else:
                                    print(f"   ❌ Results API error: {results_response.status_code}")
                            
                            elif not completed:
                                print(f"   ⏳ Still running... {progress*100:.1f}% complete")
                    else:
                        print(f"   ❌ No backtests found")
                else:
                    print(f"   ❌ API returned: {result}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        # Wait before next poll
        if poll_count < max_polls:
            print(f"   ⏳ Waiting 5 seconds before next check...")
            time.sleep(5)
    
    print(f"\n⏰ Polling completed after {max_polls} attempts")
    print(f"💡 If no results found, the backtest may not have started or failed to initialize")
    return None

if __name__ == "__main__":
    poll_for_results()