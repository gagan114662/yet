#!/usr/bin/env python3
"""
Test reading REAL backtest results from QuantConnect
"""

import sys
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI

def test_read_real_results():
    # Test with the actual project we just created
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    # Use the champion project from the last run
    project_id = "23346820"
    backtest_id = "40b479c6fe1ca10368d3c4fb96160054"
    
    print(f"Testing with:")
    print(f"  Project ID: {project_id}")
    print(f"  Backtest ID: {backtest_id}")
    print(f"  URL: https://www.quantconnect.com/project/{project_id}")
    
    print("\nReading REAL backtest results...")
    results = api.read_backtest_results(project_id, backtest_id)
    
    if results:
        print("\n*** REAL QUANTCONNECT RESULTS ***")
        print(f"CAGR: {results['cagr']:.3f}%")
        print(f"Sharpe Ratio: {results['sharpe']:.3f}")
        print(f"Total Orders: {results['total_orders']}")
        print(f"Orders/Year: {results['total_orders'] / 15:.1f}")
        print(f"Max Drawdown: {results['drawdown']:.1f}%")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Net Profit: {results['net_profit']:.1f}%")
        print(f"Alpha: {results['alpha']:.3f}")
        print(f"Beta: {results['beta']:.3f}")
        
        print(f"\nTarget Analysis:")
        print(f"  CAGR Target (25%): {'✅ PASS' if results['cagr'] >= 25 else '❌ FAIL'} ({results['cagr']:.1f}%)")
        print(f"  Trade Frequency (100+/year): {'✅ PASS' if results['total_orders']/15 >= 100 else '❌ FAIL'} ({results['total_orders']/15:.1f}/year)")
        
        return results
    else:
        print("❌ Failed to read results")
        return None

if __name__ == "__main__":
    test_read_real_results()