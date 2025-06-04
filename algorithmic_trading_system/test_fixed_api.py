#!/usr/bin/env python3
"""
Test the fixed QuantConnect API with files/update
"""

import sys
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration')

from working_qc_api import QuantConnectCloudAPI

def test_strategy_upload():
    # Initialize API
    api = QuantConnectCloudAPI(
        "357130", 
        "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
    )
    
    # Test strategy with clear indicators it's our code
    test_strategy = '''from AlgorithmImports import *
import numpy as np

class FIXED_API_TEST_STRATEGY(QCAlgorithm):
    """
    *** THIS IS THE FIXED API TEST - NOT DEFAULT TEMPLATE! ***
    If you see this comment, the files/update fix worked!
    """
    
    def initialize(self):
        # 15-YEAR VERIFIED BACKTEST
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        # Test strategy - QQQ with leverage
        self.symbol = self.add_equity("QQQ", Resolution.Daily)
        self.symbol.set_leverage(15.0)
        
        self.sma_fast = self.sma('QQQ', 5)
        self.sma_slow = self.sma('QQQ', 20)
        
        self.log("*** FIXED API TEST STRATEGY INITIALIZED! ***")
        
    def on_data(self, data):
        if not (self.sma_fast.is_ready and self.sma_slow.is_ready):
            return
            
        # Simple momentum strategy
        if self.sma_fast.current.value > self.sma_slow.current.value and not self.portfolio.invested:
            self.set_holdings("QQQ", 1.0)
            self.log("BUY: This proves the strategy code was uploaded!")
        elif self.sma_fast.current.value < self.sma_slow.current.value and self.portfolio.invested:
            self.liquidate()
            self.log("SELL: API fix confirmed working!")
'''
    
    print("Testing Fixed API with files/update...")
    result = api.deploy_strategy("FIXED_API_TEST", test_strategy)
    
    if result['success']:
        print(f"\nSUCCESS! Fixed API Test Deployed")
        print(f"URL: {result['url']}")
        print(f"Project ID: {result['project_id']}")
        print("\nVERIFICATION STEPS:")
        print("1. Click the URL above")
        print("2. Look for comment: '*** THIS IS THE FIXED API TEST - NOT DEFAULT TEMPLATE! ***'")
        print("3. Check logs for: 'FIXED API TEST STRATEGY INITIALIZED!'")
        print("4. If you see these, the fix worked!")
        return result['url']
    else:
        print(f"Test failed: {result}")
        return None

if __name__ == "__main__":
    test_strategy_upload()