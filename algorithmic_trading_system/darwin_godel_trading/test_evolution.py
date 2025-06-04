#!/usr/bin/env python3
"""
Test Darwin Gödel Trading Machine with a simple evolution
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darwin_godel_trading.dgm_core import DarwinGodelTradingMachine, TradingAgent

def create_test_agent():
    """Create a simple test agent"""
    test_path = "/tmp/dgm_test_agent"
    os.makedirs(test_path, exist_ok=True)
    
    # Simple trading strategy
    code = '''from AlgorithmImports import *

class TestAgent(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        
        self.spy = self.add_equity("SPY", Resolution.DAILY)
        self.spy.set_leverage(2.0)
        
        self.sma = self.sma("SPY", 20)
        
    def on_data(self, data):
        if not self.sma.is_ready:
            return
            
        if self.securities["SPY"].price > self.sma.current.value:
            self.set_holdings("SPY", 1.0)
        else:
            self.liquidate()
'''
    
    with open(os.path.join(test_path, "main.py"), 'w') as f:
        f.write(code)
        
    config = {
        "algorithm-language": "Python",
        "parameters": {},
        "local-id": 999999
    }
    
    with open(os.path.join(test_path, "config.json"), 'w') as f:
        json.dump(config, f)
        
    return test_path

def main():
    print("🧬 Testing Darwin Gödel Trading Machine")
    print("=" * 60)
    
    # Create test agent
    test_agent_path = create_test_agent()
    print(f"Created test agent at: {test_agent_path}")
    
    # Test agent creation and modification
    agent = TradingAgent("test_agent", test_agent_path)
    
    # Simulate performance
    agent.performance_metrics = {
        "cagr": 0.08,
        "sharpe": 0.45,
        "max_drawdown": 0.22
    }
    
    # Test modification proposal
    print("\nTesting modification proposal...")
    proposal = agent.propose_modification()
    print(f"Proposed modification: {proposal}")
    
    # Test self-modification
    print("\nTesting self-modification...")
    try:
        child = agent.self_modify(proposal)
        print(f"Created child agent: {child.agent_id}")
        print(f"Child generation: {child.generation}")
        print(f"Mutations: {child.mutations}")
        
        # Check if modification was applied
        with open(os.path.join(child.code_path, "main.py"), 'r') as f:
            child_code = f.read()
            
        print(f"\nChild code length: {len(child_code)} characters")
        
        # Look for changes
        if "set_leverage(4.0)" in child_code or "set_leverage(5.0)" in child_code:
            print("✅ Leverage modification detected")
        elif "self.rsi" in child_code:
            print("✅ RSI indicator added")
        elif "days < 1" in child_code or "days < 2" in child_code:
            print("✅ Trading frequency increased")
        else:
            print("ℹ️ Other modifications applied")
            
    except Exception as e:
        print(f"❌ Error during self-modification: {e}")
        
    # Test parent selection
    print("\n" + "=" * 60)
    print("Testing DGM parent selection...")
    
    # Create mock DGM with multiple agents
    dgm = DarwinGodelTradingMachine(test_agent_path, "/tmp", target_cagr=0.25)
    
    # Add some mock agents with different performances
    for i in range(5):
        mock_agent = TradingAgent(f"mock_{i}", test_agent_path)
        mock_agent.performance_metrics = {"cagr": 0.05 + i * 0.03}
        dgm.archive.append(mock_agent)
        
    # Test parent selection
    parents = dgm.select_parents(3)
    print(f"\nSelected {len(parents)} parents:")
    for parent in parents:
        print(f"  - {parent.agent_id}: CAGR {parent.performance_metrics.get('cagr', 0)*100:.1f}%")
        
    print("\n✅ DGM basic functionality test complete!")
    
    # Cleanup
    import shutil
    if os.path.exists("/tmp/dgm_test_agent"):
        shutil.rmtree("/tmp/dgm_test_agent")
    for i in range(10):
        path = f"/tmp/test_agent_{datetime.now().strftime('%Y%m%d')}*"
        os.system(f"rm -rf {path}")

if __name__ == "__main__":
    main()