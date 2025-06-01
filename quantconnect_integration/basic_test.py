#!/usr/bin/env python3
"""
Basic test to generate a simple QuantConnect strategy
"""

import time
from rd_agent_qc_bridge import QuantConnectIntegration

def main():
    print("Testing strategy generation...")
    
    # Create integration
    qc = QuantConnectIntegration()
    
    # Simple strategy
    strategy = {
        "name": "Simple_Test_Strategy",
        "description": "Basic momentum strategy for testing",
        "type": "momentum", 
        "start_date": "2022,1,1",
        "end_date": "2023,12,31",
        "lookback_period": 20,
        "rebalance_frequency": 5,
        "position_size": 0.1,
        "universe_size": 20,
        "min_price": 10,
        "min_volume": 5000000,
        "schedule_rule": "EveryDay",
        "max_dd_per_security": 0.05,
        "holding_period": 10,
        "indicator_setup": '"momentum": self.MOMP(symbol, 20)',
        "signal_generation_logic": '''
        indicators = self.indicators[symbol]
        momentum = indicators["momentum"].Current.Value
        
        if momentum > 0.02:
            signal = 1
        elif momentum < -0.02:
            signal = -1
        else:
            signal = 0
        '''
    }
    
    # Generate code
    print("Generating strategy code...")
    code = qc.generate_strategy_code(strategy)
    
    # Save to file
    with open("test_strategy.py", "w") as f:
        f.write(code)
        
    print(f"✅ Generated strategy code ({len(code)} characters)")
    print("Strategy saved to test_strategy.py")
    
    # Try creating a project 
    try:
        print("Creating Lean project...")
        project_path = qc.create_lean_project()
        print(f"✅ Created project: {project_path}")
        
        # Copy strategy to main.py
        with open(f"{project_path}/main.py", "w") as f:
            f.write(code)
        print("✅ Strategy copied to main.py")
        
        print("\n🎉 Basic test completed successfully!")
        print(f"You can now run: cd {project_path} && lean backtest")
        
    except Exception as e:
        print(f"❌ Error creating project: {e}")
        
if __name__ == "__main__":
    main()