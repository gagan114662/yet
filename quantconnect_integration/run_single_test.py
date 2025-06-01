#!/usr/bin/env python3
"""
Run a single strategy test with proper configuration
"""

import os
import subprocess
from rd_agent_qc_bridge import QuantConnectIntegration

def test_single_strategy():
    """Test one strategy properly"""
    
    print("🚀 Testing Strategy Against Your Targets")
    print("Target: CAGR>25%, Sharpe>1.0, MaxDD<20%, AvgProfit>0.75%")
    print("="*60)
    
    # Create strategy
    qc = QuantConnectIntegration()
    
    strategy = {
        "name": "Momentum_Test",
        "description": "Simple momentum test strategy",
        "type": "momentum",
        "start_date": "2022,1,1",
        "end_date": "2023,12,31", 
        "lookback_period": 20,
        "rebalance_frequency": 5,
        "position_size": 0.1,
        "universe_size": 20,
        "min_price": 10,
        "min_volume": 5000000,
        "indicator_setup": '"momentum": self.MOMP(symbol, 20), "rsi": self.RSI(symbol, 14)',
        "signal_generation_logic": '''
        indicators = self.indicators[symbol]
        momentum = indicators["momentum"].Current.Value
        rsi = indicators["rsi"].Current.Value
        
        # Buy: positive momentum, not overbought
        if momentum > 0.02 and rsi < 70:
            signal = 1
        # Sell: negative momentum or overbought  
        elif momentum < -0.02 or rsi > 80:
            signal = -1
        else:
            signal = 0
        '''
    }
    
    # Generate and save strategy
    code = qc.generate_strategy_code(strategy)
    project_path = qc.create_lean_project()
    
    with open(f"{project_path}/main.py", "w") as f:
        f.write(code)
    
    print(f"✅ Created project: {project_path}")
    print(f"✅ Generated strategy code ({len(code)} chars)")
    
    print("\n📋 Next Steps:")
    print(f"1. cd {project_path}")
    print("2. lean init")
    print("3. lean backtest")
    print("\nOR copy the strategy to QuantConnect cloud for immediate results!")
    
    # Show the strategy code location
    print(f"\n📄 Strategy code saved to: {project_path}/main.py")
    
    return project_path

if __name__ == "__main__":
    project = test_single_strategy()
    
    print(f"\n🎯 To test against your targets:")
    print(f"   The strategy is ready in {project}")
    print(f"   Run the backtest to see if it achieves:")
    print(f"   • CAGR > 25%")
    print(f"   • Sharpe Ratio > 1.0") 
    print(f"   • Max Drawdown < 20%")
    print(f"   • Average Profit > 0.75%")