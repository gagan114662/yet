#!/usr/bin/env python3
"""
Test script to verify RD-Agent and QuantConnect setup
"""

import os
import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test QuantConnect
        import lean
        print("✅ Lean CLI imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Lean CLI: {e}")
        
    try:
        # Test RD-Agent
        sys.path.append(str(Path(__file__).parent.parent / "RD-Agent"))
        from rdagent.core.scenario import Scenario
        print("✅ RD-Agent core imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RD-Agent: {e}")
        
    try:
        # Test integration module
        from rd_agent_qc_bridge import QuantConnectIntegration
        print("✅ Integration bridge imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import integration bridge: {e}")


def test_lean_cli():
    """Test if Lean CLI is properly configured"""
    print("\nTesting Lean CLI...")
    
    # Check if credentials exist
    cred_file = Path.home() / ".lean" / "credentials"
    if cred_file.exists():
        print("✅ Lean credentials file found")
    else:
        print("❌ Lean credentials file not found")
        
    # Test lean command
    try:
        result = subprocess.run(["lean", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Lean CLI version: {result.stdout.strip()}")
        else:
            print(f"❌ Lean CLI error: {result.stderr}")
    except Exception as e:
        print(f"❌ Failed to run Lean CLI: {e}")


def test_openrouter_config():
    """Test OpenRouter configuration"""
    print("\nTesting OpenRouter configuration...")
    
    env_file = Path(__file__).parent.parent / "RD-Agent" / ".env"
    if env_file.exists():
        print("✅ .env file found")
        
        # Check for API key
        with open(env_file, 'r') as f:
            content = f.read()
            if "sk-or-v1-" in content:
                print("✅ OpenRouter API key configured")
            else:
                print("❌ OpenRouter API key not found in .env")
                
            if "deepseek/deepseek-r1" in content:
                print("✅ DeepSeek R1 model configured")
            else:
                print("❌ DeepSeek R1 model not configured")
    else:
        print("❌ .env file not found")


def test_simple_strategy():
    """Test creating and running a simple strategy"""
    print("\nTesting simple strategy creation...")
    
    try:
        from rd_agent_qc_bridge import QuantConnectIntegration
        
        # Initialize integration
        qc = QuantConnectIntegration()
        
        # Simple test strategy
        test_strategy = {
            "name": "Test_Strategy",
            "description": "Simple test strategy",
            "type": "momentum",
            "lookback_period": 20,
            "rebalance_frequency": 5,
            "position_size": 0.1,
            "universe_size": 10,
            "min_price": 5,
            "min_volume": 1000000,
            "start_date": "2023,1,1",
            "end_date": "2023,6,30"
        }
        
        # Generate code
        code = qc.generate_strategy_code(test_strategy)
        
        if "class RDAgentStrategy" in code:
            print("✅ Strategy code generated successfully")
            print(f"   Code length: {len(code)} characters")
        else:
            print("❌ Failed to generate strategy code")
            
    except Exception as e:
        print(f"❌ Error testing strategy creation: {e}")


def main():
    """Run all tests"""
    print("=== RD-Agent + QuantConnect Setup Test ===\n")
    
    test_imports()
    test_lean_cli()
    test_openrouter_config()
    test_simple_strategy()
    
    print("\n=== Test Complete ===")
    print("\nTo run the full pipeline:")
    print("python main_pipeline.py")
    
    print("\nTo run RD-Agent model loop with UI:")
    print("cd ../RD-Agent && python -m rdagent.app.cli ui")


if __name__ == "__main__":
    main()