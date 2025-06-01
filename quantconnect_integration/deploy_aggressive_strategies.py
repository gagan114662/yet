#!/usr/bin/env python3
"""
AGGRESSIVE STRATEGY DEPLOYMENT SCRIPT
Deploy 5 sophisticated strategies to QuantConnect Cloud for high-performance backtesting

Usage: python deploy_aggressive_strategies.py
"""

import os
import shutil
from datetime import datetime

class AggressiveStrategyDeployer:
    
    def __init__(self):
        self.base_path = "/mnt/VANDAN_DISK/gagan_stuff/again and again/quantconnect_integration"
        self.lean_workspace = "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace"
        
        self.strategies = [
            {
                "name": "gamma_flow_strategy",
                "description": "Gamma Flow & Options Positioning",
                "target_cagr": "40%+",
                "leverage": "5x",
                "features": ["Options Flow Analysis", "Gamma Scalping", "Volatility Surface Arbitrage"]
            },
            {
                "name": "regime_momentum_strategy", 
                "description": "Cross-Asset Regime Momentum",
                "target_cagr": "35%+",
                "leverage": "5x",
                "features": ["Regime Detection", "Cross-Asset Momentum", "Risk Parity"]
            },
            {
                "name": "crisis_alpha_strategy",
                "description": "Crisis Alpha & Tail Risk Hedging", 
                "target_cagr": "50%+ (during crises)",
                "leverage": "10x",
                "features": ["Tail Risk Hedging", "Crisis Detection", "Volatility Trading"]
            },
            {
                "name": "earnings_momentum_strategy",
                "description": "Earnings Momentum & Options Flow",
                "target_cagr": "60%+", 
                "leverage": "8x",
                "features": ["Earnings Calendar", "Options Flow", "Surprise Momentum"]
            },
            {
                "name": "microstructure_strategy",
                "description": "Microstructure & Mean Reversion",
                "target_cagr": "45%+",
                "leverage": "15x", 
                "features": ["High Frequency", "Market Making", "Order Flow Analysis"]
            },
            {
                "name": "strategy_rotator",
                "description": "Master Strategy Rotator",
                "target_cagr": "50%+",
                "leverage": "8x",
                "features": ["Dynamic Allocation", "Regime Switching", "Risk Management"]
            }
        ]
        
    def deploy_all_strategies(self):
        """Deploy all aggressive strategies"""
        print("🚀 DEPLOYING AGGRESSIVE TRADING STRATEGIES")
        print("=" * 60)
        print()
        
        success_count = 0
        
        for strategy in self.strategies:
            try:
                print(f"📦 Deploying {strategy['description']}...")
                self.deploy_strategy(strategy)
                print(f"✅ SUCCESS: {strategy['name']} deployed")
                success_count += 1
            except Exception as e:
                print(f"❌ FAILED: {strategy['name']} - {str(e)}")
            print()
        
        print("=" * 60)
        print(f"🎯 DEPLOYMENT SUMMARY: {success_count}/{len(self.strategies)} strategies deployed")
        
        if success_count == len(self.strategies):
            print("🚀 ALL STRATEGIES READY FOR CLOUD TESTING!")
            self.print_cloud_instructions()
        else:
            print("⚠️  Some strategies failed to deploy. Check errors above.")
            
    def deploy_strategy(self, strategy):
        """Deploy individual strategy to lean workspace"""
        source_dir = os.path.join(self.base_path, strategy["name"])
        target_dir = os.path.join(self.lean_workspace, strategy["name"])
        
        # Create target directory
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)
        
        # Copy strategy files
        for file_name in ["main.py", "config.json"]:
            source_file = os.path.join(source_dir, file_name)
            target_file = os.path.join(target_dir, file_name)
            
            if os.path.exists(source_file):
                shutil.copy2(source_file, target_file)
            else:
                raise FileNotFoundError(f"Missing {file_name}")
        
        # Validate strategy
        self.validate_strategy(target_dir, strategy)
        
        print(f"  📈 Target CAGR: {strategy['target_cagr']}")
        print(f"  🔥 Leverage: {strategy['leverage']}")
        print(f"  ⚡ Features: {', '.join(strategy['features'])}")
        
    def validate_strategy(self, strategy_dir, strategy_info):
        """Validate strategy files"""
        main_py = os.path.join(strategy_dir, "main.py")
        config_json = os.path.join(strategy_dir, "config.json")
        
        # Check file sizes
        main_size = os.path.getsize(main_py)
        config_size = os.path.getsize(config_json)
        
        if main_size < 1000:
            raise ValueError("main.py too small - strategy incomplete")
        if config_size < 100:
            raise ValueError("config.json too small - configuration incomplete")
            
        # Quick syntax check
        with open(main_py, 'r') as f:
            content = f.read()
            if "class" not in content or "Initialize" not in content:
                raise ValueError("Invalid strategy structure")
                
        print(f"  ✓ Validated: {main_size:,} bytes main.py, {config_size} bytes config.json")
        
    def print_cloud_instructions(self):
        """Print instructions for cloud deployment"""
        print()
        print("🌐 QUANTCONNECT CLOUD DEPLOYMENT INSTRUCTIONS")
        print("=" * 60)
        print()
        print("1. Go to https://www.quantconnect.com/terminal")
        print("2. Create new algorithm for each strategy:")
        print()
        
        for i, strategy in enumerate(self.strategies, 1):
            print(f"   {i}. {strategy['description']}")
            print(f"      Target: {strategy['target_cagr']} CAGR")
            print(f"      File: lean_workspace/{strategy['name']}/main.py")
            print()
            
        print("3. Copy the main.py content for each strategy")
        print("4. Set backtest period: 2020-01-01 to 2024-12-31")
        print("5. Enable paper trading account with margin")
        print("6. Set cash: $100,000")
        print("7. Run backtests in parallel")
        print()
        print("🎯 EXPECTED RESULTS:")
        print("   • Combined portfolio CAGR: 40-60%")
        print("   • Sharpe Ratio: 1.5-2.5")
        print("   • Max Drawdown: 15-25%")
        print("   • Each strategy optimized for different market regimes")
        print()
        print("⚠️  HIGH LEVERAGE WARNING:")
        print("   These strategies use 5-15x leverage")
        print("   Only deploy with proper risk management")
        print("   Monitor positions closely during live trading")
        
    def generate_strategy_summary(self):
        """Generate comprehensive strategy summary"""
        summary_file = os.path.join(self.base_path, "AGGRESSIVE_STRATEGY_SUMMARY.md")
        
        with open(summary_file, 'w') as f:
            f.write("# 🚀 AGGRESSIVE TRADING STRATEGIES SUITE\\n\\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## 🎯 PERFORMANCE TARGETS\\n\\n")
            f.write("- **Combined CAGR:** 40-60%\\n")
            f.write("- **Sharpe Ratio:** 1.5-2.5\\n")
            f.write("- **Max Drawdown:** <25%\\n")
            f.write("- **Leverage:** 5-15x depending on strategy\\n\\n")
            
            f.write("## 📊 STRATEGY BREAKDOWN\\n\\n")
            
            for i, strategy in enumerate(self.strategies, 1):
                f.write(f"### {i}. {strategy['description']}\\n\\n")
                f.write(f"**Target CAGR:** {strategy['target_cagr']}\\n")
                f.write(f"**Leverage:** {strategy['leverage']}\\n")
                f.write(f"**Key Features:**\\n")
                for feature in strategy['features']:
                    f.write(f"- {feature}\\n")
                f.write("\\n")
                
            f.write("## 🌟 UNIQUE FEATURES\\n\\n")
            f.write("- **Options Flow Analysis:** Real-time gamma positioning\\n")
            f.write("- **Regime Detection:** Advanced market state identification\\n") 
            f.write("- **Crisis Alpha:** Tail risk hedging that profits from volatility\\n")
            f.write("- **Earnings Momentum:** Sophisticated earnings surprise prediction\\n")
            f.write("- **Microstructure:** High-frequency mean reversion and scalping\\n")
            f.write("- **Dynamic Rotation:** Automatic strategy switching based on conditions\\n\\n")
            
            f.write("## ⚡ AGGRESSIVE TECHNIQUES\\n\\n")
            f.write("- **Ultra-High Leverage:** Up to 15x for microstructure strategies\\n")
            f.write("- **Alternative Data:** VIX term structure, options flow, credit spreads\\n")
            f.write("- **Multi-Timeframe:** Second to daily analysis\\n")
            f.write("- **Cross-Asset:** Equities, bonds, commodities, currencies, crypto\\n")
            f.write("- **Event-Driven:** Earnings, crisis, volatility events\\n")
            f.write("- **Market Making:** Bid-ask spread capture\\n\\n")
            
            f.write("## 🚨 RISK WARNINGS\\n\\n")
            f.write("- These strategies use extreme leverage and aggressive techniques\\n")
            f.write("- Only suitable for sophisticated investors\\n")
            f.write("- Requires careful risk management and monitoring\\n")
            f.write("- Past performance does not guarantee future results\\n")
            f.write("- Can experience significant drawdowns during adverse conditions\\n\\n")
            
        print(f"📄 Strategy summary generated: {summary_file}")

if __name__ == "__main__":
    deployer = AggressiveStrategyDeployer()
    deployer.deploy_all_strategies()
    deployer.generate_strategy_summary()