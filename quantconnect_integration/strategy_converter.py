#!/usr/bin/env python3
"""
Strategy Converter for QuantConnect Cloud
Converts local Lean strategies to optimized QuantConnect Cloud format
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class StrategyConverter:
    """Converts local strategies to QuantConnect Cloud format"""
    
    def __init__(self):
        self.cloud_optimizations = {
            'data_normalization': True,
            'margin_trading': True,
            'high_frequency_data': True,
            'alternative_data': True,
            'options_data': True
        }
        
    def convert_strategy(self, local_path: str, output_path: str, strategy_name: str) -> str:
        """Convert a local strategy to cloud format"""
        with open(local_path, 'r') as f:
            content = f.read()
            
        # Apply conversions
        content = self._add_cloud_imports(content)
        content = self._optimize_for_cloud_data(content)
        content = self._add_cloud_features(content)
        content = self._fix_class_name(content, strategy_name)
        content = self._add_performance_tracking(content)
        
        # Save converted strategy
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Converted strategy saved to: {output_path}")
        return content
        
    def _add_cloud_imports(self, content: str) -> str:
        """Add cloud-specific imports and remove local ones"""
        cloud_imports = '''# QuantConnect Cloud Optimized Strategy
from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

'''
        
        # Remove existing imports
        content = re.sub(r'^from AlgorithmImports import \*\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'^import numpy as np\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'^from datetime import.*\n?', '', content, flags=re.MULTILINE)
        
        return cloud_imports + content
        
    def _optimize_for_cloud_data(self, content: str) -> str:
        """Optimize strategy for cloud data access"""
        optimizations = [
            # Use minute resolution for better signals
            (r'Resolution\.Hour', 'Resolution.Minute'),
            
            # Enable extended market hours
            (r'self\.AddEquity\(([^,]+)', r'self.AddEquity(\1, extendedMarketHours=True'),
            
            # Add data normalization
            (r'self\.AddEquity\(([^)]+)\)', 
             r'equity = self.AddEquity(\1)\n        equity.SetDataNormalizationMode(DataNormalizationMode.Adjusted)'),
             
            # Optimize for professional data feeds
            (r'# Trade SPY only for reliable data', 
             '# Using professional data feeds with extended universe'),
        ]
        
        for pattern, replacement in optimizations:
            content = re.sub(pattern, replacement, content)
            
        return content
        
    def _add_cloud_features(self, content: str) -> str:
        """Add cloud-specific features for better performance"""
        # Add to Initialize method
        cloud_features = '''
        # Cloud-specific optimizations
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.UniverseSettings.Resolution = Resolution.Minute
        self.UniverseSettings.ExtendedMarketHours = True
        self.UniverseSettings.FillForward = True
        self.UniverseSettings.Leverage = 4.0
        
        # Advanced data feeds
        self.AddData(VIX, "VIX")  # VIX for volatility signals
        
        # Options data for enhanced signals
        self.spy_option = self.AddOption("SPY")
        self.spy_option.SetFilter(lambda u: u.Strikes(-5, 5).Expiration(timedelta(days=0), timedelta(days=30)))
        
        # Performance tracking
        self.performance_tracker = {
            'trades': 0,
            'wins': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'peak_value': self.Portfolio.TotalPortfolioValue
        }
'''
        
        # Find Initialize method and add features
        init_pattern = r'(def Initialize\(self\):.*?(?=def|\Z))'
        match = re.search(init_pattern, content, re.DOTALL)
        
        if match:
            init_method = match.group(1)
            # Add features before the end of Initialize
            enhanced_init = init_method.rstrip() + cloud_features + '\n'
            content = content.replace(init_method, enhanced_init)
            
        # Add custom security initializer method
        custom_initializer = '''
    def CustomSecurityInitializer(self, security):
        """Custom security initialization for cloud optimization"""
        security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
        security.SetFeeModel(ConstantFeeModel(1.0))  # Professional trading fees
        security.SetFillModel(ImmediateFillModel())
        security.SetSlippageModel(ConstantSlippageModel(0.0001))  # 1 basis point slippage
        
'''
        
        content += custom_initializer
        return content
        
    def _fix_class_name(self, content: str, strategy_name: str) -> str:
        """Ensure proper class naming for cloud deployment"""
        # Find existing class definition
        class_pattern = r'class\s+(\w+)\s*\(QCAlgorithm\):'
        match = re.search(class_pattern, content)
        
        if match:
            old_name = match.group(1)
            if old_name != strategy_name:
                content = re.sub(class_pattern, f'class {strategy_name}(QCAlgorithm):', content)
                
        return content
        
    def _add_performance_tracking(self, content: str) -> str:
        """Add enhanced performance tracking"""
        tracking_code = '''
    def UpdatePerformanceTracking(self):
        """Update performance metrics"""
        current_value = self.Portfolio.TotalPortfolioValue
        
        # Update peak and drawdown
        if current_value > self.performance_tracker['peak_value']:
            self.performance_tracker['peak_value'] = current_value
            
        drawdown = (self.performance_tracker['peak_value'] - current_value) / self.performance_tracker['peak_value']
        self.performance_tracker['max_drawdown'] = max(self.performance_tracker['max_drawdown'], drawdown)
        
        # Log performance weekly
        if self.Time.weekday() == 0:  # Monday
            win_rate = self.performance_tracker['wins'] / max(1, self.performance_tracker['trades'])
            avg_profit = self.performance_tracker['total_profit'] / max(1, self.performance_tracker['trades'])
            
            self.Debug(f"Performance Update - Trades: {self.performance_tracker['trades']}, " +
                      f"Win Rate: {win_rate:.1%}, Avg Profit: {avg_profit:.2%}, " +
                      f"Max DD: {self.performance_tracker['max_drawdown']:.1%}")
                      
    def TrackTrade(self, symbol: str, profit_pct: float):
        """Track individual trade performance"""
        self.performance_tracker['trades'] += 1
        self.performance_tracker['total_profit'] += profit_pct
        
        if profit_pct > 0:
            self.performance_tracker['wins'] += 1
            
        self.UpdatePerformanceTracking()
        
'''
        
        content += tracking_code
        return content
        
    def create_optimized_strategies(self, base_strategies: List[str]) -> Dict[str, str]:
        """Create multiple optimized versions of strategies"""
        optimized_strategies = {}
        
        for strategy_path in base_strategies:
            if not Path(strategy_path).exists():
                logger.warning(f"Strategy not found: {strategy_path}")
                continue
                
            strategy_name = Path(strategy_path).stem
            
            # Create different optimization levels
            versions = {
                'Conservative': self._create_conservative_version,
                'Aggressive': self._create_aggressive_version,
                'UltraAggressive': self._create_ultra_aggressive_version
            }
            
            for version_name, optimizer in versions.items():
                output_name = f"{strategy_name}_{version_name}"
                output_path = f"/tmp/cloud_strategies/{output_name}.py"
                
                content = self.convert_strategy(strategy_path, output_path, output_name)
                content = optimizer(content)
                
                with open(output_path, 'w') as f:
                    f.write(content)
                    
                optimized_strategies[output_name] = output_path
                
        return optimized_strategies
        
    def _create_conservative_version(self, content: str) -> str:
        """Create conservative version with lower leverage"""
        modifications = [
            (r'self\.base_leverage = [\d.]+', 'self.base_leverage = 2.0'),
            (r'self\.max_leverage = [\d.]+', 'self.max_leverage = 3.0'),
            (r'self\.position_size = [\d.]+', 'self.position_size = 0.7'),
            (r'self\.stop_loss = [\d.]+', 'self.stop_loss = 0.02'),  # 2% stop loss
        ]
        
        for pattern, replacement in modifications:
            content = re.sub(pattern, replacement, content)
            
        return content
        
    def _create_aggressive_version(self, content: str) -> str:
        """Create aggressive version with higher leverage"""
        modifications = [
            (r'self\.base_leverage = [\d.]+', 'self.base_leverage = 3.0'),
            (r'self\.max_leverage = [\d.]+', 'self.max_leverage = 5.0'),
            (r'self\.position_size = [\d.]+', 'self.position_size = 0.9'),
            (r'self\.profit_target = [\d.]+', 'self.profit_target = 0.04'),  # 4% profit target
            (r'self\.stop_loss = [\d.]+', 'self.stop_loss = 0.015'),  # 1.5% stop loss
        ]
        
        for pattern, replacement in modifications:
            content = re.sub(pattern, replacement, content)
            
        return content
        
    def _create_ultra_aggressive_version(self, content: str) -> str:
        """Create ultra-aggressive version for maximum returns"""
        modifications = [
            (r'self\.base_leverage = [\d.]+', 'self.base_leverage = 4.0'),
            (r'self\.max_leverage = [\d.]+', 'self.max_leverage = 8.0'),
            (r'self\.position_size = [\d.]+', 'self.position_size = 0.95'),
            (r'self\.profit_target = [\d.]+', 'self.profit_target = 0.06'),  # 6% profit target
            (r'self\.stop_loss = [\d.]+', 'self.stop_loss = 0.01'),  # 1% stop loss
        ]
        
        for pattern, replacement in modifications:
            content = re.sub(pattern, replacement, content)
            
        # Add high-frequency trading elements
        hft_additions = '''
        # Ultra-aggressive high-frequency elements
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(timedelta(minutes=5)), 
                        self.HighFrequencyScalping)
                        
    def HighFrequencyScalping(self):
        """High-frequency scalping for ultra-aggressive returns"""
        if self.IsWarmingUp:
            return
            
        # Quick scalping trades on short-term momentum
        for symbol in [self.spy, self.qqq]:
            if hasattr(self, symbol.Value.lower() + '_mom_ultra_fast'):
                ultra_fast_mom = getattr(self, symbol.Value.lower() + '_mom_ultra_fast').Current.Value
                
                # Ultra-short scalping trades
                if not self.Portfolio[symbol].Invested:
                    if ultra_fast_mom > 0.001:  # Very small momentum threshold
                        self.SetHoldings(symbol, 0.5)  # Quick 50% allocation
                    elif ultra_fast_mom < -0.001:
                        self.SetHoldings(symbol, -0.5)
                else:
                    # Quick profit taking
                    profit_pct = self.Portfolio[symbol].UnrealizedProfitPercent
                    if abs(profit_pct) > 0.005:  # 0.5% profit/loss
                        self.Liquidate(symbol)
'''
        
        content += hft_additions
        return content

def main():
    """Main converter function"""
    converter = StrategyConverter()
    
    # List of strategies to convert
    strategies = [
        "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/rd_agent_strategy_20250530_171329_076469/main.py",
        "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/super_aggressive_momentum/main.py",
        "/mnt/VANDAN_DISK/gagan_stuff/again and again/lean_workspace/extreme_performance_2025/main.py"
    ]
    
    # Create optimized versions
    optimized = converter.create_optimized_strategies(strategies)
    
    print("Created optimized strategy versions:")
    for name, path in optimized.items():
        print(f"  - {name}: {path}")

if __name__ == "__main__":
    main()