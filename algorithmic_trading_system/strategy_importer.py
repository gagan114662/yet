import os
import random
from typing import Dict, List
from pathlib import Path

class StrategyImporter:
    """Import and adapt existing strategies from lean_workspace"""
    
    def __init__(self, lean_workspace_path: str = "lean_workspace"):
        self.lean_workspace_path = lean_workspace_path
        self.top_strategies = self._identify_top_strategies()
        
    def _identify_top_strategies(self) -> List[Dict]:
        """Identify the most promising strategies from analysis"""
        return [
            {
                "name": "QuantumEdgeDominator",
                "path": "quantum_edge_dominator",
                "type": "multi_factor",
                "complexity": "very_high",
                "target_cagr": 0.40,
                "target_sharpe": 1.8,
                "description": "Ultra-sophisticated multi-factor combining momentum, volatility, arbitrage"
            },
            {
                "name": "TargetCrusherUltimate", 
                "path": "target_crusher_ultimate",
                "type": "momentum_reversion",
                "complexity": "high",
                "target_cagr": 0.35,
                "target_sharpe": 1.5,
                "description": "Multi-timeframe momentum with dynamic leverage"
            },
            {
                "name": "UltimateAlphaGenerator",
                "path": "ultimate_alpha_generator", 
                "type": "leveraged_etf",
                "complexity": "moderate",
                "target_cagr": 0.35,
                "target_sharpe": 1.5,
                "description": "Leveraged ETF trading with volatility harvesting"
            },
            {
                "name": "MicrostructureStrategy",
                "path": "microstructure_strategy",
                "type": "high_frequency",
                "complexity": "very_high", 
                "target_cagr": 0.45,
                "target_sharpe": 2.5,
                "description": "High-frequency microstructure trading"
            },
            {
                "name": "ExtremePerformance2025",
                "path": "extreme_performance_2025",
                "type": "aggressive_momentum",
                "complexity": "high",
                "target_cagr": 0.45,
                "target_sharpe": 2.0,
                "description": "Aggressive momentum with 3x leveraged ETFs"
            },
            {
                "name": "EnhancedMultiTimeframe",
                "path": "MultiTimeframeStrategy",
                "type": "multi_timeframe",
                "complexity": "very_high",
                "target_cagr": 0.30,
                "target_sharpe": 1.8,
                "description": "Professional multi-timeframe with risk management"
            },
            {
                "name": "VandanStrategyBurke",
                "path": "VandanStrategyBurke", 
                "type": "options",
                "complexity": "high",
                "target_cagr": 0.50,
                "target_sharpe": 2.0,
                "description": "0-1 DTE options trading strategy"
            }
        ]
    
    def get_strategy_code(self, strategy_name: str) -> str:
        """Get the actual strategy code from lean_workspace"""
        strategy_info = next((s for s in self.top_strategies if s["name"] == strategy_name), None)
        if not strategy_info:
            return self._generate_fallback_strategy()
            
        strategy_path = os.path.join(self.lean_workspace_path, strategy_info["path"], "main.py")
        
        try:
            with open(strategy_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Strategy file not found: {strategy_path}")
            return self._generate_fallback_strategy()
    
    def generate_strategy_from_template(self) -> Dict:
        """Generate a new strategy based on existing templates"""
        # Select a random top strategy as template
        template = random.choice(self.top_strategies)
        
        # Create variation parameters
        strategy_idea = {
            "name": f"{template['name']}_Variant_{random.randint(1000, 9999)}",
            "base_template": template["name"],
            "type": template["type"],
            "complexity": template["complexity"],
            "target_cagr": template["target_cagr"],
            "target_sharpe": template["target_sharpe"],
            "description": f"Variant of {template['description']}",
            
            # Strategy parameters that can be varied
            "start_date": "2020,1,1",
            "end_date": "2023,12,31", 
            "lookback_period": random.randint(10, 50),
            "position_size": round(random.uniform(0.15, 0.35), 3),  # More aggressive sizing
            "universe_size": random.randint(50, 200),
            "min_price": round(random.uniform(5.0, 15.0), 2),
            "min_volume": random.randint(5000000, 20000000),
            "rebalance_frequency": random.choice([1, 2, 3, 5]),  # More frequent rebalancing
            "leverage": round(random.uniform(1.5, 4.0), 1),  # Add leverage for aggressive targets
            "stop_loss": round(random.uniform(0.10, 0.20), 2),  # Tight stop losses
            "profit_target": round(random.uniform(0.15, 0.30), 2),  # Reasonable profit targets
            
            # Advanced parameters for sophisticated strategies
            "volatility_lookback": random.randint(20, 60),
            "momentum_threshold": round(random.uniform(0.02, 0.08), 3),
            "rsi_oversold": random.randint(25, 35),
            "rsi_overbought": random.randint(65, 75),
            "macd_fast": random.randint(8, 15),
            "macd_slow": random.randint(20, 30),
            "bb_std": round(random.uniform(1.5, 2.5), 1)
        }
        
        # Add template-specific parameters
        if template["type"] == "leveraged_etf":
            strategy_idea.update({
                "etf_universe": ["TQQQ", "UPRO", "SQQQ", "SPXS", "QLD", "SSO"],
                "volatility_etfs": ["VXX", "SVXY", "UVXY"],
                "leverage_adjustment": True
            })
        elif template["type"] == "options":
            strategy_idea.update({
                "option_symbols": ["SPX", "SPY", "QQQ"],
                "dte_range": [0, 2],  # 0-2 days to expiration
                "delta_range": [0.15, 0.35],
                "market_strength_threshold": 3
            })
        elif template["type"] == "high_frequency":
            strategy_idea.update({
                "max_leverage": random.uniform(10, 20),
                "tick_size": 0.01,
                "order_size": random.randint(100, 1000),
                "market_making_spread": round(random.uniform(0.001, 0.005), 4)
            })
        elif template["type"] == "multi_factor":
            strategy_idea.update({
                "momentum_weight": round(random.uniform(0.2, 0.4), 2),
                "mean_reversion_weight": round(random.uniform(0.2, 0.4), 2), 
                "volatility_weight": round(random.uniform(0.1, 0.3), 2),
                "arbitrage_weight": round(random.uniform(0.1, 0.2), 2)
            })
            
        return strategy_idea
    
    def adapt_strategy_for_targets(self, strategy_idea: Dict) -> Dict:
        """Adapt strategy parameters to meet aggressive targets"""
        # Increase leverage and position sizing for higher returns
        if strategy_idea.get("target_cagr", 0) < 0.25:
            strategy_idea["leverage"] = strategy_idea.get("leverage", 2.0) * 1.2
            strategy_idea["position_size"] = min(strategy_idea.get("position_size", 0.2) * 1.3, 0.5)
            
        # Tighten risk management for better Sharpe
        if strategy_idea.get("target_sharpe", 0) < 1.0:
            strategy_idea["stop_loss"] = min(strategy_idea.get("stop_loss", 0.15), 0.12)
            strategy_idea["rebalance_frequency"] = min(strategy_idea.get("rebalance_frequency", 5), 3)
            
        # Add volatility targeting for drawdown control
        strategy_idea["volatility_target"] = 0.15  # 15% vol target
        strategy_idea["max_portfolio_drawdown"] = 0.15  # 15% max drawdown
        
        return strategy_idea
    
    def _generate_fallback_strategy(self) -> str:
        """Generate a fallback strategy if template not found"""
        return '''
from AlgorithmImports import *

class FallbackAggressiveStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Aggressive leveraged ETF universe
        self.etfs = ["TQQQ", "UPRO", "SQQQ", "SPXS"]
        for etf in self.etfs:
            self.AddEquity(etf, Resolution.Hour)
            
        # Indicators for momentum
        self.rsi = {}
        self.macd = {}
        for etf in self.etfs:
            self.rsi[etf] = self.RSI(etf, 14)
            self.macd[etf] = self.MACD(etf, 12, 26, 9)
            
        # High leverage for aggressive targets
        self.leverage = 3.0
        self.stop_loss = 0.12  # 12% stop loss
        
    def OnData(self, data):
        for etf in self.etfs:
            if not data.ContainsKey(etf) or not self.rsi[etf].IsReady:
                continue
                
            rsi = self.rsi[etf].Current.Value
            macd = self.macd[etf].Current.Value
            signal = self.macd[etf].Signal.Current.Value
            
            # Aggressive momentum signals
            if etf in ["TQQQ", "UPRO"]:  # Long ETFs
                if rsi < 30 and macd > signal:  # Oversold with momentum
                    self.SetHoldings(etf, self.leverage * 0.5)
                elif rsi > 70:  # Overbought - close
                    self.Liquidate(etf)
            else:  # Short ETFs  
                if rsi > 70 and macd < signal:  # Overbought with momentum
                    self.SetHoldings(etf, self.leverage * 0.3)
                elif rsi < 30:  # Oversold - close
                    self.Liquidate(etf)
'''

    def list_available_strategies(self) -> List[Dict]:
        """List all available strategy templates"""
        return self.top_strategies
        
    def get_random_high_performance_strategy(self) -> Dict:
        """Get a random strategy optimized for high performance"""
        strategy = self.generate_strategy_from_template()
        return self.adapt_strategy_for_targets(strategy)