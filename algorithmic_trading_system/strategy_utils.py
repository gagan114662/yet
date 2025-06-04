import uuid
import random
from typing import Dict
from strategy_importer import StrategyImporter

# Initialize strategy importer
strategy_importer = StrategyImporter()

def generate_strategy_id() -> str:
    """Generates a unique strategy ID."""
    return str(uuid.uuid4())

def generate_next_strategy() -> Dict:
    """
    Generates a new strategy idea using either lean_workspace templates or fallback strategies.
    Uses sophisticated strategies designed to meet aggressive performance targets.
    """
    # 90% chance to use high-performance strategy from lean_workspace (increased for real trading)
    if random.random() < 0.9:
        try:
            return strategy_importer.get_random_high_performance_strategy()
        except Exception as e:
            print(f"Error generating strategy from importer: {e}. Falling back to template.")
    
    # Fallback to enhanced template-based generation
    strategy_templates = [
        {
            "name": "AggressiveMomentumStrategy",
            "description": "High-leverage momentum strategy targeting 25%+ CAGR",
            "type": "aggressive_momentum",
            "indicator_setup": '\'"momentum": self.MOMP(symbol, self.lookback_period), "rsi": self.RSI(symbol, 14), "macd": self.MACD(symbol, 12, 26, 9)\'',
            "signal_generation_logic": '''
indicators = self.indicators[symbol]
momentum = indicators["momentum"].Current.Value
rsi = indicators["rsi"].Current.Value
macd = indicators["macd"].Current.Value
signal = indicators["macd"].Signal.Current.Value
trade_signal = 0

# Aggressive momentum entry with confluence
if (self.Securities[symbol].Price > 0 and 
    momentum > 0.02 and 
    rsi < 65 and 
    macd > signal):
    trade_signal = 1
elif (self.Securities[symbol].Price > 0 and 
      momentum < -0.02 and 
      rsi > 35 and 
      macd < signal):
    trade_signal = -1
'''
        },
        {
            "name": "LeveragedETFStrategy", 
            "description": "Leveraged ETF rotation targeting extreme performance",
            "type": "leveraged_etf",
            "indicator_setup": '\'"rsi": self.RSI(symbol, 14), "bb": self.BB(symbol, 20, 2), "adx": self.ADX(symbol, 14)\'',
            "signal_generation_logic": '''
indicators = self.indicators[symbol]
rsi = indicators["rsi"].Current.Value
bb_upper = indicators["bb"].UpperBand.Current.Value  
bb_lower = indicators["bb"].LowerBand.Current.Value
adx = indicators["adx"].Current.Value
price = self.Securities[symbol].Price
trade_signal = 0

# Leveraged ETF logic with trend strength
if symbol in ["TQQQ", "UPRO"]:  # Bullish ETFs
    if price < bb_lower and rsi < 35 and adx > 25:  # Oversold with strong trend
        trade_signal = 1
    elif rsi > 70:  # Exit overbought
        trade_signal = 0
elif symbol in ["SQQQ", "SPXS"]:  # Bearish ETFs  
    if price > bb_upper and rsi > 65 and adx > 25:  # Overbought with strong trend
        trade_signal = 1
    elif rsi < 30:  # Exit oversold
        trade_signal = 0
'''
        },
        {
            "name": "VolatilityHarvestingStrategy",
            "description": "VIX premium harvesting for consistent alpha",
            "type": "volatility_harvesting", 
            "indicator_setup": '\'"vix": self.RSI("VIX", 14), "bb": self.BB("VXX", 20, 2.5)\'',
            "signal_generation_logic": '''
vix_rsi = indicators.get("vix", {}).get("Current", {}).get("Value", 50)
vxx_price = self.Securities.get("VXX", {}).get("Price", 0)
bb_upper = indicators.get("bb", {}).get("UpperBand", {}).get("Current", {}).get("Value", 0)
bb_lower = indicators.get("bb", {}).get("LowerBand", {}).get("Current", {}).get("Value", 0)
trade_signal = 0

# Volatility premium harvesting logic
if vix_rsi > 70 and vxx_price > bb_upper:  # High volatility, short VXX
    trade_signal = -1  
elif vix_rsi < 30 and vxx_price < bb_lower:  # Low volatility, long SVXY
    trade_signal = 1
'''
        }
    ]

    # Select template with bias toward aggressive strategies
    strategy_idea = random.choice(strategy_templates).copy()

    # Enhanced parameters for aggressive performance
    strategy_idea.update({
        "start_date": "2020,1,1",
        "end_date": "2023,12,31",
        "lookback_period": random.randint(10, 30),  # Shorter for more responsive
        "rebalance_frequency": random.choice([1, 2, 3]),  # More frequent rebalancing
        "position_size": round(random.uniform(0.2, 0.4), 3),  # Larger positions
        "leverage": round(random.uniform(2.0, 4.0), 1),  # High leverage for targets
        "universe_size": random.randint(50, 150),
        "min_price": round(random.uniform(8.0, 25.0), 2),
        "min_volume": random.randint(5000000, 20000000),
        "stop_loss": round(random.uniform(0.08, 0.15), 2),  # Tight stops
        "profit_target": round(random.uniform(0.12, 0.25), 2),  # Reasonable targets
        "volatility_target": 0.15,  # 15% vol targeting
        "max_drawdown": 0.15,  # 15% max drawdown limit
        
        # Advanced risk management
        "position_concentration_limit": 0.25,  # Max 25% per position
        "correlation_limit": 0.7,  # Limit correlated positions
        "momentum_threshold": round(random.uniform(0.02, 0.05), 3),
        "mean_reversion_threshold": round(random.uniform(1.5, 2.5), 1),
        
        # Performance targeting
        "target_cagr": 0.25,  # 25% target
        "target_sharpe": 1.0,  # 1.0+ Sharpe target
        "target_max_dd": 0.15  # Max 15% drawdown
    })

    return strategy_idea

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    for i in range(3):
        strategy = generate_next_strategy()
        print(f"Generated Strategy Idea {i+1}:")
        for key, value in strategy.items():
            if key == "signal_generation_logic" or key == "indicator_setup":
                print(f"  {key}: {repr(value)[:80]}...") # Print truncated version for readability
            else:
                print(f"  {key}: {value}")
        print("-" * 30)
