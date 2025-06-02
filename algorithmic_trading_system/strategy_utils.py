import uuid
import random
from typing import Dict #, NamedTuple # Commenting out Strategy NamedTuple

# Using NamedTuple for a simple immutable Strategy structure
# class Strategy(NamedTuple):
#     id: str
#     parameters: Dict
#     type: str

def generate_strategy_id() -> str:
    """Generates a unique strategy ID."""
    return str(uuid.uuid4())

def generate_next_strategy() -> Dict:
    """
    Generates a new strategy idea as a dictionary.
    It randomly selects one of several predefined strategy templates and populates it with random parameters.
    """
    strategy_templates = [
        {
            "name": "MomentumStrategy",
            "description": "A momentum strategy that buys when momentum is positive and RSI is not overbought, and sells when momentum is negative and RSI is not oversold.",
            "type": "momentum",
            "indicator_setup": '\'"momentum": self.MOMP(symbol, self.lookback_period), "rsi": self.RSI(symbol, 14)\'',
            "signal_generation_logic": '''
indicators = self.indicators[symbol]
momentum = indicators["momentum"].Current.Value
rsi = indicators["rsi"].Current.Value
signal = 0
if self.Securities[symbol].Price > 0 and momentum > (0.01 + random.uniform(-0.005, 0.005)) and rsi < (70 + random.randint(-5, 5)): # Randomized threshold
    signal = 1
elif self.Securities[symbol].Price > 0 and momentum < (-0.01 + random.uniform(-0.005, 0.005)) and rsi > (30 + random.randint(-5, 5)): # Randomized threshold
    signal = -1
'''
        },
        {
            "name": "MeanReversionBB",
            "description": "A mean reversion strategy using Bollinger Bands and RSI. Buys when price crosses below lower BB and RSI is oversold. Sells when price crosses above upper BB and RSI is overbought.",
            "type": "mean_reversion",
            "indicator_setup": '\'"bb": self.BB(symbol, self.lookback_period, 2), "rsi": self.RSI(symbol, 10)\'',
            "signal_generation_logic": '''
indicators = self.indicators[symbol]
price = self.Securities[symbol].Price
upper_band = indicators["bb"].UpperBand.Current.Value
lower_band = indicators["bb"].LowerBand.Current.Value
rsi = indicators["rsi"].Current.Value
signal = 0
if self.Securities[symbol].Price > 0 and upper_band > 0 and lower_band > 0: # Ensure bands are valid
    if price < lower_band and rsi < (35 + random.randint(-5,5)): # Randomized threshold
        signal = 1
    elif price > upper_band and rsi > (65 + random.randint(-5,5)): # Randomized threshold
        signal = -1
'''
        }
    ]

    # Randomly select a template
    strategy_idea = random.choice(strategy_templates).copy() # Use .copy() to avoid modifying the original template

    # Populate with general parameters
    strategy_idea["start_date"] = "2004,1,1"
    strategy_idea["end_date"] = "2023,12,31"
    strategy_idea["lookback_period"] = random.randint(10, 60)
    strategy_idea["rebalance_frequency"] = random.choice([1, 3, 5, 7, 10, 15, 20]) # Example values
    strategy_idea["position_size"] = round(random.uniform(0.05, 0.2), 3)
    strategy_idea["universe_size"] = random.randint(20, 100)
    strategy_idea["min_price"] = round(random.uniform(5.0, 20.0), 2)
    strategy_idea["min_volume"] = random.randint(1000000, 10000000)
    strategy_idea["holding_period"] = random.randint(5, 20) # For insight duration
    strategy_idea["schedule_rule"] = random.choice(['EveryDay', 'WeekStart', 'MonthStart']) # QC Schedule options
    strategy_idea["max_dd_per_security"] = round(random.uniform(0.02, 0.10), 3)

    # strategy_id = generate_strategy_id() # Can be added if needed by other parts of the system later
    # strategy_idea["id"] = strategy_id

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
