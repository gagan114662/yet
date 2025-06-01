import uuid
from typing import Dict, NamedTuple

# Using NamedTuple for a simple immutable Strategy structure
class Strategy(NamedTuple):
    id: str
    parameters: Dict
    type: str

def generate_strategy_id() -> str:
    """Generates a unique strategy ID."""
    return str(uuid.uuid4())

def generate_next_strategy() -> Strategy:
    """
    Generates a new strategy.
    For now, this is a placeholder and will generate one of a few predefined
    simple strategy types with fixed or slightly varied parameters.
    In the future, this will be driven by the genetic algorithm.
    """
    # Placeholder: Cycle through a few types or use random choice
    # This is a very basic example.
    # We can expand this with more sophisticated parameter generation later.

    # Example: Simple moving average crossover
    strategy_type = "moving_average_crossover"
    params = {
        'short_window': 20,
        'long_window': 50
    }

    return Strategy(
        id=generate_strategy_id(),
        parameters=params,
        type=strategy_type
    )

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    strategy1 = generate_next_strategy()
    print(f"Generated Strategy 1: {strategy1}")

    strategy2 = generate_next_strategy()
    print(f"Generated Strategy 2: {strategy2}")
