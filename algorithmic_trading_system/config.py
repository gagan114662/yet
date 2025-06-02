TARGET_METRICS = {
    'annual_return': 0.18,
    'max_drawdown': -0.12,  # Negative value indicates acceptable loss
    'sharpe_ratio': 1.8,
    'win_rate': 0.55
}
REQUIRED_SUCCESSFUL_STRATEGIES = 3
PROGRESS_UPDATE_INTERVAL_SECONDS = 1800  # 30 minutes

# Lean CLI Configuration
LEAN_CLI_USER_ID = "YOUR_LEAN_USER_ID"  # Placeholder
LEAN_CLI_API_TOKEN = "YOUR_LEAN_API_TOKEN"  # Placeholder
LEAN_CLI_PATH = "/path/to/your/lean/cli"  # Placeholder, e.g., "lean" if in PATH or full path like "/Users/username/.local/bin/lean"

# Note for the user:
# Please replace the placeholder values above with your actual Lean CLI credentials and the correct path to the Lean CLI executable.
# - LEAN_CLI_USER_ID: Your user ID for QuantConnect.
# - LEAN_CLI_API_TOKEN: Your API token for QuantConnect.
# - LEAN_CLI_PATH: The command to run Lean CLI. If 'lean' is in your system's PATH, 'lean' is sufficient.
#   Otherwise, provide the full path to the executable.
