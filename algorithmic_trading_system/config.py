TARGET_METRICS = {
    'cagr': 0.25,
    'sharpe_ratio': 1.0,
    'max_drawdown': 0.20,
    'avg_profit': 0.0075
}
REQUIRED_SUCCESSFUL_STRATEGIES = 3
PROGRESS_UPDATE_INTERVAL_SECONDS = 1800  # 30 minutes

# Lean CLI Configuration
LEAN_CLI_USER_ID = " 357130"  # Placeholder
LEAN_CLI_API_TOKEN = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"  # Placeholder
LEAN_CLI_PATH = "/path/to/lean/directory"  # Placeholder, e.g., "lean" if in PATH or full path like "/Users/username/.local/bin/lean"

# Note for the user:
# Please replace the placeholder values above with your actual Lean CLI credentials and the correct path to the Lean CLI executable.
# - LEAN_CLI_USER_ID: Your user ID for QuantConnect.
# - LEAN_CLI_API_TOKEN: Your API token for QuantConnect.
# - LEAN_CLI_PATH: The command to run Lean CLI. If 'lean' is in your system's PATH, 'lean' is sufficient.
#   Otherwise, provide the full path to the executable.
