TARGET_METRICS = {
    'cagr': 0.25,           # 25% annual return target
    'sharpe_ratio': 1.0,     # 1.0+ Sharpe ratio target  
    'max_drawdown': 0.15,    # Maximum 15% drawdown (lower is better)
    'avg_profit': 0.002      # 0.2% average profit per trade
}
REQUIRED_SUCCESSFUL_STRATEGIES = 3
PROGRESS_UPDATE_INTERVAL_SECONDS = 1800  # 30 minutes

# Lean CLI Configuration - REAL CREDENTIALS
LEAN_CLI_USER_ID = "357130"  # Your QuantConnect User ID
LEAN_CLI_API_TOKEN = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"  # Your QuantConnect API Token
LEAN_CLI_PATH = "/home/vandan/.local/bin/lean"  # Detected Lean CLI path

# Real backtesting settings - 15 year period for comprehensive testing
USE_REAL_BACKTESTING = True
BACKTEST_START_DATE = "2009-01-01"  # 15-year backtest period
BACKTEST_END_DATE = "2024-01-01"    # End date for backtests
INITIAL_CAPITAL = 100000             # Starting capital for backtests

# Minimum trading activity requirements
MIN_TRADES_PER_YEAR = 100           # Minimum 100 orders per year
MIN_TOTAL_TRADES = 1500             # Minimum total trades over 15 years
