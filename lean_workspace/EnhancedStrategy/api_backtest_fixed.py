import requests
import json
import time
import os
import hmac
import hashlib
from datetime import datetime

# QuantConnect API credentials
user_id = "357130"
api_token = "400c99249479b8bca6e035d5817d85c01eafaea0a210b022e1d826196e3d4c0b"

# Base URL for QuantConnect API
base_url = "https://www.quantconnect.com/api/v2"

# Function to make API requests with proper authentication
def api_request(endpoint, method="GET", data=None):
    url = f"{base_url}/{endpoint}"
    
    # Create timestamp for authentication
    timestamp = int(time.time() * 1000)
    
    # Create hash for authentication
    message = f"{user_id}:{timestamp}:{endpoint}"
    signature = hmac.new(
        api_token.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Set up headers
    headers = {
        "Content-Type": "application/json",
        "X-Request-Timestamp": str(timestamp),
        "X-Api-Token": signature
    }
    
    # Make request
    if method == "GET":
        response = requests.get(url, headers=headers, params={"userId": user_id})
    elif method == "POST":
        response = requests.post(url, headers=headers, params={"userId": user_id}, json=data)
    
    return response.json()

# Read project files
with open("main.py", "r") as f:
    main_code = f.read()
    
with open("research_engine.py", "r") as f:
    research_code = f.read()

# Create project on QuantConnect
project_data = {
    "name": "EnhancedStrategy",
    "description": "Enhanced trading strategy with multi-timeframe scheduling",
    "language": "python"
}

print("Creating project on QuantConnect...")
project_response = api_request("projects/create", method="POST", data=project_data)
print(json.dumps(project_response, indent=2))

if not project_response.get("success"):
    print("Failed to create project:", project_response.get("errors"))
    exit(1)

project_id = project_response["projects"][0]["projectId"]
print(f"Created project with ID: {project_id}")

# Add main.py file
files_data = {
    "projectId": project_id,
    "name": "main.py",
    "content": main_code
}
print("Adding main.py file...")
main_response = api_request("files/create", method="POST", data=files_data)
print(json.dumps(main_response, indent=2))

if not main_response.get("success"):
    print("Failed to create main.py:", main_response.get("errors"))
    exit(1)

# Add research_engine.py file
files_data = {
    "projectId": project_id,
    "name": "research_engine.py",
    "content": research_code
}
print("Adding research_engine.py file...")
research_response = api_request("files/create", method="POST", data=files_data)
print(json.dumps(research_response, indent=2))

if not research_response.get("success"):
    print("Failed to create research_engine.py:", research_response.get("errors"))
    exit(1)

# Create backtest
backtest_data = {
    "projectId": project_id,
    "compileId": main_response["files"][0]["compileId"],
    "backtestName": "15-Year Backtest",
    "startDate": "2010-01-01T00:00:00Z",
    "endDate": "2025-01-01T23:59:59Z",
    "initialCash": 100000
}
print("Creating backtest...")
backtest_response = api_request("backtests/create", method="POST", data=backtest_data)
print(json.dumps(backtest_response, indent=2))

if not backtest_response.get("success"):
    print("Failed to create backtest:", backtest_response.get("errors"))
    exit(1)

backtest_id = backtest_response["backtests"][0]["backtestId"]
print(f"Created backtest with ID: {backtest_id}")

# Wait for backtest to complete
print("Waiting for backtest to complete...")
while True:
    status_response = api_request(f"backtests/read?projectId={project_id}&backtestId={backtest_id}")
    if not status_response.get("success"):
        print("Failed to get backtest status:", status_response.get("errors"))
        exit(1)
        
    status = status_response["backtests"][0]["status"]
    progress = status_response["backtests"][0]["progress"]
    print(f"Status: {status}, Progress: {progress}%")
    
    if status == "completed":
        break
        
    time.sleep(10)

# Get backtest results
results_response = api_request(f"backtests/read?projectId={project_id}&backtestId={backtest_id}")
if not results_response.get("success"):
    print("Failed to get backtest results:", results_response.get("errors"))
    exit(1)

# Extract performance metrics
results = results_response["backtests"][0]["results"]
statistics = results["statistics"]

print("\nBacktest Results:")
print(f"CAGR: {statistics.get('CompoundingAnnualReturn', 'N/A')}%")
print(f"Sharpe Ratio: {statistics.get('SharpeRatio', 'N/A')}")
print(f"Max Drawdown: {statistics.get('DrawdownPercent', 'N/A')}%")
print(f"Total Trades: {statistics.get('TotalNumberOfTrades', 'N/A')}")
print(f"Win Rate: {statistics.get('WinRate', 'N/A')}%")
print(f"Average Win: {statistics.get('AverageWinRate', 'N/A')}%")
print(f"Average Loss: {statistics.get('AverageLossRate', 'N/A')}%")

# Check if performance targets are met
cagr = float(statistics.get('CompoundingAnnualReturn', 0))
sharpe = float(statistics.get('SharpeRatio', 0))
max_drawdown = float(statistics.get('DrawdownPercent', 100))
avg_profit = float(statistics.get('AverageWinRate', 0))

print("\nPerformance Targets:")
print(f"CAGR > 25%: {'✅' if cagr > 25 else '❌'} ({cagr:.2f}%)")
print(f"Sharpe Ratio > 1: {'✅' if sharpe > 1 else '❌'} ({sharpe:.2f})")
print(f"Max Drawdown < 20%: {'✅' if max_drawdown < 20 else '❌'} ({max_drawdown:.2f}%)")
print(f"Average Profit > 0.75%: {'✅' if avg_profit > 0.75 else '❌'} ({avg_profit:.2f}%)")

# Save results to file
with open("backtest_results.json", "w") as f:
    json.dump(results, f, indent=2)
    
print("\nResults saved to backtest_results.json")
