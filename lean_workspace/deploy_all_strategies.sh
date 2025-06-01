#!/bin/bash
# Deploy all quantitative strategies to QuantConnect

echo "Deploying all professional quant strategies..."

strategies=(
    "volatility_harvester"
    "statistical_arbitrage" 
    "risk_parity_lever"
    "regime_adaptive_master"
    "multi_factor_alpha"
)

for strategy in "${strategies[@]}"; do
    echo "Pushing $strategy..."
    lean cloud push --project "$strategy"
    echo "Running backtest for $strategy..."
    lean cloud backtest "$strategy" --name "${strategy}_20yr_test"
    echo "---"
done

echo "All strategies deployed!"