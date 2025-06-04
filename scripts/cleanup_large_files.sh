#!/bin/bash
# Automated cleanup for large backtest files

echo "🧹 Running automated large file cleanup..."

# Remove files older than 30 days
find . -name "*RESULTS*.json" -mtime +30 -delete
find . -path "*/backtests/*" -name "*.json" -mtime +30 -delete
find . -name "*.log" -size +100M -mtime +7 -delete

# Compress old evolution checkpoints
find . -name "*checkpoint*.json" -mtime +7 -exec gzip {} \;

echo "✅ Cleanup completed"
