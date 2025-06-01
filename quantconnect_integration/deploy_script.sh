#!/bin/bash

# QuantConnect Cloud Deployment Script
# Automates the complete deployment process

set -e  # Exit on any error

echo "🚀 QuantConnect Cloud Deployment Suite"
echo "======================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if credentials are provided
if [ -z "$QC_USER_ID" ] || [ -z "$QC_API_TOKEN" ]; then
    echo -e "${RED}Error: Please set environment variables:${NC}"
    echo "export QC_USER_ID='your_user_id'"
    echo "export QC_API_TOKEN='your_api_token'"
    echo ""
    echo "Get these from: https://www.quantconnect.com/account"
    exit 1
fi

# Default parameters
MAX_CONCURRENT=${MAX_CONCURRENT:-3}
PRIORITY_ONLY=${PRIORITY_ONLY:-false}
CREATE_DASHBOARD=${CREATE_DASHBOARD:-true}

echo -e "${BLUE}Configuration:${NC}"
echo "User ID: $QC_USER_ID"
echo "Max Concurrent: $MAX_CONCURRENT"
echo "Priority Only: $PRIORITY_ONLY"
echo "Create Dashboard: $CREATE_DASHBOARD"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import requests, pandas, numpy, matplotlib, seaborn" 2>/dev/null || {
    echo -e "${RED}Missing dependencies. Installing...${NC}"
    pip install requests pandas numpy matplotlib seaborn
}
echo -e "${GREEN}✓ Dependencies ready${NC}"

# Create output directory
OUTPUT_DIR="deployment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Created output directory: $OUTPUT_DIR${NC}"

# Step 1: Batch Deployment
echo ""
echo -e "${BLUE}Step 1: Deploying strategies to QuantConnect Cloud...${NC}"

DEPLOY_ARGS="--user-id $QC_USER_ID --api-token $QC_API_TOKEN --max-concurrent $MAX_CONCURRENT"

if [ "$PRIORITY_ONLY" = "true" ]; then
    DEPLOY_ARGS="$DEPLOY_ARGS --priority-only"
fi

python3 batch_deployment.py $DEPLOY_ARGS

# Find the latest results file
RESULTS_FILE=$(ls -t batch_deployment_results_*.json 2>/dev/null | head -1)

if [ -z "$RESULTS_FILE" ]; then
    echo -e "${RED}Error: No results file found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Deployment completed: $RESULTS_FILE${NC}"

# Step 2: Move results to output directory
mv "$RESULTS_FILE" "$OUTPUT_DIR/"
mv deployment_report_*.md "$OUTPUT_DIR/" 2>/dev/null || true

RESULTS_FILE="$OUTPUT_DIR/$(basename $RESULTS_FILE)"

# Step 3: Analyze Results
echo ""
echo -e "${BLUE}Step 2: Analyzing results...${NC}"

ANALYSIS_ARGS="--results-file $RESULTS_FILE --output-dir $OUTPUT_DIR"

if [ "$CREATE_DASHBOARD" = "true" ]; then
    ANALYSIS_ARGS="$ANALYSIS_ARGS --create-dashboard"
fi

python3 results_analyzer.py $ANALYSIS_ARGS

echo -e "${GREEN}✓ Analysis completed${NC}"

# Step 4: Summary Report
echo ""
echo -e "${BLUE}Step 3: Generating summary...${NC}"

# Extract key metrics from results
python3 -c "
import json
import sys

try:
    with open('$RESULTS_FILE', 'r') as f:
        data = json.load(f)
    
    batch_results = data.get('batch_results', {})
    individual_results = data.get('individual_results', [])
    
    total = batch_results.get('total_strategies', 0)
    successful = batch_results.get('successful_deployments', 0)
    target_achievers = batch_results.get('target_achievers', [])
    
    print(f'Total Strategies: {total}')
    print(f'Successful Deployments: {successful}')
    print(f'Success Rate: {successful/total*100:.1f}%' if total > 0 else 'Success Rate: 0%')
    print(f'Target Achievers: {len(target_achievers)}')
    
    if target_achievers:
        print('\n🎉 TOP TARGET ACHIEVERS:')
        for i, achiever in enumerate(target_achievers[:3], 1):
            name = achiever.get('strategy_name', 'Unknown')
            cagr = achiever.get('cagr', 0) * 100
            sharpe = achiever.get('sharpe_ratio', 0)
            max_dd = abs(achiever.get('max_drawdown', 0)) * 100
            print(f'{i}. {name}')
            print(f'   CAGR: {cagr:.1f}%, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.1f}%')
    else:
        print('\n⚠️ No strategies met all targets')
        
        # Show best performers
        if individual_results:
            best_cagr = max(individual_results, key=lambda x: x.get('cagr', 0))
            best_sharpe = max(individual_results, key=lambda x: x.get('sharpe_ratio', 0))
            
            print('\nBest CAGR:', best_cagr.get('strategy_name', 'Unknown'), f\"({best_cagr.get('cagr', 0)*100:.1f}%)\")
            print('Best Sharpe:', best_sharpe.get('strategy_name', 'Unknown'), f\"({best_sharpe.get('sharpe_ratio', 0):.2f})\")

except Exception as e:
    print(f'Error reading results: {e}')
    sys.exit(1)
"

# Step 5: Create deployment package
echo ""
echo -e "${BLUE}Step 4: Creating deployment package...${NC}"

# Copy scripts to output directory
cp *.py "$OUTPUT_DIR/"
cp README.md "$OUTPUT_DIR/"

# Create summary file
cat > "$OUTPUT_DIR/DEPLOYMENT_SUMMARY.md" << EOF
# Deployment Summary

**Date**: $(date)
**Output Directory**: $OUTPUT_DIR

## Files Generated
- \`$(basename $RESULTS_FILE)\` - Raw deployment results
- \`deployment_report_*.md\` - Detailed analysis report  
- \`analysis_report_*.md\` - Comprehensive strategy analysis
- \`performance_dashboard.png\` - Visual performance dashboard (if enabled)

## Quick Start Commands
\`\`\`bash
# View detailed report
cat deployment_report_*.md

# Analyze specific strategy
python3 results_analyzer.py --results-file $(basename $RESULTS_FILE)

# Deploy additional strategies  
python3 cloud_deployment.py --user-id $QC_USER_ID --api-token [TOKEN] --strategy [NAME]
\`\`\`

## Next Steps
1. Review target-achieving strategies in the report
2. Deploy top performers to live trading
3. Implement risk management protocols
4. Monitor performance closely

---
Generated by QuantConnect Cloud Deployment Suite
EOF

echo -e "${GREEN}✓ Deployment package created in: $OUTPUT_DIR${NC}"

# Final output
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Output Location:${NC} $OUTPUT_DIR"
echo -e "${BLUE}Results File:${NC} $(basename $RESULTS_FILE)"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Review the detailed report: $OUTPUT_DIR/deployment_report_*.md"
if [ "$CREATE_DASHBOARD" = "true" ]; then
echo "2. Check the performance dashboard: $OUTPUT_DIR/performance_dashboard.png"
fi
echo "3. Identify strategies meeting your targets"
echo "4. Deploy top performers to live trading"
echo ""
echo -e "${BLUE}For help:${NC} cat $OUTPUT_DIR/README.md"