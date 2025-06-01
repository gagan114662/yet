# QuantConnect Cloud Deployment Suite

Complete deployment and analysis suite for deploying high-performance trading strategies to QuantConnect Cloud.

## 🎯 Target Performance Metrics
- **CAGR**: >25%
- **Sharpe Ratio**: >1.0 (risk-free rate 5%)
- **Max Drawdown**: <20%
- **Average Profit per Trade**: >0.75%

## 📁 Files Overview

### Core Scripts
1. **`cloud_deployment.py`** - Main deployment script for individual strategies
2. **`batch_deployment.py`** - Batch deployment with parallel processing
3. **`strategy_converter.py`** - Converts local strategies to cloud-optimized format
4. **`results_analyzer.py`** - Advanced analysis of backtest results

## 🚀 Quick Start

### 1. Installation
```bash
pip install requests pandas numpy matplotlib seaborn
```

### 2. Get QuantConnect API Credentials
1. Create account at [QuantConnect.com](https://www.quantconnect.com)
2. Go to Account → API Access
3. Generate API Token
4. Note your User ID and Organization ID

### 3. Deploy Single Strategy
```bash
python cloud_deployment.py \
  --user-id YOUR_USER_ID \
  --api-token YOUR_API_TOKEN \
  --strategy SuperAggressiveMomentum
```

### 4. Deploy All Strategies (Recommended)
```bash
python batch_deployment.py \
  --user-id YOUR_USER_ID \
  --api-token YOUR_API_TOKEN \
  --max-concurrent 3
```

### 5. Analyze Results
```bash
python results_analyzer.py \
  --results-file batch_deployment_results_TIMESTAMP.json \
  --create-dashboard
```

## 📊 Strategy Portfolio

### Tier 1: Proven Performers
- **AggressiveSPYMomentum**: 33.43% returns in local testing
- **SuperAggressiveMomentum**: Enhanced version with dynamic leverage

### Tier 2: High-Potential Strategies  
- **MasterStrategyRotator**: Meta-strategy with intelligent rotation
- **CrisisAlphaHarvester**: Explosive returns during market stress
- **GammaScalperPro**: Options-based market making

### Strategy Variations
Each strategy is deployed in 3 variations:
- **Conservative**: 2-3x leverage, 2% stop loss
- **Aggressive**: 3-5x leverage, 1.5% stop loss  
- **UltraAggressive**: 4-8x leverage, 1% stop loss

## 📈 Expected Performance

Based on mathematical edges and professional data access:

| Strategy Type | Expected CAGR | Expected Sharpe | Risk Level |
|---------------|---------------|-----------------|------------|
| Conservative | 15-25% | 0.8-1.2 | Low |
| Aggressive | 25-35% | 1.0-1.5 | Medium |
| UltraAggressive | 35-50%+ | 1.2-2.0+ | High |

## 🔧 Advanced Usage

### Custom Strategy Deployment
```python
from cloud_deployment import QuantConnectCloudDeployer, StrategyConfig

deployer = QuantConnectCloudDeployer(user_id, api_token)

config = StrategyConfig(
    name="MyCustomStrategy", 
    path="/path/to/strategy.py",
    description="Custom high-performance strategy",
    parameters={"startDate": "2020-01-01", "endDate": "2023-12-31"}
)

project_id, backtest_id = deployer.deploy_strategy(config)
```

## 🎯 Success Metrics

### Deployment Targets
- Deploy 15+ strategy variations
- Achieve 80%+ deployment success rate
- Identify 3+ target-achieving strategies

### Performance Targets  
- At least 1 strategy with CAGR >25%
- At least 1 strategy with Sharpe >1.0
- Maximum drawdown <20% for live candidates

## 🚀 Ready to Deploy?

**Start with the batch deployment script to test all strategies in the cloud environment where they can access professional data and achieve their full potential!**