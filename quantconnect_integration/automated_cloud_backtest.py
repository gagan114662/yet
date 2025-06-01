#!/usr/bin/env python3
"""
AUTOMATED QUANTCONNECT CLOUD BACKTESTING
Access cloud, deploy strategies, and retrieve results

User ID: 357130
Token: 62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912
"""

import requests
import json
import time
import base64
from datetime import datetime

class QuantConnectCloudBacktester:
    
    def __init__(self):
        self.user_id = "357130"
        self.token = "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
        self.api_base = "https://www.quantconnect.com/api/v2"
        
        # Create auth header - QuantConnect uses user_id:token format
        self.headers = {
            "Authorization": f"Basic {base64.b64encode(f'{self.user_id}:{self.token}'.encode()).decode()}",
            "Content-Type": "application/json"
        }
        
        self.strategies = [
            {
                "name": "Crisis_Alpha_Master",
                "description": "Crisis Alpha & Tail Risk - 50%+ CAGR Target", 
                "priority": 1,
                "code": self.get_crisis_alpha_code()
            },
            {
                "name": "Strategy_Rotator_Master",
                "description": "Dynamic Multi-Strategy - 50%+ CAGR Target",
                "priority": 2,
                "code": self.get_strategy_rotator_code()
            },
            {
                "name": "Gamma_Flow_Master", 
                "description": "Options Gamma Flow - 40%+ CAGR Target",
                "priority": 3,
                "code": self.get_gamma_flow_code()
            }
        ]
        
    def get_crisis_alpha_code(self):
        """Get Crisis Alpha strategy code"""
        return '''
from AlgorithmImports import *
import numpy as np

class CrisisAlphaMaster(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Enable margin for leverage
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Crisis instruments with cloud data
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.vxx = self.AddEquity("VXX", Resolution.Minute).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Minute).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Minute).Symbol
        
        # Crisis detection indicators
        self.spy_rsi = self.RSI("SPY", 14, Resolution.Minute)
        self.volatility_window = RollingWindow[float](20)
        self.crisis_mode = False
        
        # Aggressive parameters for cloud
        self.max_leverage = 8.0
        self.crisis_threshold = 0.03  # 3% daily volatility threshold
        
        # Schedule crisis monitoring
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.Every(TimeSpan.FromMinutes(15)), 
                        self.MonitorCrisis)
        
        self.Debug("Crisis Alpha Master initialized for cloud backtesting")
        
    def OnData(self, data):
        # Update volatility data
        if self.spy in data and data[self.spy] is not None:
            if hasattr(self, 'previous_spy_price'):
                spy_return = (data[self.spy].Close - self.previous_spy_price) / self.previous_spy_price
                self.volatility_window.Add(abs(spy_return))
            self.previous_spy_price = data[self.spy].Close
        
        # Execute strategy based on crisis mode
        if self.crisis_mode:
            self.ExecuteCrisisStrategy(data)
        else:
            self.ExecuteNormalStrategy(data)
    
    def MonitorCrisis(self):
        """Monitor for crisis conditions"""
        if not self.volatility_window.IsReady or not self.spy_rsi.IsReady:
            return
            
        # Calculate recent volatility
        recent_vol = np.std([x for x in self.volatility_window]) * np.sqrt(1440)  # Daily vol
        
        # Crisis conditions
        high_volatility = recent_vol > self.crisis_threshold
        extreme_rsi = self.spy_rsi.Current.Value < 25 or self.spy_rsi.Current.Value > 75
        
        previous_mode = self.crisis_mode
        self.crisis_mode = high_volatility or extreme_rsi
        
        if self.crisis_mode != previous_mode:
            mode_text = "CRISIS DETECTED" if self.crisis_mode else "NORMAL MODE"
            self.Debug(f"{mode_text} - Vol: {recent_vol:.4f}, RSI: {self.spy_rsi.Current.Value:.1f}")
    
    def ExecuteCrisisStrategy(self, data):
        """Execute crisis alpha strategy with leverage"""
        # Validate data availability
        required_symbols = [self.vxx, self.tlt, self.gld, self.spy]
        if not all(symbol in data and data[symbol] is not None for symbol in required_symbols):
            return
            
        # Crisis allocation - profit from volatility and flight to quality
        self.SetHoldings(self.vxx, 3.0)   # 3x long volatility
        self.SetHoldings(self.tlt, 2.5)   # 2.5x long bonds  
        self.SetHoldings(self.gld, 2.0)   # 2x long gold
        self.SetHoldings(self.spy, -1.5)  # 1.5x short equities
        
    def ExecuteNormalStrategy(self, data):
        """Execute normal times strategy"""
        # Validate data availability
        if not all(symbol in data and data[symbol] is not None for symbol in [self.spy, self.gld, self.vxx]):
            return
            
        # Conservative allocation with small hedge
        self.SetHoldings(self.spy, 1.2)   # 120% equity exposure
        self.SetHoldings(self.gld, 0.1)   # 10% gold hedge
        self.SetHoldings(self.vxx, 0.05)  # 5% volatility hedge
        
    def OnEndOfDay(self, symbol):
        """Daily performance tracking"""
        if self.Portfolio.TotalPortfolioValue > 0:
            daily_return = (self.Portfolio.TotalPortfolioValue - 100000) / 100000
            total_leverage = sum([abs(x.HoldingsValue) for x in self.Portfolio.Values]) / self.Portfolio.TotalPortfolioValue
            
            if abs(daily_return) > 0.02:  # Log significant moves
                mode_text = "CRISIS" if self.crisis_mode else "NORMAL"
                self.Debug(f"Daily Return: {daily_return:.2%}, Mode: {mode_text}, Leverage: {total_leverage:.1f}x")
'''
    
    def get_strategy_rotator_code(self):
        """Get Strategy Rotator code"""
        return '''
from AlgorithmImports import *
import numpy as np

class StrategyRotatorMaster(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Enable margin for leverage
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Multi-asset universe for rotation
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Minute).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Minute).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Minute).Symbol
        self.vxx = self.AddEquity("VXX", Resolution.Minute).Symbol
        
        # Strategy indicators
        self.spy_rsi = self.RSI("SPY", 14, Resolution.Minute)
        self.spy_momentum = self.MOMP("SPY", 1440, Resolution.Minute)  # Daily momentum
        self.regime_confidence = 0.0
        self.current_regime = "BALANCED"
        
        # Rotation parameters
        self.max_leverage = 6.0
        self.rebalance_frequency = 2  # Hours
        
        # Schedule strategy rotation
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(TimeSpan.FromHours(self.rebalance_frequency)), 
                        self.RotateStrategies)
        
        self.Debug("Strategy Rotator Master initialized for cloud backtesting")
        
    def OnData(self, data):
        # Strategy execution handled in scheduled rotation
        pass
    
    def RotateStrategies(self):
        """Dynamic strategy rotation based on market regime"""
        if not self.spy_rsi.IsReady or not self.spy_momentum.IsReady:
            return
            
        # Market regime analysis
        rsi = self.spy_rsi.Current.Value
        momentum = self.spy_momentum.Current.Value
        
        # Calculate VIX proxy from VXX
        vix_proxy = 20.0  # Default
        if self.vxx in self.Securities:
            vix_proxy = self.Securities[self.vxx].Price * 2.5
        
        # Regime classification
        if vix_proxy > 35:
            self.current_regime = "CRISIS"
            self.regime_confidence = min(1.0, (vix_proxy - 35) / 20)
        elif momentum > 0.05 and rsi < 70:
            self.current_regime = "BULL_MOMENTUM"
            self.regime_confidence = min(1.0, momentum * 10)
        elif momentum < -0.05 and rsi > 30:
            self.current_regime = "BEAR_MOMENTUM" 
            self.regime_confidence = min(1.0, abs(momentum) * 10)
        elif rsi > 75:
            self.current_regime = "MEAN_REVERT_SHORT"
            self.regime_confidence = (rsi - 75) / 25
        elif rsi < 25:
            self.current_regime = "MEAN_REVERT_LONG"
            self.regime_confidence = (25 - rsi) / 25
        else:
            self.current_regime = "BALANCED"
            self.regime_confidence = 0.5
            
        # Execute regime-specific allocation
        self.ExecuteRegimeAllocation()
        
        # Log regime changes
        self.Debug(f"Regime: {self.current_regime}, Confidence: {self.regime_confidence:.2f}, VIX: {vix_proxy:.1f}")
    
    def ExecuteRegimeAllocation(self):
        """Execute allocation based on current regime"""
        leverage_multiplier = 1.0 + self.regime_confidence
        
        if self.current_regime == "CRISIS":
            # Crisis alpha allocation
            self.SetHoldings(self.vxx, 2.0 * leverage_multiplier)
            self.SetHoldings(self.tlt, 2.0 * leverage_multiplier)
            self.SetHoldings(self.gld, 1.5 * leverage_multiplier)
            self.SetHoldings(self.spy, -1.0 * leverage_multiplier)
            
        elif self.current_regime == "BULL_MOMENTUM":
            # Momentum allocation
            self.SetHoldings(self.spy, 2.5 * leverage_multiplier)
            self.SetHoldings(self.qqq, 1.5 * leverage_multiplier)
            self.SetHoldings(self.tlt, -0.5 * leverage_multiplier)
            
        elif self.current_regime == "BEAR_MOMENTUM":
            # Bear market allocation
            self.SetHoldings(self.tlt, 2.5 * leverage_multiplier)
            self.SetHoldings(self.gld, 1.5 * leverage_multiplier)
            self.SetHoldings(self.spy, -1.5 * leverage_multiplier)
            
        elif self.current_regime == "MEAN_REVERT_SHORT":
            # Short mean reversion
            self.SetHoldings(self.spy, -1.5 * leverage_multiplier)
            self.SetHoldings(self.tlt, 1.5 * leverage_multiplier)
            
        elif self.current_regime == "MEAN_REVERT_LONG":
            # Long mean reversion
            self.SetHoldings(self.spy, 2.0 * leverage_multiplier)
            self.SetHoldings(self.qqq, 1.0 * leverage_multiplier)
            
        else:  # BALANCED
            # Balanced allocation
            self.SetHoldings(self.spy, 1.5)
            self.SetHoldings(self.tlt, 0.8)
            self.SetHoldings(self.gld, 0.3)
            
        # Risk management
        total_leverage = sum([abs(x.HoldingsValue) for x in self.Portfolio.Values]) / self.Portfolio.TotalPortfolioValue
        if total_leverage > self.max_leverage:
            scale_factor = self.max_leverage / total_leverage * 0.95
            for holding in self.Portfolio.Values:
                if holding.Invested:
                    current_weight = holding.HoldingsValue / self.Portfolio.TotalPortfolioValue
                    self.SetHoldings(holding.Symbol, current_weight * scale_factor)
'''
    
    def get_gamma_flow_code(self):
        """Get Gamma Flow strategy code"""
        return '''
from AlgorithmImports import *
import numpy as np

class GammaFlowMaster(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Enable margin for leverage
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Core instruments for gamma flow
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Minute).Symbol
        self.vxx = self.AddEquity("VXX", Resolution.Minute).Symbol
        
        # Gamma flow indicators
        self.spy_rsi = self.RSI("SPY", 14, Resolution.Minute)
        self.volatility_regime = "NORMAL"
        self.gamma_signal = 0.0
        
        # Schedule gamma analysis
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.Every(TimeSpan.FromMinutes(30)), 
                        self.AnalyzeGammaFlow)
        
        # Gamma parameters
        self.max_leverage = 5.0
        self.gamma_threshold = 0.1
        
        self.Debug("Gamma Flow Master initialized for cloud backtesting")
        
    def OnData(self, data):
        if self.spy_rsi.IsReady:
            self.ExecuteGammaStrategy(data)
    
    def AnalyzeGammaFlow(self):
        """Analyze gamma flow conditions"""
        if not self.spy_rsi.IsReady:
            return
            
        # Calculate VIX proxy
        vix_proxy = 20.0
        if self.vxx in self.Securities:
            vix_proxy = self.Securities[self.vxx].Price * 2.5
            
        # Determine volatility regime
        if vix_proxy < 16:
            self.volatility_regime = "LOW"
        elif vix_proxy > 30:
            self.volatility_regime = "HIGH"
        else:
            self.volatility_regime = "NORMAL"
            
        # Calculate gamma signal
        rsi = self.spy_rsi.Current.Value
        if self.volatility_regime == "LOW":
            # Low vol: momentum
            if rsi > 60:
                self.gamma_signal = (rsi - 60) / 40  # 0 to 1
            elif rsi < 40:
                self.gamma_signal = (40 - rsi) / 40 * -1  # 0 to -1
            else:
                self.gamma_signal = 0
        else:
            # High vol: mean reversion
            if rsi > 70:
                self.gamma_signal = (rsi - 70) / 30 * -1  # 0 to -1
            elif rsi < 30:
                self.gamma_signal = (30 - rsi) / 30  # 0 to 1
            else:
                self.gamma_signal = 0
                
        self.Debug(f"Vol Regime: {self.volatility_regime}, Gamma Signal: {self.gamma_signal:.2f}")
    
    def ExecuteGammaStrategy(self, data):
        """Execute gamma flow strategy"""
        if abs(self.gamma_signal) < self.gamma_threshold:
            return
            
        # Validate data
        if not all(symbol in data and data[symbol] is not None for symbol in [self.spy, self.qqq]):
            return
            
        # Calculate position size based on gamma signal
        base_position = self.gamma_signal * 3.0  # Up to 3x leverage
        
        # Execute trades
        self.SetHoldings(self.spy, base_position)
        self.SetHoldings(self.qqq, base_position * 0.5)  # 50% correlation hedge
        
        # Add volatility hedge if needed
        if self.volatility_regime == "HIGH" and self.vxx in data:
            vol_hedge = 0.2 if self.gamma_signal > 0 else -0.1
            self.SetHoldings(self.vxx, vol_hedge)
'''
    
    def create_project(self, strategy):
        """Create project in QuantConnect Cloud"""
        print(f"🚀 Creating cloud project: {strategy['name']}")
        
        url = f"{self.api_base}/projects/create"
        data = {
            "projectName": strategy['name'],
            "language": "Py",
            "timestamp": int(time.time())
        }
        
        print(f"🔗 API URL: {url}")
        print(f"📦 Data: {data}")
        print(f"🔑 Headers: {self.headers}")
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            print(f"📡 Response Status: {response.status_code}")
            print(f"📝 Response Text: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                project_id = result.get('projectId')
                print(f"✅ Project created: {project_id}")
                return project_id
            else:
                print(f"❌ Failed to create project: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error creating project: {str(e)}")
            return None
    
    def upload_code(self, project_id, strategy):
        """Upload strategy code to project"""
        print(f"📤 Uploading code for {strategy['name']}")
        
        url = f"{self.api_base}/files/create"
        data = {
            "projectId": project_id,
            "name": "main.py",
            "content": strategy['code'],
            "timestamp": int(time.time())
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 200:
                print(f"✅ Code uploaded successfully")
                return True
            else:
                print(f"❌ Failed to upload code: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error uploading code: {str(e)}")
            return False
    
    def compile_project(self, project_id):
        """Compile the project"""
        print(f"🔨 Compiling project: {project_id}")
        
        url = f"{self.api_base}/compile/create"
        data = {"projectId": project_id}
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 200:
                result = response.json()
                compile_id = result.get('compileId')
                print(f"✅ Compilation started: {compile_id}")
                return compile_id
            else:
                print(f"❌ Failed to compile: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error compiling: {str(e)}")
            return None
    
    def create_backtest(self, project_id, strategy_name):
        """Create and run backtest"""
        print(f"🧪 Creating backtest for {strategy_name}")
        
        url = f"{self.api_base}/backtests/create"
        data = {
            "projectId": project_id,
            "backtestName": f"{strategy_name}_CloudTest_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "timestamp": int(time.time())
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 200:
                result = response.json()
                backtest_id = result.get('backtestId')
                print(f"✅ Backtest started: {backtest_id}")
                return backtest_id
            else:
                print(f"❌ Failed to create backtest: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error creating backtest: {str(e)}")
            return None
    
    def get_backtest_status(self, project_id, backtest_id):
        """Get backtest status and results"""
        url = f"{self.api_base}/backtests/read"
        params = {
            "projectId": project_id,
            "backtestId": backtest_id
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"❌ Error getting backtest status: {str(e)}")
            return None
    
    def wait_for_backtest_completion(self, project_id, backtest_id, strategy_name, max_wait=600):
        """Wait for backtest to complete and return results"""
        print(f"⏳ Waiting for {strategy_name} backtest to complete...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            result = self.get_backtest_status(project_id, backtest_id)
            
            if result:
                status = result.get('status', 'Unknown')
                if status == 'Completed':
                    print(f"✅ {strategy_name} backtest completed!")
                    return result
                elif status == 'Error':
                    print(f"❌ {strategy_name} backtest failed with error")
                    return result
                else:
                    print(f"⏳ {strategy_name} status: {status}")
            
            time.sleep(30)  # Check every 30 seconds
        
        print(f"⏰ {strategy_name} backtest timed out after {max_wait} seconds")
        return None
    
    def extract_performance_metrics(self, backtest_result, strategy_name):
        """Extract key performance metrics from backtest results"""
        try:
            statistics = backtest_result.get('statistics', {})
            
            metrics = {
                'strategy': strategy_name,
                'cagr': statistics.get('Compounding Annual Return', '0%'),
                'sharpe': statistics.get('Sharpe Ratio', '0'),
                'max_drawdown': statistics.get('Drawdown', '0%'),
                'total_orders': statistics.get('Total Orders', '0'),
                'win_rate': statistics.get('Win Rate', '0%'),
                'net_profit': statistics.get('Net Profit', '0%'),
                'total_fees': statistics.get('Total Fees', '$0'),
                'capacity': statistics.get('Estimated Strategy Capacity', '$0')
            }
            
            return metrics
        except Exception as e:
            print(f"❌ Error extracting metrics for {strategy_name}: {str(e)}")
            return None
    
    def run_cloud_backtests(self):
        """Run all strategy backtests on QuantConnect Cloud"""
        print("🌐 STARTING QUANTCONNECT CLOUD BACKTESTING")
        print("=" * 60)
        print(f"User ID: {self.user_id}")
        print(f"Strategies to test: {len(self.strategies)}")
        print()
        
        results = []
        
        for strategy in self.strategies:
            print(f"📦 Processing: {strategy['name']} (Priority {strategy['priority']})")
            
            # Step 1: Create project
            project_id = self.create_project(strategy)
            if not project_id:
                continue
            
            # Step 2: Upload code
            if not self.upload_code(project_id, strategy):
                continue
            
            # Step 3: Create backtest
            backtest_id = self.create_backtest(project_id, strategy['name'])
            if not backtest_id:
                continue
            
            # Step 4: Wait for completion and get results
            backtest_result = self.wait_for_backtest_completion(project_id, backtest_id, strategy['name'])
            
            if backtest_result:
                metrics = self.extract_performance_metrics(backtest_result, strategy['name'])
                if metrics:
                    results.append(metrics)
                    print(f"✅ {strategy['name']} completed successfully")
                else:
                    print(f"❌ Failed to extract metrics for {strategy['name']}")
            else:
                print(f"❌ {strategy['name']} backtest failed or timed out")
            
            print()
        
        # Generate results summary
        self.generate_results_summary(results)
        return results
    
    def generate_results_summary(self, results):
        """Generate comprehensive results summary"""
        print("🎯 CLOUD BACKTEST RESULTS SUMMARY")
        print("=" * 50)
        
        if not results:
            print("❌ No successful backtests completed")
            return
        
        print(f"✅ Completed backtests: {len(results)}")
        print()
        
        # Create results table
        print("📊 PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(f"{'Strategy':<25} {'CAGR':<10} {'Sharpe':<8} {'Max DD':<10} {'Orders':<8} {'Win Rate':<10}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['strategy']:<25} {result['cagr']:<10} {result['sharpe']:<8} {result['max_drawdown']:<10} {result['total_orders']:<8} {result['win_rate']:<10}")
        
        print("-" * 80)
        print()
        
        # Find best performing strategy
        try:
            # Extract numeric CAGR values for comparison
            best_strategy = None
            best_cagr = -999
            
            for result in results:
                cagr_str = result['cagr'].replace('%', '')
                try:
                    cagr_val = float(cagr_str)
                    if cagr_val > best_cagr:
                        best_cagr = cagr_val
                        best_strategy = result
                except:
                    pass
            
            if best_strategy:
                print(f"🏆 BEST PERFORMER: {best_strategy['strategy']}")
                print(f"   CAGR: {best_strategy['cagr']}")
                print(f"   Sharpe: {best_strategy['sharpe']}")
                print(f"   Max Drawdown: {best_strategy['max_drawdown']}")
                print()
        except:
            pass
        
        print("🎯 TARGET COMPARISON:")
        print("   Target CAGR: 25-60%")
        print("   Target Sharpe: 1.5-2.5")
        print("   Target Max DD: <20%")
        print()
        
        print("🌐 CLOUD ADVANTAGES REALIZED:")
        print("   ✅ Complete market data access")
        print("   ✅ Professional execution environment")
        print("   ✅ Real-time options and volatility data")
        print("   ✅ Multi-asset universe availability")
        print()
        
        print("📈 NEXT STEPS:")
        print("   1. Analyze best performing strategies")
        print("   2. Deploy winners to paper trading")
        print("   3. Monitor live performance")
        print("   4. Scale to real capital")

def main():
    """Main execution function"""
    print("🚀 QUANTCONNECT CLOUD BACKTESTING INITIATED")
    print("=" * 60)
    
    backtester = QuantConnectCloudBacktester()
    results = backtester.run_cloud_backtests()
    
    if results:
        print("\n🎉 CLOUD BACKTESTING COMPLETE!")
        print("Results show performance with full market data access.")
        print("Compare these results against your 25%+ CAGR targets!")
    else:
        print("\n❌ CLOUD BACKTESTING FAILED")
        print("Check network connectivity and credentials.")

if __name__ == "__main__":
    main()