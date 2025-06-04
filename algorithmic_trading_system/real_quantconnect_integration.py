#!/usr/bin/env python3
"""
Real QuantConnect Integration - Replace mock backtesting with actual market data
Integrates with QuantConnect Cloud API for genuine strategy backtesting
"""

import os
import time
import json
import requests
import asyncio
import tempfile
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class QuantConnectBacktester:
    """
    Real QuantConnect integration for actual strategy backtesting
    """
    
    def __init__(self, user_id: str = None, api_token: str = None):
        # Set credentials
        self.user_id = user_id or "357130"
        self.api_token = api_token or "62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912"
        
        # QuantConnect API endpoints
        self.base_url = "https://www.quantconnect.com/api/v2"
        
        # Store for hash-based auth
        self.base_headers = {
            "Content-Type": "application/json"
        }
        
        # Project management
        self.project_cache = {}
        self.backtest_cache = {}
        
        logger.info(f"🔗 QuantConnect integration initialized for user {self.user_id}")
    
    def create_auth_headers(self, timestamp: str = None) -> Dict[str, str]:
        """Create QuantConnect authentication headers with proper hash"""
        if timestamp is None:
            timestamp = str(int(time.time()))
        
        # QuantConnect Basic Auth format
        auth_string = f"{self.user_id}:{self.api_token}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        headers = self.base_headers.copy()
        headers.update({
            "Timestamp": timestamp,
            "Authorization": f"Basic {encoded_auth}"
        })
        
        return headers
    
    def strategy_to_lean_algorithm(self, strategy: Dict) -> str:
        """
        Convert strategy dictionary to Lean algorithm code
        """
        strategy_type = strategy.get('type', 'momentum')
        leverage = strategy.get('leverage', 1.0)
        position_size = strategy.get('position_size', 0.1)
        stop_loss = strategy.get('stop_loss', 0.1)
        indicators = strategy.get('indicators', ['RSI', 'MACD'])
        
        # Generate algorithm based on strategy type
        if strategy_type == 'momentum':
            algorithm = self._generate_momentum_algorithm(strategy)
        elif strategy_type == 'mean_reversion':
            algorithm = self._generate_mean_reversion_algorithm(strategy)
        elif strategy_type == 'trend_following':
            algorithm = self._generate_trend_following_algorithm(strategy)
        elif strategy_type == 'breakout':
            algorithm = self._generate_breakout_algorithm(strategy)
        else:
            # Default momentum strategy
            algorithm = self._generate_momentum_algorithm(strategy)
        
        return algorithm
    
    def _generate_momentum_algorithm(self, strategy: Dict) -> str:
        """Generate momentum-based Lean algorithm"""
        leverage = strategy.get('leverage', 1.0)
        position_size = strategy.get('position_size', 0.1)
        stop_loss = strategy.get('stop_loss', 0.1)
        rsi_period = strategy.get('rsi_period', 14)
        
        algorithm_code = f'''
from AlgorithmImports import *

class EvolutionMomentumStrategy(QCAlgorithm):
    
    def Initialize(self):
        # Set strategy parameters from evolution
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Strategy parameters
        self.leverage = {leverage}
        self.position_size = {position_size}
        self.stop_loss_pct = {stop_loss}
        
        # Add equity
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Technical indicators
        self.rsi = self.RSI(self.spy, {rsi_period}, MovingAverageType.Simple, Resolution.Daily)
        self.macd = self.MACD(self.spy, 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
        
        # Risk management
        self.stop_loss_price = None
        
        # Set leverage
        self.Securities[self.spy].SetLeverage({leverage})
        
        # Schedule function
        self.Schedule.On(self.DateRules.EveryDay(self.spy), 
                        self.TimeRules.AfterMarketOpen(self.spy, 30), 
                        self.Rebalance)
    
    def Rebalance(self):
        if not self.rsi.IsReady or not self.macd.IsReady:
            return
        
        current_price = self.Securities[self.spy].Price
        
        # Momentum signals
        rsi_signal = self.rsi.Current.Value > 50  # Above neutral
        macd_signal = self.macd.Current.Value > self.macd.Signal.Current.Value  # MACD above signal
        
        # Position management
        holdings = self.Portfolio[self.spy].Quantity
        
        if rsi_signal and macd_signal and holdings == 0:
            # Enter long position
            quantity = int(self.Portfolio.TotalPortfolioValue * self.position_size / current_price)
            if quantity > 0:
                self.MarketOrder(self.spy, quantity)
                self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                self.Debug(f"Entered long: {{quantity}} shares at ${{current_price}}")
        
        elif holdings > 0:
            # Check stop loss
            if current_price <= self.stop_loss_price:
                self.Liquidate(self.spy)
                self.Debug(f"Stop loss triggered at ${{current_price}}")
            
            # Check exit signal
            elif not rsi_signal or not macd_signal:
                self.Liquidate(self.spy)
                self.Debug(f"Exit signal at ${{current_price}}")
    
    def OnData(self, data):
        pass
'''
        return algorithm_code
    
    def _generate_mean_reversion_algorithm(self, strategy: Dict) -> str:
        """Generate mean reversion Lean algorithm"""
        leverage = strategy.get('leverage', 1.0)
        position_size = strategy.get('position_size', 0.1)
        bb_period = strategy.get('bb_period', 20)
        
        algorithm_code = f'''
from AlgorithmImports import *

class EvolutionMeanReversionStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Strategy parameters
        self.leverage = {leverage}
        self.position_size = {position_size}
        
        # Add equity
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Bollinger Bands for mean reversion
        self.bb = self.BB(self.spy, {bb_period}, 2.0, MovingAverageType.Simple, Resolution.Daily)
        self.rsi = self.RSI(self.spy, 14, MovingAverageType.Simple, Resolution.Daily)
        
        # Set leverage
        self.Securities[self.spy].SetLeverage({leverage})
        
        self.Schedule.On(self.DateRules.EveryDay(self.spy), 
                        self.TimeRules.AfterMarketOpen(self.spy, 30), 
                        self.Rebalance)
    
    def Rebalance(self):
        if not self.bb.IsReady or not self.rsi.IsReady:
            return
        
        current_price = self.Securities[self.spy].Price
        holdings = self.Portfolio[self.spy].Quantity
        
        # Mean reversion signals
        below_lower_band = current_price < self.bb.LowerBand.Current.Value
        above_upper_band = current_price > self.bb.UpperBand.Current.Value
        oversold = self.rsi.Current.Value < 30
        overbought = self.rsi.Current.Value > 70
        
        if below_lower_band and oversold and holdings == 0:
            # Buy when oversold and below lower band
            quantity = int(self.Portfolio.TotalPortfolioValue * self.position_size / current_price)
            if quantity > 0:
                self.MarketOrder(self.spy, quantity)
        
        elif holdings > 0 and (above_upper_band or overbought):
            # Sell when overbought or above upper band
            self.Liquidate(self.spy)
    
    def OnData(self, data):
        pass
'''
        return algorithm_code
    
    def _generate_trend_following_algorithm(self, strategy: Dict) -> str:
        """Generate trend following Lean algorithm"""
        leverage = strategy.get('leverage', 1.0)
        position_size = strategy.get('position_size', 0.1)
        ema_period = strategy.get('ema_period', 21)
        
        algorithm_code = f'''
from AlgorithmImports import *

class EvolutionTrendFollowingStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Strategy parameters
        self.leverage = {leverage}
        self.position_size = {position_size}
        
        # Add equity
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Moving averages for trend following
        self.fast_ema = self.EMA(self.spy, 12, Resolution.Daily)
        self.slow_ema = self.EMA(self.spy, {ema_period}, Resolution.Daily)
        self.adx = self.ADX(self.spy, 14, Resolution.Daily)
        
        # Set leverage
        self.Securities[self.spy].SetLeverage({leverage})
        
        self.Schedule.On(self.DateRules.EveryDay(self.spy), 
                        self.TimeRules.AfterMarketOpen(self.spy, 30), 
                        self.Rebalance)
    
    def Rebalance(self):
        if not self.fast_ema.IsReady or not self.slow_ema.IsReady or not self.adx.IsReady:
            return
        
        current_price = self.Securities[self.spy].Price
        holdings = self.Portfolio[self.spy].Quantity
        
        # Trend following signals
        trend_up = self.fast_ema.Current.Value > self.slow_ema.Current.Value
        strong_trend = self.adx.Current.Value > 25
        
        if trend_up and strong_trend and holdings == 0:
            # Enter long in uptrend
            quantity = int(self.Portfolio.TotalPortfolioValue * self.position_size / current_price)
            if quantity > 0:
                self.MarketOrder(self.spy, quantity)
        
        elif holdings > 0 and not trend_up:
            # Exit when trend reverses
            self.Liquidate(self.spy)
    
    def OnData(self, data):
        pass
'''
        return algorithm_code
    
    def _generate_breakout_algorithm(self, strategy: Dict) -> str:
        """Generate breakout strategy Lean algorithm"""
        leverage = strategy.get('leverage', 1.0)
        position_size = strategy.get('position_size', 0.1)
        atr_period = strategy.get('atr_period', 14)
        
        algorithm_code = f'''
from AlgorithmImports import *

class EvolutionBreakoutStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Strategy parameters
        self.leverage = {leverage}
        self.position_size = {position_size}
        
        # Add equity
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Breakout indicators
        self.atr = self.ATR(self.spy, {atr_period}, MovingAverageType.Simple, Resolution.Daily)
        self.high_max = self.MAX(self.spy, 20, Resolution.Daily, Field.High)
        self.low_min = self.MIN(self.spy, 20, Resolution.Daily, Field.Low)
        
        # Set leverage
        self.Securities[self.spy].SetLeverage({leverage})
        
        self.Schedule.On(self.DateRules.EveryDay(self.spy), 
                        self.TimeRules.AfterMarketOpen(self.spy, 30), 
                        self.Rebalance)
    
    def Rebalance(self):
        if not self.atr.IsReady or not self.high_max.IsReady or not self.low_min.IsReady:
            return
        
        current_price = self.Securities[self.spy].Price
        holdings = self.Portfolio[self.spy].Quantity
        
        # Breakout signals
        breakout_level = self.high_max.Current.Value
        breakdown_level = self.low_min.Current.Value
        volatility = self.atr.Current.Value
        
        if current_price > breakout_level and holdings == 0 and volatility > 1.0:
            # Enter on upside breakout
            quantity = int(self.Portfolio.TotalPortfolioValue * self.position_size / current_price)
            if quantity > 0:
                self.MarketOrder(self.spy, quantity)
        
        elif holdings > 0 and current_price < breakdown_level:
            # Exit on breakdown
            self.Liquidate(self.spy)
    
    def OnData(self, data):
        pass
'''
        return algorithm_code
    
    async def create_project(self, strategy: Dict) -> str:
        """Create a new project in QuantConnect for the strategy"""
        strategy_id = strategy['id']
        
        if strategy_id in self.project_cache:
            return self.project_cache[strategy_id]
        
        project_name = f"Evolution_{strategy_id}_{int(time.time())}"
        
        # Create project
        create_url = f"{self.base_url}/projects/create"
        project_data = {
            "projectName": project_name,
            "language": "Py"  # QuantConnect uses "Py" not "Python"
        }
        
        try:
            # Create proper authentication headers
            auth_headers = self.create_auth_headers()
            
            response = requests.post(create_url, headers=auth_headers, json=project_data)
            
            # Debug the response
            logger.info(f"🔍 API Response: {response.status_code}")
            logger.info(f"🔍 Response headers: {dict(response.headers)}")
            logger.info(f"🔍 Response text: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                project_id = result.get('projectId')
                
                if project_id:
                    self.project_cache[strategy_id] = project_id
                    logger.info(f"✅ Created project {project_id} for strategy {strategy_id}")
                    
                    # Upload algorithm code
                    algorithm_code = self.strategy_to_lean_algorithm(strategy)
                    await self.upload_algorithm(project_id, algorithm_code)
                    
                    return project_id
                else:
                    logger.error(f"❌ Failed to get project ID from response: {result}")
                    return None
            else:
                logger.error(f"❌ Failed to create project: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception creating project: {e}")
            return None
    
    async def upload_algorithm(self, project_id: str, algorithm_code: str) -> bool:
        """Upload algorithm code to the project"""
        try:
            # Update main.py file
            update_url = f"{self.base_url}/files/update"
            file_data = {
                "projectId": project_id,
                "name": "main.py",
                "content": algorithm_code
            }
            
            response = requests.post(update_url, headers=self.create_auth_headers(), json=file_data)
            
            if response.status_code == 200:
                logger.info(f"✅ Uploaded algorithm to project {project_id}")
                return True
            else:
                logger.error(f"❌ Failed to upload algorithm: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception uploading algorithm: {e}")
            return False
    
    async def run_backtest(self, strategy: Dict) -> Dict:
        """
        Run actual backtest on QuantConnect and return real performance metrics
        """
        strategy_id = strategy['id']
        
        logger.info(f"🚀 Starting real backtest for strategy: {strategy_id}")
        
        # Create project
        project_id = await self.create_project(strategy)
        if not project_id:
            return {'error': 'Failed to create project', 'strategy_id': strategy_id}
        
        # Compile project first
        compile_success = await self.compile_project(project_id)
        if not compile_success:
            return {'error': 'Failed to compile project', 'strategy_id': strategy_id}
        
        # Run backtest
        backtest_id = await self.start_backtest(project_id, strategy)
        if not backtest_id:
            return {'error': 'Failed to start backtest', 'strategy_id': strategy_id}
        
        # Wait for completion and get results
        results = await self.wait_for_backtest_completion(backtest_id, strategy_id)
        
        return results
    
    async def compile_project(self, project_id: str) -> bool:
        """Compile the project before backtesting"""
        try:
            compile_url = f"{self.base_url}/compile/create"
            compile_data = {
                "projectId": project_id
            }
            
            response = requests.post(compile_url, headers=self.create_auth_headers(), json=compile_data)
            
            if response.status_code == 200:
                result = response.json()
                compile_id = result.get('compileId')
                
                if compile_id:
                    # Wait for compilation to complete
                    for _ in range(30):  # Wait up to 30 seconds
                        await asyncio.sleep(1)
                        
                        status_url = f"{self.base_url}/compile/read"
                        status_data = {"projectId": project_id, "compileId": compile_id}
                        status_response = requests.post(status_url, headers=self.create_auth_headers(), json=status_data)
                        
                        if status_response.status_code == 200:
                            status_result = status_response.json()
                            state = status_result.get('state', '')
                            
                            if state == 'BuildSuccess':
                                logger.info(f"✅ Project {project_id} compiled successfully")
                                return True
                            elif state == 'BuildError':
                                errors = status_result.get('logs', [])
                                logger.error(f"❌ Compilation failed: {errors}")
                                return False
                    
                    logger.error(f"❌ Compilation timeout for project {project_id}")
                    return False
                else:
                    logger.error(f"❌ No compile ID returned for project {project_id}")
                    return False
            else:
                logger.error(f"❌ Failed to start compilation: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception during compilation: {e}")
            return False
    
    async def start_backtest(self, project_id: str, strategy: Dict) -> Optional[str]:
        """Start a backtest for the project"""
        try:
            backtest_url = f"{self.base_url}/backtests/create"
            backtest_data = {
                "projectId": project_id,
                "backtestName": f"Evolution_Test_{int(time.time())}"
            }
            
            response = requests.post(backtest_url, headers=self.create_auth_headers(), json=backtest_data)
            
            if response.status_code == 200:
                result = response.json()
                backtest_id = result.get('backtestId')
                
                if backtest_id:
                    logger.info(f"✅ Started backtest {backtest_id} for project {project_id}")
                    return backtest_id
                else:
                    logger.error(f"❌ No backtest ID returned")
                    return None
            else:
                logger.error(f"❌ Failed to start backtest: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception starting backtest: {e}")
            return None
    
    async def wait_for_backtest_completion(self, backtest_id: str, strategy_id: str, timeout: int = 300) -> Dict:
        """Wait for backtest to complete and return results"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check backtest status
                status_url = f"{self.base_url}/backtests/read"
                status_data = {"backtestId": backtest_id}
                
                response = requests.post(status_url, headers=self.create_auth_headers(), json=status_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if completed
                    if result.get('completed', False):
                        logger.info(f"✅ Backtest {backtest_id} completed")
                        return self.parse_backtest_results(result, strategy_id)
                    
                    # Check progress
                    progress = result.get('progress', 0)
                    if progress > 0:
                        logger.info(f"🔄 Backtest {backtest_id} progress: {progress:.1%}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Exception checking backtest status: {e}")
                await asyncio.sleep(5)
        
        logger.error(f"❌ Backtest {backtest_id} timeout after {timeout} seconds")
        return {'error': 'Backtest timeout', 'strategy_id': strategy_id}
    
    def parse_backtest_results(self, backtest_result: Dict, strategy_id: str) -> Dict:
        """Parse QuantConnect backtest results into standard format"""
        try:
            # Extract performance statistics
            statistics = backtest_result.get('statistics', {})
            
            # Key metrics
            total_return = float(statistics.get('Total Return', '0').strip('%')) / 100
            annual_return = float(statistics.get('Annual Return', '0').strip('%')) / 100
            sharpe_ratio = float(statistics.get('Sharpe Ratio', '0'))
            max_drawdown = float(statistics.get('Max Drawdown', '0').strip('%')) / 100
            
            # Additional metrics
            win_rate = float(statistics.get('Win Rate', '0').strip('%')) / 100
            profit_loss_ratio = float(statistics.get('Profit-Loss Ratio', '0'))
            total_trades = int(statistics.get('Total Trades', '0'))
            
            # Calculate CAGR if not directly available
            if annual_return == 0 and total_return > 0:
                # Estimate CAGR from total return (assuming ~4 year backtest)
                years = 4.0
                cagr = (1 + total_return) ** (1/years) - 1
            else:
                cagr = annual_return
            
            # Parse additional data
            equity_curve = backtest_result.get('charts', {}).get('Strategy Equity', {}).get('Series', {})
            
            parsed_results = {
                'strategy_id': strategy_id,
                'cagr': cagr,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),  # Ensure positive
                'total_return': total_return,
                'win_rate': win_rate,
                'profit_factor': profit_loss_ratio,
                'total_trades': total_trades,
                'backtest_id': backtest_result.get('backtestId'),
                'start_date': backtest_result.get('periodStart'),
                'end_date': backtest_result.get('periodFinish'),
                'backtest_completed': True,
                'raw_statistics': statistics
            }
            
            logger.info(f"📊 Real backtest results for {strategy_id}:")
            logger.info(f"   CAGR: {cagr:.1%}")
            logger.info(f"   Sharpe: {sharpe_ratio:.2f}")
            logger.info(f"   Max DD: {abs(max_drawdown):.1%}")
            logger.info(f"   Total Trades: {total_trades}")
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"❌ Error parsing backtest results: {e}")
            return {
                'error': f'Failed to parse results: {e}',
                'strategy_id': strategy_id,
                'raw_result': backtest_result
            }

# Integration function to replace mock backtesting
async def real_backtest_strategy(strategy: Dict) -> Dict:
    """
    Replace the mock simulate_detailed_backtest with real QuantConnect backtesting
    """
    backtester = QuantConnectBacktester()
    
    try:
        results = await backtester.run_backtest(strategy)
        return results
    except Exception as e:
        logger.error(f"❌ Real backtest failed for {strategy.get('id', 'unknown')}: {e}")
        return {
            'error': f'Real backtest failed: {e}',
            'strategy_id': strategy.get('id', 'unknown')
        }

# Test function
async def test_real_backtesting():
    """Test real QuantConnect backtesting with a sample strategy"""
    print("🧪 TESTING REAL QUANTCONNECT INTEGRATION")
    print("=" * 60)
    
    # Test strategy
    test_strategy = {
        'id': 'test_momentum_strategy',
        'name': 'Test Momentum Strategy',
        'type': 'momentum',
        'leverage': 2.0,
        'position_size': 0.2,
        'stop_loss': 0.1,
        'indicators': ['RSI', 'MACD'],
        'rsi_period': 14
    }
    
    print("📊 Test Strategy:")
    print(f"   Type: {test_strategy['type']}")
    print(f"   Leverage: {test_strategy['leverage']}x")
    print(f"   Position Size: {test_strategy['position_size']:.1%}")
    print(f"   Stop Loss: {test_strategy['stop_loss']:.1%}")
    
    print("\n🚀 Running real backtest...")
    
    start_time = time.time()
    results = await real_backtest_strategy(test_strategy)
    execution_time = time.time() - start_time
    
    print(f"\n📈 REAL BACKTEST RESULTS (took {execution_time:.1f}s):")
    print("=" * 50)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"✅ Strategy: {results['strategy_id']}")
        print(f"📊 CAGR: {results.get('cagr', 0):.1%}")
        print(f"📈 Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"📉 Max Drawdown: {results.get('max_drawdown', 0):.1%}")
        print(f"🎯 Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"🔢 Total Trades: {results.get('total_trades', 0)}")
        print(f"🆔 Backtest ID: {results.get('backtest_id', 'N/A')}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_real_backtesting())