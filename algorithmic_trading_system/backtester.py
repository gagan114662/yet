import subprocess
import json
import os
from typing import Dict
from datetime import datetime
try:
    from quantconnect_integration.rd_agent_qc_bridge import QuantConnectIntegration
except ImportError:
    print("Warning: QuantConnect integration not available. Falling back to basic Lean CLI.")
    QuantConnectIntegration = None
import config # To access Lean CLI configurations

class Backtester:
    def __init__(self):
        """
        Initializes the Backtester with QuantConnect integration if available,
        otherwise falls back to basic Lean CLI setup.
        """
        # Try to use QuantConnect integration first
        if QuantConnectIntegration:
            try:
                self.qc_integration = QuantConnectIntegration()
                self.use_qc_integration = True
                print("Using QuantConnect integration for backtesting.")
            except Exception as e:
                print(f"Failed to initialize QuantConnect integration: {e}")
                self.use_qc_integration = False
        else:
            self.use_qc_integration = False
            
        # Fallback to basic Lean CLI setup
        if not self.use_qc_integration:
            self.lean_cli_user_id = config.LEAN_CLI_USER_ID
            self.lean_cli_api_token = config.LEAN_CLI_API_TOKEN
            self.lean_cli_path = config.LEAN_CLI_PATH
            self.lean_workspace_path = "../lean_workspace"
            self.temp_lean_project_path = os.path.join(self.lean_workspace_path, "temp_backtest_strategy")
            # Ensure the base workspace directory exists
            os.makedirs(self.temp_lean_project_path, exist_ok=True)
            print("Using basic Lean CLI for backtesting.")

    def backtest_strategy(self, strategy_idea: Dict) -> Dict:
        """
        Backtests a strategy using either QuantConnect integration or basic Lean CLI.

        Args:
            strategy_idea: A dictionary containing the strategy definition.

        Returns:
            A dictionary containing performance metrics or error information.
        """
        strategy_name = strategy_idea.get('name', 'UnnamedStrategy')
        print(f"Backtesting strategy: {strategy_name}")
        
        # Use QuantConnect integration if available
        if self.use_qc_integration:
            return self._backtest_with_qc_integration(strategy_idea)
        else:
            return self._backtest_with_lean_cli(strategy_idea)
    
    def _backtest_with_qc_integration(self, strategy_idea: Dict) -> Dict:
        """Backtest using QuantConnect integration."""
        strategy_name = strategy_idea.get('name', 'UnnamedStrategy').replace(' ', '_')
        unique_project_name = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        print(f"Preparing to backtest strategy: {unique_project_name}")
        print(f"Strategy details: {strategy_idea}")

        try:
            # 1. Create a LEAN project directory
            print(f"Attempting to create LEAN project: {unique_project_name}")
            project_path = self.qc_integration.create_lean_project(project_name=unique_project_name)
            print(f"LEAN project created at: {project_path}")

            # 2. Generate strategy code from the idea
            print("Generating strategy code...")
            strategy_code = self.qc_integration.generate_strategy_code(strategy_idea)

            # 3. Run the backtest
            print(f"Running backtest for project: {unique_project_name} at path: {project_path}...")
            results = self.qc_integration.run_backtest(strategy_code, project_path)

            if "error" in results:
                print(f"Warning: Backtest for {unique_project_name} encountered an error: {results['error']}")
            else:
                print(f"Backtest completed for {unique_project_name}. Results: {results}")

            return results

        except Exception as e:
            print(f"An unexpected error occurred during backtesting strategy {unique_project_name}: {e}")
            return {
                "error": str(e),
                "details": "Exception in Backtester._backtest_with_qc_integration",
                "project_name": unique_project_name,
                "strategy_idea": strategy_idea
            }
    
    def _backtest_with_lean_cli(self, strategy_idea: Dict) -> Dict:
        """Backtest using basic Lean CLI."""
        strategy_name = strategy_idea.get('name', 'UnnamedStrategy')
        print(f"Backtesting strategy: {strategy_name} using basic Lean CLI.")

        # Generate strategy code from strategy_idea
        strategy_code = self._generate_strategy_code_from_idea(strategy_idea)

        try:
            os.makedirs(self.temp_lean_project_path, exist_ok=True)

            # Write strategy code to main.py
            with open(os.path.join(self.temp_lean_project_path, "main.py"), "w") as f:
                f.write(strategy_code)

            # Write minimal config.json
            lean_config = {
                "algorithm-language": "Python",
                "parameters": {}
            }
            with open(os.path.join(self.temp_lean_project_path, "config.json"), "w") as f:
                json.dump(lean_config, f)

        except IOError as e:
            print(f"Error writing strategy or config files: {e}")
            return {
                'error': 'File system error preparing Lean project', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }

        # --- Construct Lean CLI Command ---
        command = [
            self.lean_cli_path,
            "backtest",
            self.temp_lean_project_path
        ]

        print(f"Executing Lean CLI command: {' '.join(command)}")

        # --- Execute Command ---
        try:
            env = os.environ.copy()
            # Run from the lean workspace directory for proper context
            process = subprocess.run(command, capture_output=True, text=True, env=env, 
                                   check=False, cwd=self.lean_workspace_path)

            # Look for backtest results in the backtests directory
            project_backtests_dir = os.path.join(self.temp_lean_project_path, "backtests")
            
            if process.returncode != 0:
                print(f"Error executing Lean CLI: {process.stderr}")
                results_json_path = self.find_results_json(project_backtests_dir)
                if results_json_path:
                    print(f"Attempting to parse results from {results_json_path} despite CLI error.")
                    return self.parse_lean_results_from_file(results_json_path, process.stderr)
                return {
                    'error': 'Lean CLI execution failed', 'details': process.stderr,
                    'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
                }

            # --- Parse Results ---
            if process.stdout:
                try:
                    lean_results_data = json.loads(process.stdout)
                    print("Successfully parsed Lean CLI JSON output from stdout.")
                    return self.parse_metrics_from_lean_json(lean_results_data, process.stdout)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON from Lean CLI stdout: {e}. Stdout was: {process.stdout[:500]}...")

            # Fallback: look for results.json in the backtests directory
            results_json_path = self.find_results_json(project_backtests_dir)
            
            if results_json_path:
                print(f"Parsing Lean CLI output from file: {results_json_path}")
                return self.parse_lean_results_from_file(results_json_path)
            else:
                print(f"Lean CLI stdout was not JSON, and no results.json found.")
                print(f"Stdout: {process.stdout[:500]}...")
                print(f"Stderr: {process.stderr[:500]}...")
                return {
                    'error': 'Lean CLI output not JSON and results.json not found', 
                    'details': f"Stdout: {process.stdout[:500]} | Stderr: {process.stderr[:500]}",
                    'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
                }

        except FileNotFoundError:
            print(f"Error: Lean CLI executable not found at '{self.lean_cli_path}'. Please check config.py.")
            return {
                'error': 'Lean CLI executable not found', 'details': f"Path '{self.lean_cli_path}' is invalid.",
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }
        except Exception as e:
            print(f"An unexpected error occurred during Lean CLI execution: {e}")
            return {
                'error': 'Unexpected error during backtest', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }

    def _generate_strategy_code_from_idea(self, strategy_idea: Dict) -> str:
        """Generate sophisticated Lean algorithm code from strategy idea."""
        # Extract strategy parameters
        strategy_type = strategy_idea.get('type', 'momentum')
        start_date = strategy_idea.get('start_date', '2020,1,1')
        end_date = strategy_idea.get('end_date', '2023,12,31')
        lookback_period = strategy_idea.get('lookback_period', 20)
        position_size = strategy_idea.get('position_size', 0.2)
        leverage = strategy_idea.get('leverage', 2.0)
        universe_size = strategy_idea.get('universe_size', 100)
        min_price = strategy_idea.get('min_price', 10.0)
        min_volume = strategy_idea.get('min_volume', 5000000)
        rebalance_frequency = strategy_idea.get('rebalance_frequency', 3)
        stop_loss = strategy_idea.get('stop_loss', 0.12)
        profit_target = strategy_idea.get('profit_target', 0.20)
        volatility_target = strategy_idea.get('volatility_target', 0.15)
        max_drawdown = strategy_idea.get('max_drawdown', 0.15)
        
        # Check if this is a strategy from lean_workspace with base_template
        if 'base_template' in strategy_idea:
            from strategy_importer import StrategyImporter
            importer = StrategyImporter()
            template_code = importer.get_strategy_code(strategy_idea['base_template'])
            if template_code and 'class' in template_code:
                # Adapt the template with real parameters
                adapted_code = self._adapt_template_code(template_code, strategy_idea)
                return adapted_code
        
        # Generate universe based on strategy type
        if strategy_type == 'leveraged_etf':
            universe = strategy_idea.get('etf_universe', ["TQQQ", "UPRO", "QLD", "SSO", "SQQQ", "SPXS"])
            universe_str = str(universe)
        elif strategy_type == 'volatility_harvesting':
            universe = ["SPY", "QQQ", "VXX", "SVXY", "UVXY"]
            universe_str = str(universe)
        elif strategy_type == 'options':
            universe = strategy_idea.get('option_symbols', ["SPY", "QQQ"])
            universe_str = str(universe)
        else:
            universe_str = f'self.universe_symbols[:min({universe_size}, len(self.universe_symbols))]'
        
        # Get strategy-specific logic
        indicator_setup = strategy_idea.get('indicator_setup', '"rsi": self.RSI(symbol, 14), "macd": self.MACD(symbol, 12, 26, 9)')
        signal_generation_logic = strategy_idea.get('signal_generation_logic', '''
indicators = self.indicators[symbol]
rsi = indicators["rsi"].Current.Value
macd = indicators["macd"].Current.Value
signal = indicators["macd"].Signal.Current.Value
trade_signal = 0

if self.Securities[symbol].Price > 0 and rsi < 35 and macd > signal:
    trade_signal = 1
elif rsi > 65 and macd < signal:
    trade_signal = -1
''')
        
        # Generate sophisticated strategy code based on type
        if strategy_type == 'leveraged_etf':
            code = self._generate_leveraged_etf_strategy(strategy_idea)
        elif strategy_type == 'volatility_harvesting':
            code = self._generate_volatility_strategy(strategy_idea) 
        elif strategy_type == 'options':
            code = self._generate_options_strategy(strategy_idea)
        elif strategy_type in ['multi_factor', 'aggressive_momentum', 'high_frequency']:
            code = self._generate_advanced_strategy(strategy_idea)
        else:
            code = self._generate_default_strategy(strategy_idea)
            
        return code
    
    def _generate_leveraged_etf_strategy(self, strategy_idea: Dict) -> str:
        """Generate leveraged ETF rotation strategy"""
        etfs = strategy_idea.get('etf_universe', ["TQQQ", "UPRO", "SQQQ", "SPXS"])
        leverage = strategy_idea.get('leverage', 3.0)
        stop_loss = strategy_idea.get('stop_loss', 0.12)
        
        return f'''
from AlgorithmImports import *

class LeveragedETFStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({strategy_idea.get('start_date', '2020,1,1')})
        self.SetEndDate({strategy_idea.get('end_date', '2023,12,31')})
        self.SetCash(100000)
        
        # Leveraged ETF universe
        self.etfs = {etfs}
        self.leverage = {leverage}
        self.stop_loss = {stop_loss}
        
        # Add ETF securities
        for etf in self.etfs:
            self.AddEquity(etf, Resolution.Hour)
            
        # Technical indicators
        self.rsi = {{}}
        self.macd = {{}}
        self.bb = {{}}
        self.adx = {{}}
        
        for etf in self.etfs:
            self.rsi[etf] = self.RSI(etf, 14)
            self.macd[etf] = self.MACD(etf, 12, 26, 9)
            self.bb[etf] = self.BB(etf, 20, 2)
            self.adx[etf] = self.ADX(etf, 14)
            
        # Risk management
        self.entry_prices = {{}}
        self.max_portfolio_loss = 0.15
        
    def OnData(self, data):
        # Portfolio drawdown protection
        if self.Portfolio.TotalUnrealizedProfit / self.Portfolio.TotalPortfolioValue < -self.max_portfolio_loss:
            self.Liquidate("Portfolio drawdown protection triggered")
            return
            
        for etf in self.etfs:
            if not data.ContainsKey(etf) or not self.rsi[etf].IsReady:
                continue
                
            rsi = self.rsi[etf].Current.Value
            macd = self.macd[etf].Current.Value
            signal = self.macd[etf].Signal.Current.Value
            bb_upper = self.bb[etf].UpperBand.Current.Value
            bb_lower = self.bb[etf].LowerBand.Current.Value
            adx = self.adx[etf].Current.Value
            price = data[etf].Price
            
            # Risk management - stop losses
            if self.Portfolio[etf].Invested:
                entry_price = self.entry_prices.get(etf, price)
                if self.Portfolio[etf].IsLong and price < entry_price * (1 - self.stop_loss):
                    self.Liquidate(etf, "Stop loss hit")
                    continue
                elif self.Portfolio[etf].IsShort and price > entry_price * (1 + self.stop_loss):
                    self.Liquidate(etf, "Stop loss hit")
                    continue
            
            # Strategy logic for leveraged ETFs
            if etf in ["TQQQ", "UPRO", "QLD", "SSO"]:  # Bull ETFs
                if (price < bb_lower and rsi < 30 and adx > 25 and macd > signal):
                    if not self.Portfolio[etf].Invested:
                        self.SetHoldings(etf, self.leverage * 0.4)
                        self.entry_prices[etf] = price
                elif rsi > 75 or macd < signal:
                    if self.Portfolio[etf].Invested:
                        self.Liquidate(etf)
                        
            elif etf in ["SQQQ", "SPXS"]:  # Bear ETFs
                if (price > bb_upper and rsi > 70 and adx > 25 and macd < signal):
                    if not self.Portfolio[etf].Invested:
                        self.SetHoldings(etf, self.leverage * 0.3)
                        self.entry_prices[etf] = price
                elif rsi < 25 or macd > signal:
                    if self.Portfolio[etf].Invested:
                        self.Liquidate(etf)
'''

    def _generate_advanced_strategy(self, strategy_idea: Dict) -> str:
        """Generate advanced multi-factor strategy"""
        return f'''
from AlgorithmImports import *
import numpy as np

class AdvancedStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({strategy_idea.get('start_date', '2020,1,1')})
        self.SetEndDate({strategy_idea.get('end_date', '2023,12,31')})
        self.SetCash(100000)
        
        # Strategy parameters
        self.leverage = {strategy_idea.get('leverage', 3.0)}
        self.position_size = {strategy_idea.get('position_size', 0.3)}
        self.stop_loss = {strategy_idea.get('stop_loss', 0.12)}
        self.universe_size = {strategy_idea.get('universe_size', 100)}
        self.min_volume = {strategy_idea.get('min_volume', 10000000)}
        
        # Universe selection
        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Multi-factor indicators
        self.indicators = {{}}
        self.momentum_scores = {{}}
        self.volatility_scores = {{}}
        
        # Risk management
        self.max_drawdown = {strategy_idea.get('max_drawdown', 0.15)}
        self.entry_prices = {{}}
        self.last_rebalance = datetime.min
        
    def CoarseSelectionFunction(self, coarse):
        # High-volume, liquid stocks only
        filtered = [x for x in coarse if x.Price > 10 and x.DollarVolume > self.min_volume]
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:self.universe_size]]
    
    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.indicators:
                self.indicators[symbol] = {{
                    "rsi": self.RSI(symbol, 14),
                    "macd": self.MACD(symbol, 12, 26, 9),
                    "bb": self.BB(symbol, 20, 2),
                    "momentum": self.MOMP(symbol, 20),
                    "adx": self.ADX(symbol, 14),
                    "atr": self.ATR(symbol, 14)
                }}
                
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.indicators:
                del self.indicators[symbol]
                
    def OnData(self, data):
        # Portfolio protection
        if self.Portfolio.TotalUnrealizedProfit / self.Portfolio.TotalPortfolioValue < -self.max_drawdown:
            self.Liquidate("Maximum drawdown exceeded")
            return
            
        # Rebalance every 2 hours
        if (self.Time - self.last_rebalance).total_seconds() < 7200:
            return
            
        # Score all securities
        scores = {{}}
        for symbol in self.indicators.keys():
            if symbol in data and self.indicators[symbol]["rsi"].IsReady:
                score = self.CalculateScore(symbol, data[symbol])
                if abs(score) > 0.3:  # Only trade high-conviction signals
                    scores[symbol] = score
                    
        # Risk management - stop losses
        for symbol in list(self.Portfolio.Keys):
            if self.Portfolio[symbol].Invested and symbol in data:
                entry_price = self.entry_prices.get(symbol, data[symbol].Price)
                current_price = data[symbol].Price
                
                if self.Portfolio[symbol].IsLong:
                    if current_price < entry_price * (1 - self.stop_loss):
                        self.Liquidate(symbol, "Stop loss")
                else:
                    if current_price > entry_price * (1 + self.stop_loss):
                        self.Liquidate(symbol, "Stop loss")
        
        # Execute top signals
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
            top_signals = sorted_scores[:min(10, len(sorted_scores))]
            
            for symbol, score in top_signals:
                weight = self.position_size * abs(score) * self.leverage
                weight = min(weight, 0.25)  # Max 25% per position
                
                if score > 0.3 and not self.Portfolio[symbol].IsLong:
                    self.SetHoldings(symbol, weight)
                    self.entry_prices[symbol] = data[symbol].Price
                elif score < -0.3 and not self.Portfolio[symbol].IsShort:
                    self.SetHoldings(symbol, -weight)
                    self.entry_prices[symbol] = data[symbol].Price
                    
        self.last_rebalance = self.Time
        
    def CalculateScore(self, symbol, bar):
        indicators = self.indicators[symbol]
        
        # Multi-factor scoring
        momentum_score = 0
        mean_reversion_score = 0
        volatility_score = 0
        
        rsi = indicators["rsi"].Current.Value
        macd = indicators["macd"].Current.Value
        signal = indicators["macd"].Signal.Current.Value
        momentum = indicators["momentum"].Current.Value
        adx = indicators["adx"].Current.Value
        
        # Momentum factor
        if momentum > 0.05 and macd > signal and adx > 25:
            momentum_score = min((momentum * 10 + (macd - signal) * 5), 1.0)
        elif momentum < -0.05 and macd < signal and adx > 25:
            momentum_score = max((momentum * 10 + (macd - signal) * 5), -1.0)
            
        # Mean reversion factor
        if rsi < 25:
            mean_reversion_score = (30 - rsi) / 30
        elif rsi > 75:
            mean_reversion_score = -(rsi - 70) / 30
            
        # Combine factors with weights
        final_score = (momentum_score * 0.6 + mean_reversion_score * 0.4)
        
        return final_score
'''
    
    def _generate_default_strategy(self, strategy_idea: Dict) -> str:
        """Generate default enhanced strategy"""
        indicator_setup = strategy_idea.get('indicator_setup', '"rsi": self.RSI(symbol, 14), "macd": self.MACD(symbol, 12, 26, 9)')
        signal_logic = strategy_idea.get('signal_generation_logic', '''
indicators = self.indicators[symbol]
rsi = indicators["rsi"].Current.Value
macd = indicators["macd"].Current.Value
signal = indicators["macd"].Signal.Current.Value
trade_signal = 0

if rsi < 35 and macd > signal:
    trade_signal = 1
elif rsi > 65 and macd < signal:
    trade_signal = -1
''')
        
        return f'''
from AlgorithmImports import *

class EnhancedStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({strategy_idea.get('start_date', '2020,1,1')})
        self.SetEndDate({strategy_idea.get('end_date', '2023,12,31')})
        self.SetCash(100000)
        
        # Enhanced parameters
        self.position_size = {strategy_idea.get('position_size', 0.2)}
        self.leverage = {strategy_idea.get('leverage', 2.0)}
        self.universe_size = {strategy_idea.get('universe_size', 100)}
        self.min_volume = {strategy_idea.get('min_volume', 5000000)}
        self.stop_loss = {strategy_idea.get('stop_loss', 0.12)}
        
        # Universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        self.indicators = {{}}
        self.last_rebalance = datetime.min
        self.entry_prices = {{}}
        
    def CoarseSelectionFunction(self, coarse):
        filtered = [x for x in coarse if x.Price > 10 and x.DollarVolume > self.min_volume]
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:self.universe_size]]
    
    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.indicators:
                self.indicators[symbol] = {{{indicator_setup}}}
        
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.indicators:
                del self.indicators[symbol]
    
    def OnData(self, data):
        # Rebalance every few days
        if (self.Time - self.last_rebalance).days < {strategy_idea.get('rebalance_frequency', 3)}:
            return
            
        signals = {{}}
        for symbol in self.indicators.keys():
            if symbol in data and self.indicators[symbol]["rsi"].IsReady:
                try:
{signal_logic}
                    signals[symbol] = trade_signal
                except:
                    signals[symbol] = 0
        
        # Risk management and execution
        self.ExecuteSignals(signals, data)
        self.last_rebalance = self.Time
    
    def ExecuteSignals(self, signals, data):
        # Stop loss management
        for symbol in list(self.Portfolio.Keys):
            if self.Portfolio[symbol].Invested and symbol in data:
                entry_price = self.entry_prices.get(symbol, data[symbol].Price)
                current_price = data[symbol].Price
                
                if self.Portfolio[symbol].IsLong and current_price < entry_price * (1 - self.stop_loss):
                    self.Liquidate(symbol)
                elif self.Portfolio[symbol].IsShort and current_price > entry_price * (1 + self.stop_loss):
                    self.Liquidate(symbol)
        
        # Position sizing with leverage
        positive_signals = [s for s in signals.values() if s > 0]
        if positive_signals:
            position_per_stock = (self.position_size * self.leverage) / len(positive_signals)
            position_per_stock = min(position_per_stock, 0.25)  # Max 25% per position
            
            for symbol, signal in signals.items():
                if signal > 0 and not self.Portfolio[symbol].IsLong:
                    self.SetHoldings(symbol, position_per_stock)
                    self.entry_prices[symbol] = data[symbol].Price
                elif signal <= 0 and self.Portfolio[symbol].IsLong:
                    self.Liquidate(symbol)
        else:
            self.Liquidate()
'''
    
    def _generate_volatility_strategy(self, strategy_idea: Dict) -> str:
        """Generate volatility harvesting strategy - placeholder"""
        return self._generate_default_strategy(strategy_idea)
    
    def _generate_options_strategy(self, strategy_idea: Dict) -> str:
        """Generate options strategy - placeholder"""
        return self._generate_default_strategy(strategy_idea)
    
    def _adapt_template_code(self, template_code: str, strategy_idea: Dict) -> str:
        """Adapt template code with real parameters from strategy_idea"""
        # Replace key parameters in the template code
        adapted_code = template_code
        
        # Update dates to use config values
        start_date = strategy_idea.get('start_date', config.BACKTEST_START_DATE)
        end_date = strategy_idea.get('end_date', config.BACKTEST_END_DATE)
        
        # Replace common date patterns
        adapted_code = adapted_code.replace('2020, 1, 1', start_date.replace(',', ', '))
        adapted_code = adapted_code.replace('2023, 12, 31', end_date.replace(',', ', '))
        adapted_code = adapted_code.replace('2020,1,1', start_date)
        adapted_code = adapted_code.replace('2023,12,31', end_date)
        
        # Update capital
        initial_capital = strategy_idea.get('initial_capital', config.INITIAL_CAPITAL)
        adapted_code = adapted_code.replace('100000', str(initial_capital))
        
        # Update key parameters if they exist in the template
        if 'leverage' in strategy_idea:
            leverage = strategy_idea['leverage']
            # Try to replace common leverage patterns
            import re
            adapted_code = re.sub(r'self\.leverage\s*=\s*[\d\.]+', f'self.leverage = {leverage}', adapted_code)
            adapted_code = re.sub(r'leverage\s*=\s*[\d\.]+', f'leverage = {leverage}', adapted_code)
        
        if 'position_size' in strategy_idea:
            position_size = strategy_idea['position_size']
            adapted_code = re.sub(r'self\.position_size\s*=\s*[\d\.]+', f'self.position_size = {position_size}', adapted_code)
            adapted_code = re.sub(r'position_size\s*=\s*[\d\.]+', f'position_size = {position_size}', adapted_code)
        
        if 'stop_loss' in strategy_idea:
            stop_loss = strategy_idea['stop_loss']
            adapted_code = re.sub(r'self\.stop_loss\s*=\s*[\d\.]+', f'self.stop_loss = {stop_loss}', adapted_code)
            adapted_code = re.sub(r'stop_loss\s*=\s*[\d\.]+', f'stop_loss = {stop_loss}', adapted_code)
        
        return adapted_code

    def find_results_json(self, search_dir: str) -> str | None:
        """Find the latest results.json file in the search directory."""
        if not os.path.isdir(search_dir):
            return None

        latest_time = 0
        results_file = None

        # Option 1: results.json directly in search_dir
        direct_results_json = os.path.join(search_dir, "results.json")
        if os.path.isfile(direct_results_json):
            return direct_results_json

        # Option 2: Search in subdirectories (timestamped folders)
        for item in os.listdir(search_dir):
            item_path = os.path.join(search_dir, item)
            if os.path.isdir(item_path):
                current_results_json = os.path.join(item_path, "results.json")
                if os.path.isfile(current_results_json):
                    try:
                        folder_time = int(item)
                    except ValueError:
                        folder_time = os.path.getmtime(current_results_json)

                    if folder_time > latest_time:
                        latest_time = folder_time
                        results_file = current_results_json

        if results_file:
            print(f"Found results.json at: {results_file}")
        return results_file

    def parse_lean_results_from_file(self, file_path: str, cli_stderr_if_any: str = None) -> Dict:
        try:
            with open(file_path, 'r') as f:
                lean_results_data = json.load(f)
            print(f"Successfully parsed Lean CLI JSON output from file: {file_path}")
            full_output_for_debugging = f"File: {file_path}"
            if cli_stderr_if_any:
                full_output_for_debugging += f"\\nCLI Stderr:\\n{cli_stderr_if_any}"

            return self.parse_metrics_from_lean_json(lean_results_data, full_output_for_debugging)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from Lean results file {file_path}: {e}")
            return {
                'error': 'Failed to parse results.json', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }
        except IOError as e:
            print(f"Failed to read results.json file {file_path}: {e}")
            return {
                'error': 'Failed to read results.json', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0
            }

    def parse_metrics_from_lean_json(self, lean_data: Dict, raw_output_for_debugging: str) -> Dict:
        """Parse metrics from Lean's JSON output structure."""
        try:
            # Default values for metrics (matching config.py format)
            metrics = {
                'cagr': 0.0,
                'max_drawdown': 1.0, # Positive value where smaller is better
                'sharpe_ratio': 0.0,
                'avg_profit': 0.0,
                'total_trades': 0,
                'lean_cli_output': raw_output_for_debugging[:2000]
            }

            # Parse Lean output structure
            stats = lean_data.get('Statistics', lean_data)

            # Map Lean metrics to our format
            metrics['sharpe_ratio'] = float(stats.get('Sharpe Ratio', stats.get('SharpeRatio', 0.0)))
            metrics['cagr'] = float(stats.get('Compounding Annual Return', stats.get('CompoundingAnnualReturn', 0.0)))
            
            # Max Drawdown - ensure it's positive (smaller is better)
            max_drawdown_lean = float(stats.get('Drawdown', stats.get('Max Drawdown', stats.get('Maximum Drawdown', 1.0))))
            metrics['max_drawdown'] = abs(max_drawdown_lean) if max_drawdown_lean != 1.0 else 1.0
            
            # Calculate average profit per trade
            total_trades = int(stats.get('Total Trades', stats.get('TotalTrades', 0)))
            metrics['total_trades'] = total_trades
            if total_trades > 0:
                total_return = metrics['cagr']
                metrics['avg_profit'] = total_return / total_trades if total_return != 0 else 0.0

            print(f"Parsed metrics: {metrics}")
            return metrics

        except (KeyError, ValueError, TypeError) as e:
            print(f"Error parsing Lean metrics: {e}. Data was: {str(lean_data)[:500]}")
            return {
                'error': 'Error parsing Lean metrics', 'details': str(e),
                'cagr': 0, 'max_drawdown': 1, 'sharpe_ratio': 0, 'avg_profit': 0, 'total_trades': 0,
                'lean_cli_output': raw_output_for_debugging[:2000]
            }

# Example usage
if __name__ == '__main__':
    from strategy_utils import generate_next_strategy

    print("Generating a sample strategy idea for backtesting...")
    strategy_idea_to_test = generate_next_strategy()
    print(f"Strategy Idea: {strategy_idea_to_test}")

    backtester = Backtester()
    print("\\nInitializing backtester and starting backtest process...")
    backtest_results = backtester.backtest_strategy(strategy_idea_to_test)

    print(f"\\nBacktester execution finished.")
    print(f"Strategy Idea Tested: {strategy_idea_to_test.get('name')}")
    print(f"Full Results: {backtest_results}")

    if "error" in backtest_results:
        print(f"Error during backtest: {backtest_results['error']}")
    elif backtest_results:
        print(f"CAGR: {backtest_results.get('cagr')}")
        print(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio')}")
        print(f"Max Drawdown: {backtest_results.get('max_drawdown')}")
        print(f"Total Trades: {backtest_results.get('total_trades')}")