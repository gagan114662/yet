# IMPORTS
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
from itertools import combinations

class TacticalAssetAllocationStrategy(QCAlgorithm):
    def Initialize(self):
        # Set strategy parameters
        self.SetStartDate(2017, 1, 1)  # Set Start Date
        self.SetEndDate(2023, 12, 31)  # Set End Date
        self.SetCash(100000)  # Set Strategy Cash
        
        # Strategy parameters
        self.rebalance_period = 21  # Trading days (approximately monthly)
        self.lookback_period = 252  # Trading days for calculating momentum (1 year)
        self.volatility_lookback = 63  # Trading days for volatility calculation (3 months)
        self.correlation_lookback = 252  # Trading days for correlation calculation (1 year)
        self.risk_free_rate = 0.05  # 5% risk-free rate as specified in requirements
        self.max_allocation = 0.25  # Maximum allocation to any single asset
        self.top_assets_count = 4  # Number of top assets to select
        self.safe_asset_allocation = 0.20  # Minimum allocation to safe assets
        
        # Track next rebalance date
        self.next_rebalance = None
        
        # Universe Selection
        # Risky Assets - Growth and Momentum ETFs
        self.risky_symbols = [
            "SPY",   # S&P 500 ETF
            "QQQ",   # Nasdaq 100 ETF
            "IWM",   # Russell 2000 Small Cap ETF
            "VGK",   # European Stocks ETF
            "EEM",   # Emerging Markets ETF
            "VNQ",   # Real Estate ETF
            "GLD",   # Gold ETF
            "USO",   # Oil ETF
            "JJC",   # Copper ETF
        ]
        
        # Safe Assets - Fixed Income and Low Volatility ETFs
        self.safe_symbols = [
            "TLT",   # 20+ Year Treasury Bond ETF
            "IEF",   # 7-10 Year Treasury Bond ETF
            "SHY",   # 1-3 Year Treasury Bond ETF
            "LQD",   # Investment Grade Corporate Bond ETF
            "USFR",  # Floating Rate Treasury ETF
            "SPLV",  # S&P 500 Low Volatility ETF
        ]
        
        # Add all symbols to universe
        self.all_symbols = self.risky_symbols + self.safe_symbols
        self.symbols = {}
        
        for symbol in self.all_symbols:
            self.symbols[symbol] = self.AddEquity(symbol, Resolution.Daily).Symbol
        
        # Create consolidators for each asset
        self.consolidators = {}
        
        # Schedule the rebalancing function to execute at market open every day
        self.Schedule.On(self.DateRules.EveryDay(), 
                         self.TimeRules.AfterMarketOpen("SPY", 30), 
                         self.Rebalance)
        
        # Warmup for the lookback periods
        self.SetWarmUp(max(self.lookback_period, self.correlation_lookback) + 10)
        
        # Indicator for tracking portfolio performance
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        self.cagr = 0
        self.avg_profit = 0
        self.performance_history = []
        self.last_portfolio_value = 100000
        self.highest_portfolio_value = 100000
        
        # Track daily returns for calculation
        self.daily_returns = []
        self.portfolio_values = []
        self.last_rebalance_date = None
        
        # For storing optimization results
        self.optimization_history = []

    def OnData(self, data):
        # Track daily portfolio performance
        if not self.IsWarmingUp:
            current_value = self.Portfolio.TotalPortfolioValue
            self.portfolio_values.append(current_value)
            
            # Calculate daily return
            if len(self.portfolio_values) > 1:
                daily_return = (current_value / self.portfolio_values[-2]) - 1
                self.daily_returns.append(daily_return)
                
                # Update max drawdown calculation
                if current_value > self.highest_portfolio_value:
                    self.highest_portfolio_value = current_value
                
                current_drawdown = 1 - (current_value / self.highest_portfolio_value)
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
                
                # If enough data, update performance metrics
                if len(self.daily_returns) > 20:
                    # Calculate Sharpe ratio
                    annual_return = np.mean(self.daily_returns) * 252
                    annual_volatility = np.std(self.daily_returns) * np.sqrt(252)
                    self.sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
                    
                    # Calculate CAGR if we have at least 252 days of data
                    if len(self.daily_returns) >= 252:
                        days = len(self.daily_returns)
                        total_return = self.portfolio_values[-1] / self.portfolio_values[0]
                        self.cagr = (total_return ** (252 / days)) - 1
                    
                    # Calculate average profit per trade
                    self.avg_profit = np.mean([r for r in self.daily_returns if r > 0])
                    
                    # Store performance metrics
                    self.performance_history.append({
                        'Date': self.Time,
                        'Sharpe': self.sharpe_ratio,
                        'MaxDrawdown': self.max_drawdown,
                        'CAGR': self.cagr,
                        'AvgProfit': self.avg_profit
                    })

    def Rebalance(self):
        # Skip if still warming up
        if self.IsWarmingUp:
            return
            
        # Check if it's time to rebalance
        if (self.next_rebalance is not None and self.Time < self.next_rebalance) or not self.Portfolio.Invested:
            if not self.Portfolio.Invested:
                # Initial allocation
                self.Debug("Initial allocation")
                self.next_rebalance = self.Time + timedelta(days=self.rebalance_period)
                self.last_rebalance_date = self.Time
                self.RebalancePortfolio()
            return
            
        # Check if it's time to rebalance
        days_since_last_rebalance = (self.Time - self.last_rebalance_date).days
        if days_since_last_rebalance >= self.rebalance_period:
            self.Debug(f"Rebalancing portfolio on {self.Time}")
            self.next_rebalance = self.Time + timedelta(days=self.rebalance_period)
            self.last_rebalance_date = self.Time
            self.RebalancePortfolio()
            
            # Log current performance
            self.Log(f"Current Performance - Sharpe: {self.sharpe_ratio:.2f}, MaxDrawdown: {self.max_drawdown:.2%}, CAGR: {self.cagr:.2%}, AvgProfit: {self.avg_profit:.2%}")

    def RebalancePortfolio(self):
        # 1. Measure performance and momentum across assets
        momentum_scores = self.CalculateMomentumScores()
        if momentum_scores is None:
            return
            
        # 2. Select top performing assets (based on momentum)
        selected_risky_assets = self.SelectTopAssets(momentum_scores, self.risky_symbols, self.top_assets_count)
        selected_safe_assets = self.SelectTopAssets(momentum_scores, self.safe_symbols, 2)
        
        # Combine selected assets
        selected_assets = selected_risky_assets + selected_safe_assets
        
        # Debug selected assets
        self.Debug(f"Selected Assets: {selected_assets}")
        
        # Early exit if not enough assets
        if len(selected_assets) < 3:
            self.Debug("Not enough selected assets to optimize portfolio")
            return
            
        # 3. Get historical data for selected assets for optimization
        # Get price data for volatility and correlation calculation
        history = self.GetHistoryForOptimization(selected_assets)
        if history is None or history.empty:
            self.Debug("No history available for optimization")
            return
        
        # Convert prices to returns
        returns = history.pct_change().dropna()
        
        # Early exit if not enough return data
        if len(returns) < self.volatility_lookback:
            self.Debug("Not enough return data for optimization")
            return
        
        # 4. Optimize portfolio weights using minimum variance with target returns
        try:
            optimal_weights = self.OptimizePortfolio(returns, selected_assets)
            
            # Enforce minimum allocation to safe assets
            safe_weight = sum(optimal_weights.get(symbol, 0) for symbol in selected_safe_assets)
            if safe_weight < self.safe_asset_allocation:
                self.Debug(f"Adjusting weights to ensure minimum safe asset allocation: {self.safe_asset_allocation}")
                # Scale down risky assets and increase safe assets
                scale_factor = (1 - self.safe_asset_allocation) / (1 - safe_weight) if safe_weight < 1 else 1
                
                # Adjust weights
                for symbol in selected_assets:
                    if symbol in selected_risky_assets:
                        optimal_weights[symbol] = optimal_weights.get(symbol, 0) * scale_factor
                    elif symbol in selected_safe_assets:
                        # Distribute the safe allocation proportionally
                        if len(selected_safe_assets) > 0:
                            original_weight = optimal_weights.get(symbol, 0)
                            safe_proportion = original_weight / safe_weight if safe_weight > 0 else 1.0 / len(selected_safe_assets)
                            optimal_weights[symbol] = self.safe_asset_allocation * safe_proportion
            
            # 5. Apply the optimal weights to the portfolio
            self.ApplyWeights(optimal_weights)
            
            # Record optimization results
            self.optimization_history.append({
                'Date': self.Time,
                'Assets': selected_assets,
                'Weights': optimal_weights
            })
            
        except Exception as e:
            self.Debug(f"Optimization failed: {str(e)}")
            # Fall back to equal weighting if optimization fails
            equal_weight = 1.0 / len(selected_assets)
            optimal_weights = {symbol: equal_weight for symbol in selected_assets}
            self.ApplyWeights(optimal_weights)

    def GetHistoryForOptimization(self, symbols):
        """Get historical price data for optimization"""
        try:
            history = self.History([self.symbols[s] for s in symbols], 
                                   max(self.lookback_period, self.correlation_lookback),
                                   Resolution.Daily)
            
            if 'close' not in history.columns.levels[0]:
                # If the 'close' column is not in the dataframe, reconstruct it
                self.Debug("Reconstructing price dataframe")
                history_dict = {}
                
                for symbol in symbols:
                    security_history = self.History(self.symbols[symbol], 
                                                  max(self.lookback_period, self.correlation_lookback),
                                                  Resolution.Daily)
                    if "close" in security_history.columns:
                        history_dict[symbol] = security_history["close"]
                
                if history_dict:
                    history = pd.DataFrame(history_dict)
                else:
                    return None
            else:
                # Extract the 'close' prices and pivot the dataframe
                history = history['close'].unstack(level=0)
                history.columns = [s.Value for s in history.columns]
            
            return history
            
        except Exception as e:
            self.Debug(f"Error getting history: {str(e)}")
            return None

    def CalculateMomentumScores(self):
        """Calculate momentum scores for all assets"""
        try:
            momentum_scores = {}
            
            # Get historical data for all assets
            history = self.History([self.symbols[s] for s in self.all_symbols], 
                                   self.lookback_period, 
                                   Resolution.Daily)
            
            if 'close' not in history.columns.levels[0]:
                # If the 'close' column is not in the dataframe, try to get individual histories
                self.Debug("Reconstructing momentum calculation")
                for symbol in self.all_symbols:
                    try:
                        security_history = self.History(self.symbols[symbol], 
                                                      self.lookback_period,
                                                      Resolution.Daily)
                        
                        if "close" in security_history.columns and len(security_history) > 0:
                            prices = security_history["close"].values
                            
                            # Calculate momentum (return over lookback_period)
                            if len(prices) > 0:
                                momentum = (prices[-1] / prices[0]) - 1
                                
                                # Adjusted momentum score (considering both return and volatility)
                                volatility = np.std(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 1
                                adjusted_momentum = momentum / volatility if volatility > 0 else 0
                                
                                momentum_scores[symbol] = adjusted_momentum
                    except Exception as inner_e:
                        self.Debug(f"Error calculating momentum for {symbol}: {str(inner_e)}")
            else:
                # Extract the 'close' prices and process
                history = history['close'].unstack(level=0)
                
                for symbol_obj in history.columns:
                    symbol = symbol_obj.Value
                    if symbol in self.all_symbols:
                        prices = history[symbol_obj].dropna().values
                        
                        if len(prices) > 0:
                            # Calculate momentum (return over lookback_period)
                            momentum = (prices[-1] / prices[0]) - 1
                            
                            # Adjusted momentum score (considering both return and volatility)
                            volatility = np.std(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 1
                            adjusted_momentum = momentum / volatility if volatility > 0 else 0
                            
                            momentum_scores[symbol] = adjusted_momentum
            
            return momentum_scores
            
        except Exception as e:
            self.Debug(f"Error calculating momentum scores: {str(e)}")
            return None

    def SelectTopAssets(self, momentum_scores, symbol_list, count):
        """Select top performing assets based on momentum scores"""
        # Filter scores to only include symbols from the provided list
        filtered_scores = {s: momentum_scores[s] for s in symbol_list if s in momentum_scores}
        
        # Sort by momentum score in descending order
        sorted_assets = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 'count' assets
        selected = [s[0] for s in sorted_assets[:count]]
        
        return selected

    def OptimizePortfolio(self, returns, selected_assets):
        """Optimize portfolio weights to minimize risk while meeting performance targets"""
        try:
            # Filter returns to only include selected assets
            asset_returns = returns[[s for s in selected_assets if s in returns.columns]].copy()
            
            # Calculate expected returns (recent performance weighted more heavily)
            short_term_returns = asset_returns.iloc[-63:].mean() * 252  # ~3 months
            long_term_returns = asset_returns.mean() * 252  # full period
            
            # Weighted average of short and long term returns (favor recent performance)
            expected_returns = (short_term_returns * 0.7) + (long_term_returns * 0.3)
            
            # Get covariance matrix
            cov_matrix = asset_returns.iloc[-self.volatility_lookback:].cov() * 252
            
            # Number of assets
            n = len(asset_returns.columns)
            
            # Initial guess (equal weights)
            initial_weights = np.array([1/n] * n)
            
            # Constraints
            # 1. Sum of weights = 1
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # 2. Expected return >= target
            target_return = max(0.25, expected_returns.mean())  # At least 25% CAGR
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: np.dot(expected_returns, x) - target_return
            })
            
            # Bounds - each asset between 0% and max_allocation
            bounds = tuple((0, self.max_allocation) for _ in range(n))
            
            # Define objective function - portfolio variance
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return portfolio_variance
            
            # Perform optimization
            result = minimize(objective, initial_weights, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
            
            # Map the optimized weights back to the asset symbols
            optimal_weights = {}
            for i, symbol in enumerate(asset_returns.columns):
                optimal_weights[symbol] = result.x[i]
            
            # Ensure weights sum to 1
            if abs(sum(optimal_weights.values()) - 1.0) > 1e-10:
                # Normalize weights
                total = sum(optimal_weights.values())
                for symbol in optimal_weights:
                    optimal_weights[symbol] /= total
            
            return optimal_weights
        
        except Exception as e:
            self.Debug(f"Optimization error: {str(e)}")
            # Return equal weights if optimization fails
            equal_weight = 1.0 / len(selected_assets)
            return {symbol: equal_weight for symbol in selected_assets}

    def ApplyWeights(self, target_weights):
        """Apply the target weights to the portfolio"""
        # Log the target allocation
        self.Debug(f"Target allocation: {target_weights}")
        
        # Apply the weights
        for symbol, weight in target_weights.items():
            if symbol in self.all_symbols:
                self.SetHoldings(self.symbols[symbol], weight)
                
    def OnEndOfAlgorithm(self):
        """Display final performance metrics"""
        self.Debug("------ Final Performance ------")
        self.Debug(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        self.Debug(f"Maximum Drawdown: {self.max_drawdown:.2%}")
        self.Debug(f"CAGR: {self.cagr:.2%}")
        self.Debug(f"Average Profit per Trade: {self.avg_profit:.2%}")
        
        # Check if the strategy meets the criteria
        criteria_met = (
            self.sharpe_ratio > 1 and
            self.cagr > 0.25 and
            self.max_drawdown <= 0.20 and
            self.avg_profit >= 0.0075
        )
        
        self.Debug(f"Criteria Met: {criteria_met}")
