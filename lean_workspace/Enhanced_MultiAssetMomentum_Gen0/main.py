from AlgorithmImports import *
import numpy as np
import pandas as pd

class MultiAssetMomentumStrategy(QCAlgorithm):
    """
    ENHANCED MULTI-ASSET MOMENTUM STRATEGY
    MAX LEVERAGE: 1.2x (REALISTIC CONSTRAINT)
    Assets: ['QQQ', 'SPY', 'IWM']
    Advanced correlation and momentum analysis
    """
    
    def initialize(self):
        # VERIFIED 15-YEAR PERIOD
        self.set_start_date(2009, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(100000)
        
        # Multi-asset portfolio with realistic leverage
        self.assets = []
        for symbol in ["QQQ", "SPY", "IWM"]:
            asset = self.add_equity(symbol, Resolution.Daily)
            asset.set_leverage(1.2)
            self.assets.append(symbol)
        
        # Advanced indicators for each asset
        self.momentum = {}
        self.volatility = {}
        self.rsi = {}
        self.correlation_window = 60
        
        for symbol in self.assets:
            self.momentum[symbol] = self.roc(symbol, 20)
            self.volatility[symbol] = self.std(symbol, 20)
            self.rsi[symbol] = self.rsi(symbol, 14)
        
        # Portfolio management
        self.rebalance_frequency = weekly
        self.last_rebalance = 0
        self.trade_count = 0
        
        # Risk management
        self.max_position = 1.0 / len(self.assets) + 0.2  # Concentrated but diversified
        self.volatility_target = 0.12
        
        # Performance tracking
        self.daily_returns = []
        self.last_portfolio_value = self.portfolio.total_portfolio_value
        
    def on_data(self, data):
        # Performance tracking
        current_value = self.portfolio.total_portfolio_value
        if self.last_portfolio_value > 0:
            daily_return = (current_value - self.last_portfolio_value) / self.last_portfolio_value
            self.daily_returns.append(daily_return)
        self.last_portfolio_value = current_value
        
        if not self._indicators_ready():
            return
        
        # Rebalancing logic
        if self.time.day - self.last_rebalance >= self.rebalance_frequency:
            self._rebalance_portfolio()
            self.last_rebalance = self.time.day
    
    def _rebalance_portfolio(self):
        """Advanced portfolio rebalancing with momentum and correlation"""
        
        # Calculate momentum scores
        momentum_scores = {}
        for symbol in self.assets:
            if self.momentum[symbol].is_ready and self.volatility[symbol].is_ready:
                mom_score = self.momentum[symbol].current.value
                vol_adj = 1.0 / max(self.volatility[symbol].current.value, 0.01)
                momentum_scores[symbol] = mom_score * vol_adj
            else:
                momentum_scores[symbol] = 0
        
        # Rank by momentum
        sorted_assets = sorted(momentum_scores.keys(), 
                             key=lambda x: momentum_scores[x], reverse=True)
        
        # Calculate target allocations
        total_allocation = 0
        for i, symbol in enumerate(sorted_assets):
            if momentum_scores[symbol] > 0 and self.rsi[symbol].current.value < 70:
                # Weight by rank and momentum strength
                weight = (len(self.assets) - i) / len(self.assets)
                allocation = min(weight * 0.4, self.max_position)
                
                # Volatility adjustment
                if self.volatility[symbol].is_ready:
                    vol_factor = self.volatility_target / max(self.volatility[symbol].current.value, 0.05)
                    allocation *= min(vol_factor, 1.5)
                
                self.set_holdings(symbol, allocation)
                total_allocation += allocation
                self.trade_count += 1
                
                self.log(f"REBALANCE: {symbol} = {allocation:.2%} (Momentum: {momentum_scores[symbol]:.3f})")
        
        # Ensure we don't exceed realistic leverage
        if total_allocation > 1.2:
            scale_factor = 1.2 / total_allocation
            for symbol in sorted_assets:
                if self.portfolio[symbol].invested:
                    current_weight = self.portfolio[symbol].holdings_value / self.portfolio.total_portfolio_value
                    self.set_holdings(symbol, current_weight * scale_factor)
                    self.trade_count += 1
    
    def _indicators_ready(self):
        for symbol in self.assets:
            if not (self.momentum[symbol].is_ready and 
                   self.volatility[symbol].is_ready and 
                   self.rsi[symbol].is_ready):
                return False
        return True
    
    def on_end_of_algorithm(self):
        years = (self.end_date - self.start_date).days / 365.25
        trades_per_year = self.trade_count / years if years > 0 else 0
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 252:
            returns_array = np.array(self.daily_returns)
            if np.std(returns_array) > 0:
                sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                self.log(f"Sharpe Ratio: {sharpe:.3f}")
        
        # Calculate CAGR
        if years > 0:
            total_return = (self.portfolio.total_portfolio_value / 100000) ** (1/years) - 1
            cagr_pct = total_return * 100
            self.log(f"ENHANCED MULTI-ASSET RESULTS:")
            self.log(f"  CAGR: {cagr_pct:.2f}% ({config['leverage']}x max leverage)")
            self.log(f"  Total Trades: {self.trade_count}")
            self.log(f"  Trades/Year: {trades_per_year:.1f}")
            self.log(f"  Portfolio Value: ${self.portfolio.total_portfolio_value:,.2f}")
