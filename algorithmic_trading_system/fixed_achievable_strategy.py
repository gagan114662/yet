from AlgorithmImports import *
import numpy as np

class FixedAchievableMultiComponentStrategy(QCAlgorithm):
    """
    FIXED MULTI-COMPONENT STRATEGY
    Target: 12% CAGR with 1.1-1.2x leverage
    
    Components:
    1. PRIMARY MOMENTUM (50%) - 3-6 month lookbacks
    2. MEAN REVERSION (30%) - Short-term oversold bounces  
    3. FACTOR STRATEGIES (20%) - Low volatility selection
    
    Fixed common QuantConnect issues:
    - Proper method naming (Initialize, OnData)
    - Error handling for asset addition
    - Simplified indicator setup
    - Cleaned variable references
    """
    
    def Initialize(self):
        # VERIFIED 15-YEAR PERIOD
        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        # Component allocations
        self.momentum_allocation = 0.5
        self.reversion_allocation = 0.3
        self.factor_allocation = 0.2
        
        # Universe setup
        self.universe = []
        self.momentum_universe = []
        self.reversion_universe = []
        self.factor_universe = []
        
        # Core assets - simplified to avoid asset availability issues
        core_symbols = ["QQQ", "SPY", "IWM"]
        for symbol in core_symbols:
            equity = self.AddEquity(symbol, Resolution.Daily)
            equity.SetLeverage(1.2)
            self.universe.append(symbol)
            self.momentum_universe.append(symbol)
        
        # Add sector ETFs with error handling
        sector_symbols = ["XLK", "XLF", "XLE"]
        for symbol in sector_symbols:
            try:
                equity = self.AddEquity(symbol, Resolution.Daily)
                equity.SetLeverage(1.15)
                self.universe.append(symbol)
                self.reversion_universe.append(symbol)
            except:
                self.Debug(f"Failed to add {symbol}")
                continue
        
        # Small cap value for factors
        factor_symbols = ["IWN", "VBR"]
        for symbol in factor_symbols:
            try:
                equity = self.AddEquity(symbol, Resolution.Daily)
                equity.SetLeverage(1.1)
                self.universe.append(symbol)
                self.factor_universe.append(symbol)
            except:
                self.Debug(f"Failed to add factor symbol {symbol}")
                continue
        
        # COMPONENT 1: MOMENTUM INDICATORS
        self.momentum_short = {}
        self.momentum_long = {}
        
        for symbol in self.momentum_universe:
            self.momentum_short[symbol] = self.ROC(symbol, 63)  # 3-month
            self.momentum_long[symbol] = self.ROC(symbol, 126)   # 6-month
        
        # COMPONENT 2: MEAN REVERSION INDICATORS
        self.rsi = {}
        self.bb = {}
        self.sma_20 = {}
        
        for symbol in self.reversion_universe:
            self.rsi[symbol] = self.RSI(symbol, 14)
            self.bb[symbol] = self.BB(symbol, 20, 2)
            self.sma_20[symbol] = self.SMA(symbol, 20)
        
        # COMPONENT 3: FACTOR INDICATORS
        self.volatility = {}
        self.price_sma = {}
        
        for symbol in self.factor_universe:
            self.volatility[symbol] = self.STD(symbol, 30)
            self.price_sma[symbol] = self.SMA(symbol, 50)
        
        # Risk management
        self.stop_loss = 0.05
        self.take_profit = 0.15
        self.position_size_limit = 0.25
        
        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.component_trades = {"momentum": 0, "reversion": 0, "factor": 0}
        self.component_wins = {"momentum": 0, "reversion": 0, "factor": 0}
        self.component_pnl = {"momentum": 0, "reversion": 0, "factor": 0}
        
        # Position tracking
        self.entry_prices = {}
        self.position_types = {}
        self.daily_returns = []
        self.monthly_returns = []
        self.last_portfolio_value = self.Portfolio.TotalPortfolioValue
        self.last_month = self.Time.month
        
        # Rebalancing timing
        self.last_rebalance_momentum = 0
        self.last_rebalance_factor = 0
        
        self.Debug("MULTI-COMPONENT STRATEGY INITIALIZED")
        self.Debug(f"Universe: {len(self.universe)} symbols")
    
    def OnData(self, data):
        # Performance tracking
        current_value = self.Portfolio.TotalPortfolioValue
        if self.last_portfolio_value > 0:
            daily_return = (current_value - self.last_portfolio_value) / self.last_portfolio_value
            self.daily_returns.append(daily_return)
            
            # Monthly returns tracking
            if self.Time.month != self.last_month:
                month_start_value = self.last_portfolio_value
                month_return = (current_value - month_start_value) / month_start_value
                self.monthly_returns.append(month_return)
                self.last_month = self.Time.month
                
        self.last_portfolio_value = current_value
        
        # Execute components with appropriate frequencies
        
        # COMPONENT 1: MOMENTUM (rebalance weekly)
        if self.Time.day % 7 == 0 and self.Time.day != self.last_rebalance_momentum:
            self._ExecuteMomentumComponent()
            self.last_rebalance_momentum = self.Time.day
        
        # COMPONENT 2: MEAN REVERSION (check daily)
        self._ExecuteMeanReversionComponent()
        
        # COMPONENT 3: FACTOR STRATEGIES (rebalance monthly)
        if self.Time.day == 1 and self.Time.day != self.last_rebalance_factor:
            self._ExecuteFactorComponent()
            self.last_rebalance_factor = self.Time.day
        
        # Risk management
        self._ManageRisk()
    
    def _ExecuteMomentumComponent(self):
        """PRIMARY MOMENTUM COMPONENT (50% allocation)"""
        
        # Check if indicators are ready
        ready_symbols = []
        for symbol in self.momentum_universe:
            if (symbol in self.momentum_short and self.momentum_short[symbol].IsReady and
                symbol in self.momentum_long and self.momentum_long[symbol].IsReady):
                ready_symbols.append(symbol)
        
        if not ready_symbols:
            return
        
        # Calculate momentum scores
        momentum_scores = {}
        for symbol in ready_symbols:
            # Combined momentum score
            score = (self.momentum_short[symbol].Current.Value * 0.6 + 
                    self.momentum_long[symbol].Current.Value * 0.4)
            
            # Quality filter
            if score > 0.02:  # 2% minimum momentum
                momentum_scores[symbol] = score
        
        if not momentum_scores:
            return
        
        # Sort by momentum score
        sorted_symbols = sorted(momentum_scores.keys(), 
                              key=lambda x: momentum_scores[x], reverse=True)
        
        # Allocate to top momentum assets
        target_positions = min(2, len(sorted_symbols))  # Top 2 to avoid over-diversification
        position_size = self.momentum_allocation / target_positions if target_positions > 0 else 0
        
        # Clear existing momentum positions not in top list
        for symbol, pos_type in list(self.position_types.items()):
            if pos_type == "momentum" and symbol not in sorted_symbols[:target_positions]:
                if self.Portfolio[symbol].Invested:
                    pnl = self._CalculatePositionPnL(symbol)
                    self.component_pnl["momentum"] += pnl
                    if pnl > 0:
                        self.component_wins["momentum"] += 1
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    self.component_trades["momentum"] += 1
                    self.Debug(f"MOMENTUM EXIT: {symbol}, PnL: {pnl:.2%}")
                    if symbol in self.position_types:
                        del self.position_types[symbol]
        
        # Enter new momentum positions
        for i in range(target_positions):
            if i < len(sorted_symbols):
                symbol = sorted_symbols[i]
                current_position = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                
                if abs(current_position - position_size) > 0.02:
                    self.SetHoldings(symbol, position_size)
                    self.entry_prices[symbol] = self.Securities[symbol].Price
                    self.position_types[symbol] = "momentum"
                    self.trade_count += 1
                    self.component_trades["momentum"] += 1
                    self.Debug(f"MOMENTUM ENTRY: {symbol}, Score: {momentum_scores[symbol]:.3f}")
    
    def _ExecuteMeanReversionComponent(self):
        """MEAN REVERSION COMPONENT (30% allocation)"""
        
        reversion_opportunities = []
        
        for symbol in self.reversion_universe:
            if (symbol in self.rsi and self.rsi[symbol].IsReady and 
                symbol in self.bb and self.bb[symbol].IsReady and
                symbol in data and data[symbol] is not None):
                
                current_price = self.Securities[symbol].Price
                rsi_value = self.rsi[symbol].Current.Value
                lower_band = self.bb[symbol].LowerBand.Current.Value
                
                # Oversold conditions
                if (rsi_value < 30 and
                    current_price < lower_band * 1.02 and
                    not self.Portfolio[symbol].Invested):
                    
                    # Quality check
                    if symbol in self.sma_20 and self.sma_20[symbol].IsReady:
                        sma_value = self.sma_20[symbol].Current.Value
                        if current_price > sma_value * 0.95:
                            reversion_opportunities.append((symbol, rsi_value))
        
        # Limit positions
        max_reversion_positions = 2  # Reduced for stability
        current_reversion_positions = sum(1 for s, t in self.position_types.items() 
                                        if t == "reversion" and self.Portfolio[s].Invested)
        
        # Sort by RSI (most oversold first)
        reversion_opportunities.sort(key=lambda x: x[1])
        
        # Take new positions
        for symbol, rsi_value in reversion_opportunities:
            if current_reversion_positions < max_reversion_positions:
                position_size = self.reversion_allocation / max_reversion_positions
                
                if position_size <= self.position_size_limit:
                    self.SetHoldings(symbol, position_size)
                    self.entry_prices[symbol] = self.Securities[symbol].Price
                    self.position_types[symbol] = "reversion"
                    self.trade_count += 1
                    self.component_trades["reversion"] += 1
                    current_reversion_positions += 1
                    self.Debug(f"REVERSION ENTRY: {symbol}, RSI: {rsi_value:.1f}")
        
        # Exit reversion positions
        for symbol, pos_type in list(self.position_types.items()):
            if pos_type == "reversion" and self.Portfolio[symbol].Invested:
                if symbol in self.rsi and self.rsi[symbol].IsReady:
                    rsi_value = self.rsi[symbol].Current.Value
                    
                    # Exit conditions
                    if rsi_value > 70 or self._CalculatePositionPnL(symbol) > 0.03:
                        pnl = self._CalculatePositionPnL(symbol)
                        self.component_pnl["reversion"] += pnl
                        if pnl > 0:
                            self.component_wins["reversion"] += 1
                        self.Liquidate(symbol)
                        self.trade_count += 1
                        self.component_trades["reversion"] += 1
                        self.Debug(f"REVERSION EXIT: {symbol}, RSI: {rsi_value:.1f}")
                        if symbol in self.position_types:
                            del self.position_types[symbol]
    
    def _ExecuteFactorComponent(self):
        """FACTOR STRATEGIES COMPONENT (20% allocation)"""
        
        factor_scores = {}
        
        for symbol in self.factor_universe:
            if (symbol in self.volatility and self.volatility[symbol].IsReady and
                symbol in self.price_sma and self.price_sma[symbol].IsReady):
                
                current_price = self.Securities[symbol].Price
                sma_price = self.price_sma[symbol].Current.Value
                volatility = self.volatility[symbol].Current.Value
                
                # Quality filter
                if current_price > sma_price and 0 < volatility < 0.25:
                    factor_scores[symbol] = 1.0 / (volatility + 0.01)
        
        if not factor_scores:
            return
        
        # Select top low-volatility assets
        sorted_symbols = sorted(factor_scores.keys(), 
                              key=lambda x: factor_scores[x], reverse=True)
        
        target_positions = min(1, len(sorted_symbols))  # Just 1 for simplicity
        position_size = self.factor_allocation if target_positions > 0 else 0
        
        # Clear existing factor positions
        for symbol, pos_type in list(self.position_types.items()):
            if pos_type == "factor" and symbol not in sorted_symbols[:target_positions]:
                if self.Portfolio[symbol].Invested:
                    pnl = self._CalculatePositionPnL(symbol)
                    self.component_pnl["factor"] += pnl
                    if pnl > 0:
                        self.component_wins["factor"] += 1
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    self.component_trades["factor"] += 1
                    self.Debug(f"FACTOR EXIT: {symbol}")
                    if symbol in self.position_types:
                        del self.position_types[symbol]
        
        # Enter new factor position
        if target_positions > 0 and len(sorted_symbols) > 0:
            symbol = sorted_symbols[0]
            current_position = abs(self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue)
            
            if abs(current_position - position_size) > 0.02:
                self.SetHoldings(symbol, position_size)
                self.entry_prices[symbol] = self.Securities[symbol].Price
                self.position_types[symbol] = "factor"
                self.trade_count += 1
                self.component_trades["factor"] += 1
                self.Debug(f"FACTOR ENTRY: {symbol}")
    
    def _ManageRisk(self):
        """Risk management for all positions"""
        
        for symbol in list(self.entry_prices.keys()):
            if self.Portfolio[symbol].Invested and symbol in self.entry_prices:
                pnl = self._CalculatePositionPnL(symbol)
                pos_type = self.position_types.get(symbol, "unknown")
                
                # Stop loss
                if pnl < -self.stop_loss:
                    self.component_pnl[pos_type] += pnl
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    self.component_trades[pos_type] += 1
                    self.Debug(f"STOP LOSS: {symbol} ({pos_type}), PnL: {pnl:.2%}")
                    if symbol in self.position_types:
                        del self.position_types[symbol]
                    if symbol in self.entry_prices:
                        del self.entry_prices[symbol]
                
                # Take profit
                elif pnl > self.take_profit:
                    self.component_pnl[pos_type] += pnl
                    self.component_wins[pos_type] += 1
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    self.component_trades[pos_type] += 1
                    self.winning_trades += 1
                    self.Debug(f"TAKE PROFIT: {symbol} ({pos_type}), PnL: {pnl:.2%}")
                    if symbol in self.position_types:
                        del self.position_types[symbol]
                    if symbol in self.entry_prices:
                        del self.entry_prices[symbol]
    
    def _CalculatePositionPnL(self, symbol):
        """Calculate P&L for a position"""
        if symbol in self.entry_prices and self.entry_prices[symbol] > 0:
            current_price = self.Securities[symbol].Price
            return (current_price - self.entry_prices[symbol]) / self.entry_prices[symbol]
        return 0
    
    def OnEndOfAlgorithm(self):
        """Calculate final performance metrics"""
        
        years = (self.EndDate - self.StartDate).days / 365.25
        trades_per_year = self.trade_count / years if years > 0 else 0
        
        # Calculate metrics
        total_return = (self.Portfolio.TotalPortfolioValue / 100000) - 1
        cagr = ((self.Portfolio.TotalPortfolioValue / 100000) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate Sharpe ratio
        sharpe = 0
        if len(self.daily_returns) > 252:
            returns_array = np.array(self.daily_returns)
            if np.std(returns_array) > 0:
                sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        
        # Calculate win rate
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        # Log results
        self.Debug("="*60)
        self.Debug("MULTI-COMPONENT STRATEGY RESULTS")
        self.Debug("="*60)
        self.Debug(f"Total CAGR: {cagr:.2f}%")
        self.Debug(f"Sharpe Ratio: {sharpe:.2f}")
        self.Debug(f"Win Rate: {win_rate:.1f}%")
        self.Debug(f"Total Trades: {self.trade_count} ({trades_per_year:.1f}/year)")
        self.Debug(f"Final Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        
        # Component breakdown
        self.Debug(f"COMPONENT BREAKDOWN:")
        for component in ["momentum", "reversion", "factor"]:
            trades = self.component_trades[component]
            wins = self.component_wins[component]
            pnl = self.component_pnl[component] * 100
            win_rate_comp = (wins / trades * 100) if trades > 0 else 0
            self.Debug(f"  {component.upper()}: {trades} trades, {win_rate_comp:.1f}% win rate, {pnl:.2f}% P&L")