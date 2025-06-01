"""
High-Performance Momentum-Reversion Strategy with Dynamic Risk Management

Target Performance Metrics:
- CAGR: >25%
- Sharpe Ratio: >1.0
- Max Drawdown: <20%
- Average Profit: >0.75% per trade
- Annual Trades: >100
"""

from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Indicators import *
import numpy as np
from datetime import timedelta, datetime

class HighPerformanceStrategy(QCAlgorithm):
    
    def Initialize(self):
        # Set start and end dates for backtest
        self.SetStartDate(2018, 1, 1)  # Set to 5 years of data for better validation
        self.SetEndDate(2023, 1, 1)
        self.SetCash(100000)
        
        # Set brokerage model
        self.SetBrokerageModel(BrokerageModel.QuantConnectBrokerage())
        
        # Universe selection - focus on liquid ETFs with different asset classes for diversification
        self.symbols = ["SPY", "QQQ", "IWM", "EEM", "GLD", "TLT", "USO", "UNG", "XLF", "XLE"]
        self.data = {}
        
        # Add symbols and indicators
        for symbol in self.symbols:
            security = self.AddEquity(symbol, Resolution.Daily)
            security.SetLeverage(2)  # Use moderate leverage to boost returns
            
            self.data[symbol] = {
                "symbol": security.Symbol,
                "rsi": self.RSI(symbol, 14, Resolution.Daily),
                "ema_fast": self.EMA(symbol, 10, Resolution.Daily),
                "ema_medium": self.EMA(symbol, 30, Resolution.Daily),
                "ema_slow": self.EMA(symbol, 50, Resolution.Daily),
                "atr": self.ATR(symbol, 14, Resolution.Daily),
                "bb": self.BB(symbol, 20, 2, Resolution.Daily),
                "macd": self.MACD(symbol, 12, 26, 9, Resolution.Daily),
                "mom": self.MOM(symbol, 20, Resolution.Daily),
                "volatility": 0,
                "score": 0
            }
        
        # Strategy parameters
        self.rebalance_days = 2  # Rebalance every 2 days to increase trade frequency
        self.max_positions = 4  # Maximum number of positions at any time
        self.position_size = 0.20  # Base position size (20% of portfolio)
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.trailing_stop_pct = 0.07  # 7% trailing stop
        self.profit_target_pct = 0.10  # 10% profit target
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen("SPY", 30), self.RebalancePortfolio)
        
        # Initialize trade tracking
        self.trade_count = 0
        self.total_profit_pct = 0
        self.trades_by_year = {}
        
        # Initialize market regime detection
        self.market_regime = "neutral"  # Can be "bullish", "bearish", or "neutral"
        self.Schedule.On(self.DateRules.WeekStart(), self.TimeRules.AfterMarketOpen("SPY", 15), self.DetectMarketRegime)
        
        # Set benchmark
        self.SetBenchmark("SPY")
        
        # Set warmup period
        self.SetWarmUp(50)
        
        # Log initialization
        self.Log("Strategy initialized with target metrics: CAGR >25%, Sharpe >1.0, Max Drawdown <20%, Avg Profit >0.75%")

    def OnData(self, data):
        """Event fired each time new data arrives"""
        if self.IsWarmingUp: return
        
        # Check stop losses and trailing stops
        self.ManageExistingPositions()
        
        # Update volatility metrics for each symbol
        self.UpdateVolatilityMetrics()

    def DetectMarketRegime(self):
        """Detect the current market regime based on price action and volatility"""
        if self.IsWarmingUp: return
        
        # Use SPY as a proxy for the overall market
        spy_data = self.data["SPY"]
        
        # Get historical data
        history = self.History(spy_data["symbol"], 50, Resolution.Daily)
        if not history.empty:
            # Calculate returns
            returns = history["close"].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Calculate trend strength
            ema_fast = spy_data["ema_fast"].Current.Value
            ema_medium = spy_data["ema_medium"].Current.Value
            ema_slow = spy_data["ema_slow"].Current.Value
            
            # Determine market regime
            if ema_fast > ema_medium and ema_medium > ema_slow and volatility < 0.20:
                self.market_regime = "bullish"
            elif ema_fast < ema_medium and ema_fast < ema_slow and volatility > 0.25:
                self.market_regime = "bearish"
            else:
                self.market_regime = "neutral"
            
            self.Log(f"Market regime detected: {self.market_regime}, Volatility: {volatility:.4f}")

    def UpdateVolatilityMetrics(self):
        """Update volatility metrics for each symbol"""
        for symbol, data in self.data.items():
            if not self.Securities.ContainsKey(data["symbol"]):
                continue
                
            # Get historical data
            history = self.History(data["symbol"], 20, Resolution.Daily)
            if not history.empty:
                # Calculate volatility
                returns = history["close"].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                data["volatility"] = volatility

    def RebalancePortfolio(self):
        """Rebalance the portfolio based on current signals and market regime"""
        if self.IsWarmingUp: return
        
        # Only rebalance every N days to reduce trading frequency
        if self.Time.day % self.rebalance_days != 0:
            return
        
        # Get current year for trade tracking
        current_year = self.Time.year
        if current_year not in self.trades_by_year:
            self.trades_by_year[current_year] = 0
        
        # Calculate scores for each symbol
        self.CalculateScores()
        
        # Sort symbols by score
        sorted_symbols = sorted([(s, self.data[s]["score"]) for s in self.symbols], key=lambda x: x[1], reverse=True)
        
        # Determine position size based on volatility
        position_sizes = self.CalculatePositionSizes()
        
        # Close positions that are no longer in our top selections
        current_holdings = [x.Key.Value for x in self.Portfolio if self.Portfolio[x.Key].Invested]
        top_symbols = [s[0] for s in sorted_symbols[:self.max_positions]]
        
        for symbol in current_holdings:
            if symbol not in top_symbols:
                self.Liquidate(symbol)
                self.Log(f"Closing position in {symbol} as it's no longer in top selections")
        
        # Open new positions or adjust existing ones
        for symbol, score in sorted_symbols[:self.max_positions]:
            if score > 0.7:  # Only invest in strong signals
                symbol_obj = self.data[symbol]["symbol"]
                current_position = self.Portfolio[symbol_obj].Quantity
                
                if current_position == 0:
                    # Calculate position size
                    size = position_sizes.get(symbol, self.position_size)
                    
                    # Adjust for market regime
                    if self.market_regime == "bearish":
                        size *= 0.5  # Reduce position size in bearish markets
                    elif self.market_regime == "bullish":
                        size *= 1.2  # Increase position size in bullish markets
                        size = min(size, 0.3)  # Cap at 30%
                    
                    # Open new position
                    self.SetHoldings(symbol_obj, size)
                    self.Log(f"Opening new position in {symbol} with size {size:.2f} and score {score:.2f}")
                    
                    # Set stop loss
                    self.StopMarketOrder(symbol_obj, -self.Portfolio[symbol_obj].Quantity, 
                                        self.Securities[symbol_obj].Close * (1 - self.stop_loss_pct))
                    
                    # Track trade
                    self.trade_count += 1
                    self.trades_by_year[current_year] += 1

    def CalculateScores(self):
        """Calculate a composite score for each symbol based on technical indicators"""
        for symbol, data in self.data.items():
            if not data["rsi"].IsReady or not data["macd"].IsReady:
                data["score"] = 0
                continue
            
            # Get indicator values
            rsi = data["rsi"].Current.Value
            macd_line = data["macd"].Current.Value
            macd_signal = data["macd"].Signal.Current.Value
            macd_hist = macd_line - macd_signal
            
            ema_fast = data["ema_fast"].Current.Value
            ema_medium = data["ema_medium"].Current.Value
            ema_slow = data["ema_slow"].Current.Value
            
            current_price = self.Securities[data["symbol"]].Close
            bb_upper = data["bb"].UpperBand.Current.Value
            bb_middle = data["bb"].MiddleBand.Current.Value
            bb_lower = data["bb"].LowerBand.Current.Value
            
            momentum = data["mom"].Current.Value
            
            # Calculate individual scores (0-1 range)
            rsi_score = 0
            if self.market_regime == "bullish":
                # In bullish regime, favor momentum
                rsi_score = (rsi - 30) / 70 if rsi > 30 else 0  # Higher RSI is better in bullish regime
            elif self.market_regime == "bearish":
                # In bearish regime, favor mean reversion
                rsi_score = (70 - rsi) / 70 if rsi < 70 else 0  # Lower RSI is better in bearish regime
            else:
                # In neutral regime, favor mean reversion from extremes
                if rsi < 30:
                    rsi_score = (30 - rsi) / 30  # Oversold
                elif rsi > 70:
                    rsi_score = 0  # Overbought (for shorting, not used here)
            
            # MACD score
            macd_score = 0
            if macd_hist > 0 and macd_line > 0:
                macd_score = 1  # Strong bullish
            elif macd_hist > 0 and macd_line < 0:
                macd_score = 0.5  # Improving
            
            # Trend score
            trend_score = 0
            if ema_fast > ema_medium and ema_medium > ema_slow:
                trend_score = 1  # Strong uptrend
            elif ema_fast > ema_medium:
                trend_score = 0.5  # Potential uptrend starting
            
            # Bollinger Band score
            bb_score = 0
            if current_price < bb_lower:
                bb_score = 1  # Oversold
            elif current_price > bb_middle and current_price < bb_upper:
                bb_score = 0.5  # Strong momentum but not overbought
            
            # Momentum score
            mom_score = 0
            if momentum > 0:
                mom_score = min(momentum / 5, 1)  # Normalize to 0-1
            
            # Combine scores based on market regime
            if self.market_regime == "bullish":
                # In bullish market, emphasize trend and momentum
                final_score = (trend_score * 0.3 + macd_score * 0.2 + 
                              mom_score * 0.2 + rsi_score * 0.15 + bb_score * 0.15)
            elif self.market_regime == "bearish":
                # In bearish market, emphasize mean reversion and risk management
                final_score = (bb_score * 0.3 + rsi_score * 0.3 + 
                              macd_score * 0.2 + trend_score * 0.1 + mom_score * 0.1)
            else:
                # In neutral market, balanced approach
                final_score = (macd_score * 0.25 + trend_score * 0.2 + 
                              bb_score * 0.2 + rsi_score * 0.2 + mom_score * 0.15)
            
            data["score"] = final_score

    def CalculatePositionSizes(self):
        """Calculate position sizes based on volatility and correlation"""
        position_sizes = {}
        
        # Get volatility for each symbol
        volatilities = {symbol: data["volatility"] for symbol, data in self.data.items() if data["volatility"] > 0}
        
        # Adjust position sizes inversely to volatility
        if volatilities:
            avg_volatility = np.mean(list(volatilities.values()))
            for symbol, vol in volatilities.items():
                if vol > 0:
                    # Inverse volatility position sizing
                    position_sizes[symbol] = self.position_size * (avg_volatility / vol)
                    # Cap position size
                    position_sizes[symbol] = min(position_sizes[symbol], self.position_size * 1.5)
                    position_sizes[symbol] = max(position_sizes[symbol], self.position_size * 0.5)
        
        return position_sizes

    def ManageExistingPositions(self):
        """Manage existing positions with trailing stops and profit targets"""
        for kvp in self.Portfolio:
            symbol = kvp.Key
            position = kvp.Value
            
            if not position.Invested:
                continue
            
            # Get current price and entry price
            current_price = self.Securities[symbol].Close
            entry_price = position.AveragePrice
            
            # Calculate current profit/loss
            if position.IsLong:
                profit_pct = (current_price / entry_price) - 1
            else:
                profit_pct = 1 - (current_price / entry_price)
            
            # Check profit target
            if profit_pct >= self.profit_target_pct:
                self.Liquidate(symbol)
                self.Log(f"Taking profit on {symbol}: {profit_pct:.2%}")
                self.total_profit_pct += profit_pct
                continue
            
            # Update trailing stop
            if profit_pct > 0.03:  # Only start trailing once we have some profit
                stop_price = current_price * (1 - self.trailing_stop_pct)
                # Calculate what the stop price would be based on entry price
                initial_stop = entry_price * (1 - self.stop_loss_pct)
                # Use the higher of the trailing stop or initial stop
                stop_price = max(stop_price, initial_stop)
                
                # Update stop order
                existing_orders = self.Transactions.GetOpenOrders(symbol)
                for order in existing_orders:
                    if order.Type == OrderType.StopMarket:
                        self.Transactions.CancelOrder(order.Id)
                
                self.StopMarketOrder(symbol, -position.Quantity, stop_price)

    def OnEndOfAlgorithm(self):
        """Summarize strategy performance at the end of the backtest"""
        # Calculate average profit per trade
        avg_profit = self.total_profit_pct / max(1, self.trade_count)
        
        # Calculate average trades per year
        years = len(self.trades_by_year)
        avg_trades_per_year = sum(self.trades_by_year.values()) / max(1, years)
        
        self.Log(f"Total trades: {self.trade_count}")
        self.Log(f"Average profit per trade: {avg_profit:.2%}")
        self.Log(f"Average trades per year: {avg_trades_per_year:.0f}")
        
        for year, count in sorted(self.trades_by_year.items()):
            self.Log(f"Year {year}: {count} trades")
