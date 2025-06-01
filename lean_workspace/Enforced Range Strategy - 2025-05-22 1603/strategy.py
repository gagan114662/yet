"""
Enforced Range Strategy
A strategy designed to work with the enforced date range (Nov 2023 to May 2025)
"""
import numpy as np
from datetime import timedelta
from QuantConnect import Resolution
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Indicators import ExponentialMovingAverage, RelativeStrengthIndex
from QuantConnect.Securities import BrokerageModel

class EnforcedRangeStrategy(QCAlgorithm):
    
    def Initialize(self):
        # We won't set dates since they're enforced by the platform
        # self.SetStartDate(2020, 1, 1)
        # self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Set brokerage model
        self.SetBrokerageModel(BrokerageModel.QuantConnectBrokerage())
        
        # Focus on volatile ETFs to increase trading opportunities
        self.symbols = ["TQQQ", "SQQQ", "UPRO", "SPXU", "LABU", "LABD", "SOXL", "SOXS"]
        self.data = {}
        
        # Add symbols and indicators
        for symbol in self.symbols:
            security = self.AddEquity(symbol, Resolution.Daily)
            security.SetLeverage(3)  # Use maximum leverage
            
            self.data[symbol] = {
                "symbol": security.Symbol,
                "rsi": self.RSI(symbol, 2, Resolution.Daily),  # Very short RSI for quick signals
                "ema_fast": self.EMA(symbol, 3, Resolution.Daily),  # Very short EMA for quick signals
                "ema_slow": self.EMA(symbol, 10, Resolution.Daily)
            }
        
        # Strategy parameters - more aggressive
        self.position_size = 0.25  # 25% position size to allow for more positions
        self.max_positions = 4    # Maximum 4 positions at once
        
        # Schedule multiple rebalancing times per day to increase trading frequency
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen("SPY", 30), self.RebalancePortfolio)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Noon, self.RebalancePortfolio)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.BeforeMarketClose("SPY", 60), self.RebalancePortfolio)
        
        # Initialize trade tracking
        self.trade_count = 0
        self.total_profit_pct = 0
        self.trades_by_year = {}
        
        # Set benchmark
        self.SetBenchmark("SPY")
        
        # Set warmup period
        self.SetWarmUp(10)
        
        # Log initialization
        self.Log("Enforced Range Strategy initialized")
        self.Log(f"Algorithm start date: {self.StartDate}")
        self.Log(f"Algorithm end date: {self.EndDate}")

    def OnData(self, data):
        """Event fired each time new data arrives"""
        if self.IsWarmingUp: return
        
        # Check for exit conditions on existing positions
        self.ManageExistingPositions()

    def RebalancePortfolio(self):
        """Portfolio rebalancing multiple times per day"""
        if self.IsWarmingUp: return
        
        # Log current state
        self.Log(f"=== REBALANCE TIME: {self.Time} ===")
        self.Log(f"Current Holdings: {[x.Key.Value for x in self.Portfolio if self.Portfolio[x.Key].Invested]}")
        
        # Get current year for trade tracking
        current_year = self.Time.year
        if current_year not in self.trades_by_year:
            self.trades_by_year[current_year] = 0
        
        # Calculate simple signals for each symbol
        signals = self.CalculateSignals()
        
        # Sort symbols by signal strength
        sorted_symbols = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Log all signals for debugging
        self.Log("All signals:")
        for symbol, signal in sorted_symbols:
            self.Log(f"  {symbol}: {signal:.4f}")
        
        # Close positions that are no longer in our top selections
        current_holdings = [x.Key.Value for x in self.Portfolio if self.Portfolio[x.Key].Invested]
        top_symbols = [s[0] for s in sorted_symbols[:self.max_positions]]
        
        for symbol in current_holdings:
            if symbol not in top_symbols:
                self.Liquidate(symbol)
                self.Log(f"Closing position in {symbol} as it's no longer in top selections")
        
        # Open new positions or adjust existing ones
        for symbol, signal in sorted_symbols[:self.max_positions]:
            # Trade on ANY signal - no threshold
            symbol_obj = self.data[symbol]["symbol"]
            current_position = self.Portfolio[symbol_obj].Quantity
            
            if current_position == 0:
                # Determine direction based on signal
                direction = 1 if signal > 0 else -1
                
                # Use fixed position size
                size = self.position_size
                
                # Open new position
                self.SetHoldings(symbol_obj, direction * size)
                self.Log(f"Opening new {('LONG' if direction > 0 else 'SHORT')} position in {symbol} with size {size:.2f} and signal {signal:.2f}")
                
                # Track trade
                self.trade_count += 1
                self.trades_by_year[current_year] += 1

    def CalculateSignals(self):
        """Calculate simple trading signals for each symbol"""
        signals = {}
        
        for symbol, data in self.data.items():
            if not data["rsi"].IsReady or not data["ema_fast"].IsReady or not data["ema_slow"].IsReady:
                signals[symbol] = 0
                continue
            
            # Get indicator values
            rsi = data["rsi"].Current.Value
            ema_fast = data["ema_fast"].Current.Value
            ema_slow = data["ema_slow"].Current.Value
            
            # Calculate trend signal (-1 to 1)
            trend_signal = (ema_fast / ema_slow - 1) * 10  # Normalize and amplify
            
            # Calculate RSI signal (-1 to 1)
            rsi_signal = 0
            if rsi < 40:  # More aggressive RSI thresholds
                rsi_signal = (40 - rsi) / 40  # Oversold (buy signal)
            elif rsi > 60:  # More aggressive RSI thresholds
                rsi_signal = -1 * (rsi - 60) / 40  # Overbought (sell signal)
            
            # Special handling for leveraged ETFs
            if symbol in ["TQQQ", "UPRO", "LABU", "SOXL"]:
                # These are bullish leveraged ETFs - amplify signals
                final_signal = (trend_signal * 0.7 + rsi_signal * 0.3) * 1.5
            elif symbol in ["SQQQ", "SPXU", "LABD", "SOXS"]:
                # These are bearish leveraged ETFs - invert and amplify signals
                final_signal = (-trend_signal * 0.7 - rsi_signal * 0.3) * 1.5
            else:
                # Regular ETFs
                final_signal = trend_signal * 0.6 + rsi_signal * 0.4
            
            signals[symbol] = final_signal
        
        return signals

    def ManageExistingPositions(self):
        """Manage existing positions with very tight profit targets and stop losses"""
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
            
            # Take profit at 2% gain - very low threshold to increase trades
            if profit_pct >= 0.02:
                self.Liquidate(symbol)
                self.Log(f"Taking profit on {symbol}: {profit_pct:.2%}")
                self.total_profit_pct += profit_pct
            
            # Cut losses at 2% loss - very low threshold to increase trades
            elif profit_pct <= -0.02:
                self.Liquidate(symbol)
                self.Log(f"Stopping loss on {symbol}: {profit_pct:.2%}")
                self.total_profit_pct += profit_pct

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
