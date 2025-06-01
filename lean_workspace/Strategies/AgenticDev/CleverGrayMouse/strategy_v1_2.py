from QuantConnect.Data import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *
from datetime import datetime, timedelta

class CryptoSwingTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        """Initialise the data and resolution required, as well as the cash and start-end dates for backtesting.
        """
        self.SetStartDate(2023, 1, 1)  # Backtest Start Date
        self.SetEndDate(datetime.now().date())  # Backtest End Date - Use date() to avoid timezone issues
        self.SetCash(100000)  # Starting Cash
        self.SetWarmUp(200)  # Warm-up period for indicators

        # Crypto Symbol - Ethereum (ETHUSD) as an example.  Change it if you want to test with Bitcoin, etc.
        self.symbol = self.AddCrypto("ETHUSD", Resolution.Minute).Symbol  # Correct way to add crypto

        # Indicator Parameters (These can be optimized)
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70

        # Indicators
        self.macd = self.MACD(self.symbol, self.fast_period, self.slow_period, self.signal_period, MovingAverageType.Exponential, Resolution.Minute)
        self.rsi = self.RSI(self.symbol, self.rsi_period, MovingAverageType.Exponential, Resolution.Minute)

        # Stop Loss and Take Profit percentages (These can be optimized)
        self.stop_loss_percentage = 0.02  # 2% Stop Loss
        self.take_profit_percentage = 0.04  # 4% Take Profit

        # Trading Window (Only trade during specific hours)
        self.trading_start_hour = 8  # 8 AM
        self.trading_end_hour = 17  # 5 PM (17:00)

        # Used to prevent day trading.  Only allow one trade per day (long or short)
        self.last_trade_date = None

        # Flag to indicate if the algorithm is warming up
        self.is_warming_up = True

        self.stop_market_ticket = None #store the stop loss ticket
        self.take_profit_ticket = None # store the take profit ticket
        self.long_pos = False
        self.short_pos = False


    def OnData(self, data: Slice):
        """OnData event handler.  This is where the trading logic is implemented.
        """

        if self.is_warming_up:
            if self.IsWarmingUp:
                return
            else:
                self.is_warming_up = False
                self.Debug("Warmup Complete")


        # Check if we are currently in a position
        if not self.Portfolio[self.symbol].Invested:
            self.CheckForEntry(data)
        else:
            self.CheckForExit(data) # Pass data to CheckForExit


    def CheckForEntry(self, data: Slice):
        """Checks for entry signals (long or short)"""

        # Trading Window Check
        current_hour = self.Time.hour
        if current_hour < self.trading_start_hour or current_hour >= self.trading_end_hour:
            return  # Sit out: Outside trading hours

        # Prevent Trading on the Same Day
        if self.last_trade_date == self.Time.date():
            return  # Sit out: Already traded today

        # Check for valid MACD and RSI values
        if not (self.macd.IsReady and self.rsi.IsReady):
            return  # Sit out: Indicators not ready


        # Long Entry Condition: MACD Histogram Crosses Above Zero and RSI Oversold
        if self.macd.Current.Value > 0 and self.macd.Previous.Value <= 0 and self.rsi.Current.Value < self.rsi_oversold:
            self.LongEntry()

        # Short Entry Condition: MACD Histogram Crosses Below Zero and RSI Overbought
        elif self.macd.Current.Value < 0 and self.macd.Previous.Value >= 0 and self.rsi.Current.Value > self.rsi_overbought:
            self.ShortEntry()



    def LongEntry(self):
        """Executes a long entry order"""
        if not self.long_pos and not self.short_pos:
            self.SetHoldings(self.symbol, 1)  # Invest 100% of portfolio
            self.Debug(f"Long Entry @ {self.Time} | Price: {self.Securities[self.symbol].Price}")

            # Set Stop Loss and Take Profit
            self.SetStopLoss(self.symbol, self.stop_loss_percentage)
            self.SetProfitTarget(self.symbol, self.take_profit_percentage)

            self.last_trade_date = self.Time.date()  # Update last trade date
            self.long_pos = True
            self.short_pos = False


    def ShortEntry(self):
        """Executes a short entry order"""
        if not self.long_pos and not self.short_pos:
            self.SetHoldings(self.symbol, -1)  # Short 100% of portfolio
            self.Debug(f"Short Entry @ {self.Time} | Price: {self.Securities[self.symbol].Price}")

            # Set Stop Loss and Take Profit
            self.SetStopLoss(self.symbol, self.stop_loss_percentage)
            self.SetProfitTarget(self.symbol, self.take_profit_percentage)

            self.last_trade_date = self.Time.date()  # Update last trade date
            self.short_pos = True
            self.long_pos = False


    def CheckForExit(self, data: Slice):
        """Checks for exit conditions (stop loss or take profit are automatically handled by QuantConnect)"""
        # Exits are handled by StopLoss and TakeProfit.  You could add other exit conditions here.
        # Example additional exit based on opposite signal:
        if self.Portfolio[self.symbol].IsLong and (self.macd.Current.Value < 0 and self.macd.Previous.Value >= 0 and self.rsi.Current.Value > self.rsi_overbought):
            self.Liquidate(self.symbol)
            self.Debug(f"Long Exit (Opposite Signal) @ {self.Time} | Price: {self.Securities[self.symbol].Price}")
            self.CancelOrders()
            self.last_trade_date = self.Time.date()
            self.long_pos = False
        elif self.Portfolio[self.symbol].IsShort and (self.macd.Current.Value > 0 and self.macd.Previous.Value <= 0 and self.rsi.Current.Value < self.rsi_oversold):
            self.Liquidate(self.symbol)
            self.Debug(f"Short Exit (Opposite Signal) @ {self.Time} | Price: {self.Securities[self.symbol].Price}")
            self.CancelOrders()
            self.last_trade_date = self.Time.date()
            self.short_pos = False

    def CancelOrders(self):
        if self.stop_market_ticket is not None and not self.stop_market_ticket.Status.IsClosed:
            self.stop_market_ticket.Cancel()
        if self.take_profit_ticket is not None and not self.take_profit_ticket.Status.IsClosed:
            self.take_profit_ticket.Cancel()



    def SetStopLoss(self, symbol, percentage):
        """Sets a stop loss order."""
        if self.Portfolio[symbol].IsLong:
            stop_price = self.Securities[symbol].Close * (1 - percentage)
            quantity = -self.Portfolio[symbol].Quantity
        elif self.Portfolio[symbol].IsShort:
            stop_price = self.Securities[symbol].Close * (1 + percentage)
            quantity = -self.Portfolio[symbol].Quantity
        else:
            return # no positions to exit from
        
        if self.stop_market_ticket is not None and not self.stop_market_ticket.Status.IsClosed:
             self.stop_market_ticket.Cancel()

        self.stop_market_ticket = self.StopMarketOrder(symbol, quantity, stop_price)
        self.Debug(f"Stop Loss set for {symbol} at {stop_price}")

    def SetProfitTarget(self, symbol, percentage):
         """Sets a take profit order."""
         if self.Portfolio[symbol].IsLong:
             take_profit_price = self.Securities[symbol].Close * (1 + percentage)
             quantity = -self.Portfolio[symbol].Quantity
         elif self.Portfolio[symbol].IsShort:
             take_profit_price = self.Securities[symbol].Close * (1 - percentage)
             quantity = -self.Portfolio[symbol].Quantity
         else:
             return # no positions to exit from

         if self.take_profit_ticket is not None and not self.take_profit_ticket.Status.IsClosed:
             self.take_profit_ticket.Cancel()

         self.take_profit_ticket = self.LimitOrder(symbol, quantity, take_profit_price)
         self.Debug(f"Take Profit set for {symbol} at {take_profit_price}")
