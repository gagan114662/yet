from AlgorithmImports import *

class LongShortEquityStrategyImproved(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        self.spy = self.AddEquity("SPY", Resolution.DAILY).Symbol
        self.aapl = self.AddEquity("AAPL", Resolution.DAILY).Symbol
        self.msft = self.AddEquity("MSFT", Resolution.DAILY).Symbol
        self.nvda = self.AddEquity("NVDA", Resolution.DAILY).Symbol

        self.symbols = [self.spy, self.aapl, self.msft, self.nvda]

        self.lookback = 20
        self.rebalance_frequency = timedelta(days=5)
        self.last_rebalance = datetime.min

        self.momentum = {}

        self.SetWarmUp(self.lookback)
        
        # Trailing stop loss and take profit
        self.trailing_stop_percentage = 0.05  # 5% trailing stop
        self.take_profit_percentage = 0.10   # 10% take profit
        self.highest_prices = {}
        self.entry_prices = {} # Track entry prices for take profit

        # Volatility filter parameters
        self.volatility_lookback = 10
        self.volatility_threshold = 0.02 # Adjust this threshold

    def OnData(self, data):

        if self.Time - self.last_rebalance < self.rebalance_frequency:
            return

        if self.IsWarmingUp:
            return

        self.CalculateMomentum(data)
        self.RebalancePortfolio(data)  # Pass data to RebalancePortfolio
        self.last_rebalance = self.Time
        self.UpdateTrailingStopsAndTakeProfit(data)


    def CalculateMomentum(self, data):
        """
        Calculates the momentum score for each symbol based on historical returns.
        """
        for symbol in self.symbols:
            if not data.ContainsKey(symbol):
                self.Log(f"No data for {symbol} at {self.Time}. Skipping momentum calculation.")
                continue

            history = self.History(symbol, self.lookback, Resolution.DAILY)

            if history.empty:
                self.Log(f"No history data for {symbol}. Skipping momentum calculation.")
                continue

            returns = history['close'].pct_change().dropna()
            # Adjusted Momentum Calculation: Exponential Weighted Moving Average
            # Gives more weight to recent returns
            weights = np.exp(np.linspace(-1., 0., num=len(returns)))
            weights /= weights.sum()
            momentum_score = np.dot(returns, weights)

            self.momentum[symbol] = momentum_score

    def CalculateVolatility(self, symbol):
        """
        Calculates the volatility of a symbol.
        """
        history = self.History(symbol, self.volatility_lookback, Resolution.DAILY)
        if history.empty:
            return None

        returns = history['close'].pct_change().dropna()
        return np.std(returns)


    def RebalancePortfolio(self, data):
        """
        Adjusts portfolio holdings based on momentum scores, incorporating volatility filter.
        Longs the top two momentum stocks and shorts the bottom two.
        """
        if not self.momentum:
            self.Log("No momentum data available. Skipping rebalancing.")
            return

        sorted_symbols = sorted(self.momentum.items(), key=lambda x: x[1], reverse=True)

        long_symbols = []
        short_symbols = []

        # Volatility filter applied during symbol selection
        for symbol, _ in sorted_symbols[:2]:
            volatility = self.CalculateVolatility(symbol)
            if volatility is not None and volatility < self.volatility_threshold:
                long_symbols.append(symbol)
            else:
                self.Log(f"Skipping {symbol} (long) due to high volatility: {volatility}")

        for symbol, _ in sorted_symbols[-2:]:
             volatility = self.CalculateVolatility(symbol)
             if volatility is not None and volatility < self.volatility_threshold:
                short_symbols.append(symbol)
             else:
                self.Log(f"Skipping {symbol} (short) due to high volatility: {volatility}")


        long_weight = 0.5 / len(long_symbols) if long_symbols else 0
        short_weight = -0.5 / len(short_symbols) if short_symbols else 0


        for holding in self.Portfolio.Values:
            if holding.Invested:
                self.Liquidate(holding.Symbol)
                self.highest_prices[holding.Symbol] = 0  # Reset highest price
                self.entry_prices[holding.Symbol] = 0 # Reset entry price


        for symbol in long_symbols:
            self.SetHoldings(symbol, long_weight)
            self.highest_prices[symbol] = self.CurrentSlice.Bars[symbol].Close  # Initialize highest price
            self.entry_prices[symbol] = self.CurrentSlice.Bars[symbol].Close # Initialize entry price

        for symbol in short_symbols:
            self.SetHoldings(symbol, short_weight)
            self.highest_prices[symbol] = self.CurrentSlice.Bars[symbol].Close  # Initialize highest price
            self.entry_prices[symbol] = self.CurrentSlice.Bars[symbol].Close # Initialize entry price

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"{orderEvent.Symbol} Order filled. Quantity: {orderEvent.FillQuantity}, Fill Price: {orderEvent.FillPrice}")

    def UpdateTrailingStopsAndTakeProfit(self, data):
        """
        Updates the trailing stop loss and take profit for each held security.
        """
        for symbol, holding in self.Portfolio.items():
            if holding.Invested and data.ContainsKey(symbol):
                current_price = data[symbol].Close
                
                # Update highest price
                if symbol in self.highest_prices:
                    self.highest_prices[symbol] = max(self.highest_prices[symbol], current_price)
                else:
                    self.highest_prices[symbol] = current_price

                # Take Profit Logic
                if symbol in self.entry_prices:
                    if holding.IsLong:
                        take_profit_price = self.entry_prices[symbol] * (1 + self.take_profit_percentage)
                        if current_price >= take_profit_price:
                            self.Liquidate(symbol)
                            self.Log(f"Take profit triggered for {symbol} at {current_price}")
                            self.highest_prices[symbol] = 0 #reset
                            self.entry_prices[symbol] = 0 #reset

                    elif holding.IsShort:
                        take_profit_price = self.entry_prices[symbol] * (1 - self.take_profit_percentage)
                        if current_price <= take_profit_price:
                            self.Liquidate(symbol)
                            self.Log(f"Take profit triggered for {symbol} at {current_price}")
                            self.highest_prices[symbol] = 0 #reset
                            self.entry_prices[symbol] = 0 #reset

                # Trailing Stop Loss Logic
                if holding.IsLong:
                    stop_loss_price = self.highest_prices[symbol] * (1 - self.trailing_stop_percentage)
                    if current_price <= stop_loss_price:
                        self.Liquidate(symbol)
                        self.Log(f"Trailing stop loss triggered for {symbol} at {current_price}")
                        self.highest_prices[symbol] = 0 #reset
                        self.entry_prices[symbol] = 0 #reset

                elif holding.IsShort:
                    stop_loss_price = self.highest_prices[symbol] * (1 + self.trailing_stop_percentage)
                    if current_price >= stop_loss_price:
                        self.Liquidate(symbol)
                        self.Log(f"Trailing stop loss triggered for {symbol} at {current_price}")
                        self.highest_prices[symbol] = 0 #reset
                        self.entry_prices[symbol] = 0 #reset
