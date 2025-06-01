from AlgorithmImports import *

class DynamicMomentumStrategy(QCAlgorithm):

    def initialize(self):
        # Set up algorithm
        self.set_start_date(2015, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(100000)

        # Parameters
        self.lookback = 126  # 6 months
        self.top_n = 5       # Number of top momentum stocks
        self.stop_loss_pct = -0.10  # 10% stop loss
        self.take_profit_pct = 0.15  # 15% take profit

        # Universe: You can expand this list
        self.symbols = [
            "AAPL", "MSFT", "AMZN", "GOOG", "META",
            "TSLA", "JPM", "JNJ", "V", "PG"
        ]

        # Add assets
        self.equities = {}
        for symbol in self.symbols:
            equity = self.add_equity(symbol, Resolution.DAILY)
            equity.set_data_normalization_mode(DataNormalizationMode.RAW)
            self.equities[symbol] = equity.symbol

        # Add VIX (use self.add_index() instead of Custom.VIX())
        self.vix_symbol = self.add_index("VIX").symbol

        # Momentum indicators
        self.mom_indicators = {
            symbol: self.momentum(symbol, self.lookback)
            for symbol in self.symbols
        }

        # Schedule rebalance once a month
        self.schedule.on(
            self.date_rules.month_start(),
            self.time_rules.after_market_open("SPY"),
            self.rebalance_portfolio
        )

    def momentum(self, symbol, period):
        """ Helper function to create momentum indicator """
        mom = self.maclaurin_series_indicator(
            symbol,
            lambda x: (x.close - x.previous_close) / x.previous_close,
            period
        )
        return SimpleMovingAverage(period)
    
    def maclaurin_series_indicator(self, symbol, func, period):
        """ Creates an indicator that applies func to each bar and computes SMA """
        indicator = SimpleMovingAverage(period)
        def update(bar):
            val = func(bar)
            if val is not None:
                indicator.update(bar.end_time, val)
        self.register_indicator(symbol, indicator, Resolution.DAILY, update)
        return indicator

    def get_momentum_score(self, symbol):
        """ Get total return over lookback period """
        history = self.history(self.equities[symbol], self.lookback, Resolution.DAILY)
        if len(history) < self.lookback:
            return None
        return float((history.close[-1] - history.close[0]) / history.close[0])

    def rebalance_portfolio(self):
        # Get current VIX level
        vix_history = self.history(self.vix_symbol, 1, Resolution.DAILY)
        vix_value = float(vix_history["close"][0]) if not vix_history.empty else 20.0
        vix_adjustment = min(1.0, 30.0 / max(1e-5, vix_value))  # reduce exposure if VIX > 30

        # Rank symbols by momentum
        scores = []
        for symbol in self.symbols:
            score = self.get_momentum_score(symbol)
            if score is not None:
                scores.append((symbol, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_stocks = scores[:self.top_n]

        if not top_stocks:
            return

        # Liquidate all positions
        self.liquidate()

        # Allocate capital dynamically
        base_weight = (1.0 / len(top_stocks)) * vix_adjustment

        for symbol, score in top_stocks:
            weight = base_weight * (1 + score)  # scale by momentum strength
            self.set_holdings(symbol, weight)

            # Optional: Attach stop-loss
            security = self.securities[symbol]
            if security.invested:
                price = security.price
                stop_price = price * (1 + self.stop_loss_pct)
                self.stop_market_order(symbol, -security.holdings.quantity, stop_price)

    def on_data(self, data: Slice):
        pass
