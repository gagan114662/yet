# region imports
# 
# from AlgorithmImports import *
  # endregion

  class SPYMomentumStrategy(QCAlgorithm):

      def initialize(self):
          self.set_start_date(2023, 1, 1)
          self.set_cash(100000)
          self.symbol = self.add_equity("SPY", Resolution.DAILY).symbol

          # Create moving average indicators
          self.sma_20 = self.sma(self.symbol, 20, Resolution.DAILY)
          self.sma_50 = self.sma(self.symbol, 50, Resolution.DAILY)

          # Warm up indicators
          self.set_warm_up(50)

      def on_data(self, data: Slice):
          # Don't trade during warm-up period
          if self.is_warming_up:
              return

          # Check if indicators are ready
          if not (self.sma_20.is_ready and self.sma_50.is_ready):
              return

          # Get current values
          short_ma = self.sma_20.current.value
          long_ma = self.sma_50.current.value

          # Trading logic: Buy when short MA > long MA
          if short_ma > long_ma and not self.portfolio[self.symbol].invested:
              self.set_holdings(self.symbol, 1.0)
              self.debug(f"BUY: SMA20={short_ma:.2f} > SMA50={long_ma:.2f}")

          elif short_ma < long_ma and self.portfolio[self.symbol].invested:
              self.liquidate(self.symbol)
              self.debug(f"SELL: SMA20={short_ma:.2f} < SMA50={long_ma:.2f}")
