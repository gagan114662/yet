from AlgorithmImports import *

class EvolutionMomentumStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Strategy parameters from evolution
        self.leverage = 2.0
        self.position_size = 0.2
        self.stop_loss_pct = 0.08
        
        # Add SPY
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Technical indicators
        self.rsi = self.RSI(self.spy, 14, MovingAverageType.Simple, Resolution.Daily)
        self.macd = self.MACD(self.spy, 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
        
        # Risk management
        self.stop_loss_price = None
        
        # Set leverage
        self.Securities[self.spy].SetLeverage(2.0)
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(self.spy), 
                        self.TimeRules.AfterMarketOpen(self.spy, 30), 
                        self.Rebalance)
        
        # Track performance
        self.benchmark_start = None
    
    def Rebalance(self):
        if not self.rsi.IsReady or not self.macd.IsReady:
            return
        
        current_price = self.Securities[self.spy].Price
        
        # Momentum signals
        rsi_signal = self.rsi.Current.Value > 50
        macd_signal = self.macd.Current.Value > self.macd.Signal.Current.Value
        
        # Position management
        holdings = self.Portfolio[self.spy].Quantity
        
        if rsi_signal and macd_signal and holdings == 0:
            # Enter long position
            portfolio_value = self.Portfolio.TotalPortfolioValue
            position_value = portfolio_value * self.position_size * self.leverage
            quantity = int(position_value / current_price)
            
            if quantity > 0:
                self.MarketOrder(self.spy, quantity)
                self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                self.Debug(f"Entered long: {quantity} shares at ${current_price:.2f}")
        
        elif holdings > 0:
            # Check stop loss
            if current_price <= self.stop_loss_price:
                self.Liquidate(self.spy)
                self.Debug(f"Stop loss triggered at ${current_price:.2f}")
                self.stop_loss_price = None
            
            # Check exit signal
            elif not rsi_signal or not macd_signal:
                self.Liquidate(self.spy)
                self.Debug(f"Exit signal at ${current_price:.2f}")
                self.stop_loss_price = None
    
    def OnData(self, data):
        # Track initial benchmark
        if self.benchmark_start is None and self.spy in data:
            self.benchmark_start = data[self.spy].Close
    
    def OnEndOfAlgorithm(self):
        # Log final performance
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:.2f}")
        if self.benchmark_start:
            spy_return = (self.Securities[self.spy].Close - self.benchmark_start) / self.benchmark_start
            self.Debug(f"SPY Return: {spy_return:.2%}")
