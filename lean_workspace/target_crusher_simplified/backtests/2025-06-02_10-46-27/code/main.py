from AlgorithmImports import *
import numpy as np
from datetime import timedelta

class TargetCrusherSimplified(QCAlgorithm):
    """
    Simplified version of Target Crusher Ultimate to isolate issues
    """
    
    def Initialize(self):
        # Backtest period
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(100000)
        
        # Enable margin trading
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # Initialize tracking dictionaries first
        self.securities = {}
        self.indicators = {}
        
        # Core liquid assets for momentum trading
        self.core_symbols = ["SPY", "QQQ"]
        
        # Add securities
        for symbol in self.core_symbols:
            try:
                security = self.AddEquity(symbol, Resolution.Hour)
                security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
                security.SetLeverage(2.0)
                self.securities[symbol] = security
            except:
                self.Debug(f"Failed to add {symbol}")
                continue
        
        # Technical indicators for each symbol
        for symbol in self.securities.keys():
            self.indicators[symbol] = {
                "rsi": self.RSI(symbol, 14, MovingAverageType.Wilders, Resolution.Daily),
                "std_20": self.STD(symbol, 20, Resolution.Daily)
            }
        
        # Portfolio management
        self.portfolio_stop_hit = False
        self.last_portfolio_value = 100000
        
        # Risk parameters
        self.portfolio_stop_loss = 0.15
        
        # Weekly risk assessment
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Friday),
            self.TimeRules.AfterMarketOpen("SPY", 60),
            self.WeeklyRiskAssessment
        )
        
        # Warm up period
        self.SetWarmup(timedelta(days=30))
        
    def WeeklyRiskAssessment(self):
        """Weekly risk check with proper safety"""
        try:
            # Calculate recent performance
            current_value = self.Portfolio.TotalPortfolioValue
            weekly_return = (current_value - self.last_portfolio_value) / self.last_portfolio_value
            self.last_portfolio_value = current_value
            
            # Simple volatility check with safety
            if "SPY" in self.indicators and "std_20" in self.indicators["SPY"]:
                if self.indicators["SPY"]["std_20"].IsReady and "SPY" in self.Securities:
                    spy_volatility = self.indicators["SPY"]["std_20"].Current.Value / self.Securities["SPY"].Price
                    self.Debug(f"SPY volatility: {spy_volatility:.4f}")
                else:
                    self.Debug("SPY volatility indicator not ready")
            else:
                self.Debug("SPY indicators not available")
                
        except Exception as e:
            self.Debug(f"Error in WeeklyRiskAssessment: {e}")
        
    def OnData(self, data):
        """Simple momentum strategy"""
        if self.IsWarmingUp or self.portfolio_stop_hit:
            return
            
        # Portfolio stop loss check
        current_value = self.Portfolio.TotalPortfolioValue
        if current_value < (1 - self.portfolio_stop_loss) * self.last_portfolio_value:
            self.portfolio_stop_hit = True
            self.Liquidate()
            self.Debug("Portfolio stop loss hit - liquidating all positions")
            return
        
        # Simple buy and hold for testing
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1.0)