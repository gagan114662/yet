from AlgorithmImports import *
import numpy as np
import pandas as pd
from scipy.stats import norm
import math

class HestonOptionsStrategy(QCAlgorithm):
    def Initialize(self):
        # Set start and end date for the backtest
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)  # Starting cash in USD
        
        # Set time zone for the algorithm
        self.SetTimeZone(TimeZones.New_York)
        
        # Add Google stock data (equity) to the algorithm
        self.symbol = self.AddEquity("GOOG", Resolution.Minute).Symbol
        
        # Add GOOG options with 1-minute resolution
        option = self.AddOption("GOOG")
        option.SetFilter(self.OptionFilter)
        
        # Set up initial placeholders for the parameters
        self.r = 0.01  # Risk-free rate (we will fetch this dynamically later)
        self.v0 = 0.04  # Initial volatility (to be calculated dynamically)
        self.kappa = 2.0  # Mean reversion speed (calibrated during warm-up)
        self.theta = 0.02  # Long-term volatility variance (calibrated during warm-up)
        self.sigma = 0.3  # Volatility of volatility (calculated dynamically)
        self.rho = -0.5  # Correlation between asset price and volatility (calculated dynamically)

        # Set up parameters for backtesting (can be adjusted)
        self.target_delta = 0.5  # Target delta for options
        
        # Warm-up period for data collection
        self.SetWarmUp(60)  # Set warm-up period to 60 minutes or adjust as needed

        # Keep track of historical data for rolling calculations
        self.price_history = []  # To store historical asset prices
        self.volatility_history = []  # To store historical volatility
    
    def OptionFilter(self, universe):
        # Filtering options based on expiry date and strike price range
        return universe.Strikes(-5, 5).Expiration(0, 30)  # Expiring within the next 30 days
    
    def OnData(self, data):
        # Skip if we are still in the warm-up period
        if self.IsWarmingUp:
            return
        
        # Check if options data is available
        if not data.OptionChains:
            return
        
        # Loop through all options contracts in the chain
        for chain in data.OptionChains:
            for option in chain.Value:
                strike = option.Strike
                expiry = option.Expiry
                bid_price = option.BidPrice
                ask_price = option.AskPrice
                
                # Calculate time to expiry in years
                T = (expiry - self.Time).days / 365
                
                # Fetch real-time data for Heston model parameters
                self.r = self.GetRiskFreeRate()  # Risk-free rate
                self.v0 = self.GetHistoricalVolatility("GOOG", 30)  # 30 days historical volatility
                self.sigma = self.GetImpliedVolatility(option)  # Implied volatility from option chain
                self.rho = self.GetCorrelation("GOOG")  # Calculate correlation
                
                # Estimate kappa and theta using historical data
                self.kappa, self.theta = self.EstimateKappaTheta()
                
                # Use the Heston model to calculate option price
                option_price = self.HestonPrice(self.symbol, strike, T, self.r, self.v0, self.kappa, self.theta, self.sigma, self.rho)
                
                # Define profitable trades
                if option_price < bid_price:  # Buy if the model price is lower than the bid price
                    self.BuyOption(option)
                elif option_price > ask_price:  # Sell if the model price is higher than the ask price
                    self.SellOption(option)
    
    def HestonPrice(self, S, K, T, r, v0, kappa, theta, sigma, rho):
        # This function returns the option price using the Heston model
        def integrand(u, S, K, T, r, v0, kappa, theta, sigma, rho):
            a = kappa * theta
            b = kappa + sigma * sigma / 2
            d = np.sqrt((rho * sigma * 1j * u - b) ** 2 - sigma ** 2 * (2 * 1j * u - u ** 2))
            g1 = (b - rho * sigma * 1j * u - d) / (b - rho * sigma * 1j * u + d)
            return np.exp(1j * u * np.log(S / K) + (r - 0.5 * v0) * T * 1j * u) * g1

        def characteristic_function(S, K, T, r, v0, kappa, theta, sigma, rho):
            result = 0
            for u in np.linspace(0.01, 100, 100):
                result += np.real(integrand(u, S, K, T, r, v0, kappa, theta, sigma, rho))
            return result

        return characteristic_function(S, K, T, r, v0, kappa, theta, sigma, rho)

    def EstimateKappaTheta(self):
        # Use historical data to estimate kappa and theta via optimization
        # This is a simplified version using price history and volatility
        kappa = 1.5  # Placeholder value, you'd calibrate this via a least squares fitting
        theta = 0.02  # Placeholder value, you'd calibrate this via a least squares fitting
        return kappa, theta

    def GetRiskFreeRate(self):
        # Fetch real-time risk-free rate (using US Treasury or another benchmark rate)
        # In this example, we can use the US10Y treasury yield
        treasury = self.AddEquity("IEF", Resolution.Daily)  # IEF is an ETF tracking the 10-year US Treasury
        return treasury.Price / 100  # We are assuming price represents yield for simplicity

    def GetHistoricalVolatility(self, symbol, window=30):
        # Calculate historical volatility from price data
        price_data = self.History(symbol, window, Resolution.Daily)
        log_returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna()
        volatility = np.std(log_returns) * np.sqrt(252)  # Annualize the volatility
        return volatility

    def GetImpliedVolatility(self, option):
        # Fetch the implied volatility from the options chain data
        # QuantConnect provides implied volatility as part of the option chain data
        return option.ImpliedVolatility if option.ImpliedVolatility is not None else 0.2  # Default to 20%

    def GetCorrelation(self, symbol):
        # Calculate correlation between asset price and volatility (simplified estimation)
        price_data = self.History(symbol, 60, Resolution.Daily)
        log_returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna()
        
        # Estimate volatility using rolling standard deviation (simple way to estimate volatility)
        volatility = log_returns.rolling(window=30).std() * np.sqrt(252)
        
        # Calculate rolling correlation
        correlation = log_returns.corr(volatility)
        return correlation.mean()  # Return average correlation over the period
    
    def BuyOption(self, option):
        # Buy the option
        self.MarketOrder(option.Symbol, 1)
        self.Debug(f"Bought option: {option.Symbol}")
    
    def SellOption(self, option):
        # Sell the option
        self.MarketOrder(option.Symbol, -1)
        self.Debug(f"Sold option: {option.Symbol}")
