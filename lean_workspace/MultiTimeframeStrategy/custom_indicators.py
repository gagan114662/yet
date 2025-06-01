# region imports
from AlgorithmImports import *
from QuantConnect import *
from QuantConnect.Indicators import *
# endregion

class RelativeStrengthIndicator(PythonIndicator):
    """
    Custom indicator that calculates the relative strength of an asset compared to a benchmark.
    Relative strength is calculated as the ratio of the asset's price to the benchmark's price,
    normalized by their respective starting prices.
    """
    
    def __init__(self, name, asset_symbol, benchmark_symbol, period):
        """
        Initialize the RelativeStrengthIndicator
        
        Parameters:
        -----------
        name : str
            The name of the indicator
        asset_symbol : Symbol
            The symbol of the asset to calculate relative strength for
        benchmark_symbol : Symbol
            The symbol of the benchmark to compare against
        period : int
            The lookback period for calculating relative strength
        """
        self.name = name
        self.asset_symbol = asset_symbol
        self.benchmark_symbol = benchmark_symbol
        self.period = period
        
        # Initialize price windows
        self.asset_prices = RollingWindow[float](period)
        self.benchmark_prices = RollingWindow[float](period)
        
        # Initialize base class
        super().__init__()
        
    def is_ready(self):
        """Check if the indicator is ready to produce values"""
        return self.asset_prices.is_ready and self.benchmark_prices.is_ready
        
    def update(self, input_data):
        """
        Update the indicator with new data
        
        Parameters:
        -----------
        input_data : Dictionary
            Dictionary containing data for the asset and benchmark symbols
        
        Returns:
        --------
        bool
            True if the indicator was updated successfully
        """
        # Check if we have data for both symbols
        if self.asset_symbol in input_data and self.benchmark_symbol in input_data:
            # Get prices
            asset_price = input_data[self.asset_symbol].close
            benchmark_price = input_data[self.benchmark_symbol].close
            
            # Add prices to windows
            self.asset_prices.add(asset_price)
            self.benchmark_prices.add(benchmark_price)
            
            # Calculate relative strength if we have enough data
            if self.is_ready():
                # Get starting prices
                asset_start = self.asset_prices[self.period - 1]
                benchmark_start = self.benchmark_prices[self.period - 1]
                
                # Get current prices
                asset_current = self.asset_prices[0]
                benchmark_current = self.benchmark_prices[0]
                
                # Calculate relative strength
                if asset_start > 0 and benchmark_start > 0:
                    asset_return = asset_current / asset_start
                    benchmark_return = benchmark_current / benchmark_start
                    
                    if benchmark_return > 0:
                        relative_strength = asset_return / benchmark_return
                        self.value = relative_strength
                        return True
                        
        return False

class VolatilityRegimeIndicator(PythonIndicator):
    """
    Custom indicator that identifies the current volatility regime.
    It compares recent volatility to historical volatility to determine
    if we're in a high, normal, or low volatility environment.
    """
    
    def __init__(self, name, symbol, short_period, long_period):
        """
        Initialize the VolatilityRegimeIndicator
        
        Parameters:
        -----------
        name : str
            The name of the indicator
        symbol : Symbol
            The symbol to calculate volatility for
        short_period : int
            The lookback period for recent volatility
        long_period : int
            The lookback period for historical volatility
        """
        self.name = name
        self.symbol = symbol
        self.short_period = short_period
        self.long_period = long_period
        
        # Initialize price window
        self.prices = RollingWindow[float](long_period + 1)
        
        # Initialize returns window
        self.returns = RollingWindow[float](long_period)
        
        # Initialize volatility indicators
        self.short_vol = StandardDeviation(short_period)
        self.long_vol = StandardDeviation(long_period)
        
        # Initialize regime value (0 = low, 1 = normal, 2 = high)
        self.regime = 1
        
        # Initialize base class
        super().__init__()
        
    def is_ready(self):
        """Check if the indicator is ready to produce values"""
        return self.short_vol.is_ready and self.long_vol.is_ready
        
    def update(self, input_data):
        """
        Update the indicator with new data
        
        Parameters:
        -----------
        input_data : Dictionary
            Dictionary containing data for the symbol
        
        Returns:
        --------
        bool
            True if the indicator was updated successfully
        """
        # Check if we have data for the symbol
        if self.symbol in input_data:
            # Get price
            price = input_data[self.symbol].close
            
            # Add price to window
            self.prices.add(price)
            
            # Calculate return if we have at least 2 prices
            if self.prices.count > 1:
                ret = (price / self.prices[1]) - 1
                self.returns.add(ret)
                
                # Update volatility indicators
                self.short_vol.update(ret)
                self.long_vol.update(ret)
                
                # Determine volatility regime if we have enough data
                if self.is_ready():
                    short_vol = self.short_vol.current.value
                    long_vol = self.long_vol.current.value
                    
                    # Determine regime
                    if short_vol > long_vol * 1.5:
                        self.regime = 2  # High volatility
                    elif short_vol < long_vol * 0.75:
                        self.regime = 0  # Low volatility
                    else:
                        self.regime = 1  # Normal volatility
                        
                    # Set indicator value to regime
                    self.value = self.regime
                    return True
                    
        return False

class TrendStrengthIndicator(PythonIndicator):
    """
    Custom indicator that measures the strength of a trend.
    It combines ADX (Average Directional Index) with price momentum
    to determine if a trend is strong, weak, or non-existent.
    """
    
    def __init__(self, name, symbol, period):
        """
        Initialize the TrendStrengthIndicator
        
        Parameters:
        -----------
        name : str
            The name of the indicator
        symbol : Symbol
            The symbol to calculate trend strength for
        period : int
            The lookback period for calculating trend strength
        """
        self.name = name
        self.symbol = symbol
        self.period = period
        
        # Initialize ADX indicator
        self.adx = AverageDirectionalIndex(period)
        
        # Initialize price window
        self.prices = RollingWindow[float](period)
        
        # Initialize trend direction (1 = up, -1 = down, 0 = none)
        self.direction = 0
        
        # Initialize trend strength (0-100)
        self.strength = 0
        
        # Initialize base class
        super().__init__()
        
    def is_ready(self):
        """Check if the indicator is ready to produce values"""
        return self.adx.is_ready and self.prices.is_ready
        
    def update(self, input_data):
        """
        Update the indicator with new data
        
        Parameters:
        -----------
        input_data : Dictionary
            Dictionary containing data for the symbol
        
        Returns:
        --------
        bool
            True if the indicator was updated successfully
        """
        # Check if we have data for the symbol
        if self.symbol in input_data:
            # Get price data
            bar = input_data[self.symbol]
            
            # Update ADX
            self.adx.update(bar)
            
            # Add price to window
            self.prices.add(bar.close)
            
            # Calculate trend strength if we have enough data
            if self.is_ready():
                # Get ADX value
                adx_value = self.adx.current.value
                
                # Determine trend direction
                start_price = self.prices[self.period - 1]
                end_price = self.prices[0]
                
                if end_price > start_price * 1.05:
                    self.direction = 1  # Up trend
                elif end_price < start_price * 0.95:
                    self.direction = -1  # Down trend
                else:
                    self.direction = 0  # No clear trend
                    
                # Calculate trend strength
                self.strength = adx_value
                
                # Set indicator value to direction * strength
                self.value = self.direction * self.strength
                return True
                
        return False

class MarketRegimeIndicator(PythonIndicator):
    """
    Custom indicator that identifies the current market regime.
    It combines price trend, volatility, and breadth indicators
    to determine if we're in a bull, bear, or neutral market.
    """
    
    def __init__(self, name, symbol, sma_period, vol_period):
        """
        Initialize the MarketRegimeIndicator
        
        Parameters:
        -----------
        name : str
            The name of the indicator
        symbol : Symbol
            The symbol to calculate market regime for
        sma_period : int
            The period for the simple moving average
        vol_period : int
            The period for volatility calculation
        """
        self.name = name
        self.symbol = symbol
        self.sma_period = sma_period
        self.vol_period = vol_period
        
        # Initialize indicators
        self.sma = SimpleMovingAverage(sma_period)
        self.volatility = StandardDeviation(vol_period)
        
        # Initialize regime value (1 = bull, 0 = neutral, -1 = bear)
        self.regime = 0
        
        # Initialize base class
        super().__init__()
        
    def is_ready(self):
        """Check if the indicator is ready to produce values"""
        return self.sma.is_ready and self.volatility.is_ready
        
    def update(self, input_data):
        """
        Update the indicator with new data
        
        Parameters:
        -----------
        input_data : Dictionary
            Dictionary containing data for the symbol
        
        Returns:
        --------
        bool
            True if the indicator was updated successfully
        """
        # Check if we have data for the symbol
        if self.symbol in input_data:
            # Get price
            price = input_data[self.symbol].close
            
            # Update indicators
            self.sma.update(price)
            
            # Calculate return for volatility
            if hasattr(self, 'last_price') and self.last_price > 0:
                ret = (price / self.last_price) - 1
                self.volatility.update(ret)
                
            self.last_price = price
            
            # Determine market regime if we have enough data
            if self.is_ready():
                # Price relative to SMA
                price_to_sma = price / self.sma.current.value
                
                # Current volatility
                vol = self.volatility.current.value
                
                # Determine regime
                if price_to_sma > 1.05 and vol < 0.015:
                    self.regime = 1  # Bull market
                elif price_to_sma < 0.95 or vol > 0.025:
                    self.regime = -1  # Bear market
                else:
                    self.regime = 0  # Neutral market
                    
                # Set indicator value to regime
                self.value = self.regime
                return True
                
        return False

class CustomIndicator:
    """
    Factory class for creating custom indicators
    """
    
    @staticmethod
    def relative_strength(algorithm, name, asset_symbol, benchmark_symbol, period):
        """
        Create a relative strength indicator
        
        Parameters:
        -----------
        algorithm : QCAlgorithm
            The algorithm instance
        name : str
            The name of the indicator
        asset_symbol : Symbol
            The symbol of the asset to calculate relative strength for
        benchmark_symbol : Symbol
            The symbol of the benchmark to compare against
        period : int
            The lookback period for calculating relative strength
            
        Returns:
        --------
        RelativeStrengthIndicator
            The relative strength indicator
        """
        indicator = RelativeStrengthIndicator(name, asset_symbol, benchmark_symbol, period)
        algorithm.register_indicator(asset_symbol, indicator)
        algorithm.register_indicator(benchmark_symbol, indicator)
        return indicator
        
    @staticmethod
    def volatility_regime(algorithm, name, symbol, short_period, long_period):
        """
        Create a volatility regime indicator
        
        Parameters:
        -----------
        algorithm : QCAlgorithm
            The algorithm instance
        name : str
            The name of the indicator
        symbol : Symbol
            The symbol to calculate volatility for
        short_period : int
            The lookback period for recent volatility
        long_period : int
            The lookback period for historical volatility
            
        Returns:
        --------
        VolatilityRegimeIndicator
            The volatility regime indicator
        """
        indicator = VolatilityRegimeIndicator(name, symbol, short_period, long_period)
        algorithm.register_indicator(symbol, indicator)
        return indicator
        
    @staticmethod
    def trend_strength(algorithm, name, symbol, period):
        """
        Create a trend strength indicator
        
        Parameters:
        -----------
        algorithm : QCAlgorithm
            The algorithm instance
        name : str
            The name of the indicator
        symbol : Symbol
            The symbol to calculate trend strength for
        period : int
            The lookback period for calculating trend strength
            
        Returns:
        --------
        TrendStrengthIndicator
            The trend strength indicator
        """
        indicator = TrendStrengthIndicator(name, symbol, period)
        algorithm.register_indicator(symbol, indicator)
        return indicator
        
    @staticmethod
    def market_regime(algorithm, name, symbol, sma_period, vol_period):
        """
        Create a market regime indicator
        
        Parameters:
        -----------
        algorithm : QCAlgorithm
            The algorithm instance
        name : str
            The name of the indicator
        symbol : Symbol
            The symbol to calculate market regime for
        sma_period : int
            The period for the simple moving average
        vol_period : int
            The period for volatility calculation
            
        Returns:
        --------
        MarketRegimeIndicator
            The market regime indicator
        """
        indicator = MarketRegimeIndicator(name, symbol, sma_period, vol_period)
        algorithm.register_indicator(symbol, indicator)
        return indicator
