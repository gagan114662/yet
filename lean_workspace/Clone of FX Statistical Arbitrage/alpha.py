#region imports
from AlgorithmImports import *
from arch.unitroot.cointegration import engle_granger
from pykalman import KalmanFilter
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import VECM
#endregion

class KalmanFilterStatisticalArbitrageAlphaModel(AlphaModel):

    def __init__(self, lookback = 30, recalibrateInterval = timedelta(7)):
        self.lookback = lookback
        self.recalibrate_interval = recalibrateInterval

        self.symbol_data = {}
        self.kalman_filter = None
        self.current_mean = None
        self.current_cov = None
        self.threshold = None
        self.trading_weight = {}
        self.state = 0

        self.rebalance_time = datetime.min

    def update(self, algorithm, data):
        insights = []

        if algorithm.time > self.rebalance_time:
            self.recalibrate()
            self.rebalance_time = algorithm.time + self.recalibrate_interval

        if not self.trading_weight: return insights
        
        # We take the latest cached price as the next time point input (does not matter if no new data point)
        data = [np.log(algorithm.securities[symbol].close) * self.trading_weight[symbol] 
            for symbol in self.symbol_data.keys() if symbol in self.trading_weight]
        spread = np.product(data)

        if not spread: return insights
            
        # If all pairs got consolidated data and updated their daily price, we update the Kalman Filter
        if all([data.updated for data in self.symbol_data.values()]):
            # Update the Kalman Filter with the spread
            (self.current_mean, self.current_cov) = self.kalman_filter.filter_update(filtered_state_mean = self.current_mean,
                                                                                  filtered_state_covariance = self.current_cov,
                                                                                  observation = spread)
            # reset the flag
            for data in self.symbol_data.values():
                data.updated = False

        # Obtain the normalized spread
        normalized_spread = spread - self.current_mean
        
        # Mean-reversion
        if normalized_spread < -self.threshold and self.state != 1:
            for symbol, weight in self.trading_weight.items():
                if algorithm.is_market_open(symbol):
                    insights.append(
                        Insight.price(symbol, timedelta(365), InsightDirection.UP, weight=weight))
            self.state = 1
            
        elif normalized_spread > self.threshold and self.state != -1:
            for symbol, weight in self.trading_weight.items():
                if algorithm.is_market_open(symbol):
                    insights.append(
                        Insight.price(symbol, timedelta(365), InsightDirection.DOWN, weight=weight))
            self.state = -1
                
        # Out of position if spread converged
        elif (self.state == 1 and normalized_spread > 0) or (self.state == -1 and normalized_spread < 0):
            algorithm.insights.cancel(list(self.symbol_data.keys()))
            self.state = 0
    
        return insights

    def recalibrate(self):
        # Get log price series of all signaled assets
        log_price = np.log(
            pd.DataFrame({symbol: data.price for symbol, data in self.symbol_data.items() if data.is_ready}))
        
        if log_price.empty: return

        # Get the weighted spread across different cointegration subspaces
        weighted_spread, weights, beta = self.get_spreads(log_price)
        
        # Set up the Kalman Filter with the weighted spread series, and obtain the adjusted mean series
        mean_series = self.set_kalman_filter(weighted_spread)

        # Obtain the normalized spread series, the first 20 in-sample will be discarded.
        normalized_spread = (weighted_spread.iloc[20:] - mean_series)

        # Set the threshold of price divergence to optimize profit
        self.set_trading_threshold(normalized_spread)

        # Set the normalize trading weight
        weights = self.get_trading_weight(beta, weights)
        for symbol, weight in zip(log_price.columns, weights):
            self.trading_weight[symbol] = weight

    def get_spreads(self, logPriceDf):
        # Initialize a VECM model following the unit test parameters, then fit to our data.
        # We allow 3 AR difference, and no deterministic term.
        vecm_result = VECM(logPriceDf, k_ar_diff=3, coint_rank=logPriceDf.shape[1]-1, deterministic='n').fit()
        # Obtain the Beta attribute. This is the cointegration subspaces' unit vectors.
        beta = vecm_result.beta
        # get the spread of different cointegration subspaces.
        spread = logPriceDf @ beta
        # Optimize the distribution across cointegration subspaces and return the weighted spread
        return self.optimize_spreads(spread, beta)

    def optimize_spreads(self, spread, beta):
        # We set the weight on each vector is between -1 and 1. While overall sum is 0.
        x0 = np.array([-1**i / beta.shape[1] for i in range(beta.shape[1])])
        bounds = tuple((-1, 1) for i in range(beta.shape[1]))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x)}]
        
        # Optimize the Portmanteau statistics
        opt = minimize(lambda w: ((w.T @ np.cov(spread.T, spread.shift(1).fillna(0).T)[spread.shape[1]:, :spread.shape[1]] @ w)\
                                 / (w.T @ np.cov(spread.T) @ w))**2,
                        x0=x0,
                        bounds=bounds,
                        constraints=constraints,
                        method="SLSQP")
        
        # Normalize the result
        opt.x = opt.x / np.sum(abs(opt.x))
        # Return the weighted spread series
        return spread @ opt.x, opt.x, beta

    def set_kalman_filter(self, weighted_spread):
        # Initialize a Kalman Filter. Using the first 20 data points to optimize its initial state. 
        # We assume the market has no regime change so that the transitional matrix and observation matrix is [1].
        self.kalman_filter = KalmanFilter(transition_matrices = [1],
                            observation_matrices = [1],
                            initial_state_mean = weighted_spread.iloc[:20].mean(),
                            observation_covariance = weighted_spread.iloc[:20].var(),
                            em_vars=['transition_covariance', 'initial_state_covariance'])
        self.kalman_filter = self.kalman_filter.em(weighted_spread.iloc[:20], n_iter=5)
        (filtered_state_means, filtered_state_covariances) = self.kalman_filter.filter(weighted_spread.iloc[:20])
        
        # Obtain the current Mean and Covariance Matrix expectations.
        self.current_mean = filtered_state_means[-1, :]
        self.current_cov = filtered_state_covariances[-1, :]
        
        # Initialize a mean series for spread normalization using the Kalman Filter's results.
        mean_series = np.array([None]*(weighted_spread.shape[0]-20))
        
        # Roll over the Kalman Filter to obtain the mean series.
        for i in range(20, weighted_spread.shape[0]):
            (self.current_mean, self.current_cov) = self.kalman_filter.filter_update(filtered_state_mean = self.current_mean,
                                                                    filtered_state_covariance = self.current_cov,
                                                                    observation = weighted_spread.iloc[i])
            mean_series[i-20] = float(self.current_mean)

        return mean_series

    def set_trading_threshold(self, normalized_spread):
        # Initialize 20 set levels for testing.
        s0 = np.linspace(0, max(normalized_spread), 20)
        
        # Calculate the profit levels using the 20 set levels.
        f_bar = np.array([None] * 20)
        for i in range(20):
            f_bar[i] = len(normalized_spread.values[normalized_spread.values > s0[i]]) \
                / normalized_spread.shape[0]
            
        # Set trading frequency matrix.
        D = np.zeros((19, 20))
        for i in range(D.shape[0]):
            D[i, i] = 1
            D[i, i+1] = -1
            
        # Set level of lambda.
        l = 1.0
        
        # Obtain the normalized profit level.
        f_star = np.linalg.inv(np.eye(20) + l * D.T @ D) @ f_bar.reshape(-1, 1)
        s_star = [f_star[i] * s0[i] for i in range(20)]
        self.threshold = s0[s_star.index(max(s_star))]

    def get_trading_weight(self, beta, weights):
        trading_weight = beta @ weights
        return trading_weight / np.sum(abs(trading_weight))

    def on_securities_changed(self, algorithm, changes):
        for removed in changes.removed_securities:
            symbolData = self.symbol_data.pop(removed.symbol, None)
            if symbolData:
                symbolData.dispose()

        for added in changes.added_securities:
            symbol = added.symbol
            if symbol not in self.symbol_data and added.type == SecurityType.FOREX:
                self.symbol_data[symbol] = SymbolData(algorithm, symbol, self.lookback)
    
class SymbolData:

    def __init__(self, algorithm, symbol, lookback):
        self.algorithm = algorithm
        self.symbol = symbol
        self.lookback = lookback
        self.updated = False

        # To store the historical daily log return
        self.window = RollingWindow[IndicatorDataPoint](lookback)

        # Use daily log return to predict cointegrating vector
        self.consolidator = QuoteBarConsolidator(timedelta(hours=1))
        self.price_ = Identity(f"{symbol} Price")
        self.price_.updated += self.on_update

        # Subscribe the consolidator and indicator to data for automatic update
        algorithm.register_indicator(symbol, self.price_, self.consolidator)
        algorithm.subscription_manager.add_consolidator(symbol, self.consolidator)

        # historical warm-up on the log return indicator
        history = algorithm.history[QuoteBar](self.symbol, self.lookback, Resolution.HOUR)
        for bar in history:
            self.consolidator.update(bar)

    def on_update(self, sender, updated):
        self.window.add(IndicatorDataPoint(updated.end_time, updated.value))
        self.updated = True

    def dispose(self):
        self.price_.updated -= self.on_update
        self.price_.reset()
        self.window.reset()
        self.algorithm.subscription_manager.remove_consolidator(self.symbol, self.consolidator)

    @property
    def is_ready(self):
        return self.window.is_ready

    @property
    def price(self):
        return pd.Series(
            data = [x.value for x in self.window],
            index = [x.end_time for x in self.window])[::-1]
