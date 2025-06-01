# region imports
from AlgorithmImports import *
from alpha import KalmanFilterStatisticalArbitrageAlphaModel
# endregion

class KalmanFilterStatisticalArbitrageAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_end_date(datetime.now())
        self.set_start_date(self.end_date - timedelta(1*365))
        self.set_cash(100000)

        self.set_brokerage_model(BrokerageName.OANDA_BROKERAGE, AccountType.MARGIN)

        self.universe_settings.resolution = Resolution.MINUTE

        # We focus on major forex pairs
        symbols = [ Symbol.create(pair, SecurityType.FOREX, Market.OANDA) for pair in
            ["AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"] ]
        self.set_universe_selection(ManualUniverseSelectionModel(symbols))

        # A custom alpha model for Kalman Filter prediction and statistical arbitrage signaling
        self.add_alpha(KalmanFilterStatisticalArbitrageAlphaModel())

        # Use the insight weights for sizing, set a very long rebalance period to avoid constant rebalancing
        self.set_portfolio_construction(InsightWeightingPortfolioConstructionModel(Expiry.END_OF_YEAR))
