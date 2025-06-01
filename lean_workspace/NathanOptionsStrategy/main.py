from AlgorithmImports import *
from scipy.optimize import minimize

class NathanOptionsOnlyStrategy(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2025, 3, 3)
        self.set_cash(50000)

        self.underlyings = ["TQQQ", "SVXY", "VXZ", "TMF", "EDZ", "UGL"]
        self.option_symbols = []

        for ticker in self.underlyings:
            option = self.add_option(ticker)
            option.set_filter(lambda u: u.strikes(-10, 5).expiration(20, 40))
            self.option_symbols.append(option.symbol)

        self.last_rebalance_time = datetime.min
        self.rebalance_interval = timedelta(days=1)

        self.targets = {}

    def constraint_settings(self):
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def run_risk_momentum_analysis(self, returns, bounds, x0, constraints):
        opt = minimize(
            lambda w: 0.5 * (w.T @ returns.cov() @ w) - x0 @ w,
            x0=x0,
            constraints=constraints,
            bounds=bounds,
            tol=1e-8,
            method="SLSQP"
        )
        return opt

    def select_deep_itm_call(self, chain, spot_price):
        # Select deep ITM call: strike < spot, delta > 0.8
        calls = [x for x in chain if x.right == OptionRight.CALL and x.strike < spot_price]
        if not calls:
            return None

        # Prefer highest delta and closest to spot price
        sorted_calls = sorted(calls, key=lambda x: abs(x.strike - spot_price))
        return sorted_calls[0] if sorted_calls else None

    def rebalance(self, data):
        self.liquidate()  # Liquidate old positions

        # Calculate optimized weights
        ret = self.history([Symbol.create(ticker, SecurityType.Equity, Market.USA) for ticker in self.underlyings],
                           252, Resolution.DAILY).close.pct_change().dropna()
        
        x0 = [1 / ret.shape[1]] * ret.shape[1]
        constraints = self.constraint_settings()
        bounds = [(0, 1)] * ret.shape[1]
        opt = self.run_risk_momentum_analysis(ret, bounds, x0, constraints)

        # Now, pick deep ITM call options for each underlying
        available_chains = {s: data.option_chains.get(s) for s in self.option_symbols}

        for idx, (underlying, opt_weight) in enumerate(zip(self.underlyings, opt.x)):
            option_symbol = self.option_symbols[idx]
            chain = available_chains.get(option_symbol)
            if chain is None or len(chain) == 0:
                self.debug(f"No option chain for {underlying} today.")
                continue

            underlying_price = self.securities[Symbol.create(underlying, SecurityType.Equity, Market.USA)].price
            contract = self.select_deep_itm_call(chain, underlying_price)
            if contract is None:
                self.debug(f"No deep ITM call found for {underlying}")
                continue

            option_price = self.securities[contract.symbol].price
            if option_price <= 0:
                self.debug(f"Option price invalid for {contract.symbol}")
                continue

            # Allocate capital proportionally
            budget = self.portfolio.total_portfolio_value * opt_weight
            quantity = int(budget / (option_price * 100))  # 100x multiplier
            if quantity > 0:
                self.market_order(contract.symbol, quantity)
                self.debug(f"Bought {quantity}x {contract.symbol}")

        self.last_rebalance_time = self.time

    def on_data(self, data):
        if (self.time - self.last_rebalance_time) >= self.rebalance_interval:
            self.rebalance(data)
