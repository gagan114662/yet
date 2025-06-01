from AlgorithmImports import *

class ParabolicShortTest(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2007,1,1)
        self.SetEndDate(2023,12,31)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        # TEST parameters
        self.lookback       = 5
        self.threshold      = 0.20    # 20% in 5 days for testing
        self.stopLossBuffer = 0.02
        self.targetMA       = 20
        self.riskPerSymbol  = 0.10

        self.stopOrders = {}
        self.inds       = {}

        self.AddUniverse(self.CoarseSelection)

    def CoarseSelection(self, coarse):
        # broaden universe for more runners
        return [c.Symbol for c in coarse if c.HasFundamentalData and c.Price>1]

    def OnSecuritiesChanged(self, changes):
        for removed in changes.RemovedSecurities:
            self.stopOrders.pop(removed.Symbol, None)
            self.inds.pop(removed.Symbol, None)
        for added in changes.AddedSecurities:
            sym = added.Symbol
            self.inds[sym] = self.SMA(sym, self.targetMA, Resolution.Daily)
            self.History(sym, self.targetMA, Resolution.Daily)

    def OnData(self, data: Slice):
        # Exit fills
        for sym, tgt in list(self.stopOrders.items()):
            if sym in data.Bars and data.Bars[sym].Close <= tgt:
                self.Liquidate(sym)
                del self.stopOrders[sym]

        # Entry scan
        for sym in self.ActiveSecurities.Keys:
            if self.Portfolio[sym].Invested or sym not in data.Bars: continue

            hist = self.History(sym, self.lookback+1, Resolution.Daily).loc[sym]
            if hist.shape[0] < self.lookback+1: continue

            past_return = hist['close'][-2]/hist['close'][0] - 1
            self.Debug(f"{sym.Value} 5d={past_return:.2%}")    # see what you're filtering

            if past_return < self.threshold:
                continue

            # place short
            qty = self.CalculateOrderQuantity(sym, -self.riskPerSymbol)
            self.MarketOrder(sym, qty)
            stop_price = hist['high'][-2] * (1 + self.stopLossBuffer)
            self.StopMarketOrder(sym, -qty, stop_price)
            self.stopOrders[sym] = self.inds[sym].Current.Value
            self.Debug(f"SHORT {sym.Value} ret={past_return:.2%} stop={stop_price:.2f}")
