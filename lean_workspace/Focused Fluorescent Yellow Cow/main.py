from AlgorithmImports import *

class OpeningRangeBreakoutStocksInPlay(QCAlgorithm):

    def Initialize(self):
        # 1) Session settings
        self.SetStartDate(2016, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # 2) Strategy parameters
        self.openingRangeMinutes = 5
        self.universeSize        = 1000
        self.maxPositions        = 20
        self.atrPeriod           = 14
        self.atrThreshold        = 0.50
        self.stopLossRiskPct     = 0.01    # risk 1% per trade
        self.stopLossAtrMult     = 1.0     # 1×ATR

        # 3) Add SPY first (so we can schedule on its hours)
        self.spy = self.AddEquity("SPY", Resolution.MINUTE).Symbol

        # 4) Universe selection at 1-min resolution
        self.AddUniverse(Resolution.MINUTE, self.CoarseSelection)

        # 5) Container for per-symbol state
        self.symbolData = {}

        # 6) Schedule the opening-range scan at 9:00 + 5 minutes
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, self.openingRangeMinutes),
            self.OnOpenRangeEnd
        )

        # 7) Flush at 1 min before close
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.BeforeMarketClose(self.spy, 1),
            self.Liquidate
        )


    def CoarseSelection(self, coarse):
        # pick top universeSize by dollar volume, price > $5
        selected = sorted(
            [c for c in coarse if c.Price > 5 and c.HasFundamentalData],
            key=lambda x: x.DollarVolume,
            reverse=True
        )[: self.universeSize]
        return [c.Symbol for c in selected]


    def OnSecuritiesChanged(self, changes):
        # add new symbols
        for sec in changes.AddedSecurities:
            self.symbolData[sec.Symbol] = SymbolData(
                self, sec.Symbol,
                self.openingRangeMinutes,
                self.atrPeriod
            )
        # remove delisted symbols
        for sec in changes.RemovedSecurities:
            self.symbolData.pop(sec.Symbol, None)


    def OnOpenRangeEnd(self):
        # filter for “stocks in play” & ATR > threshold
        candidates = [
            sd for sd in self.symbolData.values()
            if sd.RelativeVolume > 1 and sd.atr.Current.Value > self.atrThreshold
        ]
        # rank by relative volume
        candidates.sort(key=lambda sd: sd.RelativeVolume, reverse=True)
        # scan top N
        for sd in candidates[: self.maxPositions]:
            sd.Scan()


    def OnData(self, data):
        # feed each incoming TradeBar to its helper
        for sd in self.symbolData.values():
            if data.Bars.ContainsKey(sd.Symbol):
                sd.Update(data.Bars[sd.Symbol])



class SymbolData:
    def __init__(self, algorithm, symbol, openingRangeMin, atrPeriod):
        self.alg       = algorithm
        self.Symbol    = symbol
        self.rangeMin  = openingRangeMin
        self.bars      = []
        self.currentVol= 0

        # ATR (daily)
        self.atr = self.alg.ATR(symbol, atrPeriod, MovingAverageType.Simple, Resolution.DAILY)
        self.alg.RegisterIndicator(symbol, self.atr, Resolution.DAILY)

        # pre-compute avg 5-min volume from 14d of daily bars
        hist = self.alg.History(symbol, 14, Resolution.DAILY)
        if not hist.empty:
            avgDaily = hist["volume"].mean()
            self.avgVol = avgDaily * (openingRangeMin / 390)
        else:
            self.avgVol = 0

        self.entryOrder = None
        self.stopPrice  = None


    @property
    def RelativeVolume(self):
        return (self.currentVol / self.avgVol) if self.avgVol > 0 else 0


    def Update(self, bar: TradeBar):
        # collect opening-range bars & volume
        if bar.Time.hour == 9 and bar.Time.minute < self.rangeMin:
            self.bars.append(bar)
            self.currentVol += bar.Volume

        # enforce stop-loss
        if self.entryOrder and self.stopPrice is not None:
            q = self.entryOrder.Quantity
            if q > 0 and bar.Low  <= self.stopPrice: self.alg.Liquidate(self.Symbol)
            if q < 0 and bar.High >= self.stopPrice: self.alg.Liquidate(self.Symbol)


    def Scan(self):
        # need full opening range
        if len(self.bars) < self.rangeMin: return

        highs = [b.High for b in self.bars]
        lows  = [b.Low  for b in self.bars]
        first = self.bars[0].Open

        orHigh, orLow = max(highs), min(lows)

        # breakout direction
        if self.bars[-1].Close > first:
            entry = orHigh
            stop  = orHigh - self.alg.stopLossAtrMult * self.atr.Current.Value
            sign  =  1
        else:
            entry = orLow
            stop  = orLow  + self.alg.stopLossAtrMult * self.atr.Current.Value
            sign  = -1

        riskPerShare = abs(entry - stop)
        if riskPerShare == 0: return

        riskCap = self.alg.Portfolio.TotalPortfolioValue * self.alg.stopLossRiskPct
        qty     = int(riskCap / riskPerShare) * sign

        # cap by equal-weight
        maxQty  = abs(self.alg.CalculateOrderQuantity(self.Symbol, 1/self.alg.maxPositions))
        qty     = max(-maxQty, min(qty, maxQty))
        if qty == 0: return

        self.stopPrice  = stop
        self.entryOrder = self.alg.StopMarketOrder(self.Symbol, qty, entry)
