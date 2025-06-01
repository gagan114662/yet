# region imports
from AlgorithmImports import *
# endregion
using System;
using System.Collections.Generic;
using QuantConnect;
using QuantConnect.Algorithm;
using QuantConnect.Indicators;
using QuantConnect.Data.UniverseSelection;

public class AdvancedRotationStrategy : QCAlgorithm
{
    private readonly string[] _etfs = { "SPY", "QQQ", "EEM", "TLT", "VEA", "IEF" };
    private readonly Dictionary<Symbol, SimpleMovingAverage> _sma = new();
    private readonly Dictionary<Symbol, RelativeStrengthIndex> _rsi = new();
    private readonly Dictionary<Symbol, AverageTrueRange> _atr = new();
    private readonly Dictionary<Symbol, decimal> _highestHighs = new();
    private const int SmaPeriod = 50;
    private const int RsiPeriod = 14;
    private const int AtrPeriod = 14;
    private const decimal RsiBuyThreshold = 30;
    private const decimal RsiExitThreshold = 50;
    private const decimal AtrMultiplier = 2m;

    public override void Initialize()
    {
        SetStartDate(2008, 1, 1);
        SetEndDate(2023, 1, 1);
        SetCash(100000);

        // Optionally, add transaction costs
        SetSecurityInitializer(security => 
            security.SetFeeModel(new ConstantPerTradeFeeModel(1m)));

        foreach (var ticker in _etfs)
        {
            var symbol = AddEquity(ticker, Resolution.Daily).Symbol;
            _sma[symbol] = SMA(symbol, SmaPeriod);
            _rsi[symbol] = RSI(symbol, RsiPeriod);
            _atr[symbol] = ATR(symbol, AtrPeriod);
        }

        Schedule.On(DateRules.Weekly(), TimeRules.AfterMarketOpen("SPY", 30), Rebalance);
    }

    private void Rebalance()
    {
        if (_sma.Values.All(x => !x.IsReady) || _rsi.Values.All(x => !x.IsReady)) return;

        // Exit logic
        foreach (var kvp in Portfolio)
        {
            var symbol = kvp.Key;
            if (!Portfolio[symbol].Invested) continue;

            var currentPrice = Securities[symbol].Close;
            var currentRsi = _rsi[symbol].Current.Value;
            var currentAtr = _atr[symbol].Current.Value;

            if (currentRsi >= RsiExitThreshold)
            {
                Liquidate(symbol);
                _highestHighs.Remove(symbol);
                continue;
            }

            if (_highestHighs.TryGetValue(symbol, out var highPoint))
            {
                var stopPrice = highPoint - (AtrMultiplier * currentAtr);
                if (currentPrice < stopPrice)
                {
                    Liquidate(symbol);
                    _highestHighs.Remove(symbol);
                }
            }

            if (currentPrice > _highestHighs.GetValueOrDefault(symbol, 0))
                _highestHighs[symbol] = currentPrice;
        }

        // Entry logic
        var candidates = new List<Symbol>();
        foreach (var symbol in _etfs.Select(s => Symbol.Create(s, SecurityType.Equity, Market.USA)))
        {
            if (Portfolio[symbol].Invested) continue;
            if (!_sma[symbol].IsReady || !_rsi[symbol].IsReady) continue;

            var price = Securities[symbol].Close;
            if (price > _sma[symbol].Current.Value && _rsi[symbol].Current.Value < RsiBuyThreshold)
            {
                candidates.Add(symbol);
            }
        }

        var weight = 1m / Math.Max(1, candidates.Count);
        foreach (var symbol in candidates)
        {
            SetHoldings(symbol, weight);
            _highestHighs[symbol] = Securities[symbol].High;
        }
    }
}
