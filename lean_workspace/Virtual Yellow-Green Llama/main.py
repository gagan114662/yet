from AlgorithmImports import *

class GlobalMultiAssetMomentum(QCAlgorithm):
    
    def Initialize(self):
        ### Algorithm Parameters ###
        self.SetStartDate(2008, 1, 1)    # Start of backtest period (15+ years)
        self.SetEndDate(2023, 4, 1)      # End of backtest period
        self.SetCash(100000)            # Starting capital
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)  # margin for shorting
        
        # List of asset symbols to trade (diverse asset classes)
        symbols = [
            "SPY",   # S&P 500 ETF (Equity)
            "QQQ",   # Nasdaq 100 ETF (Equity)
            "IWM",   # Russell 2000 ETF (Equity)
            "EFA",   # EAFE Intl Equity ETF
            "EEM",   # Emerging Markets Equity ETF
            "TLT",   # 20+ Year Treasury Bond ETF
            "GLD",   # Gold ETF
            "USO",   # Crude Oil ETF (or use DBC commodity index ETF)
            "VNQ"    # Real Estate (REIT) ETF
            # Optionally add crypto: e.g. BTCUSD, but QC requires specific add:
        ]
        
        # Add assets with daily resolution data
        self.assets = []
        for sym in symbols:
            if sym == "BTCUSD":
                # For crypto (if included), use AddCrypto and set brokerage model if needed
                asset = self.AddCrypto(sym, Resolution.Daily, Market.Bitfinex).Symbol
            else:
                asset = self.AddEquity(sym, Resolution.Daily).Symbol
            self.assets.append(asset)
        
        # Technical Indicators: 100-day simple moving average for each asset
        self.sma = { symbol: self.SMA(symbol, 100, Resolution.Daily) for symbol in self.assets }
        
        # Warm up the SMA indicators with history
        history = self.History(self.assets, 100, Resolution.Daily)
        for symbol in self.assets:
            for time, row in history.loc[symbol].iterrows():
                self.sma[symbol].Update(time, row["close"])
        
        # Trailing stop tracking
        self.highest_price = { symbol: 0 for symbol in self.assets }   # for longs
        self.lowest_price  = { symbol: float('inf') for symbol in self.assets }  # for shorts
        
        # Risk management parameters
        self.maxGrossLeverage = 1.5   # maximum total gross leverage (150% exposure)
        self.trailing_pct = 0.15      # 15% trailing stop
        
        self.Log("Initialized strategy with assets: " + ", ".join([str(x) for x in self.assets]))
    
    def OnData(self, data: Slice):
        # Ensure indicators are ready
        if self.IsWarmingUp:
            return
        
        # Calculate volatility (e.g. 60-day ATR % or std dev) for each asset
        vol = {}
        for symbol in self.assets:
            # Use ATR as volatility proxy (percentage of price)
            # Alternatively, could use standard deviation of returns
            atr_indicator = self.ATR(symbol, 20, MovingAverageType.Simple)
            # Need to update ATR with history or check if ready
            if not atr_indicator.IsReady:
                continue  # skip until ATR ready
            atr = atr_indicator.Current.Value
            price = data[symbol].Close if symbol in data and data[symbol] else None
            if price:
                vol[symbol] = atr / price  # ATR as fraction of price (approx vol)
        
        if len(vol) == 0:
            return  # wait until we have volatility data for all
        
        # Determine raw target weights based on momentum signal and inverse volatility
        target_weights = {}
        for symbol in self.assets:
            if symbol not in data or data[symbol] is None:
                continue  # no data update for this symbol
            
            price = data[symbol].Close
            sma = self.sma[symbol]
            if not sma.IsReady:
                continue
            
            current_holdings = self.Portfolio[symbol].Quantity
            trend_up = price > sma.Current.Value    # bullish trend signal
            trend_down = price < sma.Current.Value  # bearish trend signal
            
            # Update trailing price extremes for any existing positions
            if current_holdings > 0:  # currently long
                # update highest price seen
                self.highest_price[symbol] = max(self.highest_price[symbol], price, sma.Current.Value)
                # Check trailing stop for long
                if price < self.highest_price[symbol] * (1 - self.trailing_pct):
                    # Price dropped more than trailing_pct from high – exit long
                    self.Liquidate(symbol, "Trailing stop hit for long")
                    continue  # skip further processing this symbol this time
            elif current_holdings < 0:  # currently short
                # update lowest price seen
                self.lowest_price[symbol] = min(self.lowest_price[symbol], price, sma.Current.Value)
                # Check trailing stop for short
                if price > self.lowest_price[symbol] * (1 + self.trailing_pct):
                    # Price rose more than trailing_pct from low – exit short
                    self.Liquidate(symbol, "Trailing stop hit for short")
                    continue
        
            # Determine desired position (long, short, or zero) based on trend
            desired_weight = 0
            if trend_up and not trend_down:   # (we use strict > or <, no ==, so one is True)
                # Uptrend signal – target long weight proportional to 1/vol
                desired_weight = 1.0 / vol.get(symbol, 1) 
            elif trend_down and not trend_up:
                # Downtrend signal – target short weight proportional to 1/vol (negative weight)
                desired_weight = -1.0 / vol.get(symbol, 1)
            else:
                # If price == SMA (rare) or no clear trend, go flat
                desired_weight = 0
            
            target_weights[symbol] = desired_weight
        
        if len(target_weights) == 0:
            return
        
        # Normalize weights to target max gross leverage
        # Calculate current sum of absolute weights
        total_abs = sum(abs(w) for w in target_weights.values())
        if total_abs > 1e-6:
            # Scale all weights to achieve desired gross exposure (<= maxGrossLeverage)
            # First, normalize to 1.0 gross exposure:
            for symbol, w in target_weights.items():
                target_weights[symbol] = w / total_abs
            # Then scale to maxGrossLeverage (so sum(abs) = maxGrossLeverage)
            for symbol, w in target_weights.items():
                target_weights[symbol] = w * self.maxGrossLeverage
        
        # Execute trades to adjust holdings
        for symbol, weight in target_weights.items():
            current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
            # If significant change in desired weight or reversing position, execute trade
            if abs(weight - current_weight) > 0.02:  # threshold to avoid tiny adjustments
                # Use SetHoldings to automatically manage ordering and crossing zero
                self.SetHoldings(symbol, weight)
                # Reset trailing extreme for new positions
                if weight > 0:
                    # New long or increased long position
                    self.highest_price[symbol] = data[symbol].Close  # reset high at entry price
                    self.lowest_price[symbol] = float('inf')         # reset short tracker
                elif weight < 0:
                    # New short position
                    self.lowest_price[symbol] = data[symbol].Close   # reset low at entry
                    self.highest_price[symbol] = 0
                else:
                    # Exiting to zero
                    self.highest_price[symbol] = 0
                    self.lowest_price[symbol] = float('inf')
