
from AlgorithmImports import *

class BreakoutScalpingStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Trade multiple liquid ETFs for more opportunities
        self.symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        self.securities = {}
        self.indicators = {}
        
        for symbol in self.symbols:
            security = self.AddEquity(symbol, Resolution.Hour)
            security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            self.securities[symbol] = security
            
            # Breakout detection indicators
            self.indicators[symbol] = {
                "sma_short": self.SMA(symbol, 5),
                "sma_long": self.SMA(symbol, 20),
                "atr": self.ATR(symbol, 14),
                "bb": self.BB(symbol, 20, 1.5),  # Tighter bands
                "rsi": self.RSI(symbol, 10),     # Faster RSI
                "momentum": self.MOMP(symbol, 5)  # Very short momentum
            }
            
        self.trade_count = 0
        self.positions = {}
        self.stop_losses = {}
        
        # Schedule frequent checks
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(timedelta(hours=2)),
                        self.CheckBreakouts)
                        
    def CheckBreakouts(self):
        for symbol in self.symbols:
            if all(indicator.IsReady for indicator in self.indicators[symbol].values()):
                self.ProcessSymbol(symbol)
                
    def ProcessSymbol(self, symbol):
        indicators = self.indicators[symbol]
        price = self.Securities[symbol].Price
        
        sma_short = indicators["sma_short"].Current.Value
        sma_long = indicators["sma_long"].Current.Value
        atr = indicators["atr"].Current.Value
        bb_upper = indicators["bb"].UpperBand.Current.Value
        bb_lower = indicators["bb"].LowerBand.Current.Value
        bb_middle = indicators["bb"].MiddleBand.Current.Value
        rsi = indicators["rsi"].Current.Value
        momentum = indicators["momentum"].Current.Value
        
        current_holdings = self.Portfolio[symbol].Quantity
        
        # UPSIDE BREAKOUT CONDITIONS
        if (price > bb_upper and 
            sma_short > sma_long and 
            momentum > 0.005 and 
            rsi > 50 and rsi < 80 and
            current_holdings <= 0):
            
            # Enter long position
            position_size = 0.15  # Smaller positions for more trades
            self.SetHoldings(symbol, position_size)
            self.stop_losses[symbol] = price - (2 * atr)  # 2 ATR stop
            self.trade_count += 1
            
        # DOWNSIDE BREAKOUT CONDITIONS  
        elif (price < bb_lower and
              sma_short < sma_long and
              momentum < -0.005 and
              rsi < 50 and rsi > 20 and
              current_holdings >= 0):
              
            # Enter short position
            position_size = -0.15
            self.SetHoldings(symbol, position_size)
            self.stop_losses[symbol] = price + (2 * atr)  # 2 ATR stop
            self.trade_count += 1
            
        # PROFIT TAKING / STOP MANAGEMENT
        elif current_holdings != 0:
            # Long position management
            if current_holdings > 0:
                # Take profit at middle band or stop loss
                if price <= bb_middle or price <= self.stop_losses.get(symbol, 0):
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    if symbol in self.stop_losses:
                        del self.stop_losses[symbol]
                        
            # Short position management  
            elif current_holdings < 0:
                # Take profit at middle band or stop loss
                if price >= bb_middle or price >= self.stop_losses.get(symbol, float('inf')):
                    self.Liquidate(symbol)
                    self.trade_count += 1
                    if symbol in self.stop_losses:
                        del self.stop_losses[symbol]
                        
    def OnData(self, data):
        # Additional intraday momentum trades
        for symbol in self.symbols:
            if symbol in data and self.indicators[symbol]["momentum"].IsReady:
                momentum = self.indicators[symbol]["momentum"].Current.Value
                current_holdings = self.Portfolio[symbol].Quantity
                
                # Quick momentum scalps
                if abs(momentum) > 0.01 and abs(current_holdings) < 0.05:
                    quick_position = 0.1 if momentum > 0 else -0.1
                    self.SetHoldings(symbol, quick_position)
                    self.trade_count += 1
                    
    def OnEndOfAlgorithm(self):
        years = (self.EndDate - self.StartDate).days / 365.25
        trades_per_year = self.trade_count / years
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Trades Per Year: {trades_per_year:.1f}")
