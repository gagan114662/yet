from AlgorithmImports import *

class RiskFreeOptionHedge(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2018, 2, 1)
        self.SetCash(10000)
        
        self.equity = self.AddEquity("AAPL", Resolution.Minute)
        self.option = self.AddOption("AAPL", Resolution.Minute)
        self.option.SetFilter(-2, 2, 10, 30)
        
        self.long_option = None
        self.short_stock = None
        self.initial_balance = None

    def OnData(self, data):
        if not self.Portfolio.Invested and self.long_option is None:
            option_chain = data.OptionChains.get(self.option.Symbol, None)
            if option_chain:
                contracts = sorted(option_chain, key=lambda x: abs(x.Strike - data[self.equity.Symbol].Price))
                self.Debug({'time':self.Time, 'contracts':contracts[0]})
                if len(contracts) >= 5:
                    selected_contract = contracts[0]
                    option_price = selected_contract.LastPrice
                    stock_price = data[self.equity.Symbol].Price
                    hedge_ratio = 100 / stock_price
                    
                    self.long_option = self.MarketOrder(selected_contract.Symbol, 1)
                    self.short_stock = self.MarketOrder(self.equity.Symbol, -hedge_ratio)
                    self.initial_balance = self.Portfolio.Cash
        
        if self.long_option and self.short_stock:
            current_balance = self.Portfolio.Cash
            if current_balance - self.initial_balance >= 5:
                self.Liquidate()
