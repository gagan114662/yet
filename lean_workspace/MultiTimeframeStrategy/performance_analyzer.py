import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class PerformanceAnalyzer:
    """
    Utility class to analyze the performance of the multi-timeframe strategy
    by processing backtest results from QuantConnect.
    """
    
    def __init__(self, backtest_dir):
        """
        Initialize the analyzer with the backtest directory
        
        Parameters:
        -----------
        backtest_dir : str
            Path to the backtest directory containing results
        """
        self.backtest_dir = backtest_dir
        self.orders = None
        self.equity_curve = None
        self.trades_by_timeframe = {
            'daily': [],
            'weekly': [],
            'monthly': []
        }
        
    def load_backtest_data(self):
        """Load order and equity data from backtest results"""
        # Find the most recent backtest
        backtest_dirs = [d for d in os.listdir(self.backtest_dir) 
                        if os.path.isdir(os.path.join(self.backtest_dir, d)) and d.startswith('20')]
        
        if not backtest_dirs:
            print("No backtest directories found")
            return False
            
        # Sort by date (newest first)
        backtest_dirs.sort(reverse=True)
        latest_backtest = os.path.join(self.backtest_dir, backtest_dirs[0])
        
        # Find order events file
        order_files = [f for f in os.listdir(latest_backtest) if f.endswith('order-events.json')]
        if not order_files:
            print("No order events file found")
            return False
            
        # Load order events
        order_file = os.path.join(latest_backtest, order_files[0])
        with open(order_file, 'r') as f:
            self.orders = json.load(f)
            
        # Find equity curve file
        equity_files = [f for f in os.listdir(latest_backtest) if f.endswith('equity.json')]
        if equity_files:
            equity_file = os.path.join(latest_backtest, equity_files[0])
            with open(equity_file, 'r') as f:
                self.equity_curve = json.load(f)
                
        print(f"Loaded data from backtest: {backtest_dirs[0]}")
        return True
        
    def classify_trades_by_timeframe(self):
        """Classify trades by timeframe based on order properties"""
        if not self.orders:
            print("No order data loaded")
            return
            
        # Group orders by ID to form complete trades
        orders_by_id = {}
        for order in self.orders:
            if order['status'] == 'filled':
                order_id = order['orderId']
                if order_id not in orders_by_id:
                    orders_by_id[order_id] = []
                orders_by_id[order_id].append(order)
        
        # Process each order to determine timeframe
        for order_id, order_events in orders_by_id.items():
            # Get the first filled order event
            order = order_events[0]
            
            # Convert timestamp to datetime
            order_time = datetime.fromtimestamp(order['time'])
            symbol = order['symbolPermtick']
            
            # Classify based on order properties and timing
            # This is a heuristic approach - in a real system, orders would be tagged with their strategy
            if "SPY" in symbol and order['quantity'] < 100:
                # Small SPY orders are likely from daily RSI strategy
                self.trades_by_timeframe['daily'].append(order)
            elif "GLD" in symbol:
                # GLD trades are from daily Bollinger Bands strategy
                self.trades_by_timeframe['daily'].append(order)
            elif ("QQQ" in symbol or "EFA" in symbol or "TLT" in symbol) and order_time.weekday() == 0:
                # QQQ/EFA/TLT trades on Mondays are likely from weekly strategy
                self.trades_by_timeframe['weekly'].append(order)
            elif order_time.day <= 5:
                # Orders in first 5 days of month are likely monthly strategy
                self.trades_by_timeframe['monthly'].append(order)
            else:
                # Default to daily for other orders
                self.trades_by_timeframe['daily'].append(order)
                
        # Print summary
        print("\nTrades by Timeframe:")
        for timeframe, trades in self.trades_by_timeframe.items():
            print(f"  {timeframe.capitalize()}: {len(trades)} trades")
            
    def analyze_performance(self):
        """Analyze performance metrics by timeframe"""
        if not self.orders:
            print("No order data loaded")
            return
            
        # Calculate overall statistics
        total_orders = len([o for o in self.orders if o['status'] == 'filled'])
        unique_symbols = set([o['symbolPermtick'] for o in self.orders])
        
        print("\nOverall Performance:")
        print(f"  Total Orders: {total_orders}")
        print(f"  Unique Symbols Traded: {len(unique_symbols)}")
        print(f"  Symbols: {', '.join(unique_symbols)}")
        
        # Calculate fees
        total_fees = sum([float(o.get('orderFeeAmount', 0)) for o in self.orders])
        print(f"  Total Fees: ${total_fees:.2f}")
        
        # Analyze equity curve if available
        if self.equity_curve:
            equity_series = pd.Series(self.equity_curve['equity'])
            returns = equity_series.pct_change().dropna()
            
            print("\nEquity Performance:")
            print(f"  Starting Equity: ${equity_series.iloc[0]:.2f}")
            print(f"  Ending Equity: ${equity_series.iloc[-1]:.2f}")
            print(f"  Total Return: {(equity_series.iloc[-1]/equity_series.iloc[0] - 1)*100:.2f}%")
            print(f"  Sharpe Ratio: {returns.mean()/returns.std() * np.sqrt(252):.2f}")
            print(f"  Max Drawdown: {self.calculate_max_drawdown(equity_series)*100:.2f}%")
            
    def calculate_max_drawdown(self, equity_series):
        """Calculate maximum drawdown from equity curve"""
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown.min()
        
    def plot_equity_curve(self):
        """Plot equity curve with annotations for different timeframe trades"""
        if not self.equity_curve:
            print("No equity curve data available")
            return
            
        # Convert to DataFrame
        equity_df = pd.DataFrame({
            'date': [datetime.fromtimestamp(t) for t in self.equity_curve['time']],
            'equity': self.equity_curve['equity']
        })
        equity_df.set_index('date', inplace=True)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'], label='Portfolio Equity')
        
        # Add markers for trades by timeframe
        colors = {'daily': 'green', 'weekly': 'blue', 'monthly': 'red'}
        for timeframe, trades in self.trades_by_timeframe.items():
            if trades:
                trade_times = [datetime.fromtimestamp(t['time']) for t in trades]
                trade_equities = [equity_df.loc[equity_df.index >= t]['equity'].iloc[0] for t in trade_times]
                plt.scatter(trade_times, trade_equities, color=colors[timeframe], 
                           label=f'{timeframe.capitalize()} Trades', alpha=0.7)
        
        plt.title('Portfolio Equity with Multi-Timeframe Trade Markers')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(self.backtest_dir, 'equity_curve_with_trades.png'))
        print(f"Equity curve plot saved to {os.path.join(self.backtest_dir, 'equity_curve_with_trades.png')}")
        
    def run_analysis(self):
        """Run the complete analysis workflow"""
        if self.load_backtest_data():
            self.classify_trades_by_timeframe()
            self.analyze_performance()
            self.plot_equity_curve()
            
if __name__ == "__main__":
    # Example usage
    analyzer = PerformanceAnalyzer("backtests")
    analyzer.run_analysis()
