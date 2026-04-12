import pandas as pd
import numpy as np

class Performance:
    def __init__(self, history: list[dict], fill_history: list[dict] = None):
        self.df = pd.DataFrame(history)
        self.df.set_index('timestamp', inplace=True)
        self.fills = pd.DataFrame(fill_history) if fill_history else pd.DataFrame()
    
    def total_return(self) -> float:
        initial = self.df['total_value'].iloc[0]
        final = self.df['total_value'].iloc[-1]
        return (final / initial - 1) * 100
    
    def sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        daily_returns = self.df['total_value'].pct_change().dropna()
        excess_returns = daily_returns - (risk_free_rate / 252)
        if excess_returns.std() == 0:
            return 0.0
        return float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())
    
    def max_drawdown(self) -> float:
        rolling_max = self.df['total_value'].cummax()
        drawdown = (self.df['total_value'] - rolling_max) / rolling_max
        return float(drawdown.min() * 100)
    
    def buy_and_hold_return(self, data_handler, ticker: str) -> float:
        df = data_handler.data[ticker]
        initial_price = _scalar(df['Close'].iloc[0])
        final_price = _scalar(df['Close'].iloc[-1])
        return (final_price / initial_price - 1) * 100

    def report(self, data_handler=None, ticker: str = None):
        print(f"Total Return:   {self.total_return():.2f}%")
        print(f"Sharpe Ratio:   {self.sharpe_ratio():.3f}")
        print(f"Max Drawdown:   {self.max_drawdown():.2f}%")
        if data_handler and ticker:
            bh = self.buy_and_hold_return(data_handler, ticker)
            print(f"Buy & Hold:      {bh:.2f}%")
            print(f"Alpha:           {self.total_return() - bh:.2f}%")
            print(f"Impl Shortfall:  {self.implementation_shortfall():.2f} bps")

    def implementation_shortfall(self) -> float:
        if self.fills.empty:
            return 0.0
        
        buys = self.fills[self.fills['direction'] == 'BUY']
        if buys.empty:
            return 0.0
        
        avg_fill = (buys['fill_price'] * buys['quantity']).sum() / buys['quantity'].sum()
        avg_decision = (buys['decision_price'] * buys['quantity']).sum() / buys['quantity'].sum()
        
        return (avg_fill - avg_decision) / avg_decision * 10000  # basis points

def _scalar(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)