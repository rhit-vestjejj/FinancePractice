from .base import BaseStrategy
from collections import defaultdict
from ..data.data import MarketEvent
from ..portfolio.portfolio import OrderEvent

class MovingAverageCrossover(BaseStrategy):
    def __init__(self, tickers, short_window=20, long_window=50, algo="MARKET", execution_bars=1, stop_loss_pct: float = 0.05, position_pct: float = 0.95 ):
        super().__init__(tickers, algo, execution_bars, stop_loss_pct, position_pct)
        self.short_window = short_window
        self.long_window = long_window
        self.prices = defaultdict(list)
        self.invested = defaultdict(bool)
    
    def generate_signal(self, event: MarketEvent, portfolio_value: float) -> OrderEvent | None:
        ticker = event.ticker
        self.prices[ticker].append(event.close)

        if len(self.prices[ticker]) < self.long_window:
            return None
        
        quantity = int((portfolio_value * self.position_pct) / event.close)
        
        short_ma = sum(self.prices[ticker][-self.short_window:]) / self.short_window
        long_ma = sum(self.prices[ticker][-self.long_window:]) / self.long_window
        
        if short_ma > long_ma and not self.invested[ticker]:
            self.entry_prices[ticker] = event.close
            self.current_quantities[ticker] = quantity
            self.invested[ticker] = True
            return self.create_order(ticker, 'BUY', quantity, event.close)
        
        elif short_ma < long_ma and self.invested[ticker]:
            self.entry_prices.pop(ticker, None)
            self.current_quantities.pop(ticker, None)
            self.invested[ticker] = False
            return self.create_order(ticker, 'SELL', quantity, event.close)
        
        return None
    
    def on_stop_loss(self, ticker: str):
        self.invested[ticker] = False