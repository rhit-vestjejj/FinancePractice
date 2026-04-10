from dataclasses import dataclass
from collections import defaultdict
from ..data.data import MarketEvent
from ..portfolio.portfolio import OrderEvent
import pandas as pd

class Strategy:
    def __init__(self, tickers: list[str], short_window: int = 20, long_window: int = 50, algo: str = "MARKET"):
        self.tickers = tickers
        self.short_window = short_window
        self.long_window = long_window
        # store price history per ticker
        self.prices = defaultdict(list)
        # track current position state per ticker
        self.invested = defaultdict(bool)
        self.algo = algo
    
    def on_market(self, event: MarketEvent) -> OrderEvent | None:
        ticker = event.ticker
        self.prices[ticker].append(event.close)
        
        # need at least long_window prices before trading
        if len(self.prices[ticker]) < self.long_window:
            return None
        
        # calculate moving averages
        short_ma = sum(self.prices[ticker][-self.short_window:]) / self.short_window
        long_ma = sum(self.prices[ticker][-self.long_window:]) / self.long_window
        
        # generate signals
        if short_ma > long_ma and not self.invested[ticker]:
            self.invested[ticker] = True
            return OrderEvent(
                ticker=ticker,
                direction='BUY',
                quantity=100,
                execution_algo=self.algo,
                execution_bars=20
            )
        
        elif short_ma < long_ma and self.invested[ticker]:
            self.invested[ticker] = False
            return OrderEvent(
                ticker=ticker,
                direction='BUY',
                quantity=100,
                execution_algo=self.algo,
                execution_bars=20
            )
                    
        return None