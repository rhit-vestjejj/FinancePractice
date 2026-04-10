import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketEvent:
    ticker: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

class DataHandler:
    def __init__(self, tickers: list[str], start: str, end: str):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = {}
        self.current_index = 0
        self._load_data()
    
    def _load_data(self):
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start, end=self.end)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)
            self.data[ticker] = df
    
    def get_next_bars(self) -> Optional[list[MarketEvent]]:
        events = []
        
        # get the first ticker to check index bounds
        first = self.data[self.tickers[0]]
        
        if self.current_index >= len(first):
            return None
        
        for ticker in self.tickers:
            df = self.data[ticker]
            row = df.iloc[self.current_index]
            event = MarketEvent(
                ticker=ticker,
                timestamp=df.index[self.current_index],
                open=_scalar(row['Open']),
                high=_scalar(row['High']),
                low=_scalar(row['Low']),
                close=_scalar(row['Close']),
                volume=_scalar(row['Volume'])
            )
            events.append(event)
        
        self.current_index += 1
        return events
    
    def has_more_data(self) -> bool:
        first = self.data[self.tickers[0]]
        return self.current_index < len(first)
    
def _scalar(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)