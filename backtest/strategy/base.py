from abc import ABC, abstractmethod
from ..data.data import MarketEvent
from ..portfolio.portfolio import OrderEvent

class BaseStrategy(ABC):
    def __init__(self, tickers: list[str], algo: str = "MARKET", execution_bars: int = 1, stop_loss_pct: float = 0.05, position_pct: float = 0.95):
        self.tickers = tickers
        self.algo = algo
        self.execution_bars = execution_bars
        self.stop_loss_pct = stop_loss_pct
        self.position_pct = position_pct
        self.entry_prices = {}
        self.current_quantities = {}
    
    @abstractmethod
    def generate_signal(self, event: MarketEvent, portfolio_value: float) -> OrderEvent | None:
        pass
    
    def on_market(self, event: MarketEvent, portfolio_value: float) -> OrderEvent | None:
        # stop loss runs first, always, for every strategy
        stop_order = self.check_stop_loss(event)
        if stop_order:
            return stop_order
        
        # then delegate to the strategy's specific logic
        return self.generate_signal(event, portfolio_value)

    def create_order(self, ticker: str, direction: str, quantity: int, decision_price: float) -> OrderEvent:
        return OrderEvent(
            ticker=ticker,
            direction=direction,
            quantity=quantity,
            execution_algo=self.algo,
            execution_bars=self.execution_bars,
            decision_price=decision_price
        )
    
    def check_stop_loss(self, event: MarketEvent) -> OrderEvent | None:
        ticker = event.ticker
        if ticker not in self.entry_prices:
            return None
        
        loss = (event.close - self.entry_prices[ticker]) / self.entry_prices[ticker]
        if loss < -self.stop_loss_pct:
            decision_price = self.entry_prices.get(ticker, 0.0)
            quantity = self.current_quantities.pop(ticker, 0)
            self.entry_prices.pop(ticker)
            self.on_stop_loss(ticker)
            return self.create_order(ticker, 'SELL', quantity, decision_price)
                    
        return None
    
    # in base.py
    def on_stop_loss(self, ticker: str):
        pass  # base does nothing, subclasses override if needed