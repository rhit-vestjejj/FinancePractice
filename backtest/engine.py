from queue import Queue
from .data.data import DataHandler, MarketEvent
from .portfolio.portfolio import Portfolio, OrderEvent, FillEvent
from .execution.execution import ExecutionHandler
from .strategy.strategy import Strategy
from .performance.performance import Performance

class Engine:
    def __init__(self, tickers: list[str], start: str, end: str, initial_capital: float = 100000.0):
        self.tickers = tickers
        self.events = Queue()
        self.data_handler = DataHandler(tickers, start, end)
        self.portfolio = Portfolio(initial_capital)
        self.execution = ExecutionHandler()
        self.strategy = Strategy(tickers, algo="TWAP")
        self.current_bars = {}
    
    def run(self):
        while self.data_handler.has_more_data():
            bars = self.data_handler.get_next_bars()
            if bars:
                for bar in bars:
                    self.current_bars[bar.ticker] = bar
                    self.events.put(bar)
            
            while not self.events.empty():
                event = self.events.get()
                self._process_event(event)
        
        self.portfolio.summary()
        print("---")
        perf = Performance(self.portfolio.history)
        perf.report(self.data_handler, self.tickers[0])
    
    def _process_event(self, event):
        if isinstance(event, MarketEvent):
            self._on_market(event)
        elif isinstance(event, OrderEvent):
            self._on_order(event)
        elif isinstance(event, FillEvent):
            self._on_fill(event)
    
    def _on_market(self, event: MarketEvent):
        self.execution.update_volume(event)
        fills = self.execution.process_pending(event)
        for fill in fills:
            self.events.put(fill)
        self.portfolio.update_market(event)
        order = self.strategy.on_market(event)
        if order:
            self.events.put(order)

    def _on_order(self, event: OrderEvent):
        bar = self.current_bars[event.ticker]
        fill = self.execution.execute_order(event, bar)
        if fill is not None:
            self.events.put(fill)
    
    def _on_fill(self, event: FillEvent):
        self.portfolio.on_fill(event)


if __name__ == "__main__":
    engine = Engine(
        tickers=["SPY"],
        start="2020-01-01",
        end="2024-01-01",
        initial_capital=100000.0
    )
    engine.run()