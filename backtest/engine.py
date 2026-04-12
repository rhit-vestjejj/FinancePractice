from queue import Queue
from .data.data import DataHandler, MarketEvent
from .portfolio.portfolio import Portfolio, OrderEvent, FillEvent
from .execution.execution import ExecutionHandler
from .strategy.moving_average import MovingAverageCrossover
from .performance.performance import Performance
import math

class Engine:
    def __init__(self, 
                 tickers: list[str], 
                 start: str, end: str, 
                 initial_capital: float = 100000.0, 
                 short_window: float = 5, 
                 long_window: float = 30,
                 stop_loss_pct: float = 0.05,
                 position_pct: float = 0.5,
                 strategy: MovingAverageCrossover = None
                 ):
        self.tickers = tickers
        self.events = Queue()
        self.data_handler = DataHandler(tickers, start, end)
        self.portfolio = Portfolio(initial_capital)
        self.execution = ExecutionHandler()
        self.strategy = MovingAverageCrossover(tickers, 
                                               algo="VWAP", 
                                               execution_bars=5, 
                                               short_window=short_window, 
                                               long_window=long_window, 
                                               stop_loss_pct=stop_loss_pct,
                                               position_pct=position_pct)
        self.current_bars = {}
        self.peak_value = initial_capital
        self.max_drawdown_pct = 0.20  # stop trading if portfolio drops 20% from peak
        self.trading_halted = False
        self.halted_value = None
        self.recovery_pct = 0.10  # resume after 10% recovery from halt point
        self.sharpe = 0

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
        
        # self.portfolio.summary()
        # print("---")
        perf = Performance(self.portfolio.history, self.portfolio.fill_history)
        self.shape = perf.sharpe_ratio()
        self.sharpe = perf.sharpe_ratio()
        # perf.report(self.data_handler, self.tickers[0])
    
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
        self.check_circuit_breaker()
        if not self.trading_halted:
            order = self.strategy.on_market(event, self.portfolio.total_value)
            if order:
                self.events.put(order)

    def _on_order(self, event: OrderEvent):
        bar = self.current_bars[event.ticker]
        self.execution.execute_order(event, bar)
    
    def _on_fill(self, event: FillEvent):
        self.portfolio.on_fill(event)

    def check_circuit_breaker(self):
        current_value = self.portfolio.total_value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (current_value - self.peak_value) / self.peak_value
        if drawdown < -self.max_drawdown_pct and not self.trading_halted:
            self.trading_halted = True
            self.halted_value = current_value
            # generate sell orders for all positions
            for ticker, quantity in self.portfolio.positions.items():
                if quantity > 0:
                    order = OrderEvent(
                        ticker=ticker,
                        direction='SELL',
                        quantity=quantity,
                        execution_algo='MARKET',
                        execution_bars=1
                    )
                    self.events.put(order)
            # print(f"Trading paused at {self.portfolio.history[-1]['timestamp']}")


        if self.trading_halted and self.halted_value:
            recovery = (current_value - self.halted_value) / self.halted_value
            if recovery > self.recovery_pct:
                self.trading_halted = False
                self.halted_value = None
                # print(f"Trading resumed at {self.portfolio.history[-1]['timestamp']}")


if __name__ == "__main__":
    engine = Engine(
        tickers=["SPY"],
        start="2010-01-01",
        end="2015-01-01",
        initial_capital=100000.0
    )
    engine.run()
