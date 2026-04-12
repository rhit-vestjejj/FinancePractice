from dataclasses import dataclass
from ..data.data import MarketEvent

@dataclass
class OrderEvent:
    ticker: str
    direction: str  # 'BUY' or 'SELL'
    quantity: int
    execution_algo: str = 'MARKET'  
    execution_bars: int = 1

@dataclass  
class FillEvent:
    ticker: str
    direction: str
    quantity: int
    fill_price: float
    commission: float

class Portfolio:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}      # ticker -> quantity held
        self.holdings = {}       # ticker -> current market value
        self.total_value = initial_capital
        self.history = []        # track portfolio value over time
    
    def update_market(self, event: MarketEvent):
        # update current market value of each position
        if event.ticker in self.positions:
            quantity = self.positions[event.ticker]
            self.holdings[event.ticker] = quantity * event.close
        
        # recalculate total portfolio value
        self.total_value = self.cash + sum(self.holdings.values())
        
        # record snapshot
        self.history.append({
            'timestamp': event.timestamp,
            'cash': self.cash,
            'total_value': self.total_value
        })
    
    def on_fill(self, event: FillEvent):
        ticker = event.ticker
        
        # update position
        if ticker not in self.positions:
            self.positions[ticker] = 0
            
        if event.direction == 'BUY':
            cost = event.fill_price * event.quantity + event.commission
            if cost < self.cash:
                self.positions[ticker] += event.quantity
                self.cash -= (event.fill_price * event.quantity) + event.commission
                
        if event.direction == 'SELL':
            current = self.positions.get(ticker, 0)
            if event.quantity > current:
                event = FillEvent(ticker, 'SELL', current, event.fill_price, event.commission)
    
    def get_position(self, ticker: str) -> int:
        return self.positions.get(ticker, 0)
    
    def summary(self):
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value:     ${self.total_value:,.2f}")
        print(f"Cash:            ${self.cash:,.2f}")
        print(f"PnL:             ${self.total_value - self.initial_capital:,.2f}")
        print(f"Return:          {((self.total_value / self.initial_capital) - 1) * 100:.2f}%")
        print(f"Positions:       {self.positions}")