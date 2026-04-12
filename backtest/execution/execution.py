from dataclasses import dataclass
from pathlib import Path
from ..data.data import MarketEvent
from ..portfolio.portfolio import OrderEvent, FillEvent
import pandas as pd
from collections import defaultdict, deque

@dataclass
class PendingOrder:
    ticker: str
    direction: str
    remaining_quantity: int
    total_quantity: int
    bars_remaining: int
    algo: str
    decision_price: float

class ExecutionHandler:
    def __init__(self, commission_rate: float = 0.001):
        # 0.1% commission per trade, realistic for retail
        self.commission_rate = commission_rate
        self.volume_buffer = defaultdict(lambda: deque(maxlen=20))
        self.pending_orders: list[PendingOrder] = []

    def execute_order(self, order: OrderEvent, bar: MarketEvent) -> None:
        self.pending_orders.append(
            PendingOrder(
                ticker=order.ticker,
                direction=order.direction,
                remaining_quantity=order.quantity,
                total_quantity=order.quantity,
                bars_remaining=1 if order.execution_algo == "MARKET" else order.execution_bars,
                algo=order.execution_algo,
                decision_price=order.decision_price
            )
        )
        
    def process_pending(self, bar: MarketEvent):
        pending: list[FillEvent] = []
        fill_price = bar.open

        for pending_order in self.pending_orders:
            if pending_order.bars_remaining == 0:
                continue

            if pending_order.algo == "TWAP":
                slice = pending_order.remaining_quantity / pending_order.bars_remaining
                commission = round(fill_price * slice * self.commission_rate, 2)

                pending.append(FillEvent(
                    ticker=pending_order.ticker,
                    direction=pending_order.direction,
                    quantity=slice,
                    fill_price=fill_price,
                    commission=commission,
                    decision_price=pending_order.decision_price
                ))

                pending_order.remaining_quantity -= slice
                pending_order.bars_remaining -= 1

            if pending_order.algo == "VWAP":
                avg_volume = self.get_average_volume(pending_order.ticker)
                current_volume = bar.volume
                volume_weight = current_volume / avg_volume
                raw_slice = (pending_order.remaining_quantity / pending_order.bars_remaining ) * volume_weight
                slice = int(min(raw_slice, pending_order.remaining_quantity))
                commission = round(fill_price * slice * self.commission_rate, 2)

                pending.append(FillEvent(
                    ticker=pending_order.ticker,
                    direction=pending_order.direction,
                    quantity=slice,
                    fill_price=fill_price,
                    commission=commission,
                    decision_price=pending_order.decision_price
                ))

                pending_order.remaining_quantity -= slice
                pending_order.bars_remaining -= 1

        self.pending_orders = [o for o in self.pending_orders if o.bars_remaining > 0]

        return pending



    def update_volume(self, bar: MarketEvent):
        self.volume_buffer[bar.ticker].append(bar.volume)

    def get_average_volume(self, ticker: str) -> float | None:
        volumes = self.volume_buffer[ticker]
        if not volumes:
            return None
        return sum(volumes) / len(volumes)
    
    def get_volume_history(self, ticker: str) -> list[float]:
        return list(self.volume_buffer[ticker])