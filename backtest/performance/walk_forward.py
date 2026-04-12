from dateutil.relativedelta import relativedelta
from datetime import datetime
from itertools import product
from ..engine import Engine
from .performance import Performance
import math
import numpy as np

class WalkForward:
    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: str,
        test_window: int,  # months
        min_train: int,    # months
    ):
        self.tickers = tickers
        self.start = datetime.strptime(start, "%Y-%m-%d")
        self.end = datetime.strptime(end, "%Y-%m-%d")
        self.test_window = test_window
        self.min_train = min_train
    
    def generate_windows(self) -> list[tuple]:
        windows = []
        n = 0
        
        while True:
            train_start = self.start
            train_end = self.start + relativedelta(months=self.min_train + n * self.test_window)
            test_start = train_end
            test_end = test_start + relativedelta(months=self.test_window)
            
            if test_end > self.end:
                break
            
            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d")
            ))
            n += 1
        
        return windows
    
    def train(self, tr_start, tr_end):
        """
        param_grid = {
            'short_window': [5, 10,],
            'long_window':  [30, 50,],
            'stop_loss_pct': [0.05, 0.10],
            'position_pct': [0.50, 0.75]
        }
        """
        
        param_grid = {
            'short_window': [5, 10, 20],
            'long_window':  [30, 50, 100],
            'stop_loss_pct': [0.05, 0.10, 0.20],
            'position_pct': [0.50, 0.75, 0.95]
        }

        keys = param_grid.keys()
        values = param_grid.values()
        max_sharpe = -math.inf
        best_combo = None
        for combo in product(*values):
            params = dict(zip(keys, combo))
            # params is now {'short_window': 5, 'long_window': 30, ...}
            if params['short_window'] < params['long_window']:
                engine = Engine(
                    tickers=self.tickers,
                    start=tr_start,
                    end=tr_end,
                    initial_capital=100000.0,
                    short_window=params['short_window'],
                    long_window=params['long_window'],
                    stop_loss_pct=params['stop_loss_pct'],
                    position_pct=params['position_pct']
                )
                engine.run()
                
                if engine.sharpe > max_sharpe:
                    max_sharpe = engine.sharpe
                    best_combo = {
                        'short_window' : params['short_window'],
                        'long_window' : params['long_window'],
                        'stop_loss_pct' : params['stop_loss_pct'],
                        'position_pct' : params['position_pct']
                    }
        print("best sharpe: ", max_sharpe)
        print("best combo: ", best_combo)

        return best_combo
    
    def test(self, te_start, te_end, best_combo):
        engine = Engine(
                    tickers=self.tickers,
                    start=te_start,
                    end=te_end,
                    initial_capital=100000.0,
                    short_window=best_combo['short_window'],
                    long_window=best_combo['long_window'],
                    stop_loss_pct=best_combo['stop_loss_pct'],
                    position_pct=best_combo['position_pct']
                )

        engine.run()

        perf = Performance(engine.portfolio.history, engine.portfolio.fill_history)
        sharpe = perf.sharpe_ratio()
        alpha = perf.total_return() - perf.buy_and_hold_return(engine.data_handler, engine.tickers[0])
        is_bps = perf.implementation_shortfall()

        return sharpe, alpha, is_bps


    def run(self):
        windows = self.generate_windows()
        results = []
        
        for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            print(f"\nWindow {i+1}: Train {tr_start} -> {tr_end} | Test {te_start} -> {te_end}")
            best_params = self.train(tr_start, tr_end)
            sharpe, alpha, is_bps = self.test(te_start, te_end, best_params)
            results.append({'window': i+1, 'sharpe': sharpe, 'alpha': alpha, 'is_bps': is_bps})
            print(f"Test Sharpe: {sharpe:.3f} | Alpha: {alpha:.2f}% | IS: {is_bps:.2f} bps")
        
        print("\n=== WALK-FORWARD SUMMARY ===")
        sharpes = [r['sharpe'] for r in results]
        alphas = [r['alpha'] for r in results]
        print(f"Avg Test Sharpe:        {sum(sharpes)/len(sharpes):.3f}")
        print(f"Avg Test Alpha:         {sum(alphas)/len(alphas):.2f}%")
        print(f"Positive Alpha Windows: {sum(1 for a in alphas if a > 0)}/{len(results)}")
        print(f"Best Window Alpha:      {max(alphas):.2f}%")
        print(f"Worst Window Alpha:     {min(alphas):.2f}%")
        is_list = [r['is_bps'] for r in results]
        print(f"Avg Impl Shortfall: {sum(is_list)/len(is_list):.2f} bps") 


if __name__ == "__main__":
    wf = WalkForward(
        tickers=["SPY"],
        start="2010-01-01",
        end="2024-01-01",
        test_window=12,
        min_train=24
    )
    wf.run()