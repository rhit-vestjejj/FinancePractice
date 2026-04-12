import yfinance as yf
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def compute_hedge_ratio(spot_ticker: str, futures_ticker: str, period: str = "1y") -> float:
    # pull data
    # compute returns
    # compute correlation and stds
    # return h = rho * (sigma_s / sigma_f)
    spot_prices = yf.download(spot_ticker, period=period, progress=False)['Close'].squeeze().dropna()
    futures_prices = yf.download(futures_ticker, period=period, progress=False)['Close'].squeeze().dropna()

    spots_returns = spot_prices.pct_change().dropna()
    futures_returns = futures_prices.pct_change().dropna()
    spots_returns, futures_returns = spots_returns.align(futures_returns, join='inner')


    rho = np.corrcoef(spots_returns, futures_returns)[0, 1]
    sigma_s = np.std(spots_returns)
    sigma_f = np.std(futures_returns)

    h = rho * (sigma_s / sigma_f)

    return h

def contracts_needed(h: float, spot_position: int, spot_price: float, 
                     futures_price: float, contract_size: int) -> int:
    # scale hedge ratio to actual number of contracts

    spot_position_value = spot_position * spot_price
    futures_contracts_value = futures_price * contract_size
    
    N = h * (spot_position_value / futures_contracts_value)

    return N

def _scalar(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


h = compute_hedge_ratio("SPY", "ES=F")
spy = yf.download("SPY", period="1d", progress=False)
es = yf.download("ES=F", period="1d", progress=False)
spot_price = _scalar(spy['Close'].iloc[-1])
futures_price = _scalar(es['Close'].iloc[-1])

N = contracts_needed(h, 10000, spot_price, futures_price, 50)
print(f"Hedge Ratio: ", h)
print(f"Contracts Needed: ", N)