import numpy as np
import math
import yfinance as yf
import pandas as pd

def forward_price(S: float, r: float, T: float) -> float:
    """
    Forward price for non-dividend paying stock
    S = spot price
    r = risk free rate (annual, continuous compounding)
    T = time to maturity in years
    """
    return S * np.exp(r * T)

def forward_price_dividend(S: float, r: float, T: float, I: float) -> float:
    """
    Forward price for dividend paying stock
    I = present value of dividends during contract life
    """
    return (S - I) * np.exp(r * T)

def hedge_ratio(rho: float, sigma_s: float, sigma_f: float) -> float:
    """
    Minimum variance hedge ratio
    rho = correlation between spot and futures returns
    sigma_s = std of spot returns
    sigma_f = std of futures returns
    """
    return rho * (sigma_s / sigma_f)

def _scalar(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)

print("forward pirce: ", forward_price(100, 0.05, 1))
print("forward price divident: ", forward_price_dividend(100, 0.05, 1, 3))
print("hedge:ratio: ", hedge_ratio(0.8, 0.02, 0.015))

spy = yf.download("SPY", period="1d", progress=False)
S = _scalar(spy['Close'].iloc[-1])
r = 0.05  # approximate current risk free rate
T = 0.25  # 3 months

F = forward_price(S, r, T)
print(f"SPY Spot: {S:.2f}")
print(f"Theoretical Forward: {F:.2f}")

es = yf.download("ES=F", period="1d", progress=False)
print("ES: ", _scalar(es['Close'].iloc[-1]))

# SPY dividend yield ~1.3%
dividend_yield = 0.013
I = S * dividend_yield * T  # approximate PV of dividends over 3 months
F_div = forward_price_dividend(S, r, T, I)
print(f"Forward with dividends: {F_div:.2f}")