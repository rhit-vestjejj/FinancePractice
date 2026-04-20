import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Optional


# ─────────────────────────────────────────────
# Hull Ch 15 — The Greek Letters
# ─────────────────────────────────────────────
# Each Greek measures sensitivity of option price V
# to one underlying parameter, all else held equal.
# Notation follows Hull exactly.
# ─────────────────────────────────────────────


def _d1_d2(S: float, K: float, T: float, r: float,
           sigma: float, q: float = 0.0) -> tuple[float, float]:
    """
    Compute d1 and d2 from the Black-Scholes formula.

    Parameters
    ----------
    S     : current stock price
    K     : strike price
    T     : time to expiration in years
    r     : risk-free rate (continuously compounded)
    sigma : volatility of S
    q     : continuous dividend yield (default 0)

    Returns
    -------
    (d1, d2) as a tuple of floats
    """

    d_1 = (np.log(S / K) + (((r - q) + (sigma ** 2 / 2)) * T)) / (sigma * (T ** (1/2)))

    d_2 = d_1  - sigma * (T ** (1/2))
    
    return (d_1, d_2)


def delta(S: float, K: float, T: float, r: float,
          sigma: float, option_type: str = 'call',
          q: float = 0.0) -> float:
    """
    Compute the Black-Scholes delta of a European option.

    Delta = dV/dS. For a call (with dividends): e^{-qT} N(d1).
    For a put: e^{-qT} (N(d1) - 1).

    Parameters
    ----------
    S           : current stock price
    K           : strike price
    T           : time to expiration in years
    r           : risk-free rate
    sigma       : volatility
    option_type : 'call' or 'put'
    q           : continuous dividend yield

    Returns
    -------
    float : delta value
    """

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    
    if option_type == 'call':
        delt = np.exp(-q * T) * norm.cdf(d1)
    elif option_type == 'put':
        delt = np.exp(-q * T) * (norm.cdf(d1) - 1)
    else:
        delt = 0

    return float(delt)


def gamma(S: float, K: float, T: float, r: float,
          sigma: float, q: float = 0.0) -> float:
    """
    Compute the Black-Scholes gamma of a European option.

    Gamma = d^2V/dS^2 = e^{-qT} * N'(d1) / (S * sigma * sqrt(T)).
    Same for calls and puts.

    Parameters
    ----------
    S     : current stock price
    K     : strike price
    T     : time to expiration in years
    r     : risk-free rate
    sigma : volatility
    q     : continuous dividend yield

    Returns
    -------
    float : gamma value
    """

    d1, d2 = _d1_d2(S, K, T, r , sigma, q)

    gamma = N_prime(d1) * np.exp(-q * T) / (S * sigma * (T ** (1/2)))

    return gamma


def theta(S: float, K: float, T: float, r: float,
          sigma: float, option_type: str = 'call',
          q: float = 0.0) -> float:
    """
    Compute the Black-Scholes theta of a European option (per calendar day).

    Theta = dV/dt. Hull defines it as the rate of change of V with
    respect to the passage of time (so typically negative for long options).
    Divide by 365 to get per-calendar-day decay.

    For a call:
        theta = [-S*N'(d1)*sigma*e^{-qT}/(2*sqrt(T))
                 + q*S*N(d1)*e^{-qT}
                 - r*K*e^{-rT}*N(d2)] / 365

    For a put: analogous with N(-d1), N(-d2).

    Parameters
    ----------
    S           : current stock price
    K           : strike price
    T           : time to expiration in years
    r           : risk-free rate
    sigma       : volatility
    option_type : 'call' or 'put'
    q           : continuous dividend yield

    Returns
    -------
    float : theta per calendar day
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option_type == 'call':
        theta = -S * N_prime(d1) * sigma * np.exp(-q * T) / (2 * (T ** (1/2))) + q * S * norm.cdf(d1) * np.exp(-q * T) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        theta = -S * N_prime(d1) * sigma * np.exp(-q * T) / (2 * (T ** (1/2))) + q * S * norm.cdf(-d1) * np.exp(-q * T) - r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        theta = 0
    
    return float(theta)

def N_prime(x):
    return (1 / ((2 * np.pi) ** (1/2))) * np.exp(-(x ** 2) / 2)

def vega(S: float, K: float, T: float, r: float,
         sigma: float, q: float = 0.0) -> float:
    """
    Compute the Black-Scholes vega of a European option.

    Vega = dV/d(sigma) = S * sqrt(T) * N'(d1) * e^{-qT}.
    Same for calls and puts.
    Hull quotes vega per 1% change in vol — divide by 100.

    Parameters
    ----------
    S     : current stock price
    K     : strike price
    T     : time to expiration in years
    r     : risk-free rate
    sigma : volatility
    q     : continuous dividend yield

    Returns
    -------
    float : vega (per 1% change in vol)
    """

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    vega = S * (T ** (1/2)) * N_prime(d1) * np.exp(-q * T) / 100

    return vega


def rho(S: float, K: float, T: float, r: float,
        sigma: float, option_type: str = 'call',
        q: float = 0.0) -> float:
    """
    Compute the Black-Scholes rho of a European option.

    Rho = dV/dr.
    For a call: K * T * e^{-rT} * N(d2).
    For a put:  -K * T * e^{-rT} * N(-d2).
    Hull quotes rho per 1% change in r — divide by 100.

    Parameters
    ----------
    S           : current stock price
    K           : strike price
    T           : time to expiration in years
    r           : risk-free rate
    sigma       : volatility
    option_type : 'call' or 'put'
    q           : continuous dividend yield

    Returns
    -------
    float : rho (per 1% change in r)
    """

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        rho = 0

    return rho / 100



def delta_hedge_pnl(ticker: str = 'AAPL',
                    K: float = 150.0,
                    T: float = 0.25,
                    r: float = 0.05,
                    sigma: float = 0.25,
                    n_shares: int = 100) -> pd.DataFrame:
    """
    Simulate a delta-neutral hedge over a short historical window.

    Fetches recent daily prices for `ticker`, computes the BS call delta
    each day, computes the required hedge position (shares to short),
    and tracks daily P&L from the hedged portfolio.

    The hedged portfolio is: long 1 call (on n_shares) + short delta*n_shares shares.

    Parameters
    ----------
    ticker   : stock ticker
    K        : strike price
    T        : initial time to expiry (years); decrements by 1/252 each day
    r        : risk-free rate
    sigma    : assumed constant volatility
    n_shares : number of shares underlying one contract

    Returns
    -------
    pd.DataFrame with columns:
        Date, S, delta, hedge_shares, option_pnl, hedge_pnl, total_pnl
    """
    data = yf.download(ticker, period='3mo', interval='1d', auto_adjust=False, progress=False)

    if data.empty:
        raise ValueError(f"No price data returned for {ticker}")

    close_data = data['Close']
    if isinstance(close_data, pd.DataFrame):
        if ticker in close_data.columns:
            S_series = close_data[ticker].dropna()
        else:
            S_series = close_data.iloc[:, 0].dropna()
    else:
        S_series = close_data.dropna()

    rows = []
    prev_S = None
    prev_delta = None
    t = T

    for date, S in S_series.items():
        S = float(S)
        d = delta(S, K, t, r, sigma, 'call')
        hedge_shares = -d * n_shares

        if prev_S is None:
            option_pnl = 0.0
            hedge_pnl = 0.0
        else:
            dS = S - prev_S
            option_pnl = prev_delta * dS * n_shares
            hedge_pnl = (-prev_delta) * dS

        total_pnl = option_pnl + hedge_pnl

        rows.append({
            'Date': date,
            'S': S,
            'delta': d,
            'hedge_shares': hedge_shares,
            'option_pnl': option_pnl,
            'hedge_pnl': hedge_pnl,
            'total_pnl': total_pnl
        })

        prev_S = S
        prev_delta = d
        t = max(t - 1 / 252, 1e-6)

    return pd.DataFrame(rows)


def greek_surface(greek_fn,
                  S: float = 100.0,
                  r: float = 0.05,
                  sigma: float = 0.25,
                  q: float = 0.0,
                  option_type: str = 'call') -> pd.DataFrame:
    """
    Compute a Greek across a grid of (K, T) values and return as a DataFrame.

    Grid: K in [0.7*S, 1.3*S] with 20 steps; T in [0.05, 1.0] with 20 steps.

    Parameters
    ----------
    greek_fn    : one of delta, gamma, theta, vega, rho
    S           : current stock price
    r           : risk-free rate
    sigma       : volatility
    q           : continuous dividend yield
    option_type : 'call' or 'put'

    Returns
    -------
    pd.DataFrame indexed by K (rows) and T (columns)
    """
    K_grid = np.linspace(0.7 * S, 1.3 * S, 20)
    T_grid = np.linspace(0.05, 1.0, 20)
    
    surface = []

    for K in K_grid:
        row = []
        for T in T_grid:
            if greek_fn.__name__ in ['delta', 'theta', 'rho']:
                value = greek_fn(S, K, T, r, sigma, option_type, q)
            else:
                value = greek_fn(S, K, T, r, sigma, q)
            row.append(value)
        surface.append(row)

    df = pd.DataFrame(surface, index=K_grid, columns=T_grid)
    df.index.name = 'K'
    df.columns.name = 'T'

    return df


# Reference parameters from Hull Example 19.1
# S=49, K=50, r=5%, sigma=20%, T=20 weeks (~0.3846 yr), q=0
S, K, T, r, sigma = 49.0, 50.0, 20/52, 0.05, 0.20

# --- Test _d1_d2 ---
d1, d2 = _d1_d2(S, K, T, r, sigma)
print(f"d1 = {d1:.4f}  (expected ~  0.0542)")
print(f"d2 = {d2:.4f}  (expected ~ -0.0699)")

# --- Test delta ---
c_delta = delta(S, K, T, r, sigma, 'call')
p_delta = delta(S, K, T, r, sigma, 'put')
print(f"\nCall delta = {c_delta:.4f}  (expected ~ 0.5216)")
print(f"Put  delta = {p_delta:.4f}  (expected ~ -0.4784)")
# Sanity check: call_delta - put_delta should equal 1.0 (put-call parity on delta)
print(f"Delta parity check (call - put): {c_delta - p_delta:.4f}  (expected 1.0000)")

# --- Test gamma ---
g = gamma(S, K, T, r, sigma)
print(g)
print(f"\nGamma = {g:.4f}  (expected ~ 0.0655)")

# --- Test theta ---
c_theta = theta(S, K, T, r, sigma, 'call')
p_theta = theta(S, K, T, r, sigma, 'put')
print(f"\nCall theta = {c_theta:.4f}  (expected ~ -0.0141 per day)")
print(f"Put  theta = {p_theta:.4f}  (expected ~ -0.0077 per day)")

# --- Test vega ---
v = vega(S, K, T, r, sigma)
print(f"\nVega = {v:.4f}  (expected ~ 0.1217 per 1% vol)")

# --- Test rho ---
c_rho = rho(S, K, T, r, sigma, 'call')
p_rho = rho(S, K, T, r, sigma, 'put')
print(f"\nCall rho = {c_rho:.4f}  (expected ~  0.0891 per 1% rate)")
print(f"Put  rho = {p_rho:.4f}  (expected ~ -0.0982 per 1% rate)")

# --- Test delta_hedge_pnl ---
print("\n--- Delta Hedge Simulation (AAPL) ---")
df = delta_hedge_pnl('AAPL', K=150.0, T=0.25, r=0.05, sigma=0.25)
print(df.to_string(index=False))

# --- Test greek_surface ---
print("\n--- Gamma Surface (sample corner values) ---")
surf = greek_surface(gamma, S=100.0, r=0.05, sigma=0.25)
print(surf.iloc[[0, -1], [0, -1]])  # corners of the grid