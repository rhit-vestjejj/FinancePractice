import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

from ..options.black_scholes import bs_call, bs_put


# ── Black-Scholes helpers (needed internally) ──────────────────────────────

def _bs_call_price(S, K, T, r, sigma):
    """
    Compute the Black-Scholes theoretical call price.

    Parameters
    ----------
    S     : float - current stock price
    K     : float - strike price
    T     : float - time to expiration in years
    r     : float - continuously compounded risk-free rate
    sigma : float - annualized volatility

    Returns
    -------
    float - call option price
    """
    return bs_call(S, K, r, T, sigma)


def _bs_put_price(S, K, T, r, sigma):
    """
    Compute the Black-Scholes theoretical put price.

    Parameters
    ----------
    Same as _bs_call_price.

    Returns
    -------
    float - put option price
    """
    return bs_put(S, K, r, T, sigma)


# ── Core implied vol solver ────────────────────────────────────────────────

def implied_volatility(market_price, S, K, T, r, option_type='call',
                       tol=1e-6, max_iter=500):
    """
    Invert the Black-Scholes formula to recover implied volatility using
    Brent's method (scipy.optimize.brentq).

    Parameters
    ----------
    market_price : float - observed market option price
    S            : float - current underlying price
    K            : float - strike
    T            : float - time to expiration in years
    r            : float - risk-free rate
    option_type  : str   - 'call' or 'put'
    tol          : float - convergence tolerance
    max_iter     : int   - iteration cap

    Returns
    -------
    float - implied volatility (annualized), or np.nan if no solution found
    """

    if option_type == 'call':
        try:
            result = brentq(lambda sigma: _bs_call_price(S, K, T, r, sigma) - market_price, a=1e-6, b=10, xtol=tol, maxiter=max_iter)
        except Exception as e:
            print(f"K={K}, T={T}, market_price={market_price}, error={e}")
            return np.nan
    elif option_type == 'put':
        try:
            result = brentq(lambda sigma: _bs_put_price(S, K, T, r, sigma) - market_price, a=1e-6, b=10, xtol=tol, maxiter=max_iter)
        except Exception as e:
            print(f"K={K}, T={T}, market_price={market_price}, error={e}")
            return np.nan
    else:
        result = np.nan
    
    return float(result)


# ── Fetch live option chain ────────────────────────────────────────────────

def fetch_option_chain(ticker, expiry=None):
    """
    Pull a live option chain via yfinance for the given ticker and nearest
    (or specified) expiry; return calls and puts as separate DataFrames with
    columns [strike, lastPrice, bid, ask, impliedVolatility].

    Parameters
    ----------
    ticker : str  - e.g. 'SPY'
    expiry : str  - 'YYYY-MM-DD', or None for the nearest available expiry

    Returns
    -------
    (calls_df, puts_df, S, expiry_str)
        calls_df, puts_df : pd.DataFrame
        S                 : float - current spot price
        expiry_str        : str   - expiry date used
    """

    ticker = yf.Ticker(ticker)

    S = ticker.fast_info['last_price']

    expiries = ticker.options

    if expiry is None:
        expiry = next(e for e in expiries if (pd.to_datetime(e) - pd.Timestamp.today()).days > 7) 
    
    chain = ticker.option_chain(expiry)

    calls = chain.calls
    puts = chain.puts

    cols = ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']
    calls = calls[cols]
    puts = puts[cols]

    return (calls, puts, S, expiry)



# ── Test 1: _bs_call_price ─────────────────────────────────────────────
# Hull Example 13.1 (pg 296 6th ed):
# S=42, K=40, T=0.5, r=0.10, sigma=0.20
# Expected call price ≈ 4.76
call_price = _bs_call_price(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
print(f"BS Call Price: {call_price:.4f}  (expected ~4.76)")

# ── Test 2: implied_volatility round-trip ──────────────────────────────
# If the market price IS the BS price, recovered IV should equal input sigma
# Use same params: S=42, K=40, T=0.5, r=0.10, sigma=0.20
# Feed call_price back in — expected IV ≈ 0.20
iv_recovered = implied_volatility(
    market_price=call_price, S=42, K=40, T=0.5, r=0.10, option_type='call'
)
print(f"Round-trip IV: {iv_recovered:.6f}  (expected ~0.200000)")

# ── Test 3: IV monotonicity check ─────────────────────────────────────
# Higher market price -> higher IV (for fixed S, K, T, r)
# Use prices: 4.0, 5.0, 6.5 with S=42, K=40, T=0.5, r=0.10
# Expected: iv_a < iv_b < iv_c  (all > 0)
iv_a = implied_volatility(4.0,  S=42, K=40, T=0.5, r=0.10)
iv_b = implied_volatility(5.0,  S=42, K=40, T=0.5, r=0.10)
iv_c = implied_volatility(6.5,  S=42, K=40, T=0.5, r=0.10)
print(f"IV monotonicity: {iv_a:.4f} < {iv_b:.4f} < {iv_c:.4f}  (expected True)")
print(f"  Monotone: {iv_a < iv_b < iv_c}")
